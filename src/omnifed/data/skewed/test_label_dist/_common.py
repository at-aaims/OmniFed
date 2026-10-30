# Copyright (c) 2026, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared helpers for the Caltech-256 label-skew tests.

Each ``test_*.py`` here pairs with one ``conf/datamodule/skewed-data-sim/cal256-*.yaml``.
It builds by hand what Hydra builds from that yaml (same datasets, template
loaders, and partitioner, then the same ``build_skewed_datamodule`` call),
runs the shared checks in :class:`SkewSuite`, and adds scheme-specific ones.
The feature-skew tests in ``test_feature_dist/`` reuse these helpers and checks.

Run from the repo root (``OMNIFED_DATA_DIR`` holds ``Caltech256/{train,test}``)::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets python -m pytest src/omnifed/data/skewed/test_label_dist -v

Without ``OMNIFED_DATA_DIR`` only the config-composition tests run (Hydra only,
no images); everything else is skipped.
"""

from __future__ import annotations

import contextlib
import functools
import os
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple
from unittest import mock

import numpy as np
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision import datasets, transforms

from src.omnifed.data.skewed.base import SkewedPartitioner, build_skewed_datamodule

# ── settings mirrored from cal256-common.yaml (+ conf/datamodule/{base,dataloader}.yaml) ──

DATA_ROOT = Path(os.environ.get("OMNIFED_DATA_DIR", "")) / "Caltech256"
HAVE_DATA = (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "test").is_dir()
CONF_DIR = Path(__file__).resolve().parents[5] / "conf"  # test_label_dist/ skewed/ data/ omnifed/ src/ <repo>
CONFIG_GROUP = "skewed-data-sim"
BUILDER_TARGET = "src.omnifed.data.skewed.base.build_skewed_datamodule"

NUM_CLIENTS = 4  # simulated trainers; the runners set this via env in real jobs
NUM_CLASSES = 257
TRAIN_BATCH, EVAL_BATCH = 32, 64
LOADER_KW = dict(num_workers=2, pin_memory=True, persistent_workers=True)  # dataloader.yaml
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

needs_data = unittest.skipUnless(
    HAVE_DATA, "set OMNIFED_DATA_DIR to the folder containing Caltech256/{train,test}"
)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


@functools.lru_cache(maxsize=1)
def load_datasets() -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Scan the image folders once per process (shared by every test file)."""
    train = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=train_transform())
    test = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=eval_transform())
    return train, test


def build_datamodule(partitioner: SkewedPartitioner, train_set=None, eval_set=None):
    """What Hydra does for a ``cal256-*.yaml``, written out by hand."""
    if train_set is None or eval_set is None:
        train_set, eval_set = load_datasets()
    # Hydra first builds full-dataset loaders from `train:` / `eval:` ...
    train_template = DataLoader(train_set, batch_size=TRAIN_BATCH, shuffle=True, **LOADER_KW)
    eval_loader = DataLoader(eval_set, batch_size=EVAL_BATCH, shuffle=False, **LOADER_KW)
    # ... then calls the yaml's _target_ with them and the instantiated partitioner.
    return build_skewed_datamodule(train=train_template, eval=eval_loader, partitioner=partitioner)


@contextlib.contextmanager
def fl_env(client_idx: int, num_clients: int = NUM_CLIENTS) -> Iterator[None]:
    """Environment the Slurm/Ray runners set for one rank (-1 = server)."""
    env = {
        "OMNIFED_NUM_FEDERATED_CLIENTS": str(num_clients),
        "OMNIFED_FEDERATED_CLIENT_INDEX": str(client_idx),
    }
    with mock.patch.dict(os.environ, env):  # restores the full environment on exit
        os.environ.pop("OMNIFED_PARTITION_SEED", None)  # use the default seed (0)
        yield


def compose_config(config_name: str) -> Any:
    """Compose ``base.yaml`` with ``datamodule=skewed-data-sim/<config_name>``."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(
            config_name="base", overrides=[f"datamodule={CONFIG_GROUP}/{config_name}"]
        )


def describe(make_partitioner: Callable[[], SkewedPartitioner]) -> None:
    """Build one client's datamodule as a trainer rank would, and print it."""
    cid = int(os.environ.get("OMNIFED_FEDERATED_CLIENT_INDEX", 0))
    with fl_env(cid):
        train_set, eval_set = load_datasets()
        dm = build_datamodule(make_partitioner(), train_set, eval_set)  # logs partition_id
        info = make_partitioner().summary(train_set)
    print(f"train loader: {type(dm.train).__name__}, {len(dm.train.dataset)} samples, "
          f"{len(dm.train)} batches of {dm.train.batch_size}")
    print(f"eval loader:  {type(dm.eval).__name__}, {len(dm.eval.dataset)} samples")
    print(f"partition_id={info['partition_id']} seed={info['seed']} sizes={info['sizes']} "
          f"classes/client={info['classes_per_client']} coverage={info['coverage']:.3f} "
          f"mean_hellinger={info['mean_hellinger']:.3f}")


class SkewSuite:
    """
    Checks shared by every skew scheme (label and feature). Mix into
    ``unittest.TestCase`` and set:

    * ``CONFIG``: yaml name in ``conf/datamodule/skewed-data-sim`` (no extension)
    * ``make_partitioner()``: the partitioner that yaml describes, built in Python
    * ``FULL_COVERAGE``: whether every training sample lands on some client
    """

    CONFIG: str
    FULL_COVERAGE: bool = True

    @staticmethod
    def make_partitioner() -> SkewedPartitioner:  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def setUpClass(cls) -> None:
        if HAVE_DATA:
            cls.train_set, cls.eval_set = load_datasets()
            cls.train_labels = np.asarray(cls.train_set.targets)

    # helpers ---------------------------------------------------------------

    def summary(self, partitioner: SkewedPartitioner | None = None) -> Dict[str, Any]:
        with fl_env(0):
            return (partitioner or self.make_partitioner()).summary(self.train_set)

    # config ------------------------------------------------------------------

    def test_config_composes(self) -> None:
        """The yaml resolves its defaults and describes make_partitioner(). No data needed."""
        try:
            from hydra.utils import instantiate
        except ImportError:
            self.skipTest("hydra not installed")
        dm = compose_config(self.CONFIG).datamodule

        self.assertEqual(dm._target_, BUILDER_TARGET)
        self.assertEqual(dm.train.dataset._target_, "torchvision.datasets.ImageFolder")
        self.assertEqual(dm.train.batch_size, TRAIN_BATCH)
        self.assertEqual(dm.eval.batch_size, EVAL_BATCH)
        # from /datamodule/base -> dataloader.yaml; proves the absolute default resolved
        self.assertTrue(dm.train.shuffle)
        self.assertFalse(dm.eval.shuffle)
        self.assertEqual(dm.train.num_workers, LOADER_KW["num_workers"])
        # same scheme and parameters as the hand-built partitioner
        self.assertEqual(repr(instantiate(dm.partitioner)), repr(self.make_partitioner()))

    # loaders -----------------------------------------------------------------

    @needs_data
    def test_trainer_gets_stateful_loader(self) -> None:
        with fl_env(1):
            dm = build_datamodule(self.make_partitioner(), self.train_set, self.eval_set)

        self.assertIsInstance(dm.train, StatefulDataLoader)
        # a Subset (label skew) or a Subset wrapper (feature skew): both expose these
        self.assertTrue(hasattr(dm.train.dataset, "indices"), type(dm.train.dataset))
        self.assertIs(dm.train.dataset.dataset, self.train_set)
        self.assertEqual(dm.train.batch_size, TRAIN_BATCH)
        self.assertEqual(dm.train.num_workers, LOADER_KW["num_workers"])

        self.assertIs(type(dm.eval), DataLoader)  # eval: global test set, not partitioned
        self.assertIs(dm.eval.dataset, self.eval_set)

        x, y = next(iter(dm.train))
        self.assertEqual(tuple(x.shape[1:]), (3, 224, 224))
        own_classes = set(self.train_labels[dm.train.dataset.indices].tolist())
        self.assertTrue(set(y.tolist()) <= own_classes)

    @needs_data
    def test_server_gets_no_train_loader(self) -> None:
        with fl_env(-1):
            dm = build_datamodule(self.make_partitioner(), self.train_set, self.eval_set)
        self.assertIsNone(dm.train)
        self.assertIs(dm.eval.dataset, self.eval_set)

    # partition -----------------------------------------------------------------

    @needs_data
    def test_clients_disjoint_and_agree(self) -> None:
        ids, slices = set(), []
        for cid in range(NUM_CLIENTS):  # each iteration = a separate rank
            with fl_env(cid):
                p = self.make_partitioner()
                slices.append(p.client_indices(self.train_set))
                ids.add(p.partition_id(self.train_set))

        self.assertEqual(len(ids), 1, "ranks computed different partitions")
        flat = np.concatenate(slices)
        self.assertEqual(np.unique(flat).size, flat.size, "a sample is on two clients")
        self.assertTrue(all(s.size > 0 for s in slices), "a client got no data")
        if self.FULL_COVERAGE:
            self.assertEqual(flat.size, len(self.train_set), "some samples went unused")
        else:
            self.assertLess(flat.size, len(self.train_set))

    @needs_data
    def test_matches_hydra_config(self) -> None:
        try:
            from hydra.utils import instantiate
        except ImportError:
            self.skipTest("hydra not installed")
        cfg = compose_config(self.CONFIG)
        with fl_env(2):
            from_yaml = instantiate(cfg.datamodule)
            by_hand = build_datamodule(self.make_partitioner(), self.train_set, self.eval_set)

        a, b = from_yaml.train, by_hand.train
        self.assertIs(type(a), StatefulDataLoader)
        self.assertIs(type(b), StatefulDataLoader)
        for attr in ("batch_size", "num_workers", "pin_memory", "drop_last"):
            self.assertEqual(getattr(a, attr), getattr(b, attr), attr)
        self.assertEqual(a.dataset.dataset.root, b.dataset.dataset.root)
        self.assertTrue(np.array_equal(a.dataset.indices, b.dataset.indices))
        self.assertIs(type(from_yaml.eval), DataLoader)
        self.assertEqual(from_yaml.eval.batch_size, by_hand.eval.batch_size)
        self.assertEqual(len(from_yaml.eval.dataset), len(by_hand.eval.dataset))


LabelSkewSuite = SkewSuite  # name used by the label-skew tests
