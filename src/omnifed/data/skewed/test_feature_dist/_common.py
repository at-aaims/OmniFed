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
Shared helpers for the Caltech-256 feature-skew tests.

Each ``test_*.py`` here pairs with one
``conf/datamodule/skewed-data-sim/cal256-*-feature-skew.yaml``. It inherits

* the loader / partition / Hydra checks every skew scheme gets
  (:class:`SkewSuite` from ``test_label_dist/_common.py``), and
* the feature-skew checks in :class:`FeatureSkewSuite` below: transforms are
  deterministic, picklable (DataLoader workers), shape-preserving, identical on
  every rank, applied to exactly the client's samples, and covered by
  ``partition_id``; the yaml's ``input_mean``/``input_std`` match its Normalize.

Run from the repo root (``OMNIFED_DATA_DIR`` holds ``Caltech256/{train,test}``)::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets python -m pytest src/omnifed/data/skewed/test_feature_dist -v

Without ``OMNIFED_DATA_DIR`` the transform tests (synthetic images) and the
config-composition tests still run; tests that read Caltech images are skipped.
"""

from __future__ import annotations

import functools
import pickle
from typing import List

import numpy as np
import torch
from torchvision import datasets

from src.omnifed.data.skewed.feature_distribution import ClientTransform, ClientTransformDataset
from src.omnifed.data.skewed.test_label_dist._common import (
    DATA_ROOT,
    MEAN,
    NUM_CLIENTS,
    STD,
    SkewSuite,
    compose_config,
    eval_transform,
    fl_env,
    needs_data,
)

__all__ = [
    "MEAN",
    "STD",
    "NUM_CLIENTS",
    "FeatureSkewSuite",
    "LabelsOnly",
    "deterministic_train_set",
    "synthetic_image",
    "fl_env",
    "needs_data",
]


@functools.lru_cache(maxsize=1)
def deterministic_train_set() -> datasets.ImageFolder:
    """Caltech train images with the eval (non-random) transform, for exact comparisons."""
    return datasets.ImageFolder(str(DATA_ROOT / "train"), transform=eval_transform())


def synthetic_image(seed: int = 0, size: int = 64, channels: int = 3) -> torch.Tensor:
    """A random ``[C, size, size]`` image in [0, 1] (pixel space)."""
    return torch.rand((channels, size, size), generator=torch.Generator().manual_seed(seed))


class LabelsOnly:
    """A dataset stand-in with only labels: enough to partition, no images."""

    def __init__(self, n: int = 2000, num_classes: int = 10, seed: int = 0) -> None:
        self.targets = np.random.default_rng(seed).integers(0, num_classes, n).tolist()

    def __len__(self) -> int:
        return len(self.targets)


class FeatureSkewSuite(SkewSuite):
    """
    Checks shared by every feature-skew scheme. Mix into ``unittest.TestCase``
    and set ``CONFIG`` and ``make_partitioner()`` (see :class:`SkewSuite`).
    """

    FULL_COVERAGE = True  # the default IID split assigns every sample

    def transforms(self, num_clients: int = NUM_CLIENTS) -> List[ClientTransform]:
        p = self.make_partitioner()
        return [p.client_transform(i, num_clients) for i in range(num_clients)]

    # transforms (synthetic images, no data needed) -----------------------------

    def test_transforms_deterministic_shape_preserving_picklable(self) -> None:
        x = synthetic_image()
        for t in self.transforms():
            out = t(x, 7)
            self.assertEqual(out.shape, x.shape, t)
            self.assertTrue(torch.isfinite(out).all(), t)
            self.assertTrue(torch.equal(out, t(x, 7)), f"{t} is not deterministic")
            clone = pickle.loads(pickle.dumps(t))  # what DataLoader workers receive
            self.assertTrue(torch.equal(out, clone(x, 7)), f"{t} changes when pickled")

    def test_transforms_identical_on_every_rank(self) -> None:
        with fl_env(0):  # same seed, fresh objects = separate ranks
            a = [t.describe() for t in self.transforms()]
            b = [t.describe() for t in self.transforms()]
        self.assertEqual(a, b)

    def test_split_decides_assignment_and_id_covers_transforms(self) -> None:
        ds = LabelsOnly()
        with fl_env(0):
            p = self.make_partitioner()
            for mine, split in zip(p.partition(ds), p.split.partition(ds)):
                self.assertTrue(np.array_equal(mine, split))
            self.assertNotEqual(p.partition_id(ds), p.split.partition_id(ds))
            features = p.summary(ds)["client_features"]
        self.assertEqual(len(features), NUM_CLIENTS)

    def test_yaml_normalize_matches_partitioner(self) -> None:
        try:
            dm = compose_config(self.CONFIG).datamodule
        except ImportError:
            self.skipTest("hydra not installed")
        normalize = dm.train.dataset.transform.transforms[-1]
        self.assertEqual(normalize._target_, "torchvision.transforms.Normalize")
        self.assertEqual(list(dm.partitioner.input_mean), list(normalize.mean))
        self.assertEqual(list(dm.partitioner.input_std), list(normalize.std))

    # real Caltech samples ------------------------------------------------------

    @needs_data
    def test_client_samples_are_transformed(self) -> None:
        base = deterministic_train_set()
        with fl_env(1):
            p = self.make_partitioner()
            ds = p.client_dataset(base)
            t = p.client_transform(1, NUM_CLIENTS)
        self.assertIsInstance(ds, ClientTransformDataset)
        self.assertIs(ds.dataset, base)

        idx = int(ds.indices[0])
        x, y = ds[0]
        raw, y0 = base[idx]
        self.assertEqual(y, y0)  # labels untouched
        mean, std = torch.tensor(MEAN).view(-1, 1, 1), torch.tensor(STD).view(-1, 1, 1)
        expected = (t(raw * std + mean, idx).clamp(0, 1) - mean) / std  # pixel space round trip
        self.assertTrue(torch.allclose(x, expected, atol=1e-5))
        self.assertTrue(torch.equal(x, ds[0][0]), "reading a sample twice changed it")
