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
Tests on the real FLamby datasets, one file per dataset, sharing :class:`FlambySuite`.

Each test class first downloads its dataset with
:func:`~src.omnifed.data.skewed.flamby_download.download_flamby_dataset`: data
already in ``$OMNIFED_DATA_DIR/flamby/<dataset>`` is reused, otherwise FLamby's
download scripts run (once; needs ``OMNIFED_FLAMBY_ACCEPT_LICENSE=1``). Then it
loads it only through the datamodule (which never downloads) and checks, on
the real data, that

* the pooled dataset matches FLamby's per-center datasets,
* ``conf/datamodule/flamby/flamby.yaml`` gives every rank a ``StatefulDataLoader``
  over its own centers (rank and client count from the env vars only),
* fewer / more clients than centers merge / split centers and cover the data,
* every scheme in ``conf/datamodule/flamby/scheme/`` covers the data (label
  schemes refuse datasets without class labels),
* partitions are deterministic and seeded, loaders resume from ``state_dict``.

Run from the repo root (``-rs`` lists why a class was skipped)::

    export OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1
    python -m pytest src/omnifed/data/skewed/test_flamby -v -rs
    python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_ixi.py -v -rs   # one dataset

Skips only for what a test cannot fix itself: licence not accepted, a dataset
that must be prepared by hand (Camelyon16, KiTS19), or a missing package (the
skip message gives the ``pip install`` line).
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from src.omnifed.data.skewed import flamby_download as fd
from src.omnifed.data.skewed import flamby_partition as fp
from src.omnifed.data.skewed.test_label_dist._common import fl_env

__all__ = ["CONF_DIR", "SCHEMES", "FlambySuite", "compose_flamby", "build", "fl_env"]

CONF_DIR = Path(__file__).resolve().parents[5] / "conf"
SCHEME_DIR = CONF_DIR / "datamodule" / "flamby" / "scheme"
SCHEMES = sorted(p.stem for p in SCHEME_DIR.glob("*.yaml"))
LABEL_SCHEMES = {"natural-dirichlet-label", "pooled-dirichlet-label"}


def compose_flamby(dataset: str, scheme: Optional[str] = None, overrides: Sequence[str] = ()):
    """``base`` + ``datamodule=flamby/flamby`` with this dataset (and scheme)."""
    from hydra import compose, initialize_config_dir

    ov = ["datamodule=flamby/flamby", f"datamodule.dataset={dataset}"]
    if scheme is not None:
        ov.append(f"datamodule/flamby/scheme@datamodule.partitioner={scheme}")
    for o in overrides:  # relative to the datamodule; keep a leading +/~
        prefix = o[0] if o[0] in "+~" else ""
        ov.append(f"{prefix}datamodule.{o[len(prefix):]}")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name="base", overrides=ov)


def build(dataset: str, num_clients: int, client_idx: int, scheme: Optional[str] = None, overrides=()):
    """The datamodule rank ``client_idx`` of ``num_clients`` gets from flamby.yaml."""
    from hydra.utils import instantiate

    cfg = compose_flamby(dataset, scheme, overrides)
    with fl_env(client_idx, num_clients):
        return instantiate(cfg.datamodule)


def partitioner_for(dataset: str, scheme: str, overrides: Sequence[str] = ()):
    from hydra.utils import instantiate

    return instantiate(compose_flamby(dataset, scheme, overrides).datamodule.partitioner)


def _import_error(e: BaseException) -> Optional[ImportError]:
    """The ImportError behind ``e`` (Hydra wraps errors from targets), if any."""
    while e is not None and not isinstance(e, ImportError):
        e = e.__cause__ or e.__context__
    return e


class FlambySuite:
    """
    Mixin: subclass with ``unittest.TestCase`` and set ``DATASET``.

    Hooks: :meth:`check_batch` (shapes / dtypes of one train batch) and
    ``DETERMINISTIC_ITEMS`` (``False`` when ``__getitem__`` is random, e.g.
    augmentation or patch sampling, so resumed batches are compared by count).
    """

    DATASET: str = ""
    DETERMINISTIC_ITEMS: bool = True
    NO_WORKERS = ("num_workers=0",)

    # ── setup: download if missing (flamby_download), then load once ──────

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = fd.FLAMBY_DATASETS[cls.DATASET]
        needs_dir = not cls.spec.bundled and cls.spec.takes_data_path
        if needs_dir and not os.environ.get(fd.DATA_DIR_ENV):
            raise unittest.SkipTest(f"set {fd.DATA_DIR_ENV} (data goes to $OMNIFED_DATA_DIR/flamby/{cls.DATASET})")
        try:
            t0 = time.perf_counter()
            cls.path = fd.download_flamby_dataset(cls.DATASET)  # no-op when already downloaded
            if time.perf_counter() - t0 > 5:
                print(f"\n[{cls.DATASET}] ready after {time.perf_counter() - t0:.0f}s at {cls.path}")
        except PermissionError as e:  # licence not accepted
            raise unittest.SkipTest(str(e))
        except ImportError as e:  # FLamby submodule missing
            raise unittest.SkipTest(str(e))
        except RuntimeError as e:
            if cls.spec.manual:  # must be prepared by hand
                raise unittest.SkipTest(str(e))
            raise
        try:
            cls.train = fp.FlambyPooledDataset(cls.DATASET, train=True)
            cls.test = fp.FlambyPooledDataset(cls.DATASET, train=False)
        except PermissionError as e:
            raise unittest.SkipTest(str(e))
        except Exception as e:
            missing = _import_error(e)
            if missing is not None:
                hint = "" if "pip install" in str(missing) else f" pip install {cls.spec.pip}"
                raise unittest.SkipTest(f"{cls.DATASET}: {missing}{hint}")
            if cls.spec.manual:
                raise unittest.SkipTest(f"{cls.DATASET} not prepared ({e}): {cls.spec.manual}")
            raise
        cls.C = len(cls.train.centers)

    # ── hooks ──────────────────────────────────────────────────────────────

    def check_batch(self, x: Any, y: Any) -> None:
        self.assertEqual(len(x), len(y))

    # ── helpers ────────────────────────────────────────────────────────────

    def flamby_center(self, center: int, train: bool = True):
        module = fp._import_flamby(self.spec)
        kwargs = {"data_path": str(self.path)} if self.spec.takes_data_path and self.path else {}
        return getattr(module, self.spec.cls)(center=center, train=train, pooled=False, **kwargs)

    def parts(self, partitioner, num_clients: int) -> List[np.ndarray]:
        with fl_env(0, num_clients):
            return partitioner.partition(self.train)

    def assert_exact_cover(self, parts: Sequence[np.ndarray], n: Optional[int] = None) -> None:
        flat = np.concatenate([np.asarray(p, dtype=np.int64) for p in parts])
        n = len(self.train) if n is None else n
        self.assertEqual((flat.size, np.unique(flat).size), (n, n), "partition must cover every sample once")

    def centers_of(self, idx) -> set:
        return set(self.train.center_ids[np.asarray(idx, dtype=np.int64)].tolist())

    # ── data ───────────────────────────────────────────────────────────────

    def test_downloaded_once_and_reused(self) -> None:
        self.assertTrue(fd.is_flamby_downloaded(self.DATASET))
        self.assertEqual(fd.flamby_dataset_dir(self.DATASET), self.path)  # what the datamodule loads
        if self.path is None:  # bundled or FLamby-configured
            self.assertIsNone(fd.download_flamby_dataset(self.DATASET))
            return
        if not self.spec.manual:
            marker = self.path / fd._MARKER
            self.assertTrue(marker.is_file())
            before = marker.stat().st_mtime
            self.assertEqual(fd.download_flamby_dataset(self.DATASET), self.path)  # no second download
            self.assertEqual(marker.stat().st_mtime, before)

    def test_pooled_matches_flamby_centers(self) -> None:
        self.assertEqual(self.train.centers, list(range(self.spec.num_centers)))
        for split, pooled in ((True, self.train), (False, self.test)):
            sizes = [len(self.flamby_center(c, split)) for c in range(self.C)]
            self.assertEqual(np.bincount(pooled.center_ids, minlength=self.C).tolist(), sizes)
        x, y = self.train[0][:2]
        x0, y0 = self.flamby_center(0)[0][:2]
        if self.DETERMINISTIC_ITEMS:
            self.assertTrue(torch.equal(torch.as_tensor(x), torch.as_tensor(x0)))

    def test_labels(self) -> None:
        if self.train.has_class_labels:
            self.assertEqual(len(self.train.targets), len(self.train))
            self.assertGreater(len(np.unique(self.train.targets)), 1)
        else:
            self.assertTrue(np.all(self.train.targets == 0))  # placeholders

    # ── natural partition rules ────────────────────────────────────────────

    def test_clients_equal_centers(self) -> None:
        parts = self.parts(fp.FlambyPartition(), self.C)
        self.assert_exact_cover(parts)
        for c, idx in enumerate(parts):
            self.assertEqual(self.centers_of(idx), {c})

    def test_fewer_clients_merge_centers(self) -> None:
        for k in sorted({1, max(1, self.C - 1), max(1, self.C // 2)}):
            parts = self.parts(fp.FlambyPartition(), k)
            self.assert_exact_cover(parts)
            owners = [[i for i, p in enumerate(parts) if c in self.centers_of(p)] for c in range(self.C)]
            self.assertTrue(all(len(o) == 1 for o in owners), "a merged center stays whole")

    def test_more_clients_split_centers(self) -> None:
        k = self.C + 2
        p = fp.FlambyPartition()
        parts = self.parts(p, k)
        self.assert_exact_cover(parts)
        with fl_env(0, k):
            plan = p.summary(self.train)["client_centers"]
        self.assertEqual(plan[: self.C], [[c] for c in range(self.C)])  # clients 0..C-1 keep centers 0..C-1
        for i, idx in enumerate(parts):
            self.assertEqual(self.centers_of(idx), set(plan[i]))

    # ── config: every rank, server, schemes ───────────────────────────────

    def test_each_rank_from_yaml(self) -> None:
        for cid in range(self.C):
            dm = build(self.DATASET, self.C, cid, overrides=self.NO_WORKERS)
            self.assertIsInstance(dm.train, StatefulDataLoader)
            self.assertIs(dm.train.dataset.dataset.__class__, fp.FlambyPooledDataset)
            pooled = dm.train.dataset.dataset
            self.assertEqual(set(pooled.center_ids[np.asarray(dm.train.dataset.indices)].tolist()), {cid})
            self.assertIsInstance(dm.eval.dataset, Subset)  # own center's test split
            test = dm.eval.dataset.dataset
            self.assertEqual(set(test.center_ids[np.asarray(dm.eval.dataset.indices)].tolist()), {cid})
            self.assertEqual(dm.train.batch_size, self.spec.batch_size)
            if cid == 0:
                x, y = next(iter(dm.train))[:2]
                self.check_batch(x, y)
                next(iter(dm.eval))

    def test_server_gets_pooled_eval_only(self) -> None:
        dm = build(self.DATASET, self.C, -1, overrides=self.NO_WORKERS)
        self.assertIsNone(dm.train)
        self.assertEqual(len(dm.eval.dataset), len(self.test))

    def test_default_workers_rebuild_dataset(self) -> None:
        if self.spec.num_workers == 0:
            self.skipTest("dataset loads in the main process by default")
        dm = build(self.DATASET, self.C, 0)  # workers get the dataset pickled (spawn on macOS)
        self.assertEqual(dm.train.num_workers, self.spec.num_workers)
        x, y = next(iter(dm.train))[:2]
        self.check_batch(x, y)

    def test_every_scheme(self) -> None:
        self.assertIn("natural", SCHEMES)
        k = self.C + 1
        for scheme in SCHEMES:
            with self.subTest(scheme=scheme):
                partitioner = partitioner_for(self.DATASET, scheme)
                self.assertTrue(partitioner.stateful_loader)
                if scheme in LABEL_SCHEMES and not self.train.has_class_labels:
                    with self.assertRaisesRegex(ValueError, "class labels"):
                        self.parts(partitioner, k)
                    continue
                self.assert_exact_cover(self.parts(partitioner, k))

    def test_scheme_from_yaml_for_a_rank(self) -> None:
        dm = build(self.DATASET, self.C + 1, 1, "pooled-dirichlet-quantity", self.NO_WORKERS)
        self.assertIsInstance(dm.train, StatefulDataLoader)
        self.assertIs(dm.eval.dataset.__class__, fp.FlambyPooledDataset)  # pooled schemes: pooled eval

    def test_deterministic_and_seeded(self) -> None:
        k = self.C + 1
        a = self.parts(partitioner_for(self.DATASET, "pooled-dirichlet-quantity"), k)
        b = self.parts(partitioner_for(self.DATASET, "pooled-dirichlet-quantity"), k)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        c = self.parts(partitioner_for(self.DATASET, "pooled-dirichlet-quantity", ["+partitioner.seed=12345"]), k)
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_stateful_loader_resumes(self) -> None:
        dm = build(self.DATASET, self.C, 0, overrides=self.NO_WORKERS + ("batch_size=1",))
        n = len(dm.train)
        if n < 3:
            self.skipTest("client 0 has fewer than 3 batches")
        it = iter(dm.train)
        next(it)
        state = dm.train.state_dict()
        expected = next(it)
        resumed = build(self.DATASET, self.C, 0, overrides=self.NO_WORKERS + ("batch_size=1",)).train
        resumed.load_state_dict(state)
        rest = list(resumed) if n <= 50 else None
        got = next(iter(resumed)) if rest is None else rest[0]
        if rest is not None:
            self.assertEqual(len(rest), n - 1)
        if self.DETERMINISTIC_ITEMS:
            self.assertTrue(torch.equal(torch.as_tensor(got[0]), torch.as_tensor(expected[0])))
