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
Shared helpers for the Caltech-256 quantity-skew tests.

Each ``test_*.py`` here pairs with one
``conf/datamodule/skewed-data-sim/cal256-*-quantity-skew.yaml``. It inherits

* the loader / partition / Hydra checks every skew scheme gets
  (:class:`SkewSuite` from ``test_label_dist/_common.py``), and
* the quantity-skew checks in :class:`QuantitySkewSuite` below: sizes sum to
  the dataset, respect the minimum, the split is disjoint and complete, labels
  stay IID, and the result is the same on every rank.

Run from the repo root (``OMNIFED_DATA_DIR`` holds ``Caltech256/{train,test}``)::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets python -m pytest src/omnifed/data/skewed/test_quantity_skew -v

Quantity skew only decides sizes, so most checks use a synthetic label set and
run without Caltech images.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.omnifed.data.skewed.test_label_dist._common import NUM_CLIENTS, SkewSuite, fl_env

__all__ = ["NUM_CLIENTS", "LabelsOnly", "QuantitySkewSuite", "size_spread", "fl_env"]


class LabelsOnly:
    """A dataset stand-in with only labels: enough to partition, no images."""

    def __init__(self, n: int = 20000, num_classes: int = 10, seed: int = 0) -> None:
        self.targets = np.random.default_rng(seed).integers(0, num_classes, n).tolist()

    def __len__(self) -> int:
        return len(self.targets)


def size_spread(sizes: List[int]) -> float:
    """Coefficient of variation of client sizes (0 = all equal)."""
    s = np.asarray(sizes, dtype=float)
    return float(s.std() / s.mean())


class QuantitySkewSuite(SkewSuite):
    """
    Checks shared by every quantity-skew scheme. Mix into ``unittest.TestCase``
    and set ``CONFIG`` and ``make_partitioner()`` (see :class:`SkewSuite`).
    """

    FULL_COVERAGE = True  # sizes always sum to the full dataset

    def summary_of(self, partitioner=None, dataset=None, num_clients: int = NUM_CLIENTS) -> Dict[str, Any]:
        with fl_env(0, num_clients):
            return (partitioner or self.make_partitioner()).summary(dataset or LabelsOnly())

    # synthetic labels, no data needed -------------------------------------------

    def test_sizes_sum_to_dataset_and_respect_minimum(self) -> None:
        p = self.make_partitioner()
        s = self.summary_of(p)
        self.assertEqual(sum(s["sizes"]), len(LabelsOnly()))
        self.assertEqual(s["coverage"], 1.0)
        self.assertGreaterEqual(min(s["sizes"]), p.min_samples_per_client)

    def test_split_is_disjoint_and_complete(self) -> None:
        ds = LabelsOnly()
        with fl_env(0):
            parts = self.make_partitioner().partition(ds)
        flat = np.concatenate(parts)
        self.assertEqual(flat.size, len(ds))
        self.assertEqual(np.unique(flat).size, len(ds))

    def test_labels_stay_iid(self) -> None:
        s = self.summary_of()
        for size, h in zip(s["sizes"], s["hellinger"]):
            if size >= 1000:  # tiny clients are noisy by chance alone
                self.assertLess(h, 0.1, f"client of {size} samples has a skewed label mix")

    def test_same_on_every_rank_and_seed_matters(self) -> None:
        ds = LabelsOnly()
        with fl_env(0):
            a, b = self.make_partitioner(), self.make_partitioner()
            self.assertEqual(a.partition_id(ds), b.partition_id(ds))
        other = type(a)(**{**{k: v for k, v in vars(a).items() if not k.startswith("_")}, "seed": 12345})
        with fl_env(0):
            self.assertNotEqual(other.partition_id(ds), a.partition_id(ds))

    def test_client_sizes_api(self) -> None:
        sizes = self.make_partitioner().client_sizes(20000, NUM_CLIENTS, np.random.default_rng(0))
        self.assertEqual(int(sizes.sum()), 20000)
        self.assertEqual(sizes.shape, (NUM_CLIENTS,))
