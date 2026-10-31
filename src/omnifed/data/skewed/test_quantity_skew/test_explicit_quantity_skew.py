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
Explicit quantity skew on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-explicit-quantity-skew.yaml``. Shared checks come
from :class:`QuantitySkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.quantity_skew import ExplicitQuantitySkew
from src.omnifed.data.skewed.test_label_dist._common import describe
from src.omnifed.data.skewed.test_quantity_skew._common import QuantitySkewSuite


class TestExplicitQuantitySkew(QuantitySkewSuite, unittest.TestCase):
    CONFIG = "cal256-explicit-quantity-skew"

    @staticmethod
    def make_partitioner():
        return ExplicitQuantitySkew(proportions=[0.4, 0.3, 0.2, 0.1], min_samples_per_client=10, stateful_loader=True)

    def test_sizes_match_proportions(self) -> None:
        self.assertEqual(self.summary_of()["sizes"], [8000, 6000, 4000, 2000])  # of 20000

    def test_weights_are_normalized(self) -> None:
        a = self.summary_of(ExplicitQuantitySkew(proportions=[2, 1, 1]), num_clients=3)["sizes"]
        b = self.summary_of(ExplicitQuantitySkew(proportions=[0.5, 0.25, 0.25]), num_clients=3)["sizes"]
        self.assertEqual(a, b)

    def test_one_entry_per_client(self) -> None:
        with self.assertRaises(ValueError):
            self.summary_of(ExplicitQuantitySkew(proportions=[1, 1, 1]))  # 3 entries, 4 clients

    def test_proportions_must_be_positive(self) -> None:
        for bad in ([], [1, 0, 1, 1], [1, -1, 1, 1]):
            with self.assertRaises(ValueError):
                ExplicitQuantitySkew(proportions=bad)


if __name__ == "__main__":
    describe(TestExplicitQuantitySkew.make_partitioner)
