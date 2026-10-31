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
Power-law quantity skew on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-power-law-quantity-skew.yaml``. Shared checks come
from :class:`QuantitySkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.quantity_skew import PowerLawQuantitySkew
from src.omnifed.data.skewed.test_label_dist._common import describe
from src.omnifed.data.skewed.test_quantity_skew._common import (
    NUM_CLIENTS,
    LabelsOnly,
    QuantitySkewSuite,
)


class TestPowerLawQuantitySkew(QuantitySkewSuite, unittest.TestCase):
    CONFIG = "cal256-power-law-quantity-skew"

    @staticmethod
    def make_partitioner():
        return PowerLawQuantitySkew(exponent=1.0, shuffle_order=True, min_samples_per_client=10, stateful_loader=True)

    def test_sizes_follow_the_power_law(self) -> None:
        sizes = self.summary_of(PowerLawQuantitySkew(exponent=1.0, shuffle_order=False))["sizes"]
        for rank in range(NUM_CLIENTS):  # size_r proportional to 1 / (rank + 1)
            self.assertAlmostEqual(sizes[0] / sizes[rank], rank + 1, delta=0.01 * (rank + 1))

    def test_shuffle_changes_order_not_sizes(self) -> None:
        fixed = self.summary_of(PowerLawQuantitySkew(exponent=1.0, shuffle_order=False))["sizes"]
        shuffled = self.summary_of(PowerLawQuantitySkew(exponent=1.0, shuffle_order=True))["sizes"]
        self.assertEqual(sorted(fixed), sorted(shuffled))
        self.assertEqual(fixed, sorted(fixed, reverse=True))

    def test_exponent_zero_gives_equal_sizes(self) -> None:
        sizes = self.summary_of(PowerLawQuantitySkew(exponent=0.0))["sizes"]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_too_steep_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.summary_of(PowerLawQuantitySkew(exponent=6.0), LabelsOnly(n=50000), 20)


if __name__ == "__main__":
    describe(TestPowerLawQuantitySkew.make_partitioner)
