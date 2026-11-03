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
Log-normal quantity skew (LEAF-style) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-lognormal-quantity-skew.yaml``. Shared checks come
from :class:`QuantitySkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.quantity_skew import LogNormalQuantitySkew
from src.omnifed.data.skewed.test_label_dist._common import describe
from src.omnifed.data.skewed.test_quantity_skew._common import (
    LabelsOnly,
    QuantitySkewSuite,
    size_spread,
)


class TestLogNormalQuantitySkew(QuantitySkewSuite, unittest.TestCase):
    CONFIG = "cal256-lognormal-quantity-skew"

    @staticmethod
    def make_partitioner():
        return LogNormalQuantitySkew(sigma=1.0, max_retries=1000, min_samples_per_client=10, stateful_loader=True)

    def test_sigma_zero_gives_equal_sizes(self) -> None:
        sizes = self.summary_of(LogNormalQuantitySkew(sigma=0.0))["sizes"]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_larger_sigma_more_unequal(self) -> None:
        ds = LabelsOnly(n=50000)
        spread = [
            size_spread(self.summary_of(LogNormalQuantitySkew(sigma=s, min_samples_per_client=1), ds, 20)["sizes"])
            for s in (0.5, 2.0)
        ]
        self.assertLess(spread[0], spread[1])


if __name__ == "__main__":
    describe(TestLogNormalQuantitySkew.make_partitioner)
