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
Dirichlet quantity skew (NIID-Bench) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-dirichlet-quantity-skew.yaml``. Shared checks come
from :class:`QuantitySkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.quantity_skew import DirichletQuantitySkew
from src.omnifed.data.skewed.test_label_dist._common import describe
from src.omnifed.data.skewed.test_quantity_skew._common import (
    LabelsOnly,
    QuantitySkewSuite,
    size_spread,
)


class TestDirichletQuantitySkew(QuantitySkewSuite, unittest.TestCase):
    CONFIG = "cal256-dirichlet-quantity-skew"

    @staticmethod
    def make_partitioner():
        return DirichletQuantitySkew(beta=0.5, max_retries=1000, min_samples_per_client=10, stateful_loader=True)

    def test_smaller_beta_more_unequal(self) -> None:
        ds = LabelsOnly(n=50000)
        spread = {
            b: size_spread(self.summary_of(DirichletQuantitySkew(beta=b, min_samples_per_client=1), ds, 20)["sizes"])
            for b in (0.3, 10.0)
        }
        self.assertGreater(spread[0.3], spread[10.0])

    def test_unreachable_minimum_raises(self) -> None:
        p = DirichletQuantitySkew(beta=0.01, max_retries=5)
        with self.assertRaises(RuntimeError):
            self.summary_of(p, LabelsOnly(n=50000), 50)


if __name__ == "__main__":
    describe(TestDirichletQuantitySkew.make_partitioner)
