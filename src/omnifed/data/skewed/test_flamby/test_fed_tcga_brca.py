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
Fed-TCGA-BRCA: 6 regions, 39 clinical features, survival target (event, time).

Ships with FLamby (no download); FLamby asks for its licence once, answered by
OMNIFED_FLAMBY_ACCEPT_LICENSE=1. No class labels, so label schemes must refuse it.

Shared checks: :class:`~._common.FlambySuite`. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_tcga_brca.py -v -rs
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.test_flamby._common import FlambySuite


class TestFedTcgaBrca(FlambySuite, unittest.TestCase):
    DATASET = "fed_tcga_brca"
    DETERMINISTIC_ITEMS = True

    def check_batch(self, x, y) -> None:
        self.assertEqual(tuple(x.shape[1:]), (39,))
        self.assertEqual(tuple(y.shape[1:]), (2,))  # (event, time)
        self.assertFalse(self.train.has_class_labels)


if __name__ == "__main__":
    unittest.main()
