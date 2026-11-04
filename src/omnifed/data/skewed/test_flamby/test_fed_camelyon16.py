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
Fed-Camelyon16: 2 hospitals, whole-slide images as tile features, binary label.

Cannot be downloaded automatically: prepare the features as described at
https://owkin.github.io/FLamby/fed_camelyon.html into
$OMNIFED_DATA_DIR/flamby/fed_camelyon16; until then these tests skip.

Shared checks: :class:`~._common.FlambySuite`. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_camelyon16.py -v -rs
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.test_flamby._common import FlambySuite


class TestFedCamelyon16(FlambySuite, unittest.TestCase):
    DATASET = "fed_camelyon16"
    DETERMINISTIC_ITEMS = True

    def check_batch(self, x, y) -> None:
        self.assertEqual(x.dim(), 3)  # (slides, tiles padded by FLamby's collate_fn, features)
        self.assertEqual(len(x), len(y))
        self.assertTrue(self.train.has_class_labels)


if __name__ == "__main__":
    unittest.main()
