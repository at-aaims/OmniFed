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
Fed-KiTS19: 6 centers, CT kidney tumour segmentation.

Cannot be downloaded automatically: download kits19 and run FLamby's nnU-Net
preprocessing (https://owkin.github.io/FLamby/fed_kits19.html); FLamby's own
config gives the data location. Needs: pip install batchgenerators nibabel
nnunet==1.7.0. Until prepared these tests skip. Train items are augmented.

Shared checks: :class:`~._common.FlambySuite`. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_kits19.py -v -rs
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.test_flamby._common import FlambySuite


class TestFedKits19(FlambySuite, unittest.TestCase):
    DATASET = "fed_kits19"
    DETERMINISTIC_ITEMS = False

    def check_batch(self, x, y) -> None:
        self.assertEqual(x.dim(), 5)  # (batch, 1, D, H, W) patches
        self.assertEqual(len(x), len(y))
        self.assertFalse(self.train.has_class_labels)


if __name__ == "__main__":
    unittest.main()
