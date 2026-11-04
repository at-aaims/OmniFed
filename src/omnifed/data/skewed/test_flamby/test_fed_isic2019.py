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
Fed-ISIC2019: 6 centers, dermoscopy images, 8 classes.

Downloads ~9 GB if missing (resumes if interrupted), keeps FLamby's 23,247
images and resizes them with FLamby's script. Needs: pip install albumentations
(any release; efficientnet-pytorch is not needed for the data: see
flamby_compat). Train items are randomly augmented.

Shared checks: :class:`~._common.FlambySuite`. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_isic2019.py -v -rs
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.test_flamby._common import FlambySuite


class TestFedIsic2019(FlambySuite, unittest.TestCase):
    DATASET = "fed_isic2019"
    DETERMINISTIC_ITEMS = False

    def test_isic_config(self) -> None:
        from hydra import compose, initialize_config_dir

        from src.omnifed.data.skewed.test_flamby._common import CONF_DIR

        with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
            dm = compose(config_name="base", overrides=["datamodule=flamby/isic2019-natural"]).datamodule
        self.assertEqual((dm.dataset, dm.batch_size), ("fed_isic2019", 64))
        self.assertTrue(dm.partitioner._target_.endswith("FlambyPartition"))

    def check_batch(self, x, y) -> None:
        self.assertEqual(tuple(x.shape[1:]), (3, 200, 200))
        self.assertEqual(y.dtype, torch.int64)
        self.assertTrue(bool(((y >= 0) & (y < 8)).all()))
        self.assertTrue(self.train.has_class_labels)


if __name__ == "__main__":
    unittest.main()
