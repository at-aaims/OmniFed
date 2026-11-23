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
Fed-LIDC-IDRI: 4 scanner manufacturers, CT lung nodule segmentation.

Downloads FLamby's 1,009 CT scans (~133 GB of DICOM, from NCI Imaging Data Commons
or TCIA) and converts them with FLamby's own code if missing: hours, so do it
once beforehand (OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m
src.omnifed.data.skewed.flamby_download fed_lidc_idri); for a quick try set
OMNIFED_FLAMBY_LIDC_LIMIT=16. Needs: pip install nibabel pydicom dicom-numpy
networkx scipy scikit-image idc-index. Items are randomly sampled patches.

Shared checks: :class:`~._common.FlambySuite`. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_lidc_idri.py -v -rs
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.test_flamby._common import FlambySuite


class TestFedLidcIdri(FlambySuite, unittest.TestCase):
    DATASET = "fed_lidc_idri"
    DETERMINISTIC_ITEMS = False

    def test_lidc_config(self) -> None:
        from hydra import compose, initialize_config_dir

        from src.omnifed.data.skewed.test_flamby._common import CONF_DIR

        with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
            dm = compose(config_name="base", overrides=["datamodule=flamby/lidc-idri-natural"]).datamodule
        self.assertEqual((dm.dataset, dm.batch_size), ("fed_lidc_idri", 1))
        self.assertTrue(dm.partitioner._target_.endswith("FlambyPartition"))

    def check_batch(self, x, y) -> None:
        self.assertEqual(x.dim(), 5)  # (patches, 1, D, W, H) from FLamby's collate_fn
        self.assertEqual(tuple(x.shape), tuple(y.shape))
        self.assertFalse(self.train.has_class_labels)


if __name__ == "__main__":
    unittest.main()
