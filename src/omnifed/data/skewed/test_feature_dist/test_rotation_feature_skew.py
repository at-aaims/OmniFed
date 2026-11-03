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
Rotation feature skew (rotated MNIST/CIFAR, e.g. IFCA) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-rotation-feature-skew.yaml``. Shared checks come
from :class:`FeatureSkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.feature_distribution import Rotate, RotationFeatureSkew
from src.omnifed.data.skewed.test_feature_dist._common import (
    MEAN,
    STD,
    FeatureSkewSuite,
    synthetic_image,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestRotationFeatureSkew(FeatureSkewSuite, unittest.TestCase):
    CONFIG = "cal256-rotation-feature-skew"

    @staticmethod
    def make_partitioner():
        return RotationFeatureSkew(
            angles=[0, 90, 180, 270], input_mean=MEAN, input_std=STD, stateful_loader=True
        )

    def test_client_i_rotates_by_angles_i(self) -> None:
        x = synthetic_image()
        for i, t in enumerate(self.transforms(num_clients=8)):  # angles cycle every 4 clients
            self.assertTrue(torch.equal(t(x, 0), torch.rot90(x, i % 4, dims=(-2, -1))), t)

    def test_other_angles_and_non_square_keep_shape(self) -> None:
        rect = torch.rand(3, 32, 48)
        for angle in (30, 90, 270):
            self.assertEqual(Rotate(angle)(rect, 0).shape, rect.shape)


if __name__ == "__main__":
    describe(TestRotationFeatureSkew.make_partitioner)
