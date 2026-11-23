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
Corruption feature skew (ImageNet-C style) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-corruption-feature-skew.yaml``. Shared checks come
from :class:`FeatureSkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.feature_distribution import CORRUPTIONS, Corrupt, CorruptionFeatureSkew
from src.omnifed.data.skewed.test_feature_dist._common import (
    MEAN,
    STD,
    FeatureSkewSuite,
    synthetic_image,
)
from src.omnifed.data.skewed.test_label_dist._common import describe

# the list in cal256-corruption-feature-skew.yaml
CLIENT_CORRUPTIONS = [
    "none",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "contrast",
    "pixelate",
    "brightness",
    "shot_noise",
]


class TestCorruptionFeatureSkew(FeatureSkewSuite, unittest.TestCase):
    CONFIG = "cal256-corruption-feature-skew"

    @staticmethod
    def make_partitioner():
        return CorruptionFeatureSkew(
            corruptions=CLIENT_CORRUPTIONS, severity=3, input_mean=MEAN, input_std=STD, stateful_loader=True
        )

    def test_clients_cycle_through_corruptions(self) -> None:
        names = [t.describe().split("(")[0] for t in self.transforms(num_clients=10)]
        self.assertEqual(names, [CLIENT_CORRUPTIONS[i % len(CLIENT_CORRUPTIONS)] for i in range(10)])

    def test_every_corruption_changes_the_image(self) -> None:
        x = synthetic_image()
        for name in CORRUPTIONS:
            for severity in range(1, 6):
                out = Corrupt(name, severity, (0, 0, 1))(x, 0)
                self.assertEqual(out.shape, x.shape, name)
                self.assertTrue(torch.isfinite(out).all(), name)
                self.assertEqual(torch.equal(out, x), name == "none", f"{name} severity {severity}")

    def test_higher_severity_distorts_more(self) -> None:
        x = synthetic_image()
        for name in CORRUPTIONS:
            if name == "none":
                continue
            mse = [float(((Corrupt(name, s, (0, 0, 1))(x, 0).clamp(0, 1) - x) ** 2).mean()) for s in (1, 5)]
            self.assertLess(mse[0], mse[1], name)

    def test_jpeg_output_is_8bit(self) -> None:
        out = Corrupt("jpeg_compression", 3, (0, 0, 1))(synthetic_image(), 0) * 255
        self.assertTrue(torch.allclose(out, out.round(), atol=1e-4))



if __name__ == "__main__":
    describe(TestCorruptionFeatureSkew.make_partitioner)
