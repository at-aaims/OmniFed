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
Style / colour feature skew on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-style-feature-skew.yaml``. Shared checks come
from :class:`FeatureSkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.feature_distribution import StyleFeatureSkew
from src.omnifed.data.skewed.test_feature_dist._common import (
    MEAN,
    NUM_CLIENTS,
    STD,
    FeatureSkewSuite,
    synthetic_image,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestStyleFeatureSkew(FeatureSkewSuite, unittest.TestCase):
    CONFIG = "cal256-style-feature-skew"

    @staticmethod
    def make_partitioner():
        return StyleFeatureSkew(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, grayscale_prob=0.0,
            input_mean=MEAN, input_std=STD, stateful_loader=True,
        )

    def test_factors_within_ranges_and_differ_per_client(self) -> None:
        ts = self.transforms()
        for t in ts:
            for factor in (t.brightness, t.contrast, t.saturation):
                self.assertTrue(0.6 <= factor <= 1.4, t)
            self.assertLessEqual(abs(t.hue), 0.1)
            self.assertFalse(t.grayscale)
        self.assertEqual(len({t.describe() for t in ts}), NUM_CLIENTS)

    def test_grayscale_clients(self) -> None:
        t = StyleFeatureSkew(grayscale_prob=1.0).client_transform(0, 4)
        out = t(synthetic_image(), 0)
        self.assertTrue(torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2]))

    def test_seed_changes_styles(self) -> None:
        a = StyleFeatureSkew(seed=1).client_transform(0, 4).describe()
        b = StyleFeatureSkew(seed=2).client_transform(0, 4).describe()
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    describe(TestStyleFeatureSkew.make_partitioner)
