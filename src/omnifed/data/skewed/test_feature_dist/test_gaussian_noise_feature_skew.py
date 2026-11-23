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
Gaussian-noise feature skew (NIID-Bench) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-gaussian-noise-feature-skew.yaml``. Shared checks come
from :class:`FeatureSkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.feature_distribution import GaussianNoiseFeatureSkew
from src.omnifed.data.skewed.test_feature_dist._common import (
    MEAN,
    NUM_CLIENTS,
    STD,
    FeatureSkewSuite,
    synthetic_image,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestGaussianNoiseFeatureSkew(FeatureSkewSuite, unittest.TestCase):
    CONFIG = "cal256-gaussian-noise-feature-skew"

    @staticmethod
    def make_partitioner():
        return GaussianNoiseFeatureSkew(
            sigma=0.1, fixed_per_sample=True, input_mean=MEAN, input_std=STD, stateful_loader=True
        )

    def test_noise_grows_with_client_index(self) -> None:
        x = torch.full((3, 64, 64), 0.5)
        stds = [float((t(x, 3) - x).std()) for t in self.transforms()]
        self.assertEqual(stds[0], 0.0)  # client 0 is clean
        for i, s in enumerate(stds[1:], start=1):
            expected = 0.1 * i / (NUM_CLIENTS - 1)
            self.assertAlmostEqual(s, expected, delta=0.1 * expected)

    def test_noise_fixed_per_sample_but_differs_across_samples(self) -> None:
        t, x = self.transforms()[-1], synthetic_image()
        self.assertTrue(torch.equal(t(x, 1), t(x, 1)))
        self.assertFalse(torch.equal(t(x, 1), t(x, 2)))

    def test_fresh_noise_mode(self) -> None:
        t = GaussianNoiseFeatureSkew(sigma=0.1, fixed_per_sample=False).client_transform(3, 4)
        x = synthetic_image()
        self.assertFalse(torch.equal(t(x, 1), t(x, 1)))


if __name__ == "__main__":
    describe(TestGaussianNoiseFeatureSkew.make_partitioner)
