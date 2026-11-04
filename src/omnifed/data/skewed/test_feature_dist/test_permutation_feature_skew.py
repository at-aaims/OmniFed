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
Permutation feature skew (permuted MNIST, patch shuffle) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-permutation-feature-skew.yaml``. Shared checks come
from :class:`FeatureSkewSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.feature_distribution import PermutationFeatureSkew, PermutePatches
from src.omnifed.data.skewed.test_feature_dist._common import (
    MEAN,
    STD,
    FeatureSkewSuite,
    synthetic_image,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestPermutationFeatureSkew(FeatureSkewSuite, unittest.TestCase):
    CONFIG = "cal256-permutation-feature-skew"

    @staticmethod
    def make_partitioner():
        return PermutationFeatureSkew(
            patch_size=32, include_identity=True, input_mean=MEAN, input_std=STD, stateful_loader=True
        )

    @staticmethod
    def patches(x: torch.Tensor, p: int = 32) -> torch.Tensor:
        c, h, w = x.shape
        return x.reshape(c, h // p, p, w // p, p).permute(1, 3, 0, 2, 4).reshape(-1, c * p * p)

    def test_client0_identity_others_shuffle_patches(self) -> None:
        x = synthetic_image(size=224)
        ts = self.transforms()
        self.assertTrue(torch.equal(ts[0](x, 0), x))
        outs = [t(x, 0) for t in ts[1:]]
        for out in outs:
            self.assertFalse(torch.equal(out, x))
            # same patches, different places
            a, b = self.patches(x), self.patches(out)
            self.assertTrue(torch.equal(a[a[:, 0].argsort()], b[b[:, 0].argsort()]))
        self.assertFalse(torch.equal(outs[0], outs[1]), "two clients share a permutation")

    def test_same_permutation_for_every_sample(self) -> None:
        t = self.transforms()[1]
        x = synthetic_image(size=224)
        self.assertTrue(torch.equal(t(x, 0), t(x, 12345)))

    def test_indivisible_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            PermutePatches(32, (0, 0, 1))(torch.rand(3, 50, 50), 0)


if __name__ == "__main__":
    describe(TestPermutationFeatureSkew.make_partitioner)
