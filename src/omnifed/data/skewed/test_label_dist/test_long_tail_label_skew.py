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
Long-tail + Dirichlet label skew (Cao et al., 2019; CReFF) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-long-tail-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.omnifed.data.skewed.label_distribution import DirichletLabelSkew, LongTailLabelSkew
from src.omnifed.data.skewed.test_label_dist._common import LabelSkewSuite, describe, needs_data


class TestLongTailLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-long-tail-label-skew"
    FULL_COVERAGE = False

    @staticmethod
    def make_partitioner():
        return LongTailLabelSkew(
            base=DirichletLabelSkew(alpha=0.5, min_samples_per_client=10),
            imbalance_factor=10,
            class_order="index",
            stateful_loader=True,
        )

    @needs_data
    def test_long_tail_profile(self) -> None:
        kept = np.asarray(self.summary()["class_counts"]).sum(axis=0)  # per class, all clients
        orig = np.bincount(self.train_labels, minlength=kept.size)
        self.assertTrue(np.all(kept >= 1), "a class vanished")
        self.assertTrue(np.all(kept <= orig))
        self.assertEqual(kept[0], orig[0])  # head class (index order) keeps everything
        self.assertLess(kept.sum(), orig.sum())
        # tail ends near n_max / imbalance_factor (capped by what the class has)
        self.assertLessEqual(kept[-1], int(orig.max() / 10) + 1)  # +1: float rounding


if __name__ == "__main__":
    describe(TestLongTailLabelSkew.make_partitioner)
