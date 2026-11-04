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
Similarity-controlled label skew (Karimireddy et al., 2020) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-similarity-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.label_distribution import SimilarityLabelSkew
from src.omnifed.data.skewed.test_label_dist._common import LabelSkewSuite, describe, needs_data


class TestSimilarityLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-similarity-label-skew"
    FULL_COVERAGE = True

    @staticmethod
    def make_partitioner():
        return SimilarityLabelSkew(similarity=0.2, stateful_loader=True)

    @needs_data
    def test_equal_sizes_and_similarity_controls_skew(self) -> None:
        s = self.summary()
        self.assertLessEqual(max(s["sizes"]) - min(s["sizes"]), 2)  # IID part + sorted part
        hell = {v: self.summary(SimilarityLabelSkew(similarity=v))["mean_hellinger"] for v in (0.0, 0.2, 1.0)}
        self.assertGreater(hell[0.0], hell[0.2])
        self.assertGreater(hell[0.2], hell[1.0])


if __name__ == "__main__":
    describe(TestSimilarityLabelSkew.make_partitioner)
