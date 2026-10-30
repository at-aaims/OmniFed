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
Shard label skew (McMahan et al., 2017) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-shard-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.label_distribution import IIDLabelDistribution, ShardLabelSkew
from src.omnifed.data.skewed.test._common import NUM_CLASSES, LabelSkewSuite, describe, needs_data


class TestShardLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-shard-label-skew"
    FULL_COVERAGE = True

    @staticmethod
    def make_partitioner():
        return ShardLabelSkew(shards_per_client=2, stateful_loader=True)

    @needs_data
    def test_equal_sizes_and_skewed(self) -> None:
        s = self.summary()
        self.assertLessEqual(max(s["sizes"]) - min(s["sizes"]), 2)  # 2 shards of +/-1
        self.assertLess(max(s["classes_per_client"]), NUM_CLASSES)  # nobody sees every class
        iid = self.summary(IIDLabelDistribution())
        self.assertGreater(s["mean_hellinger"], iid["mean_hellinger"])


if __name__ == "__main__":
    describe(TestShardLabelSkew.make_partitioner)
