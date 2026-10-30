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
IID baseline on Caltech-256: no label skew, near-equal client sizes.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-iid.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.label_distribution import IIDLabelDistribution
from src.omnifed.data.skewed.test._common import LabelSkewSuite, describe, needs_data


class TestIIDLabelDistribution(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-iid"
    FULL_COVERAGE = True

    @staticmethod
    def make_partitioner():
        return IIDLabelDistribution(stateful_loader=True)

    @needs_data
    def test_clients_match_global_distribution(self) -> None:
        s = self.summary()
        self.assertLessEqual(max(s["sizes"]) - min(s["sizes"]), 1)
        self.assertLess(s["mean_hellinger"], 0.15)  # only sampling noise
        self.assertGreaterEqual(min(s["classes_per_client"]), 250)


if __name__ == "__main__":
    describe(TestIIDLabelDistribution.make_partitioner)
