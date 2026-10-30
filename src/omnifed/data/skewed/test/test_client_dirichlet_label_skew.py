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
Per-client Dirichlet label skew (Hsu et al., 2019) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-client-dirichlet-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.label_distribution import ClientDirichletLabelSkew
from src.omnifed.data.skewed.test._common import LabelSkewSuite, describe, needs_data


class TestClientDirichletLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-client-dirichlet-label-skew"
    FULL_COVERAGE = True

    @staticmethod
    def make_partitioner():
        return ClientDirichletLabelSkew(
            alpha=0.5, prior="global", samples_per_client=None, stateful_loader=True
        )

    @needs_data
    def test_equal_sizes_and_alpha_controls_skew(self) -> None:
        s = self.summary()
        self.assertLessEqual(max(s["sizes"]) - min(s["sizes"]), 1)  # fixed quota N // K
        hell = {a: self.summary(ClientDirichletLabelSkew(alpha=a))["mean_hellinger"] for a in (0.1, 0.5, 1000.0)}
        self.assertGreater(hell[0.1], hell[0.5])
        self.assertGreater(hell[0.5], hell[1000.0])


if __name__ == "__main__":
    describe(TestClientDirichletLabelSkew.make_partitioner)
