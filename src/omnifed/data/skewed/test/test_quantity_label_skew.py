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
Quantity-based label skew, #C=k (NIID-Bench), on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-quantity-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.label_distribution import QuantityLabelSkew
from src.omnifed.data.skewed.test._common import LabelSkewSuite, describe, needs_data


class TestQuantityLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-quantity-label-skew"
    FULL_COVERAGE = False

    @staticmethod
    def make_partitioner():
        return QuantityLabelSkew(classes_per_client=K_CLASSES, stateful_loader=True)

    @needs_data
    def test_every_client_has_exactly_k_classes(self) -> None:
        s = self.summary()
        self.assertEqual(set(s["classes_per_client"]), {K_CLASSES})
        # 4 clients x 32 classes cannot cover 257 classes: the rest is dropped
        self.assertLess(s["coverage"], 1.0)


K_CLASSES = 32  # classes_per_client in the yaml


if __name__ == "__main__":
    describe(TestQuantityLabelSkew.make_partitioner)
