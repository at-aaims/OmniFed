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
Dirichlet label skew (Yurochkin et al., 2019; NIID-Bench) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-dirichlet-label-skew.yaml``. Shared checks
(config composes, StatefulDataLoader per trainer, server has no train loader,
ranks agree and are disjoint, matches the Hydra-built loader) come from
:class:`LabelSkewSuite`; see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import torch

from src.omnifed.data.skewed.label_distribution import DirichletLabelSkew
from src.omnifed.data.skewed.test_label_dist._common import (
    LabelSkewSuite,
    build_datamodule,
    describe,
    fl_env,
    needs_data,
)


class TestDirichletLabelSkew(LabelSkewSuite, unittest.TestCase):
    CONFIG = "cal256-dirichlet-label-skew"
    FULL_COVERAGE = True

    @staticmethod
    def make_partitioner():
        return DirichletLabelSkew(
            alpha=0.5,
            balance=True,
            min_samples_per_client=10,
            max_retries=1000,
            stateful_loader=True,
        )

    @needs_data
    def test_skew_increases_as_alpha_decreases(self) -> None:
        hell = {a: self.summary(DirichletLabelSkew(alpha=a))["mean_hellinger"] for a in (0.1, 0.5, 1000.0)}
        self.assertGreater(hell[0.1], hell[0.5])
        self.assertGreater(hell[0.5], hell[1000.0])
        self.assertTrue(all(n >= 10 for n in self.summary()["sizes"]))

    @needs_data
    def test_stateful_resume(self) -> None:
        with fl_env(0):
            loader = build_datamodule(self.make_partitioner(), self.train_set, self.eval_set).train
            it = iter(loader)
            for _ in range(3):
                next(it)
            state = loader.state_dict()
            expected = next(it)[1]

            resumed = build_datamodule(self.make_partitioner(), self.train_set, self.eval_set).train
            resumed.load_state_dict(state)
            got = next(iter(resumed))[1]
        # labels are deterministic given the sampler order (images use random augmentation)
        self.assertTrue(torch.equal(expected, got))


if __name__ == "__main__":
    describe(TestDirichletLabelSkew.make_partitioner)
