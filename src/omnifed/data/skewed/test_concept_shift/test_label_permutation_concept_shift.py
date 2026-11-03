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
Label-permutation concept shift (clustered / multi-task FL) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-label-permutation-concept-shift.yaml``. Shared checks come
from :class:`ConceptShiftSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.omnifed.data.skewed.concept_shift import LabelPermutationConceptShift
from src.omnifed.data.skewed.test_concept_shift._common import (
    CLASSES,
    NUM_CLIENTS,
    ConceptShiftSuite,
    TinyLabelled,
    fl_env,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestLabelPermutationConceptShift(ConceptShiftSuite, unittest.TestCase):
    CONFIG = "cal256-label-permutation-concept-shift"
    EVAL_FOLLOWS_CONCEPT = True

    @staticmethod
    def make_partitioner():
        return LabelPermutationConceptShift(
            num_groups=2, include_identity=True, eval_follows_concept=True, stateful_loader=True
        )

    def test_other_groups_relabel_every_class(self) -> None:
        ts = self.target_transforms()
        self.assertEqual(ts[0].changed, 0)
        for t in (ts[1], ts[3]):
            self.assertEqual(sorted(t.perm.tolist()), list(range(CLASSES.size)))  # one-to-one
            self.assertEqual(t.changed, CLASSES.size)  # no class keeps its label
        self.assertTrue(np.array_equal(ts[1].perm, ts[3].perm))

    def test_observed_counts_are_permuted_counts(self) -> None:
        with fl_env(0):
            p = self.make_partitioner()
            s = p.summary(TinyLabelled())
        for cid in range(NUM_CLIENTS):
            perm = p.client_target_transform(cid, NUM_CLIENTS, CLASSES).perm
            original = np.asarray(s["class_counts"][cid])
            observed = np.asarray(s["observed_class_counts"][cid])
            self.assertTrue(np.array_equal(observed[perm], original))
        self.assertEqual(s["relabelled_fraction"], [0.0, 1.0, 0.0, 1.0])

    def test_without_identity_group(self) -> None:
        p = LabelPermutationConceptShift(num_groups=2, include_identity=False)
        self.assertEqual(p.client_target_transform(0, NUM_CLIENTS, CLASSES).changed, CLASSES.size)


if __name__ == "__main__":
    describe(TestLabelPermutationConceptShift.make_partitioner)
