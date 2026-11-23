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
Label-swap concept shift (clustered FL, e.g. CFL) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-label-swap-concept-shift.yaml``. Shared checks come
from :class:`ConceptShiftSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.omnifed.data.skewed.concept_shift import LabelSwapConceptShift
from src.omnifed.data.skewed.test_concept_shift._common import (
    CLASSES,
    NUM_CLIENTS,
    ConceptShiftSuite,
    TinyLabelled,
    fl_env,
)
from src.omnifed.data.skewed.test_label_dist._common import describe

CALTECH = np.arange(257)  # Caltech-256 label values (ImageFolder: 0..256)


class TestLabelSwapConceptShift(ConceptShiftSuite, unittest.TestCase):
    CONFIG = "cal256-label-swap-concept-shift"
    EVAL_FOLLOWS_CONCEPT = True

    @staticmethod
    def make_partitioner():
        return LabelSwapConceptShift(
            num_groups=2, swap_pairs=32, include_identity=True, eval_follows_concept=True, stateful_loader=True
        )

    def test_groups_share_swaps_group0_identity(self) -> None:
        ts = self.target_transforms(classes=CALTECH)
        self.assertEqual([t.changed for t in ts], [0, 64, 0, 64])  # clients 0, 2: group 0
        self.assertTrue(np.array_equal(ts[1].perm, ts[3].perm))

    def test_swaps_are_disjoint_pairs(self) -> None:
        perm = self.target_transforms(classes=CALTECH)[1].perm
        self.assertTrue(np.array_equal(perm[perm], np.arange(perm.size)))  # swapping twice undoes it

    def test_groups_differ(self) -> None:
        p = LabelSwapConceptShift(num_groups=3, swap_pairs=4)
        a, b = (p.client_target_transform(i, 3, CALTECH).perm for i in (1, 2))
        self.assertFalse(np.array_equal(a, b))

    def test_too_many_pairs_raises(self) -> None:
        with self.assertRaises(ValueError):
            LabelSwapConceptShift(swap_pairs=6).client_target_transform(1, NUM_CLIENTS, np.arange(10))

    def test_relabelled_fraction(self) -> None:
        with fl_env(0):
            s = LabelSwapConceptShift(num_groups=2, swap_pairs=2).summary(TinyLabelled())
        self.assertEqual(s["relabelled_fraction"][0], 0.0)
        self.assertGreater(s["relabelled_fraction"][1], 0.0)  # 4 of 257 classes swapped



if __name__ == "__main__":
    describe(TestLabelSwapConceptShift.make_partitioner)
