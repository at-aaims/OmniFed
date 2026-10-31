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
Label-noise concept shift (noisy-label FL, e.g. FedCorr) on Caltech-256.

Pairs with ``conf/datamodule/skewed-data-sim/cal256-label-noise-concept-shift.yaml``. Shared checks come
from :class:`ConceptShiftSuite` (and the loader / partition / Hydra checks it
inherits); see ``_common.py`` for how to run.
"""

from __future__ import annotations

import unittest

from src.omnifed.data.skewed.concept_shift import LabelNoiseConceptShift, NoisyLabels
from src.omnifed.data.skewed.test_concept_shift._common import (
    CLASSES,
    NUM_CLIENTS,
    ConceptShiftSuite,
    TinyLabelled,
    fl_env,
)
from src.omnifed.data.skewed.test_label_dist._common import describe


class TestLabelNoiseConceptShift(ConceptShiftSuite, unittest.TestCase):
    CONFIG = "cal256-label-noise-concept-shift"
    EVAL_FOLLOWS_CONCEPT = False

    @staticmethod
    def make_partitioner():
        return LabelNoiseConceptShift(
            noise_type="symmetric",
            noisy_client_fraction=0.5,
            noise_rate=[0.3, 0.5],
            eval_follows_concept=False,
            stateful_loader=True,
        )

    def test_noisy_clients_and_rates(self) -> None:
        p = self.make_partitioner()
        noisy = p.noisy_clients(NUM_CLIENTS).tolist()
        self.assertEqual(len(noisy), 2)
        for cid in range(NUM_CLIENTS):
            rate = p.client_noise_rate(cid, NUM_CLIENTS)
            if cid in noisy:
                self.assertTrue(0.3 <= rate <= 0.5)
            else:
                self.assertEqual(rate, 0.0)

    def test_realised_noise_matches_rate(self) -> None:
        with fl_env(0):
            p = self.make_partitioner()
            s = p.summary(TinyLabelled(n=20000))
        for cid in range(NUM_CLIENTS):
            self.assertAlmostEqual(s["relabelled_fraction"][cid], p.client_noise_rate(cid, NUM_CLIENTS), delta=0.02)

    def test_symmetric_flips_to_other_classes(self) -> None:
        t = NoisyLabels(CLASSES, 1.0, "symmetric", (0, 1, 2))
        outs = {t(15, i) for i in range(10000)}  # reaches all 256 other classes
        self.assertEqual(outs, set(CLASSES.tolist()) - {15})

    def test_pairflip_goes_to_next_class(self) -> None:
        t = NoisyLabels(CLASSES, 1.0, "pairflip", (0, 1, 2))
        self.assertEqual([t(int(c), 7) for c in CLASSES], CLASSES[1:].tolist() + [int(CLASSES[0])])

    def test_flips_fixed_per_sample(self) -> None:
        t = NoisyLabels(CLASSES, 0.5, "symmetric", (0, 1, 2))
        first = [t(15, i) for i in range(200)]
        self.assertEqual(first, [t(15, i) for i in range(200)])
        self.assertTrue(len(set(first)) > 1)

    def test_invalid_settings_raise(self) -> None:
        for kwargs in ({"noise_type": "uniform"}, {"noise_rate": [0.5, 0.2]}, {"noisy_client_fraction": 1.5}):
            with self.assertRaises(ValueError):
                LabelNoiseConceptShift(**kwargs)


if __name__ == "__main__":
    describe(TestLabelNoiseConceptShift.make_partitioner)
