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
Shared helpers for the Caltech-256 concept-shift tests.

Each ``test_*.py`` here pairs with one
``conf/datamodule/skewed-data-sim/cal256-*-concept-shift.yaml``. It inherits

* the loader / partition / Hydra checks every skew scheme gets
  (:class:`SkewSuite` from ``test_label_dist/_common.py``), adapted so that a
  trainer's batches may carry relabelled targets and its eval set may be
  relabelled (``eval_follows_concept``), and
* the concept-shift checks in :class:`ConceptShiftSuite` below.

Run from the repo root (``OMNIFED_DATA_DIR`` holds ``Caltech256/{train,test}``)::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets python -m pytest src/omnifed/data/skewed/test_concept_shift -v

The label logic is plain integer work, so most checks use a small synthetic
labelled dataset and run without Caltech images.
"""

from __future__ import annotations

import pickle
from typing import Any, List

import numpy as np

from src.omnifed.data.skewed.base import ClientTransformDataset
from src.omnifed.data.skewed.test_label_dist._common import NUM_CLIENTS, SkewSuite, fl_env

__all__ = ["NUM_CLIENTS", "CLASSES", "ConceptShiftSuite", "TinyLabelled", "fl_env"]

#: Label values of :class:`TinyLabelled`: as many classes as Caltech-256, but
#: deliberately not 0..C-1, so mappings must work on the dataset's own values.
CLASSES = np.arange(257) * 10 + 5


class TinyLabelled:
    """Synthetic ``(input, label)`` dataset; input ``"x<i>"`` identifies the sample."""

    def __init__(self, n: int = 4000, seed: int = 0) -> None:
        self.targets = CLASSES[np.random.default_rng(seed).integers(0, CLASSES.size, n)].tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        return (f"x{i}", self.targets[i])


class ConceptShiftSuite(SkewSuite):
    """
    Checks shared by every concept-shift scheme. Mix into ``unittest.TestCase``
    and set ``CONFIG``, ``make_partitioner()`` and ``EVAL_FOLLOWS_CONCEPT``.
    """

    FULL_COVERAGE = True  # the default IID split assigns every sample
    EVAL_FOLLOWS_CONCEPT: bool

    def target_transforms(self, num_clients: int = NUM_CLIENTS, classes=CLASSES) -> List[Any]:
        p = self.make_partitioner()
        return [p.client_target_transform(i, num_clients, classes) for i in range(num_clients)]

    # SkewSuite hooks: relabelled targets, relabelled eval ------------------------

    def allowed_train_labels(self, client_dataset) -> set:
        t = client_dataset.target_transform
        return {t(int(self.train_labels[i]), int(i)) for i in client_dataset.indices}

    def check_trainer_eval(self, eval_loader) -> None:
        if self.EVAL_FOLLOWS_CONCEPT:
            self.assertIsInstance(eval_loader.dataset, ClientTransformDataset)
            self.assertIs(eval_loader.dataset.dataset, self.eval_set)
        else:
            super().check_trainer_eval(eval_loader)

    # label transforms (synthetic, no data needed) ---------------------------------

    def test_target_transforms_deterministic_valid_picklable(self) -> None:
        for t in self.target_transforms():
            clone = pickle.loads(pickle.dumps(t))  # what DataLoader workers receive
            for index, y in enumerate(CLASSES.tolist() * 5):
                out = t(y, index)
                self.assertIn(out, CLASSES, t)
                self.assertEqual(out, t(y, index), f"{t} is not deterministic")
                self.assertEqual(out, clone(y, index), f"{t} changes when pickled")
            self.assertEqual(t(12345, 0), 12345, "labels outside the class list must pass through")

    def test_transforms_identical_on_every_rank(self) -> None:
        with fl_env(0):  # same seed, fresh objects = separate ranks
            a = [t.describe() for t in self.target_transforms()]
            b = [t.describe() for t in self.target_transforms()]
        self.assertEqual(a, b)

    def test_split_decides_assignment_and_id_covers_labels(self) -> None:
        ds = TinyLabelled()
        with fl_env(0):
            p = self.make_partitioner()
            for mine, split in zip(p.partition(ds), p.split.partition(ds)):
                self.assertTrue(np.array_equal(mine, split))
            self.assertNotEqual(p.partition_id(ds), p.split.partition_id(ds))
            s = p.summary(ds)
        for key in ("client_features", "observed_class_counts", "relabelled_fraction"):
            self.assertEqual(len(s[key]), NUM_CLIENTS, key)
        self.assertEqual(
            [sum(c) for c in s["observed_class_counts"]], s["sizes"], "relabelling must not lose samples"
        )

    def test_inputs_untouched_labels_transformed(self) -> None:
        ds = TinyLabelled()
        with fl_env(1):
            p = self.make_partitioner()
            client = p.client_dataset(ds)
        t = p.client_target_transform(1, NUM_CLIENTS, CLASSES)
        self.assertIsInstance(client, ClientTransformDataset)
        for i in range(min(300, len(client))):
            idx = int(client.indices[i])
            x, y = client[i]
            self.assertEqual(x, f"x{idx}")
            self.assertEqual(y, t(ds.targets[idx], idx))

    def test_eval_follows_option(self) -> None:
        test = TinyLabelled(n=500, seed=9)
        p = self.make_partitioner()
        ev = p.client_eval_dataset(test, 1, NUM_CLIENTS, CLASSES)
        if not self.EVAL_FOLLOWS_CONCEPT:
            self.assertIs(ev, test)
            return
        t = p.client_target_transform(1, NUM_CLIENTS, CLASSES)
        self.assertEqual(len(ev), len(test))
        for i in range(len(test)):
            self.assertEqual(ev[i], (f"x{i}", t(test.targets[i], i)))

    def test_label_types_preserved(self) -> None:
        import torch

        t = self.target_transforms()[1]
        self.assertIsInstance(t(15, 0), int)
        self.assertIsInstance(t(np.int64(15), 0), np.int64)
        out = t(torch.tensor(15), 0)
        self.assertTrue(torch.is_tensor(out) and out.dtype == torch.int64)
