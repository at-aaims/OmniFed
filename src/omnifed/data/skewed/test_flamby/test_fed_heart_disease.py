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
Fed-Heart-Disease: 4 hospitals, 13 tabular features, binary label (~40 KB).

Besides the shared checks (:class:`~._common.FlambySuite`), the small size makes
it the dataset for the download module itself (:mod:`.flamby_download`): the
licence gate, several processes downloading at once into a fresh folder, the
command line, and the datamodule refusing to download. Also: the example
configs naming a dataset and a scheme, and the config against a hand-built
datamodule. Run::

    OMNIFED_DATA_DIR=/Users/ssq/Desktop/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_heart_disease.py -v -rs
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.omnifed.data.skewed import flamby_download as fd
from src.omnifed.data.skewed import flamby_partition as fp
from src.omnifed.data.skewed.test_flamby._common import FlambySuite, build, compose_flamby, fl_env

REPO = Path(__file__).resolve().parents[5]


class TestFedHeartDisease(FlambySuite, unittest.TestCase):
    DATASET = "fed_heart_disease"

    def check_batch(self, x, y) -> None:
        self.assertEqual(tuple(x.shape[1:]), (13,))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(set(torch.as_tensor(y).flatten().tolist()) <= {0.0, 1.0})

    def test_binary_labels(self) -> None:
        self.assertTrue(self.train.has_class_labels)
        self.assertEqual(sorted(set(self.train.targets.tolist())), [0, 1])

    def test_yaml_matches_hand_built(self) -> None:
        from_yaml = build(self.DATASET, 4, 2, overrides=self.NO_WORKERS)
        with fl_env(2, 4):
            by_hand = fp.build_flamby_datamodule(self.DATASET, num_workers=0, verbose=False)
        self.assertTrue(np.array_equal(from_yaml.train.dataset.indices, by_hand.train.dataset.indices))
        self.assertEqual(list(from_yaml.eval.dataset.indices), list(by_hand.eval.dataset.indices))

    def test_example_configs_name_dataset_and_scheme(self) -> None:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from src.omnifed.data.skewed.label_distribution import DirichletLabelSkew
        from src.omnifed.data.skewed.quantity_skew import DirichletQuantitySkew
        from src.omnifed.data.skewed.test_flamby._common import CONF_DIR

        with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
            heart = compose(config_name="base", overrides=["datamodule=flamby/heart-disease-pooled-dirichlet-label"])
            ixi = compose(config_name="base", overrides=["datamodule=flamby/ixi-natural-dirichlet-quantity"])
        self.assertEqual(heart.datamodule.dataset, "fed_heart_disease")
        p = instantiate(heart.datamodule.partitioner)
        self.assertIsInstance(p, DirichletLabelSkew)
        self.assertEqual(p.alpha, 0.3)
        self.assertEqual(ixi.datamodule.dataset, "fed_ixi")
        q = instantiate(ixi.datamodule.partitioner)
        self.assertIsInstance(q, fp.FlambyPartition)
        self.assertIsInstance(q.within_center, DirichletQuantitySkew)
        with fl_env(1, 6):  # the heart example builds a loader for a rank
            dm = instantiate(heart.datamodule, num_workers=0)
        self.assertEqual(type(dm.train).__name__, "StatefulDataLoader")

    def test_dataset_and_scheme_from_command_line(self) -> None:
        # seed is not in the scheme yaml (commented out there), so it is added with "+"
        cfg = compose_flamby(self.DATASET, "pooled-iid", ["+partitioner.seed=3", "batch_size=2"]).datamodule
        self.assertEqual((cfg.dataset, cfg.batch_size, cfg.partitioner.seed), (self.DATASET, 2, 3))
        self.assertTrue(cfg.partitioner._target_.endswith("IIDLabelDistribution"))

    def test_two_clients_merge_four_hospitals(self) -> None:
        with fl_env(0, 2):
            plan = fp.FlambyPartition().summary(self.train)["client_centers"]
        self.assertEqual(sorted(c for cs in plan for c in cs), [0, 1, 2, 3])
        dm = build(self.DATASET, 2, 1, overrides=self.NO_WORKERS)
        self.assertEqual(set(self.test.center_ids[np.asarray(dm.eval.dataset.indices)].tolist()), set(plan[1]))


class TestHeartDiseaseDownload(unittest.TestCase):
    """flamby_download into temporary folders (your $OMNIFED_DATA_DIR is untouched)."""

    def setUp(self) -> None:
        if not fd._license_accepted():
            self.skipTest(f"downloads need {fd.FLAMBY_LICENSE_ENV}=1")
        self.data_dir = Path(tempfile.mkdtemp(prefix="omnifed_flamby_"))
        self.addCleanup(shutil.rmtree, self.data_dir, ignore_errors=True)
        self._env = mock.patch.dict(os.environ, {fd.DATA_DIR_ENV: str(self.data_dir)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.target = self.data_dir / "flamby" / "fed_heart_disease"

    def test_datamodule_never_downloads(self) -> None:
        self.assertFalse(fd.is_flamby_downloaded("fed_heart_disease"))
        with self.assertRaisesRegex(FileNotFoundError, "flamby_download fed_heart_disease"):
            fd.flamby_dataset_dir("fed_heart_disease")
        with self.assertRaisesRegex(Exception, "flamby_download"):  # Hydra wraps the error
            build("fed_heart_disease", 4, 0, overrides=("num_workers=0",))
        with self.assertRaises(FileNotFoundError):
            fp.FlambyPooledDataset("fed_heart_disease")
        self.assertFalse(self.target.exists())

    def test_licence_required_to_download(self) -> None:
        with mock.patch.dict(os.environ, {fd.FLAMBY_LICENSE_ENV: ""}):
            with self.assertRaisesRegex(PermissionError, fd.FLAMBY_LICENSE_ENV):
                fd.download_flamby_dataset("fed_heart_disease")
        self.assertFalse((self.target / fd._MARKER).exists())

    def test_download_then_load(self) -> None:
        self.assertEqual(fd.download_flamby_dataset("fed_heart_disease"), self.target)
        self.assertTrue(fd.is_flamby_downloaded("fed_heart_disease"))
        self.assertTrue(any(self.target.glob("*.data")))  # FLamby's raw files
        with mock.patch.dict(os.environ, {fd.FLAMBY_LICENSE_ENV: ""}):  # loading needs no licence
            self.assertEqual(fd.download_flamby_dataset("fed_heart_disease"), self.target)  # no-op
            dm = build("fed_heart_disease", 4, 3, overrides=("num_workers=0",))
        self.assertEqual(set(dm.train.dataset.dataset.center_ids[np.asarray(dm.train.dataset.indices)].tolist()), {3})

    def test_concurrent_downloads_run_once(self) -> None:
        code = (
            "import sys; sys.path.insert(0, {repo!r});"
            "from src.omnifed.data.skewed import flamby_download as fd;"
            "print(fd.download_flamby_dataset('fed_heart_disease'))"
        ).format(repo=str(REPO))
        env = dict(os.environ, **{fd.DATA_DIR_ENV: str(self.data_dir), fd.FLAMBY_LICENSE_ENV: "1"})
        procs = [
            subprocess.Popen([sys.executable, "-c", code], env=env, cwd=str(REPO), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for _ in range(4)
        ]
        outs = [p.communicate(timeout=600)[0] for p in procs]
        self.assertEqual([p.returncode for p in procs], [0, 0, 0, 0], "\n".join(outs))
        self.assertEqual(sum("[flamby] downloading" in o for o in outs), 1, "\n".join(outs))
        self.assertTrue((self.target / fd._MARKER).is_file())

    def test_command_line(self) -> None:
        env = dict(os.environ, **{fd.DATA_DIR_ENV: str(self.data_dir), fd.FLAMBY_LICENSE_ENV: "1"})
        cmd = [sys.executable, "-m", "src.omnifed.data.skewed.flamby_download"]

        def run(*args):
            return subprocess.run(cmd + list(args), env=env, cwd=str(REPO), text=True, capture_output=True, timeout=600)

        listing = run("--list")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertRegex(listing.stdout, r"fed_heart_disease .* missing")
        self.assertIn("fed_ixi", listing.stdout)
        first = run("fed_heart_disease", "fed_tcga_brca")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("fed_heart_disease: ready", first.stdout)
        self.assertIn("fed_tcga_brca: ready", first.stdout)
        again = run("fed_heart_disease")
        self.assertNotIn("[flamby] downloading", again.stdout)  # already there
        self.assertRegex(run("--list").stdout, r"fed_heart_disease .* ready")
        self.assertNotEqual(run("fed_nope").returncode, 0)


if __name__ == "__main__":
    unittest.main()
