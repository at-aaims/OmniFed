"""Tests for hybrid round-end checkpoint helpers."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from src.omnifed.checkpoint.hybrid_round_checkpoint import (
    is_training_complete,
    load_manifest,
    load_round_model_state,
    manifest_path,
    model_shard_path,
    resolve_experiment_checkpoint_dir,
    resume_start_round,
    save_round_checkpoint,
    should_resume,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class TestHybridRoundCheckpoint(unittest.TestCase):
    def test_resolve_experiment_dir(self) -> None:
        cfg = OmegaConf.create(
            {
                "slurm": {
                    "checkpoint_dir": "/lustre/ckpts",
                    "experiment_id": "llama150_7_v1",
                }
            }
        )
        self.assertEqual(
            resolve_experiment_checkpoint_dir(cfg),
            os.path.join("/lustre/ckpts", "llama150_7_v1"),
        )

    def test_save_load_round_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = os.path.join(tmp, "exp1")
            model = _Tiny()
            save_round_checkpoint(
                exp_dir=exp_dir,
                round_idx=0,
                rank=2,
                model=model,
                target_global_rounds=3,
                is_manifest_writer=True,
                experiment_id="exp1",
                topology_num_clients=6,
            )
            self.assertTrue(os.path.isfile(model_shard_path(exp_dir, 0, 2)))
            manifest = load_manifest(exp_dir)
            assert manifest is not None
            self.assertEqual(manifest["last_completed_round"], 0)
            self.assertEqual(manifest["next_round_idx"], 1)
            self.assertEqual(manifest["status"], "in_progress")

            model2 = _Tiny()
            model2.w.data.zero_()
            self.assertTrue(load_round_model_state(model2, exp_dir, 0, 2))
            self.assertTrue(torch.allclose(model2.w, model.w))

            cfg = OmegaConf.create({"slurm": {"resume": True}})
            self.assertEqual(resume_start_round(cfg, exp_dir), 1)

            save_round_checkpoint(
                exp_dir=exp_dir,
                round_idx=2,
                rank=2,
                model=model,
                target_global_rounds=3,
                is_manifest_writer=True,
                experiment_id="exp1",
                topology_num_clients=6,
            )
            manifest = load_manifest(exp_dir)
            assert manifest is not None
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["next_round_idx"], 3)
            self.assertTrue(is_training_complete(exp_dir))

    def test_is_training_complete_false_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_training_complete(tmp))

    def test_should_resume_parses_string_booleans(self) -> None:
        self.assertFalse(should_resume(OmegaConf.create({"slurm": {"resume": "false"}})))
        self.assertFalse(should_resume(OmegaConf.create({"slurm": {"resume": "False"}})))
        self.assertTrue(should_resume(OmegaConf.create({"slurm": {"resume": "true"}})))
        self.assertTrue(should_resume(OmegaConf.create({"slurm": {"resume": True}})))


if __name__ == "__main__":
    unittest.main()
