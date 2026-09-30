"""Hydra preset parity: layout‑only hybrid config (Phase C) vs file‑preset sibling."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.omnifed.engine_communication import validate_hierarchical_slurm_topology_alignment
from src.omnifed.hierarchical.hydra_loader import load_hierarchical_cfg_for_engine


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestHybridPhaseCPresetCompose(unittest.TestCase):
    def test_layout_preset_has_no_topology_file_and_matches_world_size(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="old_way/test_hybrid_layout_fedavg")

        tcp = OmegaConf.select(cfg, "engine.hierarchical.topology_config")
        self.assertTrue(tcp is None or str(tcp).strip().lower() in ("null", "", "none"))
        self.assertEqual(int(cfg.topology.num_facilities), 2)
        self.assertEqual(int(cfg.topology.mpi_ranks_per_facility), 3)

        ws = validate_hierarchical_slurm_topology_alignment(cfg, topology_node_count=7)
        self.assertEqual(ws, 7)
        self.assertEqual(int(OmegaConf.select(cfg, "topology.num_clients")), 6)

        hcfg = load_hierarchical_cfg_for_engine(cfg)
        self.assertEqual(int(hcfg.topology.world_size), 7)
        self.assertEqual(
            OmegaConf.to_container(hcfg.topology.communicators, resolve=True),
            {"intra_facility": "torch_mpi", "global_aggregation": "grpc"},
        )

        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg_file = compose(config_name="old_way/test_hybrid_engine_contract")
        from src.omnifed.engine_communication import hierarchical_world_size_from_cfg

        self.assertEqual(
            hierarchical_world_size_from_cfg(cfg),
            hierarchical_world_size_from_cfg(cfg_file),
        )


if __name__ == "__main__":
    unittest.main()
