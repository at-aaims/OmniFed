"""Hydra load + layout-generated topology (Phase A Step 2)."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.hybrid.hydra_loader import load_hybrid_cfg
from src.omnifed.hybrid.topology_builder import build_hybrid_topology


class TestHydraBuiltTopology(unittest.TestCase):
    def test_built_symmetric_2x3_matches_manual_try1(self) -> None:
        manual = load_hybrid_cfg("try1_hybrid_topo.yaml")
        built = load_hybrid_cfg("built_symmetric_2x3.yaml")
        self.assertEqual(
            OmegaConf.to_container(manual.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )
        self.assertEqual(manual.training.dataset_total_clients, 6)
        self.assertEqual(built.training.dataset_total_clients, 6)

    def test_built_asymmetric_2_8_world_size(self) -> None:
        cfg = load_hybrid_cfg("built_asymmetric_2_8.yaml")
        self.assertEqual(cfg.topology.world_size, 11)
        self.assertEqual(cfg.training.dataset_total_clients, 10)
        expected = build_hybrid_topology(
            num_facilities=2,
            mpi_ranks_per_facility=[2, 8],
        )
        self.assertEqual(
            OmegaConf.to_container(cfg.topology, resolve=True),
            expected,
        )

    def test_try1_still_loads_without_layout(self) -> None:
        cfg = load_hybrid_cfg("try1_hybrid_topo.yaml")
        self.assertFalse(OmegaConf.is_missing(cfg.topology, "world_size"))
        self.assertEqual(cfg.topology.world_size, 7)


if __name__ == "__main__":
    unittest.main()
