"""Phase B Step 6: engine.communication_mode and Slurm task count resolution."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.engine_communication import (
    communication_mode,
    hybrid_slurm_world_size_from_conf_name,
    resolve_slurm_ntasks,
)


class TestEngineCommunication(unittest.TestCase):
    def test_default_classic(self) -> None:
        cfg = OmegaConf.create({"engine": {"mode": "slurm"}})
        self.assertEqual(communication_mode(cfg), "classic")

    def test_invalid_mode(self) -> None:
        cfg = OmegaConf.create({"engine": {"communication_mode": "lora"}})
        with self.assertRaises(ValueError):
            communication_mode(cfg)

    def test_resolve_classic(self) -> None:
        cfg = OmegaConf.create({"engine": {"communication_mode": "classic"}})
        self.assertEqual(resolve_slurm_ntasks(cfg, 15), 15)

    def test_resolve_hybrid_builtin_world(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {"topology_config": "built_symmetric_2x3.yaml"},
                }
            }
        )
        self.assertEqual(resolve_slurm_ntasks(cfg, 7), 7)

    def test_resolve_hybrid_mismatch_errors(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {"topology_config": "built_symmetric_2x3.yaml"},
                }
            }
        )
        with self.assertRaises(ValueError):
            resolve_slurm_ntasks(cfg, 16)

    def test_hybrid_world_size_yaml_matches_compose(self) -> None:
        from src.omnifed.hybrid.hydra_loader import (
            hybrid_slurm_world_size_from_topology_yaml,
            load_hybrid_cfg,
        )

        for name in ("built_symmetric_2x3.yaml", "try1_hybrid_topo.yaml"):
            self.assertEqual(
                hybrid_slurm_world_size_from_topology_yaml(name),
                int(load_hybrid_cfg(name).topology.world_size),
            )


if __name__ == "__main__":
    unittest.main()
