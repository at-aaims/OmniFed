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

    def test_resolve_hybrid_runtime_layout_world(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                        },
                    },
                },
            }
        )
        self.assertEqual(resolve_slurm_ntasks(cfg, 7), 7)

    def test_resolve_hybrid_runtime_layout_mismatch_errors(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                        },
                    },
                },
            }
        )
        with self.assertRaises(ValueError):
            resolve_slurm_ntasks(cfg, 5)

    def test_resolve_hybrid_topology_num_clients_must_match_world_size_minus_one(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {"topology_config": "built_symmetric_2x3.yaml"},
                },
                "topology": {"num_clients": 5},
            }
        )
        with self.assertRaises(ValueError) as ar:
            resolve_slurm_ntasks(cfg, 7)
        self.assertIn("topology.num_clients", str(ar.exception))

    def test_validate_hybrid_vs_slurm_ntasks_mismatch(self) -> None:
        from src.omnifed.engine_communication import validate_hybrid_slurm_topology_alignment

        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {"topology_config": "built_symmetric_2x3.yaml"},
                },
                "topology": {"num_clients": 6},
            }
        )
        with self.assertRaises(ValueError) as ar:
            validate_hybrid_slurm_topology_alignment(
                cfg, topology_node_count=7, slurm_ntasks=6
            )
        self.assertIn("SLURM_NTASKS", str(ar.exception))

    def test_layout_and_preset_same_world_size_passes_duplicate_knobs(self) -> None:
        from src.omnifed.engine_communication import validate_hybrid_slurm_topology_alignment

        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {
                        "topology_config": "built_symmetric_2x3.yaml",
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                        },
                    },
                },
                "topology": {"num_clients": 6},
            }
        )
        ws = validate_hybrid_slurm_topology_alignment(cfg, topology_node_count=7)
        self.assertEqual(ws, 7)

    def test_layout_and_preset_mismatched_world_size_errors(self) -> None:
        from src.omnifed.engine_communication import validate_hybrid_slurm_topology_alignment

        cfg = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {
                        "topology_config": "built_symmetric_2x3.yaml",
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 2,
                            "dedicated_rpc_server": True,
                        },
                    },
                },
                "topology": {"num_clients": 6},
            }
        )
        with self.assertRaises(ValueError) as ctx:
            validate_hybrid_slurm_topology_alignment(cfg, topology_node_count=7)
        self.assertIn("topology_config=", str(ctx.exception))

    def test_hybrid_runtime_layout_world_matches_yaml(self) -> None:
        from src.omnifed.hybrid.hydra_loader import (
            hybrid_slurm_world_size_from_engine_layout,
            hybrid_slurm_world_size_from_topology_yaml,
            load_hybrid_cfg,
        )

        cfg = OmegaConf.create(
            {
                "engine": {
                    "hybrid": {
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                            "rpc_port": 50051,
                            "facility_mpi_base_port": 28250,
                            "facility_mpi_port_stride": 40,
                        },
                    },
                },
            }
        )
        self.assertEqual(
            hybrid_slurm_world_size_from_engine_layout(cfg),
            hybrid_slurm_world_size_from_topology_yaml("built_symmetric_2x3.yaml"),
        )

        for name in ("built_symmetric_2x3.yaml", "try1_hybrid_topo.yaml"):
            self.assertEqual(
                hybrid_slurm_world_size_from_topology_yaml(name),
                int(load_hybrid_cfg(name).topology.world_size),
            )


if __name__ == "__main__":
    unittest.main()
