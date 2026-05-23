"""Hydra load + layout-generated topology (Phase A Step 2)."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.engine_communication import validate_hybrid_slurm_topology_alignment
from src.omnifed.hybrid.hydra_loader import load_hybrid_cfg, load_hybrid_cfg_for_engine
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

    def test_load_hybrid_cfg_for_engine_runtime_matches_built_yaml(self) -> None:
        built = load_hybrid_cfg("built_symmetric_2x3.yaml")
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "hybrid": {
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                            "rpc_addr": "127.0.0.1",
                            "rpc_port": 50051,
                            "facility_mpi_addr": "127.0.0.1",
                            "facility_mpi_base_port": 28250,
                            "facility_mpi_port_stride": 40,
                            "facility_name_prefix": "fac",
                        },
                        "training": {"dataset_total_clients": 6},
                    },
                },
            }
        )
        rt = load_hybrid_cfg_for_engine(engine_like)
        self.assertEqual(
            OmegaConf.to_container(rt.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )
        self.assertEqual(
            OmegaConf.to_container(rt.training, resolve=True),
            OmegaConf.to_container(built.training, resolve=True),
        )

    def test_runtime_layout_overrides_topology_config_when_both_presets_agree_on_world_size(self) -> None:
        """If ``layout`` and ``topology_config`` both exist, shapes must imply the same ``world_size``."""
        built = load_hybrid_cfg("built_symmetric_2x3.yaml")
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "communication_mode": "hybrid",
                    "hybrid": {
                        "topology_config": "built_symmetric_2x3.yaml",
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                            "rpc_addr": "127.0.0.1",
                            "rpc_port": 50051,
                            "facility_mpi_addr": "127.0.0.1",
                            "facility_mpi_base_port": 28250,
                            "facility_mpi_port_stride": 40,
                            "facility_name_prefix": "fac",
                        },
                    },
                },
                "topology": {"num_clients": 6},
            }
        )
        validate_hybrid_slurm_topology_alignment(
            engine_like, topology_node_count=7, slurm_ntasks=None
        )
        rt = load_hybrid_cfg_for_engine(engine_like)
        self.assertEqual(
            OmegaConf.to_container(rt.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )

    def test_runtime_layout_conflict_with_topology_preset_errors(self) -> None:
        """Redundant knobs must not disagree on hybrid ``world_size`` (Phase B)."""
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
        with self.assertRaises(ValueError) as ar:
            validate_hybrid_slurm_topology_alignment(cfg, topology_node_count=7)
        self.assertIn("topology_config=", str(ar.exception))
        self.assertIn("world_size", str(ar.exception))

    def test_runtime_layout_merges_communicator_labels_into_topology(self) -> None:
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "hybrid": {
                        "layout": {
                            "num_facilities": 2,
                            "mpi_ranks_per_facility": 3,
                            "dedicated_rpc_server": True,
                            "communicators": {
                                "global_aggregation": "grpc_experimental_branch",
                            },
                        },
                        "training": {"dataset_total_clients": 6},
                    },
                },
            }
        )
        rt = load_hybrid_cfg_for_engine(engine_like)
        self.assertEqual(rt.topology.communicators.intra_facility, "torch_mpi")
        self.assertEqual(rt.topology.communicators.global_aggregation, "grpc_experimental_branch")


if __name__ == "__main__":
    unittest.main()
