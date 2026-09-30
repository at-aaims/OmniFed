"""Hydra load + layout-generated topology (Phase A Step 2)."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.engine_communication import validate_hierarchical_slurm_topology_alignment
from src.omnifed.hierarchical.hydra_loader import load_hierarchical_cfg, load_hierarchical_cfg_for_engine
from src.omnifed.hierarchical.topology_builder import build_hierarchical_topology


class TestHydraBuiltTopology(unittest.TestCase):
    def test_built_symmetric_2x3_matches_manual_try1(self) -> None:
        manual = load_hierarchical_cfg("try1_hybrid_topo.yaml")
        built = load_hierarchical_cfg("built_symmetric_2x3.yaml")
        self.assertEqual(
            OmegaConf.to_container(manual.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )
        self.assertEqual(manual.training.dataset_total_clients, 6)
        self.assertEqual(built.training.dataset_total_clients, 6)

    def test_built_asymmetric_2_8_world_size(self) -> None:
        cfg = load_hierarchical_cfg("built_asymmetric_2_8.yaml")
        self.assertEqual(cfg.topology.world_size, 11)
        self.assertEqual(cfg.training.dataset_total_clients, 10)
        expected = build_hierarchical_topology(
            num_facilities=2,
            mpi_ranks_per_facility=[2, 8],
        )
        self.assertEqual(
            OmegaConf.to_container(cfg.topology, resolve=True),
            expected,
        )

    def test_load_hierarchical_cfg_for_engine_runtime_matches_built_yaml(self) -> None:
        built = load_hierarchical_cfg("built_symmetric_2x3.yaml")
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "hierarchical": {
                        "training": {"dataset_total_clients": 6},
                    },
                },
                "topology": {
                    "_target_": "src.omnifed.topology.HierarchicalTopology",
                    "num_facilities": 2,
                    "mpi_ranks_per_facility": 3,
                    "dedicated_rpc_server": True,
                    "rpc_addr": "127.0.0.1",
                    "rpc_port": 50051,
                    "facility_mpi_addr": "127.0.0.1",
                    "facility_mpi_base_port": 28250,
                    "facility_mpi_port_stride": 40,
                    "facility_name_prefix": "fac",
                    "num_clients": 6,
                },
            }
        )
        rt = load_hierarchical_cfg_for_engine(engine_like)
        self.assertEqual(
            OmegaConf.to_container(rt.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )
        self.assertEqual(
            OmegaConf.to_container(rt.training, resolve=True),
            OmegaConf.to_container(built.training, resolve=True),
        )

    def test_runtime_layout_overrides_topology_config_when_both_presets_agree_on_world_size(self) -> None:
        built = load_hierarchical_cfg("built_symmetric_2x3.yaml")
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "hierarchical": {
                        "topology_config": "built_symmetric_2x3.yaml",
                        "training": {"dataset_total_clients": 6},
                    },
                },
                "topology": {
                    "_target_": "src.omnifed.topology.HierarchicalTopology",
                    "num_facilities": 2,
                    "mpi_ranks_per_facility": 3,
                    "dedicated_rpc_server": True,
                    "num_clients": 6,
                    "rpc_addr": "127.0.0.1",
                    "rpc_port": 50051,
                    "facility_mpi_addr": "127.0.0.1",
                    "facility_mpi_base_port": 28250,
                    "facility_mpi_port_stride": 40,
                    "facility_name_prefix": "fac",
                },
            }
        )
        validate_hierarchical_slurm_topology_alignment(
            engine_like, topology_node_count=7, slurm_ntasks=None
        )
        rt = load_hierarchical_cfg_for_engine(engine_like)
        self.assertEqual(
            OmegaConf.to_container(rt.topology, resolve=True),
            OmegaConf.to_container(built.topology, resolve=True),
        )

    def test_runtime_layout_conflict_with_topology_preset_errors(self) -> None:
        cfg = OmegaConf.create(
            {
                "engine": {
                    "hierarchical": {
                        "topology_config": "built_symmetric_2x3.yaml",
                    },
                },
                "topology": {
                    "_target_": "src.omnifed.topology.HierarchicalTopology",
                    "num_facilities": 2,
                    "mpi_ranks_per_facility": 2,
                    "dedicated_rpc_server": True,
                    "num_clients": 6,
                },
            }
        )
        with self.assertRaises(ValueError) as ar:
            validate_hierarchical_slurm_topology_alignment(cfg, topology_node_count=7)
        self.assertIn("world_size", str(ar.exception))

    def test_runtime_layout_merges_communicator_labels_into_topology(self) -> None:
        engine_like = OmegaConf.create(
            {
                "engine": {
                    "hierarchical": {
                        "training": {"dataset_total_clients": 6},
                    },
                },
                "topology": {
                    "_target_": "src.omnifed.topology.HierarchicalTopology",
                    "num_facilities": 2,
                    "mpi_ranks_per_facility": 3,
                    "dedicated_rpc_server": True,
                    "communicators": {
                        "global_aggregation": "grpc_experimental_branch",
                    },
                },
            }
        )
        rt = load_hierarchical_cfg_for_engine(engine_like)
        self.assertEqual(rt.topology.communicators.intra_facility, "torch_mpi")
        self.assertEqual(
            rt.topology.communicators.global_aggregation, "grpc_experimental_branch"
        )


if __name__ == "__main__":
    unittest.main()
