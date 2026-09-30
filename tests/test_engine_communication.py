"""Slurm ntasks and hierarchical dispatch follow topology, not communication_mode."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.engine_communication import (
    is_hierarchical_cfg,
    resolve_slurm_ntasks,
    validate_hierarchical_slurm_topology_alignment,
)


def _hierarchical_cfg(**topo_extra):
    topo = {
        "_target_": "src.omnifed.topology.HierarchicalTopology",
        "num_facilities": 2,
        "mpi_ranks_per_facility": 3,
        "dedicated_rpc_server": True,
        "num_clients": 6,
    }
    topo.update(topo_extra)
    return OmegaConf.create({"engine": {"mode": "slurm"}, "topology": topo})


class TestEngineCommunication(unittest.TestCase):
    def test_centralized_is_not_hierarchical(self) -> None:
        cfg = OmegaConf.create(
            {
                "topology": {
                    "_target_": "src.omnifed.topology.CentralizedTopology",
                    "num_clients": 6,
                }
            }
        )
        self.assertFalse(is_hierarchical_cfg(cfg))
        self.assertEqual(resolve_slurm_ntasks(cfg, 7), 7)

    def test_decentralized_is_not_hierarchical(self) -> None:
        cfg = OmegaConf.create(
            {
                "topology": {
                    "_target_": "src.omnifed.topology.DecentralizedTopology",
                    "num_clients": 6,
                }
            }
        )
        self.assertFalse(is_hierarchical_cfg(cfg))
        self.assertEqual(resolve_slurm_ntasks(cfg, 6), 6)

    def test_resolve_hierarchical_world(self) -> None:
        cfg = _hierarchical_cfg()
        self.assertTrue(is_hierarchical_cfg(cfg))
        self.assertEqual(resolve_slurm_ntasks(cfg, 7), 7)

    def test_resolve_hierarchical_mismatch_errors(self) -> None:
        cfg = _hierarchical_cfg()
        with self.assertRaises(ValueError):
            resolve_slurm_ntasks(cfg, 5)

    def test_num_clients_must_match_trainers(self) -> None:
        cfg = _hierarchical_cfg(num_clients=5)
        with self.assertRaises(ValueError) as ar:
            resolve_slurm_ntasks(cfg, 7)
        self.assertIn("topology.num_clients", str(ar.exception))

    def test_validate_vs_slurm_ntasks_mismatch(self) -> None:
        cfg = _hierarchical_cfg()
        with self.assertRaises(ValueError) as ar:
            validate_hierarchical_slurm_topology_alignment(
                cfg, topology_node_count=7, slurm_ntasks=6
            )
        self.assertIn("SLURM_NTASKS", str(ar.exception))

    def test_facility_world_matches_conf_hybrid_preset(self) -> None:
        from src.omnifed.hierarchical.hydra_loader import (
            hierarchical_slurm_world_size_from_engine_layout,
            hierarchical_slurm_world_size_from_topology_yaml,
            load_hierarchical_cfg,
        )

        cfg = _hierarchical_cfg()
        self.assertEqual(
            hierarchical_slurm_world_size_from_engine_layout(cfg),
            hierarchical_slurm_world_size_from_topology_yaml("built_symmetric_2x3.yaml"),
        )
        for name in ("built_symmetric_2x3.yaml", "try1_hybrid_topo.yaml"):
            self.assertEqual(
                hierarchical_slurm_world_size_from_topology_yaml(name),
                int(load_hierarchical_cfg(name).topology.world_size),
            )


if __name__ == "__main__":
    unittest.main()
