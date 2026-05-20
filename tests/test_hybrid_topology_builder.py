"""Tests for hybrid topology generation (Phase A Step 1)."""

import unittest

from src.omnifed.hybrid.topology_builder import (
    build_hybrid_topology,
    validate_hybrid_topology_dict,
)


class TestBuildHybridTopology(unittest.TestCase):
    def test_symmetric_2x3_matches_try1_layout(self) -> None:
        """Same global ranks / members as conf_hybrid/topology/try1_hybrid_topo.yaml."""
        topo = build_hybrid_topology(
            num_facilities=2,
            mpi_ranks_per_facility=3,
            rpc_port=50051,
            facility_mpi_base_port=28250,
            facility_mpi_port_stride=40,
        )
        expected = {
            "world_size": 7,
            "rpc": {
                "server_rank": 0,
                "addr": "127.0.0.1",
                "port": 50051,
                "client_ranks": [1, 2],
            },
            "facilities": [
                {
                    "name": "fac1",
                    "mpi": {
                        "addr": "127.0.0.1",
                        "port": 28250,
                        "world_size": 3,
                        "members": [1, 3, 4],
                        "leader_rank": 1,
                    },
                },
                {
                    "name": "fac2",
                    "mpi": {
                        "addr": "127.0.0.1",
                        "port": 28290,
                        "world_size": 3,
                        "members": [2, 5, 6],
                        "leader_rank": 2,
                    },
                },
            ],
        }
        self.assertEqual(topo, expected)

    def test_asymmetric_2_and_8(self) -> None:
        """Matches asym_hybrid_topo-style sizes (2 + 8 MPI ranks)."""
        topo = build_hybrid_topology(
            num_facilities=2,
            mpi_ranks_per_facility=[2, 8],
            facility_mpi_base_port=28250,
            facility_mpi_port_stride=40,
        )
        self.assertEqual(topo["world_size"], 11)
        self.assertEqual(topo["rpc"]["client_ranks"], [1, 2])
        self.assertEqual(topo["facilities"][0]["mpi"]["members"], [1, 3])
        self.assertEqual(topo["facilities"][0]["mpi"]["world_size"], 2)
        self.assertEqual(
            topo["facilities"][1]["mpi"]["members"],
            [2, 4, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(topo["facilities"][1]["mpi"]["world_size"], 8)
        validate_hybrid_topology_dict(topo)

    def test_validate_rejects_leader_not_first_in_members(self) -> None:
        bad = {
            "world_size": 3,
            "rpc": {
                "server_rank": 0,
                "addr": "127.0.0.1",
                "port": 50051,
                "client_ranks": [1],
            },
            "facilities": [
                {
                    "name": "fac1",
                    "mpi": {
                        "addr": "127.0.0.1",
                        "port": 28250,
                        "world_size": 2,
                        "members": [2, 1],
                        "leader_rank": 1,
                    },
                },
            ],
        }
        with self.assertRaises(ValueError):
            validate_hybrid_topology_dict(bad)

    def test_int_mpi_ranks_requires_positive(self) -> None:
        with self.assertRaises(ValueError):
            build_hybrid_topology(num_facilities=2, mpi_ranks_per_facility=0)

    def test_list_length_must_match_facilities(self) -> None:
        with self.assertRaises(ValueError):
            build_hybrid_topology(
                num_facilities=2,
                mpi_ranks_per_facility=[3],
            )


if __name__ == "__main__":
    unittest.main()
