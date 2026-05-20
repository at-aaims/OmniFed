"""Env overrides for multi-node hybrid smoke addresses."""

import os
import unittest
from unittest.mock import patch

from src.omnifed.hybrid.addr_env import apply_hybrid_addr_env_overrides
from src.omnifed.hybrid.hydra_loader import load_hybrid_cfg


class TestHybridAddrOverrides(unittest.TestCase):
    def test_apply_three_hosts_symmetric_topo(self) -> None:
        cfg = load_hybrid_cfg("built_symmetric_2x3.yaml")
        with patch.dict(
            os.environ,
            {
                "OMNIFED_HYBRID_RPC_ADDR": "10.0.0.1",
                "OMNIFED_HYBRID_RPC_PORT": "60051",
                "OMNIFED_HYBRID_FACILITY_MPI_ADDRS": "10.0.0.2,10.0.0.3",
            },
            clear=False,
        ):
            apply_hybrid_addr_env_overrides(cfg)
        self.assertEqual(str(cfg.topology.rpc.addr), "10.0.0.1")
        self.assertEqual(int(cfg.topology.rpc.port), 60051)
        self.assertEqual(str(cfg.topology.facilities[0].mpi.addr), "10.0.0.2")
        self.assertEqual(str(cfg.topology.facilities[1].mpi.addr), "10.0.0.3")


if __name__ == "__main__":
    unittest.main()
