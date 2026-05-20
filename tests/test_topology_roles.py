"""Facility membership helpers used by hybrid smoke / future slurm_worker."""

import unittest

from omegaconf import OmegaConf

from src.omnifed.hybrid.topology_builder import build_hybrid_topology
from src.omnifed.hybrid.topology_roles import (
    facility_local_rank,
    find_facility_for_global_rank,
)


class TestTopologyRoles(unittest.TestCase):
    def test_find_and_local_rank(self) -> None:
        t = OmegaConf.create(build_hybrid_topology(num_facilities=2, mpi_ranks_per_facility=3))
        f1 = find_facility_for_global_rank(t, 3)
        assert f1 is not None
        self.assertEqual(f1.name, "fac1")
        self.assertEqual(facility_local_rank(f1, 1), 0)
        self.assertEqual(facility_local_rank(f1, 3), 1)
        f2 = find_facility_for_global_rank(t, 6)
        assert f2 is not None
        self.assertEqual(f2.name, "fac2")
        self.assertIsNone(find_facility_for_global_rank(t, 99))


if __name__ == "__main__":
    unittest.main()
