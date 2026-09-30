"""GrpcServer sample-weighted aggregation: n_i on the payload RPC."""

from __future__ import annotations

import unittest

import torch

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.grpc_server import GrpcServer


class TestGrpcServerSampleWeighted(unittest.TestCase):
    def test_payload_sum_weights_by_num_samples(self) -> None:
        """sum(n_i x_i) / sum(n_i) on one RPC — no Path A/B flag."""
        server = GrpcServer(world_size=2, communicate_params=False)
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        server._accumulate_into_session(
            session_state, "1", {"w": torch.tensor([1.0, 2.0])}, num_samples=2
        )
        server._accumulate_into_session(
            session_state, "2", {"w": torch.tensor([3.0, 4.0])}, num_samples=6
        )
        session_state["total_samples"] = 8
        self.assertTrue(server.perform_aggregation_if_ready(session_state, session_id))
        # (2*[1,2] + 6*[3,4]) / 8 = [20, 28] / 8
        torch.testing.assert_close(
            session_state["result"]["w"], torch.tensor([2.5, 3.5])
        )


    def test_sum_skips_normalize_when_no_sample_count(self) -> None:
        """Heartbeat / BN-style SUM with total_samples=0 stays a raw SUM."""
        server = GrpcServer(
            world_size=2,
            communicate_params=False,
        )
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        session_state["data"] = {
            "1": {"signal": torch.tensor([1.0])},
            "2": {"signal": torch.tensor([1.0])},
        }
        session_state["total_samples"] = 0

        done = server.perform_aggregation_if_ready(session_state, session_id)
        self.assertTrue(done)
        torch.testing.assert_close(
            session_state["result"]["signal"], torch.tensor([2.0])
        )

    def test_reset_partial_session_on_reduction_mismatch(self) -> None:
        server = GrpcServer(world_size=3, communicate_params=False)
        sid = server.current_aggregation_session
        st = server.aggregation_state[sid]
        st["reduction_type"] = AggregationOp.MAX.value
        st["data"]["1"] = {"x": torch.tensor([1.0])}

        server._reset_aggregation_session(sid)
        self.assertIsNone(st["reduction_type"])
        self.assertEqual(st["data"], {})
        self.assertEqual(st["total_samples"], 0)

    def test_aggregation_clears_submitted_inputs(self) -> None:
        server = GrpcServer(world_size=2, communicate_params=False)
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        session_state["data"] = {
            "1": {"w": torch.tensor([1.0])},
            "2": {"w": torch.tensor([3.0])},
        }

        done = server.perform_aggregation_if_ready(session_state, session_id)
        self.assertTrue(done)
        self.assertEqual(session_state["data"], {})
        self.assertEqual(session_state["participants"], {"1", "2"})
        torch.testing.assert_close(
            session_state["result"]["w"], torch.tensor([4.0])
        )

    def test_session_dropped_after_all_participants_fetch(self) -> None:
        server = GrpcServer(world_size=3, communicate_params=False)
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        session_state["data"] = {
            "server": {"w": torch.tensor([0.0])},
            "1": {"w": torch.tensor([1.0])},
            "2": {"w": torch.tensor([2.0])},
        }

        self.assertTrue(
            server.perform_aggregation_if_ready(session_state, session_id)
        )
        self.assertIn(session_id, server.aggregation_state)

        server.mark_aggregation_result_delivered(session_id, "server")
        server.mark_aggregation_result_delivered(session_id, "1")
        self.assertIn(session_id, server.aggregation_state)

        server.mark_aggregation_result_delivered(session_id, "2")
        self.assertNotIn(session_id, server.aggregation_state)

    def test_running_sum_does_not_keep_all_client_copies(self) -> None:
        server = GrpcServer(world_size=3, communicate_params=False)
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        server._accumulate_into_session(
            session_state, "1", {"w": torch.tensor([1.0, 2.0])}
        )
        server._accumulate_into_session(
            session_state, "2", {"w": torch.tensor([3.0, 4.0])}
        )
        self.assertEqual(session_state["data"], {})
        self.assertEqual(session_state["participants"], {"1", "2"})
        torch.testing.assert_close(
            session_state["accum"]["w"], torch.tensor([4.0, 6.0])
        )
        self.assertFalse(
            server.perform_aggregation_if_ready(session_state, session_id)
        )
        server._accumulate_into_session(
            session_state, "server", {"w": torch.tensor([0.0, 0.0])}
        )
        self.assertTrue(
            server.perform_aggregation_if_ready(session_state, session_id)
        )
        torch.testing.assert_close(
            session_state["result"]["w"], torch.tensor([4.0, 6.0])
        )
        self.assertIsNone(session_state["accum"])


if __name__ == "__main__":
    unittest.main()
