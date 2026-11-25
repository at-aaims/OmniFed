import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from src.omnifed.algorithm_torchtitan import DiLoCo, FedAvg
from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.exchange.legacy import (
    LegacyModelExchange,
)

""" 
conda activate omnifed
cd /home/user/omnifed

python -m unittest discover \
  -s tests \
  -p 'test_grpc_legacy_model_exchange.py' \
  -v
"""

def make_state(value):
    return {
        "weight": torch.tensor([value, value + 2.0]),
        "bias": torch.tensor([value]),
    }


class SimulatedServerCommunicator:
    """Supply client averages and execute the real server callback."""

    def __init__(self, averages):
        self.averages = iter(averages)
        self.processor = None
        self.processor_calls = 0
        self.token_calls = 0
        self.servicer = self

    def set_next_result_processor(self, processor):
        if self.processor is not None:
            raise AssertionError("Previous processor was not consumed")
        self.processor = processor

    def aggregate(self, message, reduction):
        if reduction != AggregationOp.SUM:
            raise AssertionError("Expected SUM")

        if torch.is_tensor(message):
            # Server contributes zero to the token-count collective.
            torch.testing.assert_close(
                message,
                torch.tensor([0.0], dtype=torch.float64),
            )
            self.token_calls += 1
            return torch.tensor([4.0], dtype=torch.float64)

        # Server must contribute zeros, not its previous model.
        for tensor in message.values():
            torch.testing.assert_close(
                tensor,
                torch.zeros_like(tensor),
            )

        if self.processor is None:
            raise AssertionError(
                "Server callback must be registered before aggregation"
            )

        average = next(self.averages)
        processor = self.processor
        self.processor = None
        self.processor_calls += 1

        return processor(average)


class TestLegacyModelExchange(unittest.TestCase):
    def assert_state_close(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(
                actual[name],
                expected[name],
            )

    def test_client_fedavg_weighting_and_returned_model(self):
        local_state = make_state(12.0)
        expected_global = make_state(13.0)

        communicator = Mock()
        communicator.aggregate.side_effect = [
            # Total tokens across clients.
            torch.tensor([4.0], dtype=torch.float64),
            # Updated model returned by the server.
            expected_global,
        ]

        exchange = LegacyModelExchange(
            communicator,
            chunk_size_mb=1,
        )

        result = exchange.exchange_client_model(
            local_state=local_state,
            local_units=1.0,
            algorithm=FedAvg(),
            round_id=0,
        )

        # Client receives and reconstructs the server's result.
        self.assert_state_close(result, expected_global)

        calls = communicator.aggregate.call_args_list
        self.assertEqual(len(calls), 2)

        # First operation submits this client's token count.
        torch.testing.assert_close(
            calls[0].args[0],
            torch.tensor([1.0], dtype=torch.float64),
        )

        # Client weight is 1 / 4.
        submitted = calls[1].args[0]
        self.assert_state_close(
            submitted,
            {
                name: tensor * 0.25
                for name, tensor in local_state.items()
            },
        )

        for call in calls:
            self.assertEqual(call.args[1], AggregationOp.SUM)

    def test_server_fedavg_returns_average(self):
        average = make_state(13.0)
        communicator = SimulatedServerCommunicator([average])

        exchange = LegacyModelExchange(
            communicator,
            chunk_size_mb=1,
        )

        result = exchange.aggregate_server_round(
            global_state=make_state(10.0),
            algorithm=FedAvg(),
            round_id=0,
        )

        self.assert_state_close(result, average)
        self.assertEqual(communicator.token_calls, 1)
        self.assertEqual(communicator.processor_calls, 1)
        self.assertIsNone(communicator.processor)

    def test_server_diloco_preserves_velocity_between_rounds(self):
        communicator = SimulatedServerCommunicator(
            [make_state(13.0), make_state(15.0)]
        )

        exchange = LegacyModelExchange(
            communicator,
            chunk_size_mb=1,
        )

        algorithm = DiLoCo(
            num_clients=2,
            outer_lr=0.7,
            outer_momentum=0.9,
            nesterov=False,
        )

        first = exchange.aggregate_server_round(
            global_state=make_state(10.0),
            algorithm=algorithm,
            round_id=0,
        )

        # delta = 13 - 10 = 3
        # velocity = 0.7 * 3 = 2.1
        # global = 10 + 2.1 = 12.1
        self.assert_state_close(first, make_state(12.1))

        second = exchange.aggregate_server_round(
            global_state=first,
            algorithm=algorithm,
            round_id=1,
        )

        # delta = 15 - 12.1 = 2.9
        # velocity = 0.9 * 2.1 + 0.7 * 2.9 = 3.92
        # global = 12.1 + 3.92 = 16.02
        self.assert_state_close(second, make_state(16.02))

        for velocity in algorithm.velocity.values():
            torch.testing.assert_close(
                velocity,
                torch.full_like(velocity, 3.92),
            )

        self.assertEqual(communicator.token_calls, 2)
        self.assertEqual(communicator.processor_calls, 2)


if __name__ == "__main__":
    unittest.main()