"""Test transport selection without launching Slurm, Ray, or gRPC."""

""" 
conda activate omnifed
cd /home/user/omnifed

python -m unittest discover \
  -s tests \
  -p 'test_grpc_transport_selection.py' \
  -v 
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

from src.omnifed.communicator.grpc_factory import (
    create_grpc_communicator,
    create_model_exchange,
)
from src.omnifed.communicator.exchange.legacy import (
    LegacyModelExchange,
)


class TestGrpcTransportSelection(unittest.TestCase):
    def test_legacy_selects_existing_communicator(self):
        arguments = {
            "rank": 1,
            "world_size": 3,
            "master_addr": "server.example",
            "master_port": 50051,
            "max_send_message_length": 128 * 1024 * 1024,
            "max_receive_message_length": 128 * 1024 * 1024,
            "aggregation_timeout": 3600.0,
            "client_timeout": 3600.0,
            "max_retries": 120,
            "retry_delay": 5.0,
        }

        # Patch where the factory looks up the class.
        with patch(
            "src.omnifed.communicator.grpc_factory.GrpcCommunicator"
        ) as legacy_class:
            communicator = create_grpc_communicator(
                transport_mode="legacy",
                **arguments,
            )

            legacy_class.assert_called_once_with(**arguments)
            self.assertIs(communicator, legacy_class.return_value)
            self.assertEqual(
                communicator.transport_mode,
                "legacy",
            )

    def test_missing_mode_defaults_to_legacy(self):
        with patch(
            "src.omnifed.communicator.grpc_factory.GrpcCommunicator"
        ) as legacy_class:
            communicator = create_grpc_communicator(
                rank=0,
                world_size=3,
            )

            legacy_class.assert_called_once()
            self.assertEqual(
                communicator.transport_mode,
                "legacy",
            )

    def test_optimized_does_not_fall_back_to_legacy(self):
        with patch(
            "src.omnifed.communicator.grpc_factory.GrpcCommunicator"
        ) as legacy_class:
            with self.assertRaisesRegex(
                NotImplementedError,
                "Optimized",
            ):
                create_grpc_communicator(
                    transport_mode="optimized",
                    rank=0,
                    world_size=3,
                )

            legacy_class.assert_not_called()

    def test_invalid_mode_is_rejected(self):
        with patch(
            "src.omnifed.communicator.grpc_factory.GrpcCommunicator"
        ) as legacy_class:
            with self.assertRaises(ValueError):
                create_grpc_communicator(
                    transport_mode="typo",
                    rank=0,
                    world_size=3,
                )

            legacy_class.assert_not_called()

    def test_legacy_selects_matching_model_exchange(self):
        communicator = SimpleNamespace(
            transport_mode="legacy"
        )

        exchange = create_model_exchange(
            communicator,
            chunk_size_mb=64,
        )

        self.assertIsInstance(exchange, LegacyModelExchange)
        self.assertIs(exchange.communicator, communicator)
        self.assertEqual(exchange.chunk_size_mb, 64)
        self.assertTrue(callable(exchange.exchange_client_model))
        self.assertTrue(callable(exchange.aggregate_server_round))

    def test_optimized_model_exchange_is_not_legacy(self):
        communicator = SimpleNamespace(
            transport_mode="optimized"
        )

        with self.assertRaises(NotImplementedError):
            create_model_exchange(
                communicator,
                chunk_size_mb=64,
            )

    def test_experiment_configs_reach_factory(self):
        config_directory = str(
            Path(__file__).resolve().parents[1] / "conf"
        )

        with initialize_config_dir(
            config_dir=config_directory,
            version_base=None,
        ):
            for experiment in (
                "experiment/single_level_torchtitan_fedavg",
                "experiment/single_level_torchtitan_diloco",
            ):
                for mode in ("legacy", "optimized"):
                    with self.subTest(
                        experiment=experiment,
                        mode=mode,
                    ):
                        cfg = compose(
                            config_name=experiment,
                            overrides=[
                                "topology.local_comm."
                                f"transport_mode={mode}"
                            ],
                        )

                        self.assertEqual(
                            cfg.topology.local_comm._target_,
                            "src.omnifed.communicator.grpc_factory."
                            "create_grpc_communicator",
                        )

                        # Verify Hydra forwards the configured mode.
                        with patch(
                            "src.omnifed.communicator.grpc_factory."
                            "create_grpc_communicator",
                            return_value=SimpleNamespace(),
                        ) as factory:
                            instantiate(
                                cfg.topology.local_comm,
                                rank=0,
                                world_size=3,
                            )

                            factory.assert_called_once()
                            self.assertEqual(
                                factory.call_args.kwargs[
                                    "transport_mode"
                                ],
                                mode,
                            )


if __name__ == "__main__":
    unittest.main()