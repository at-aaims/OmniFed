import multiprocessing as mp
import queue
import socket
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from src.omnifed.algorithm_torchtitan import DiLoCo, FedAvg
from src.omnifed.communicator.grpc_factory import (
    create_grpc_communicator,
    create_model_exchange,
)
from src.omnifed.communicator.tensor_chunks import (
    ModelStateChunkAssembler,
    iter_model_state_chunks,
)

"""
conda activate omnifed
cd /home/user/omnifed

python -m unittest discover \
  -s tests \
  -p 'test_grpc_legacy_exchange_integration.py' \
  -v

"""

def make_state(value):
    # 300,000 FP32 values = 1.2 MB.
    # With a 1 MiB chunk limit, this tensor must be sliced.
    return {
        "large_weight": torch.full(
            (300_000,), float(value), dtype=torch.float32
        ),
        "small_bias": torch.tensor(
            [value, value + 2.0], dtype=torch.float32
        ),
    }


def assert_state_close(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError("Returned parameter names differ")

    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
        )


def find_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_local_exchange(algorithm_name, result_queue):
    """Run actual gRPC traffic in an isolated process."""
    communicators = []

    try:
        # Avoid excessive CPU thread creation in a small test.
        torch.set_num_threads(1)

        port = find_free_port()

        for rank in range(3):
            communicator = create_grpc_communicator(
                transport_mode="legacy",
                rank=rank,
                world_size=3,
                master_addr="127.0.0.1",
                master_port=port,
                aggregation_timeout=15.0,
                client_timeout=15.0,
                max_retries=2,
                retry_delay=0.1,
                max_workers=8,
            )
            communicator.set_agg_device("cpu")
            communicators.append(communicator)
            communicator.setup()

        server, client_1, client_2 = [
            create_model_exchange(
                communicator,
                chunk_size_mb=1,
            )
            for communicator in communicators
        ]

        if algorithm_name == "fedavg":
            server_algorithm = FedAvg()
            client_algorithms = [FedAvg(), FedAvg()]
            expected_values = [13.5]
        elif algorithm_name == "diloco":
            server_algorithm = DiLoCo(
                num_clients=2,
                outer_lr=0.7,
                outer_momentum=0.9,
                nesterov=False,
            )
            client_algorithms = [
                DiLoCo(num_clients=2),
                DiLoCo(num_clients=2),
            ]
            expected_values = [12.1, 16.02]
        else:
            raise ValueError(algorithm_name)

        global_state = make_state(10.0)

        # Reuse these communicators and server algorithm across rounds.
        for round_id, expected_value in enumerate(expected_values):
            local_1 = make_state(12.0 + 2.0 * round_id)
            local_2 = make_state(14.0 + 2.0 * round_id)

            # Unequal token counts check FedAvg's weighting.
            # DiLoCo uses equal client weights regardless of token counts.
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        server.aggregate_server_round,
                        global_state=global_state,
                        algorithm=server_algorithm,
                        round_id=round_id,
                    ),
                    executor.submit(
                        client_1.exchange_client_model,
                        local_state=local_1,
                        local_units=1.0,
                        algorithm=client_algorithms[0],
                        round_id=round_id,
                    ),
                    executor.submit(
                        client_2.exchange_client_model,
                        local_state=local_2,
                        local_units=3.0,
                        algorithm=client_algorithms[1],
                        round_id=round_id,
                    ),
                ]

                results = [
                    future.result(timeout=30)
                    for future in futures
                ]

            expected = make_state(expected_value)

            # All participants must receive the same updated model.
            for result in results:
                assert_state_close(result, expected)

            global_state = results[0]

        if algorithm_name == "diloco":
            for velocity in server_algorithm.velocity.values():
                torch.testing.assert_close(
                    velocity,
                    torch.full_like(velocity, 3.92),
                )

    except Exception:
        result_queue.put(("error", traceback.format_exc()))
    else:
        result_queue.put(("ok", algorithm_name))
    finally:
        for communicator in reversed(communicators):
            try:
                communicator.close()
            except Exception:
                traceback.print_exc()


class TestMultiChunkReconstruction(unittest.TestCase):
    def test_sliced_tensor_round_trip(self):
        state = {
            "large_weight": torch.arange(
                300_000, dtype=torch.float32
            ),
            "small_bias": torch.tensor([3.0, 7.0]),
        }

        chunks = list(
            iter_model_state_chunks(
                state,
                chunk_size_mb=1,
            )
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            any(
                ".__omnifed_slice__." in key
                for chunk in chunks
                for key in chunk
            )
        )

        for chunk in chunks:
            size_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in chunk.values()
            )
            self.assertLessEqual(size_bytes, 1024 * 1024)

        assembler = ModelStateChunkAssembler(state)

        for chunk in chunks:
            assembler.add_chunk(chunk)

        assert_state_close(assembler.finish(), state)

    def test_missing_slice_is_rejected(self):
        state = {
            "weight": torch.arange(
                300_000, dtype=torch.float32
            )
        }

        chunks = list(
            iter_model_state_chunks(state, chunk_size_mb=1)
        )

        assembler = ModelStateChunkAssembler(state)

        # Deliberately omit the final slice.
        for chunk in chunks[:-1]:
            assembler.add_chunk(chunk)

        with self.assertRaises(RuntimeError):
            assembler.finish()


class TestLocalGrpcExchange(unittest.TestCase):
    def run_integration(self, algorithm_name):
        # A separate process gives the test a hard deadline even
        # if an RPC or a worker thread gets stuck.
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=run_local_exchange,
            args=(algorithm_name, result_queue),
        )

        process.start()

        try:
            process.join(timeout=60)

            if process.is_alive():
                self.fail(
                    f"{algorithm_name}: local exchange exceeded "
                    "60 seconds; possible deadlock"
                )

            self.assertEqual(
                process.exitcode,
                0,
                f"Child process exited with {process.exitcode}",
            )

            try:
                status, detail = result_queue.get(timeout=5)
            except queue.Empty:
                self.fail("Child process returned no test result")

            self.assertEqual(status, "ok", detail)

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

                if process.is_alive():
                    process.kill()
                    process.join()

            result_queue.close()

    def test_fedavg_one_server_two_clients(self):
        self.run_integration("fedavg")

    def test_diloco_two_rounds(self):
        self.run_integration("diloco")


if __name__ == "__main__":
    unittest.main()