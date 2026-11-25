"""
This preserves the existing synchronous sequence, client weighting, and server-side FedAvg/DiLoCo update.
"""

from contextlib import nullcontext

import torch

from ..base import AggregationOp
from ..tensor_chunks import (
    ModelStateChunkAssembler,
    iter_model_state_chunks,
)
from .base import ModelExchange


class LegacyModelExchange(ModelExchange):
    """Existing synchronous model-chunk exchange."""

    def __init__(self, communicator, *, chunk_size_mb):
        self.communicator = communicator
        self.chunk_size_mb = int(chunk_size_mb)

        if self.chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be positive")

    def exchange_client_model(
        self,
        *,
        local_state,
        local_units,
        algorithm,
        round_id,
        global_step=None,
        timer=None,
    ):
        context = (
            timer.measure(
                phase="grpc_client_token_count_aggregation",
                round_id=round_id,
                iteration="",
                global_step=global_step,
            )
            if timer is not None
            else nullcontext()
        )

        with context:
            total_units = self.communicator.aggregate(
                torch.tensor(
                    [float(local_units)],
                    dtype=torch.float64,
                ),
                AggregationOp.SUM,
            ).item()

        weight = algorithm.client_weight(
            local_units=float(local_units),
            total_units=total_units,
            round_id=round_id,
        )

        assembler = ModelStateChunkAssembler(local_state)

        for chunk_id, chunk in enumerate(
            iter_model_state_chunks(
                local_state,
                self.chunk_size_mb,
            )
        ):
            prepared_chunk = algorithm.prepare_client_chunk(
                chunk=chunk,
                weight=weight,
                round_id=round_id,
            )

            context = (
                timer.measure(
                    phase="grpc_client_send_wait_receive_model_chunk",
                    round_id=round_id,
                    iteration="",
                    global_step=global_step,
                    extra=(
                        f"chunk_id={chunk_id},"
                        f"num_tensors={len(prepared_chunk)}"
                    ),
                )
                if timer is not None
                else nullcontext()
            )

            with context:
                updated_chunk = self.communicator.aggregate(
                    prepared_chunk,
                    AggregationOp.SUM,
                )

            assembler.add_chunk(updated_chunk)

        return assembler.finish()

    def aggregate_server_round(
        self,
        *,
        global_state,
        algorithm,
        round_id,
        timer=None,
    ):
        context = (
            timer.measure(
                "server_collect_token_counts",
                round_id,
            )
            if timer is not None
            else nullcontext()
        )

        with context:
            self.communicator.aggregate(
                torch.tensor([0.0], dtype=torch.float64),
                AggregationOp.SUM,
            )

        assembler = ModelStateChunkAssembler(global_state)

        for chunk_id, chunk in enumerate(
            iter_model_state_chunks(
                global_state,
                self.chunk_size_mb,
            )
        ):
            previous_chunk = {
                name: tensor.detach().clone()
                for name, tensor in chunk.items()
            }

            def process_averaged_chunk(
                averaged_parameters,
                previous_parameters=previous_chunk,
                current_round=round_id,
            ):
                return algorithm.server_step(
                    averaged_parameters=averaged_parameters,
                    previous_global_parameters=previous_parameters,
                    round_id=current_round,
                )

            # Register before the server contributes, so clients
            # receive the algorithm-updated result.
            self.communicator.servicer.set_next_result_processor(
                process_averaged_chunk
            )

            server_chunk = algorithm.prepare_server_chunk(
                chunk=chunk,
                round_id=round_id,
            )

            context = (
                timer.measure(
                    "server_collect_algorithm_and_return_chunk",
                    round_id,
                    extra=(
                        f"chunk_id={chunk_id},"
                        f"num_tensors={len(server_chunk)}"
                    ),
                )
                if timer is not None
                else nullcontext()
            )

            with context:
                updated_chunk = self.communicator.aggregate(
                    server_chunk,
                    AggregationOp.SUM,
                )

            assembler.add_chunk(updated_chunk)

        return assembler.finish()