from __future__ import annotations

import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate

from src.omnifed.algorithm_torchtitan import (
    BaseTorchTitanAlgorithm,
)
from src.omnifed.communicator import (
    AggregationOp,
)
from src.omnifed.communicator.grpc import (
    GrpcCommunicator,
)
from src.omnifed.data.federated_shards import topology_has_server
from src.omnifed.torchtitan.frozen_config import (
    load_frozen_run_config,
    parse_frozen_config_argument,
)
from src.omnifed.torchtitan.profiling import TimingRecorder
from src.omnifed.torchtitan import (
    ModelStateChunkAssembler,
    TorchTitanRole,
    create_torchtitan_backend,
    iter_model_state_chunks,
    resolve_torchtitan_role,
)
from src.omnifed.torchtitan.file_sync import (
    get_initial_model_paths,
    wait_for_file,
)
from src.omnifed.utils import print


def create_torchtitan_algorithm(
    cfg,
) -> BaseTorchTitanAlgorithm:
    algorithm_cfg = getattr(
        cfg,
        "algorithm_torchtitan",
        None,
    )

    if algorithm_cfg is None:
        raise ValueError(
            "TorchTitan execution requires an "
            "algorithm_torchtitan configuration. "
            "For example: "
            "`algorithm_torchtitan=fedavg`."
        )

    algorithm = instantiate(
        algorithm_cfg,
        _recursive_=True,
    )

    if not isinstance(
        algorithm,
        BaseTorchTitanAlgorithm,
    ):
        raise TypeError(
            "algorithm_torchtitan._target_ must create a "
            "BaseTorchTitanAlgorithm instance; got "
            f"{type(algorithm)!r}"
        )

    algorithm.validate()

    return algorithm


def create_federated_communicator(cfg, federated_rank: int, server_addr: str):
    communicator = GrpcCommunicator(
        rank=federated_rank,
        world_size=int(cfg.topology.num_clients) + 1,
        master_addr=server_addr,
        master_port=int(cfg.torchtitan.federated.server_port),
        max_send_message_length=128 * 1024 * 1024,
        max_receive_message_length=128 * 1024 * 1024,
        aggregation_timeout=float(
            cfg.torchtitan.federated.aggregation_timeout
        ),
        client_timeout=float(
            cfg.torchtitan.federated.aggregation_timeout
        ),
        max_retries=120,
        retry_delay=5.0,
    )
    communicator.setup()
    return communicator


# _TENSOR_SLICE_SEPARATOR = ".__omnifed_slice__."



# def maybe_dist_barrier() -> None:
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()


def exchange_model_with_federated_server(
    cfg,
    communicator,
    result,
    federated_algorithm: BaseTorchTitanAlgorithm,
    timer=None,
    round_id=None,
    global_step=None,
):

    with timer.measure("client_load_consolidated_tensor_checkpoint", round_id) if timer else nullcontext():
        payload = torch.load(
            result["checkpoint_path"],
            map_location="cpu",
        )

    local_state = payload["model"]
    local_tokens = float(payload["num_tokens"])

    with timer.measure(
        phase="grpc_client_token_count_aggregation", 
        round_id=round_id,
        iteration="",
        global_step=global_step,
    ) if timer else nullcontext():
        total_tokens = communicator.aggregate(
            torch.tensor([local_tokens], dtype=torch.float64),
            AggregationOp.SUM,
        ).item()

    weight = federated_algorithm.client_weight(
        local_units=local_tokens,
        total_units=total_tokens,
        round_id=round_id,
    )
    # aggregated_state = {}
    assembler = ModelStateChunkAssembler(local_state)

    chunk_size = int(cfg.torchtitan.federated.grpc_chunk_size_mb)

    for chunk_id, chunk in enumerate(iter_model_state_chunks(local_state, chunk_size)):
        prepared_chunk = (
            federated_algorithm.prepare_client_chunk(
                chunk=chunk,
                weight=weight,
                round_id=round_id,
            )
        )

        with timer.measure(
            phase="grpc_client_send_wait_receive_model_chunk",
            round_id=round_id,
            iteration="",
            global_step=global_step,
            extra=(
                f"chunk_id={chunk_id},"
                f"num_tensors={len(prepared_chunk)}"
            ),
        ) if timer else nullcontext():
            result_chunk = communicator.aggregate(
                prepared_chunk,
                AggregationOp.SUM,
            )
        # aggregated_state.update(result_chunk)
        assembler.add_chunk(result_chunk)

    # return aggregated_state
    return assembler.finish()

def run_torchtitan_client(
    cfg: Any,
    role: TorchTitanRole,
) -> None:
    os.environ["RANK"] = str(role.torchtitan_rank)
    os.environ["WORLD_SIZE"] = str(role.torchtitan_world_size)
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    # os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = os.environ["CLIENT_MASTER_ADDR"]
    os.environ["MASTER_PORT"] = os.environ["CLIENT_MASTER_PORT"]
    os.environ["CLIENT_ID"] = str(role.client_id)
    os.environ["CLIENT_LEADER_RANK"] = str(
        cfg.torchtitan.subclusters.leader_rank
    )

    federated_algorithm = create_torchtitan_algorithm(
        cfg
    )

    print(
        f"[client {role.client_id}] "
        f"algorithm_torchtitan="
        f"{federated_algorithm.name}",
        flush=True,
    )

    checkpoint_root = Path(os.environ["CLIENT_CHECKPOINT_ROOT"])
    
    timer = TimingRecorder(
        root=Path(cfg.torchtitan.subclusters.checkpoint_root) / f"job_{os.environ['SLURM_JOB_ID']}",
        role="client",
        rank=role.torchtitan_rank,
    )

    backend = create_torchtitan_backend(
        cfg=cfg,
    )
    backend.timer = timer

    communicator = None
    federated = topology_has_server(cfg.topology)
    if federated and role.is_client_leader:
        communicator = create_federated_communicator(
            cfg=cfg,
            federated_rank=role.federated_rank,
            server_addr=os.environ["SERVER_ADDR"],
        )

    initial_path, initial_ready_path = get_initial_model_paths(cfg)

    try:
        if not federated and role.is_client_leader:
            print(
                f"[client {role.client_id}] topology has no server: no OmniFed gRPC; "
                "leader publishes the initial model for this Titan",
                flush=True,
            )
            global_state = backend.initialize_global_model_on_cpu()
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = initial_path.with_suffix(".tmp")
            torch.save({"model": global_state}, temporary_path)
            os.replace(temporary_path, initial_path)
            initial_ready_path.touch()

        wait_for_file(initial_ready_path)
        print(
            f"[client {role.client_id} rank {role.torchtitan_rank}] "
            f"starting initial load: {initial_path}",
            flush=True,
        )
        backend.load_global_model(
            model_path=str(initial_path),
            round_id=-1,
        )
        print(
            f"[client {role.client_id} rank {role.torchtitan_rank}] "
            "finished initial load",
            flush=True,
        )
        current_global_model_path = str(initial_path)

        # Federated rounds.
        for round_id in range(int(cfg.global_rounds)):
            backend.current_round_id = round_id
            with timer.measure("torchtitan_local_training_total", round_id):
                local_steps = int(
                    cfg.torchtitan.federated.local_steps
                )

                federated_algorithm.before_local_training(
                    backend=backend,
                    global_model_path=current_global_model_path,
                    round_id=round_id,
                )

                train_result = federated_algorithm.train_local(
                    backend=backend,
                    local_steps=local_steps,
                    round_id=round_id,
                )

                federated_algorithm.after_local_training(
                    backend=backend,
                    train_result=train_result,
                    round_id=round_id,
                )
                

            with timer.measure("save_sharded_checkpoint_and_consolidate", round_id):
                local_result = backend.save_and_consolidate(
                    round_id=round_id,
                    num_tokens=int(train_result["num_tokens"]),
                )

            global_path = (
                checkpoint_root
                / f"round_{round_id}"
                / "global_model.pt"
            )

            global_ready_path = global_path.with_suffix(".ready")

            if role.is_client_leader:
                if federated:
                    assert communicator is not None
                    with timer.measure(
                        phase="grpc_round_communication_total",
                        round_id=round_id,
                        iteration="",
                        global_step=int(train_result["step"]),
                        extra=f"client_id={role.client_id}",
                    ):
                        aggregated_state = exchange_model_with_federated_server(
                            cfg=cfg,
                            communicator=communicator,
                            result=local_result,
                            federated_algorithm=federated_algorithm,
                            timer=timer,
                            round_id=round_id,
                            global_step=int(train_result["step"]),
                        )
                else:
                    payload = torch.load(
                        local_result["checkpoint_path"],
                        map_location="cpu",
                    )
                    aggregated_state = payload["model"]

                with timer.measure("client_write_global_tensor_model", round_id):
                    global_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    temporary_path = global_path.with_suffix(".tmp")

                    torch.save(
                        {"model": aggregated_state},
                        temporary_path,
                    )

                    # Atomic publish: model.pt appears only after the full file is written.
                    os.replace(temporary_path, global_path)

                    # Signal non-leader ranks that this round's global model is ready.
                    global_ready_path.touch()

            # Non-leader ranks wait through Lustre, not raw dist.barrier().
            with timer.measure("client_wait_for_global_model_file", round_id):
                wait_for_file(global_ready_path)

            # Wait for this client's leader to finish writing.
            # dist.barrier()
            # maybe_dist_barrier()

            # Convert global tensors back into PP/TP DTensors.
            with timer.measure("convert_global_tensor_to_dtensor", round_id):
                print(
                    f"[client {role.client_id} rank {role.torchtitan_rank}] "
                    f"starting global load for round {round_id}: {global_path}",
                    flush=True,
                )

                backend.load_global_model(
                    model_path=str(global_path),
                    round_id=round_id,
                )

                print(
                    f"[client {role.client_id} rank {role.torchtitan_rank}] "
                    f"finished global load for round {round_id}",
                    flush=True,
                )

                # This global model becomes the reference for the next round.
                current_global_model_path = str(global_path)

            # dist.barrier()
            # maybe_dist_barrier()

    finally:

        # try:
        #     maybe_dist_barrier()
        # except Exception:
        #     pass

        if communicator is not None:
            communicator.close()

        backend.close()


def run_torchtitan_federated_server(
    cfg: Any,
    role: TorchTitanRole,
) -> None:
    server_addr = subprocess.check_output(
        ["hostname"], text=True
    ).strip()

    federated_algorithm = create_torchtitan_algorithm(
        cfg
    )

    print(
        f"[server] algorithm_torchtitan="
        f"{federated_algorithm.name}",
        flush=True,
    )

    communicator = create_federated_communicator(
        cfg=cfg,
        federated_rank=0,
        server_addr=server_addr,
    )

    timer = TimingRecorder(
        root=Path(cfg.torchtitan.subclusters.checkpoint_root) / f"job_{os.environ['SLURM_JOB_ID']}",
        role="server",
        rank=0,
    )

    initial_path, initial_ready_path = get_initial_model_paths(cfg)

    with timer.measure("server_initialize_global_model", "initial"):
        backend = create_torchtitan_backend(cfg=cfg)
        global_state = backend.initialize_global_model_on_cpu()

        initial_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_path = initial_path.with_suffix(".tmp")
        torch.save(
            {
                "model": global_state,
                "initialized_by": "server",
            },
            temporary_path,
        )
        os.replace(temporary_path, initial_path)

        # Clients may only load after the complete checkpoint is visible.
        initial_ready_path.touch()

    print(
        f"[server] published initial global model: {initial_path}",
        flush=True,
    )

    chunk_size = int(cfg.torchtitan.federated.grpc_chunk_size_mb)

    for round_id in range(int(cfg.global_rounds)):
        # Make the current global state available to client leaders.
        # communicator.broadcast(global_state)

        # Server participates with zero samples and zero-valued tensors.
        with timer.measure("server_collect_token_counts", round_id):
            communicator.aggregate(
                torch.tensor([0.0], dtype=torch.float64),
                AggregationOp.SUM,
            )

        # new_global_state = {}
        assembler = ModelStateChunkAssembler(global_state)

        for chunk_id, chunk in enumerate(iter_model_state_chunks(global_state, chunk_size)):
            server_chunk = (
                federated_algorithm.prepare_server_chunk(
                    chunk=chunk,
                    round_id=round_id,
                )
            )

            with timer.measure(
                "server_collect_algorithm_and_return_chunk",
                round_id,
                extra=(
                    f"chunk_id={chunk_id},"
                    f"num_tensors={len(server_chunk)}"
                ),
            ):
                aggregated_chunk = communicator.aggregate(
                    server_chunk,
                    AggregationOp.SUM,
                )
            # new_global_state.update(aggregated_chunk)
            assembler.add_chunk(aggregated_chunk)

        # Preserve the model that entered this round.
        # FedMom, DiLoCo and similar algorithms may need it.
        previous_global_state = global_state

        # Reconstruct the state returned by federated aggregation.
        aggregated_state = assembler.finish()

        # Apply optional algorithm-specific server processing.
        # FedAvg returns aggregated_state unchanged.
        global_state = federated_algorithm.finalize_global_state(
            aggregated_state=aggregated_state,
            previous_global_state=previous_global_state,
            round_id=round_id,
        )

        with timer.measure("server_save_global_model", round_id):
            output = (
                Path(cfg.torchtitan.subclusters.checkpoint_root)
                / f"job_{os.environ['SLURM_JOB_ID']}"
                / "server"
                / f"round_{round_id}"
                / "model.pt"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": global_state}, output)

    communicator.close()


def main() -> None:
    config_path = (
        parse_frozen_config_argument()
    )
    frozen = load_frozen_run_config(
        config_path
    )

    cfg = frozen.cfg
    role = resolve_torchtitan_role(cfg)

    if role.is_server:
        run_torchtitan_federated_server(
            cfg=cfg,
            role=role,
        )
    else:
        run_torchtitan_client(
            cfg=cfg,
            role=role,
        )



if __name__ == "__main__":
    main()
