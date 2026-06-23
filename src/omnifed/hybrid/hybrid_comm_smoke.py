"""
Phase A Step 3: minimal hybrid smoke test (torch per-facility + one gRPC round).

No dataset. Use on Slurm with ``SLURM_PROCID`` or pass ``--global-rank``.
Non-server ranks sleep briefly so rank 0 can bind the gRPC port first.

Backends: ``gloo`` (CPU, dev), ``nccl`` (GPU / ROCm RCCL — one visible GPU per task;
set ``LOCAL_RANK`` / ``SLURM_LOCALID`` as usual on Slurm).

Example (local, matches ``generic_hybrid_comm.sh`` stagger)::

    for r in $(seq 0 6); do
      python -m src.omnifed.hybrid.hybrid_comm_smoke \\
        --config built_symmetric_2x3.yaml --global-rank $r --backend gloo &
      sleep 2
    done
    wait

Frontier RCCL: export ``HYBRID_SMOKE_BACKEND=nccl`` and use the Slurm smoke script
(one GPU per node; server rank keeps the model on CPU for gRPC).
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from src.omnifed.hybrid.communicator import global_grpc as rpc_comm
from src.omnifed.hybrid.communicator import torch_mpi
from src.omnifed.hybrid.addr_env import apply_hybrid_addr_env_overrides
from src.omnifed.hybrid.hydra_loader import load_hybrid_cfg
from src.omnifed.hybrid.topology_roles import (
    facility_local_rank,
    find_facility_for_global_rank,
)


def _resolve_global_rank(explicit: int | None) -> int:
    if explicit is not None and explicit >= 0:
        return int(explicit)
    env = os.environ.get("SLURM_PROCID")
    if env is not None:
        return int(env)
    raise ValueError("Set --global-rank or SLURM_PROCID")


def _worker_device_for_backend(backend: str) -> torch.device:
    """CPU for gloo; first visible GPU (LOCAL_RANK / SLURM_LOCALID) for nccl."""
    b = backend.lower()
    if b == "gloo":
        return torch.device("cpu")
    if b == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NCCL/RCCL smoke needs CUDA (ROCm). "
                "Use --backend gloo on CPU-only nodes, or set HIP_VISIBLE_DEVICES."
            )
        local_rank = int(
            os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        )
        n = torch.cuda.device_count()
        if n < 1:
            raise RuntimeError("nccl backend requested but torch.cuda.device_count()==0")
        gpu_id = local_rank % n
        torch.cuda.set_device(gpu_id)
        return torch.device("cuda", gpu_id)
    raise ValueError(f"Unsupported backend {backend!r} (use gloo or nccl)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="built_symmetric_2x3.yaml")
    p.add_argument(
        "--global-rank",
        type=int,
        default=-1,
        help="Hybrid global rank (default: use SLURM_PROCID)",
    )
    p.add_argument("--backend", type=str, default="gloo")
    p.add_argument(
        "--server-wait-sec",
        type=float,
        default=float(os.environ.get("HYBRID_SMOKE_SERVER_WAIT_SEC", "90")),
        help="Rank 0: keep gRPC server up this many seconds, then stop.",
    )
    p.add_argument(
        "--rank-stagger-sec",
        type=float,
        default=float(os.environ.get("HYBRID_SMOKE_RANK_STAGGER_SEC", "2")),
        help="Per-rank delay for non-server ranks before torch/grpc client setup.",
    )
    args = p.parse_args()

    cfg = load_hybrid_cfg(args.config)
    apply_hybrid_addr_env_overrides(cfg)
    topo = cfg.topology
    g = _resolve_global_rank(None if args.global_rank < 0 else args.global_rank)

    rpc_server_rank = int(topo.rpc.server_rank)
    rpc_addr = str(topo.rpc.addr)
    rpc_port = int(topo.rpc.port)
    rpc_client_ranks = {int(x) for x in topo.rpc.client_ranks}
    rpc_total = 1 + len(rpc_client_ranks)

    # gRPC server rank: keep model on CPU so CentralServer stays protobuf/CPU-friendly.
    # Workers use NCCL on GPU when --backend nccl (Frontier RCCL).
    model = nn.Linear(8, 4)
    worker_device: torch.device | None = None
    if g != rpc_server_rank:
        worker_device = _worker_device_for_backend(args.backend)
        model = model.to(worker_device)

    print(
        f"[smoke] rank={g} hostname={os.uname().nodename} "
        f"backend={args.backend} "
        f"device={worker_device or torch.device('cpu')}",
        flush=True,
    )

    if g == rpc_server_rank:
        print(f"[smoke] rank {g}: gRPC server (daemon) on port {rpc_port}", flush=True)
        comm = rpc_comm.GrpcCommunicator(
            model=model,
            id=g,
            total_clients=rpc_total,
            master_addr=rpc_addr,
            master_port=rpc_port,
            accumulate_updates=True,
            daemon_server=True,
        )
        time.sleep(max(0.0, args.server_wait_sec))
        comm.grpc_shutdown()
        print(f"[smoke] rank {g}: server shut down cleanly.", flush=True)
        return

    # Non-server: wait so server is listening (Slurm simultaneous start)
    delay = args.rank_stagger_sec * float(g)
    if delay > 0:
        time.sleep(delay)

    facility = find_facility_for_global_rank(topo, g)
    if facility is None:
        print(f"[smoke] rank {g}: not in any facility, exiting.", flush=True)
        return

    local_rank = facility_local_rank(facility, g)
    mpi_ws = int(facility.mpi.world_size)
    mpi_addr = str(facility.mpi.addr)
    mpi_port = int(facility.mpi.port)
    is_grpc_client = g in rpc_client_ranks

    print(
        f"[smoke] rank {g}: facility={facility.name} local_rank={local_rank} "
        f"mpi_group={mpi_ws} grpc_client={is_grpc_client}",
        flush=True,
    )

    mpi_comm = torch_mpi.TorchMPICommunicator(
        id=local_rank,
        total_clients=mpi_ws,
        backend=args.backend,
        master_addr=mpi_addr,
        master_port=str(mpi_port),
    )

    rpc_c = None
    try:
        # One collective on parameters
        model = mpi_comm.aggregate(msg=model, communicate_params=True, compute_mean=True)

        if is_grpc_client:
            rpc_c = rpc_comm.GrpcCommunicator(
                model=model,
                id=g,
                total_clients=rpc_total,
                master_addr=rpc_addr,
                master_port=rpc_port,
                accumulate_updates=True,
            )

        if rpc_c is not None:
            model = rpc_c.aggregate(
                msg=model,
                batch_samples=1,
                communicate_params=True,
                compute_mean=True,
            )

        model = mpi_comm.broadcast(msg=model, id=0)
        print(f"[smoke] rank {g}: done (torch + optional gRPC + broadcast).", flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rpc_c is not None:
            rpc_c.grpc_shutdown()


if __name__ == "__main__":
    main()
