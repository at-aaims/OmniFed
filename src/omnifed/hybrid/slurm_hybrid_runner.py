"""
Hybrid ``engine.communication_mode=hybrid`` Slurm training (Phase B step 7).
"""
from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.flora.communicator import grpc_communicator as FloraGrpcComm
from src.flora.communicator import torch_mpi
from src.omnifed.communicator import AggregationOp
from src.omnifed.engine_communication import hybrid_topology_config_for_slurm
from src.omnifed.hybrid.addr_env import apply_hybrid_addr_env_overrides
from src.omnifed.hybrid.comm_bridge import HybridCommBridge
from src.omnifed.hybrid.grpc_leader_comm import GrpcLeaderCommunicator
from src.omnifed.hybrid.hybrid_slurm_sync import install_hybrid_slurm_sync
from src.omnifed.hybrid.hydra_loader import load_hybrid_cfg
from src.omnifed.hybrid.slurm_hostlist import (
    apply_hosts_to_hybrid_topology,
    slurm_job_hosts_ordered,
)
from src.omnifed.hybrid.topology_roles import facility_local_rank, find_facility_for_global_rank
from src.omnifed.hybrid.torch_mpi_adapter import TorchMPIAdapter
from src.omnifed.utils import print

__all__ = ["run_hybrid_training"]


def _worker_device(backend: str, local_rank_fallback: int) -> torch.device:
    b = backend.lower()
    if b == "gloo":
        return torch.device("cpu")
    if b == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("nccl backend requires CUDA/ROCm.")
        lr = int(
            os.environ.get(
                "LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(local_rank_fallback))
            )
        )
        n = torch.cuda.device_count()
        if n < 1:
            raise RuntimeError("nccl requested but torch.cuda.device_count()==0")
        gid = lr % n
        torch.cuda.set_device(gid)
        return torch.device("cuda", gid)
    raise ValueError(f"Unsupported hybrid backend {backend!r} (use gloo or nccl)")


def _safe_len_train(dm) -> int:
    try:
        return len(dm.train) if dm.train is not None else 0
    except Exception:
        return 0


def run_hybrid_training(cfg, hydra_out_dir: str, ckpt_dir: str) -> None:
    _ = ckpt_dir
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))

    htc = hybrid_topology_config_for_slurm(cfg)
    if not htc:
        raise ValueError("engine.hybrid.topology_config missing in frozen cfg.")

    hcfg = load_hybrid_cfg(htc)
    apply_hybrid_addr_env_overrides(hcfg)
    topo = hcfg.topology

    if int(topo.world_size) != world:
        raise RuntimeError(
            f"Hybrid topology world_size={topo.world_size} != SLURM_NTASKS={world}"
        )

    hosts = slurm_job_hosts_ordered()
    apply_hosts_to_hybrid_topology(topo, hosts)

    rpc_server_rank = int(topo.rpc.server_rank)
    rpc_addr = str(topo.rpc.addr)
    rpc_port = int(topo.rpc.port)
    client_ranks = {int(x) for x in topo.rpc.client_ranks}
    rpc_total_clients = 1 + len(client_ranks)

    print(
        f"[hybrid] rank={rank}/{world} rpc_server={rpc_server_rank} "
        f"rpc_addr={rpc_addr}:{rpc_port} hosts_patched=1",
        flush=True,
    )

    if rank == rpc_server_rank:
        _run_grpc_server_only(cfg, rank, hydra_out_dir, rpc_port, rpc_total_clients)
        return

    delay = 2.0 * float(rank)
    if delay > 0:
        time.sleep(delay)

    facility = find_facility_for_global_rank(topo, rank)
    if facility is None:
        print(f"[hybrid][FATAL] rank {rank} not in any facility.", flush=True)
        os._exit(1)

    local_rank = facility_local_rank(facility, rank)
    mpi_ws = int(facility.mpi.world_size)
    mpi_addr = str(facility.mpi.addr)
    mpi_port = str(facility.mpi.port)

    backend = str(
        OmegaConf.select(cfg, "topology.local_comm.backend", default="gloo")
    ).lower()
    device = _worker_device(backend, local_rank)

    center = instantiate(cfg.topology, _recursive_=False)
    center.setup(
        default_algorithm_cfg=cfg.algorithm,
        default_model_cfg=cfg.model,
        default_datamodule_cfg=cfg.datamodule,
    )
    node_cfgs = list(center)
    node_cfg = node_cfgs[rank]

    node_name = node_cfg.name
    node_log_dir = os.path.join(hydra_out_dir, node_name)
    os.makedirs(node_log_dir, exist_ok=True)

    model = instantiate(cfg.model).to(device, non_blocking=True)
    datamodule = instantiate(cfg.datamodule)
    algorithm = instantiate(node_cfg.algorithm, log_dir=node_log_dir)

    if hasattr(datamodule, "setup"):
        datamodule.setup()

    print(
        f"[hybrid] rank={rank} facility={facility.name} local_rank={local_rank} "
        f"device={device} backend={backend} leader={rank in client_ranks}",
        flush=True,
    )

    mpi = torch_mpi.TorchMPICommunicator(
        id=local_rank,
        total_clients=mpi_ws,
        backend=backend,
        master_addr=mpi_addr,
        master_port=mpi_port,
    )
    bridge = HybridCommBridge()
    local_comm = TorchMPIAdapter(
        mpi,
        rank=local_rank,
        world_size=mpi_ws,
        master_addr=mpi_addr,
        master_port=int(mpi_port),
    )

    global_comm: Optional[GrpcLeaderCommunicator] = None
    if rank in client_ranks:
        global_comm = GrpcLeaderCommunicator(
            bridge=bridge,
            global_rank=rank,
            world_size=rpc_total_clients,
            master_addr=rpc_addr,
            master_port=rpc_port,
            rpc_total_clients=rpc_total_clients,
        )
        global_comm.attach_model(model)

    local_comm.setup()
    if global_comm is not None:
        global_comm.setup()

    if getattr(center, "global_comm", None) is not None:
        warnings.warn("Hybrid Slurm ignores topology.global_comm on workers.", UserWarning)

    model = local_comm.broadcast(model, src=0)

    local_iters = _safe_len_train(datamodule)
    group_max = local_comm.aggregate(
        dict(
            iters_per_epoch=torch.tensor(local_iters, dtype=torch.int, device=device),
            epochs_per_round=torch.tensor(
                getattr(algorithm, "max_epochs_per_round", 1),
                dtype=torch.int,
                device=device,
            ),
        ),
        AggregationOp.MAX,
    )
    iters_per_epoch = int(group_max["iters_per_epoch"].detach().cpu().item())
    epochs_per_round = int(group_max["epochs_per_round"].detach().cpu().item())
    total_rounds = int(cfg.global_rounds)

    algorithm.setup(
        local_comm,
        global_comm,
        model,
        datamodule,
        iters_per_epoch,
        epochs_per_round,
        total_rounds,
    )
    install_hybrid_slurm_sync(algorithm, bridge)
    algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)

    try:
        for r in range(algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
    finally:
        if global_comm is not None:
            global_comm.close()
        local_comm.close()

    results = algorithm.get_experiment_data()
    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    out_pkl = os.path.join(node_results_dir, f"node_{rank:03d}_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)

    from src.omnifed.slurm_worker import _to_jsonable

    out_json = os.path.join(node_results_dir, f"node_{rank:03d}_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
    print(f"[hybrid] rank={rank} wrote {out_json}", flush=True)
    print(f"[hybrid] rank={rank} finished.", flush=True)


def _run_grpc_server_only(
    cfg,
    rank: int,
    hydra_out_dir: str,
    rpc_port: int,
    rpc_total_clients: int,
) -> None:
    model = instantiate(cfg.model)
    model = model.to(torch.device("cpu"))

    extra = float(
        OmegaConf.select(cfg, "engine.hybrid.server_run_extra_sec", default=120.0)
    )
    per_round = float(
        OmegaConf.select(cfg, "engine.hybrid.server_sec_per_round", default=180.0)
    )
    rounds = int(cfg.global_rounds)
    nap = extra + rounds * per_round

    comm = FloraGrpcComm.GrpcCommunicator(
        model=model,
        id=0,
        total_clients=rpc_total_clients,
        master_addr="0.0.0.0",
        master_port=int(rpc_port),
        accumulate_updates=True,
        daemon_server=True,
    )
    print(
        f"[hybrid] rank={rank} gRPC daemon listening ~{nap:.0f}s "
        f"(extra={extra}, per_round={per_round}, rounds={rounds})",
        flush=True,
    )
    time.sleep(max(30.0, nap))
    comm.grpc_shutdown()
    print(f"[hybrid] rank={rank} gRPC server shut down.", flush=True)

    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    stub = {"role": "hybrid_grpc_server", "rank": rank}
    from src.omnifed.slurm_worker import _to_jsonable

    out_json = os.path.join(node_results_dir, f"node_{rank:03d}_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(stub), f, indent=2)
