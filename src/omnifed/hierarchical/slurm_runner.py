"""Hierarchical Slurm two-hop training (inner TorchMPI, outer classic gRPC)."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import types
import warnings
from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from src.omnifed.communicator import AggregationOp
from src.omnifed.communicator.base import BaseCommunicator
from src.omnifed.communicator.grpc import GrpcCommunicator
from src.omnifed.communicator.torch_mpi import TorchMPICommunicator
from src.omnifed.communicator.torch_mpi_adapter import TorchMPIAdapter
from src.omnifed.engine_communication import (
    hierarchical_topology_config_for_slurm,
    validate_hierarchical_slurm_topology_alignment,
)
from src.omnifed.hierarchical.aggregate_config import (
    hierarchical_aggregate_payload_from_cfg,
    hierarchical_communicate_params_from_cfg,
)
from src.omnifed.hierarchical.addr_env import apply_hierarchical_addr_env_overrides
from src.omnifed.hierarchical.comm_bridge import HierarchicalCommBridge
from src.omnifed.hierarchical.hydra_loader import (
    engine_has_facility_topology,
    load_hierarchical_cfg_for_engine,
)
from src.omnifed.hierarchical.outer_grpc import outer_grpc_rank, outer_grpc_world_size
from src.omnifed.hierarchical.run_summary import write_hybrid_slurm_per_round_summary
from src.omnifed.hierarchical.slurm_hostlist import (
    apply_hosts_to_hybrid_topology,
    slurm_job_hosts_ordered,
)
from src.omnifed.hierarchical.slurm_sync import install_slurm_sync
from src.omnifed.hierarchical.topology_roles import (
    facility_local_rank,
    find_facility_for_global_rank,
    hybrid_rank_to_centralized_node_index,
)
from src.omnifed.checkpoint.hybrid_round_checkpoint import (
    load_manifest,
    load_round_model_state,
    resume_start_round,
    save_round_checkpoint,
    should_resume,
)
from src.omnifed.summary.startup import (
    emit_model_startup,
    instantiate_model_timed,
    move_model_to_device_timed,
)
from src.omnifed.utils import print

__all__ = ["run_hierarchical_training"]


class _SoloCommunicator(BaseCommunicator):
    """1-rank no-op local hop so the dedicated server can keep grpc as global_comm."""

    def __init__(self) -> None:
        super().__init__(rank=0, world_size=1, master_addr="127.0.0.1", master_port=0)

    def _setup(self) -> None:
        return None

    def broadcast(self, msg, src: int = 0):
        return msg

    def aggregate(self, msg, reduction):
        return msg

    def close(self) -> None:
        return None


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


def _leader_done_dir(hydra_out_dir: str) -> str:
    return os.path.join(hydra_out_dir, "engine", "hybrid_grpc_leader_done")


def _reset_leader_done_dir(hydra_out_dir: str) -> None:
    d = _leader_done_dir(hydra_out_dir)
    shutil.rmtree(d, ignore_errors=True)


def _write_leader_done_marker(hydra_out_dir: str, rank: int) -> None:
    d = _leader_done_dir(hydra_out_dir)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"rank_{int(rank)}.done")
    with open(p, "w", encoding="utf-8") as f:
        f.write("ok\n")


def _instantiate_server_datamodule(cfg):
    raw = OmegaConf.to_container(cfg.datamodule, resolve=False)
    dm_cfg = OmegaConf.create(raw)
    with open_dict(dm_cfg):
        dm_cfg.train = None
    return instantiate(dm_cfg)


def _make_outer_grpc(
    *,
    grpc_rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    communicate_params: bool,
) -> GrpcCommunicator:
    comm = GrpcCommunicator(
        rank=grpc_rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=int(master_port),
        communicate_params=communicate_params,
        compressor=None,
    )
    comm.setup()
    return comm


def _write_node_results(hydra_out_dir: str, rank: int, payload) -> None:
    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    out_pkl = os.path.join(node_results_dir, f"node_{rank:03d}_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)
    from src.omnifed.execution.slurm.slurm_worker import _to_jsonable

    out_json = os.path.join(node_results_dir, f"node_{rank:03d}_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[hybrid] rank={rank} wrote {out_json}", flush=True)


def run_hierarchical_training(cfg, hydra_out_dir: str, ckpt_dir: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))

    if not engine_has_facility_topology(cfg) and not hierarchical_topology_config_for_slurm(cfg):
        raise ValueError(
            "engine.hierarchical.layout or engine.hierarchical.topology_config required in frozen cfg for hybrid."
        )

    hcfg = load_hierarchical_cfg_for_engine(cfg)
    apply_hierarchical_addr_env_overrides(hcfg)
    topo = hcfg.topology

    validate_hierarchical_slurm_topology_alignment(
        cfg,
        topology_node_count=int(topo.world_size),
        slurm_ntasks=world,
    )

    hosts = slurm_job_hosts_ordered()
    apply_hosts_to_hybrid_topology(topo, hosts)

    rpc_server_rank = int(topo.rpc.server_rank)
    rpc_addr = str(topo.rpc.addr)
    rpc_port = int(topo.rpc.port)
    client_ranks = {int(x) for x in topo.rpc.client_ranks}
    grpc_world = outer_grpc_world_size(client_ranks)

    print(
        f"[hybrid] rank={rank}/{world} rpc_server={rpc_server_rank} "
        f"rpc_addr={rpc_addr}:{rpc_port} grpc_world={grpc_world} hosts_patched=1 "
        f"aggregate_payload={hierarchical_aggregate_payload_from_cfg(cfg)!r}",
        flush=True,
    )

    if rank == rpc_server_rank:
        _run_grpc_server_only(
            cfg,
            rank,
            hydra_out_dir,
            rpc_addr=rpc_addr,
            rpc_port=rpc_port,
            rpc_server_rank=rpc_server_rank,
            client_ranks=client_ranks,
            grpc_world=grpc_world,
        )
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
    nc_raw = OmegaConf.select(cfg, "topology.num_clients", default=None)
    if nc_raw is None:
        ranks = cfg.topology.mpi_ranks_per_facility
        if isinstance(ranks, int):
            nc = int(cfg.topology.num_facilities) * int(ranks)
        else:
            nc = sum(int(x) for x in ranks)
    else:
        nc = int(nc_raw)
    if len(node_cfgs) != nc + 1 or len(node_cfgs) != world:
        raise RuntimeError(
            f"Hierarchical node count mismatch: len(node_configs)={len(node_cfgs)} "
            f"num_clients={nc} world={world}"
        )
    cen_idx = hybrid_rank_to_centralized_node_index(
        rank,
        rpc_server_rank=rpc_server_rank,
        world_size=world,
        num_clients=nc,
    )
    node_cfg = node_cfgs[cen_idx]

    node_name = node_cfg.name
    node_log_dir = os.path.join(hydra_out_dir, node_name)
    os.makedirs(node_log_dir, exist_ok=True)

    os.environ["OMNIFED_NUM_FEDERATED_CLIENTS"] = str(int(nc))
    os.environ["OMNIFED_FEDERATED_CLIENT_INDEX"] = str(int(cen_idx) - 1)
    os.environ["OMNIFED_CENTRALIZED_NODE_INDEX"] = str(int(cen_idx))

    model, model_load_s = instantiate_model_timed(cfg.model)
    model, model_to_device_s = move_model_to_device_timed(
        model, device, non_blocking=True
    )
    startup_log_dir = os.path.join(hydra_out_dir, "engine")
    emit_model_startup(
        rank=rank,
        log_dir=startup_log_dir,
        model_load_s=model_load_s,
        model_to_device_s=model_to_device_s,
    )
    datamodule = instantiate(cfg.datamodule)
    algorithm = instantiate(node_cfg.algorithm, log_dir=node_log_dir)
    emit_model_startup(
        rank=rank,
        log_dir=startup_log_dir,
        model_load_s=model_load_s,
        model_to_device_s=model_to_device_s,
        algorithm=algorithm,
        write_csv=False,
    )

    if hasattr(datamodule, "setup"):
        datamodule.setup()

    print(
        f"[hybrid] rank={rank} facility={facility.name} local_rank={local_rank} "
        f"device={device} backend={backend} leader={rank in client_ranks}",
        flush=True,
    )

    mpi = TorchMPICommunicator(
        id=local_rank,
        total_clients=mpi_ws,
        backend=backend,
        master_addr=mpi_addr,
        master_port=mpi_port,
    )
    bridge = HierarchicalCommBridge()
    communicate_params = hierarchical_communicate_params_from_cfg(cfg)
    local_comm = TorchMPIAdapter(
        mpi,
        rank=local_rank,
        world_size=mpi_ws,
        master_addr=mpi_addr,
        master_port=int(mpi_port),
        communicate_params=communicate_params,
    )

    global_comm: Optional[GrpcCommunicator] = None
    if rank in client_ranks:
        global_comm = _make_outer_grpc(
            grpc_rank=outer_grpc_rank(
                rank, client_ranks, server_rank=rpc_server_rank
            ),
            world_size=grpc_world,
            master_addr=rpc_addr,
            master_port=rpc_port,
            communicate_params=communicate_params,
        )

    local_comm.setup()
    if getattr(center, "global_comm", None) is not None:
        warnings.warn("Hybrid Slurm ignores topology.global_comm on workers.", UserWarning)

    if global_comm is not None:
        model = global_comm.broadcast(model)
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
    if global_comm is not None:
        global_comm.set_logger(algorithm)
    install_slurm_sync(
        algorithm,
        bridge,
        communicate_params=communicate_params,
    )
    payload = hierarchical_aggregate_payload_from_cfg(cfg)
    algorithm.set_aggregate_payload(payload)
    algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)

    start_round = resume_start_round(cfg, ckpt_dir)
    if start_round > 0:
        manifest = load_manifest(ckpt_dir)
        if manifest is None:
            print(
                f"[hybrid] rank={rank} resume requested but no manifest in {ckpt_dir}; "
                "starting round 0",
                flush=True,
            )
            start_round = 0
        else:
            last = int(manifest["last_completed_round"])
            saved_payload = manifest.get("aggregate_payload")
            current_payload = hierarchical_aggregate_payload_from_cfg(cfg)
            if saved_payload is not None and str(saved_payload) != str(current_payload):
                raise ValueError(
                    f"[hybrid] rank={rank} checkpoint aggregate_payload={saved_payload!r} "
                    f"does not match current config {current_payload!r}; "
                    "start a new experiment_id or match engine.hierarchical.aggregate_payload."
                )
            if load_round_model_state(algorithm.local_model, ckpt_dir, last, rank):
                print(
                    f"[hybrid] rank={rank} loaded checkpoint round_{last:03d}; "
                    f"next round_idx={start_round}",
                    flush=True,
                )
            else:
                print(
                    f"[hybrid] rank={rank} WARN: missing shard for round {last}; "
                    "starting round 0",
                    flush=True,
                )
                start_round = 0
    elif should_resume(cfg):
        print(
            f"[hybrid] rank={rank} slurm.resume=true but no prior manifest; fresh start",
            flush=True,
        )

    manifest_writer_rank = min(client_ranks) if client_ranks else None
    exp_id = OmegaConf.select(cfg, "slurm.experiment_id", default=None)

    try:
        for r in range(start_round, algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
            save_round_checkpoint(
                exp_dir=ckpt_dir,
                round_idx=r,
                rank=rank,
                model=algorithm.local_model,
                target_global_rounds=total_rounds,
                is_manifest_writer=(manifest_writer_rank is not None and rank == manifest_writer_rank),
                experiment_id=str(exp_id) if exp_id else None,
                topology_num_clients=int(cfg.topology.num_clients),
                aggregate_payload=hierarchical_aggregate_payload_from_cfg(cfg),
            )
            if manifest_writer_rank is not None and rank == manifest_writer_rank:
                print(
                    f"[hybrid] checkpoint manifest updated after round {r} -> {ckpt_dir}",
                    flush=True,
                )
        if rank in client_ranks:
            _write_leader_done_marker(hydra_out_dir, rank)
    finally:
        if global_comm is not None:
            global_comm.close()
        local_comm.close()

    results = algorithm.get_experiment_data()
    _write_node_results(hydra_out_dir, rank, results)

    if client_ranks and rank == min(client_ranks):
        write_hybrid_slurm_per_round_summary(
            hydra_out_dir,
            topo=topo,
            world_size=world,
            rpc_server_rank=rpc_server_rank,
            rank_writer=rank,
        )

    print(f"[hybrid] rank={rank} finished.", flush=True)


def _run_grpc_server_only(
    cfg,
    rank: int,
    hydra_out_dir: str,
    *,
    rpc_addr: str,
    rpc_port: int,
    rpc_server_rank: int,
    client_ranks: set[int],
    grpc_world: int,
) -> None:
    """Dedicated server: dummy local hop + classic GrpcCommunicator as global_comm."""
    communicate_params = hierarchical_communicate_params_from_cfg(cfg)
    print(
        f"[hybrid] rank={rank} outer GrpcCommunicator server "
        f"world_size={grpc_world} port={rpc_port} "
        f"aggregate_payload={hierarchical_aggregate_payload_from_cfg(cfg)!r}",
        flush=True,
    )

    _reset_leader_done_dir(hydra_out_dir)

    # Listen before trainers finish MPI init / data load.
    global_comm = _make_outer_grpc(
        grpc_rank=outer_grpc_rank(rank, client_ranks, server_rank=rpc_server_rank),
        world_size=grpc_world,
        master_addr=rpc_addr,
        master_port=rpc_port,
        communicate_params=communicate_params,
    )
    local_comm = _SoloCommunicator()
    local_comm.setup()

    node_log_dir = os.path.join(hydra_out_dir, "server")
    os.makedirs(node_log_dir, exist_ok=True)

    model, model_load_s = instantiate_model_timed(cfg.model)
    model, model_to_device_s = move_model_to_device_timed(model, torch.device("cpu"))
    emit_model_startup(
        rank=rank,
        log_dir=os.path.join(hydra_out_dir, "engine"),
        model_load_s=model_load_s,
        model_to_device_s=model_to_device_s,
    )
    datamodule = _instantiate_server_datamodule(cfg)
    algorithm = instantiate(cfg.algorithm, log_dir=node_log_dir)
    emit_model_startup(
        rank=rank,
        log_dir=os.path.join(hydra_out_dir, "engine"),
        model_load_s=model_load_s,
        model_to_device_s=model_to_device_s,
        algorithm=algorithm,
        write_csv=False,
    )
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    model = global_comm.broadcast(model)
    model = local_comm.broadcast(model)

    epochs_per_round = int(getattr(algorithm, "max_epochs_per_round", 1))
    total_rounds = int(cfg.global_rounds)
    algorithm.setup(
        local_comm,
        global_comm,
        model,
        datamodule,
        1,
        epochs_per_round,
        total_rounds,
    )
    global_comm.set_logger(algorithm)
    install_slurm_sync(
        algorithm,
        HierarchicalCommBridge(),
        communicate_params=communicate_params,
    )
    # Dedicated server does not train (n_i=0). FedAvg inner hop would scale
    # every parameter by 0, then track_model_operation("local_agg") aborts.
    # Keep the model on the inner hop; outer gRPC is the real wait (server
    # still submits with facility weight 0 as per FedAvg).
    def _keep_server_model_on_inner_hop(self, comm, weight):
        return self.local_model

    algorithm._aggregate_within_group = types.MethodType(
        _keep_server_model_on_inner_hop, algorithm
    )
    algorithm.set_aggregate_payload(hierarchical_aggregate_payload_from_cfg(cfg))

    try:
        for r in range(algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
    finally:
        global_comm.close()
        local_comm.close()

    _write_node_results(hydra_out_dir, rank, algorithm.get_experiment_data())
    print(f"[hybrid] rank={rank} outer gRPC server finished.", flush=True)
