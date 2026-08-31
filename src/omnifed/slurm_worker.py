# src/omnifed/slurm_worker.py
from __future__ import annotations
import argparse, json, os, signal, time, subprocess, pickle, warnings
from typing import Any, Dict, Optional
import numpy as np
from dataclasses import is_dataclass, asdict
import csv

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torch.distributed as dist

from src.omnifed.device_resolver import (
    cuda_device_for_local_rank,
    resolve_slurm_devices,
)
from src.omnifed.engine_communication import communication_mode
from src.omnifed.classic.classic_grad_config import (
    classic_aggregate_payload_from_cfg,
    format_classic_grad_policy,
)
from src.omnifed.communicator import AggregationOp
from src.omnifed.utils import print  # pretty printer used elsewhere

import threading
from datetime import datetime

def _first_host_from_nodelist() -> str:
    out = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
        text=True,
    )
    return out.strip().splitlines()[0]


def _grpc_server_ready_marker(hydra_out_dir: str) -> str:
    """Shared Lustre marker: rank 0 creates it after gRPC listen; clients poll before connect."""
    return os.path.join(hydra_out_dir, "engine", ".grpc_server_ready")


def _write_grpc_server_ready_marker(
    hydra_out_dir: str, *, master_addr: str, master_port: str
) -> str:
    path = _grpc_server_ready_marker(hydra_out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{master_addr}:{master_port}\n")
    return path


def _wait_for_grpc_server_ready_marker(
    hydra_out_dir: str,
    *,
    rank: int,
    timeout_s: Optional[float] = None,
    poll_s: float = 1.0,
) -> None:
    """Block until rank 0 writes ``engine/.grpc_server_ready`` (multi-node Slurm startup)."""
    if timeout_s is None:
        timeout_s = float(os.environ.get("OMNIFED_GRPC_READY_TIMEOUT_SEC", "900"))
    poll_s = float(os.environ.get("OMNIFED_GRPC_READY_POLL_SEC", str(poll_s)))
    path = _grpc_server_ready_marker(hydra_out_dir)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.isfile(path):
            print(
                f"[slurm_worker] rank={rank}: gRPC server ready marker found ({path})",
                flush=True,
            )
            return
        time.sleep(poll_s)
    raise RuntimeError(
        f"[slurm_worker] rank={rank}: timed out after {timeout_s:.0f}s waiting for "
        f"gRPC server ready marker {path}. Rank 0 may not have started listening."
    )


def _uses_grpc_centralized_server(local_comm) -> bool:
    return hasattr(local_comm, "is_server") and hasattr(local_comm, "master_port")


def _classic_comm_backend_name(local_comm) -> str:
    name = type(local_comm).__name__.lower()
    if "grpc" in name:
        return "grpc"
    if "torchdist" in name:
        return "torchdist"
    return name


def install_preemption_handlers(on_checkpoint):
    def _h(signum, frame):
        try:
            on_checkpoint(reason=f"signal:{signum}")
        finally:
            time.sleep(2)
            os._exit(0)
    signal.signal(getattr(signal, "SIGUSR1"), _h)
    signal.signal(getattr(signal, "SIGTERM"), _h)


def _safe_len_train(dm) -> int:
    try:
        return len(dm.train) if dm.train is not None else 0
    except Exception:
        return 0


def _resolve_slurm_device(
    node_cfg,
    *,
    rank: int,
    local_rank: int,
    local_comm,
) -> torch.device:
    """Model device for this Slurm task (classic path). See ``resolve_slurm_devices``."""
    return resolve_slurm_devices(
        node_cfg, rank=rank, local_rank=local_rank, local_comm=local_comm
    ).model_device


def _resolve_device_auto(rank: Optional[int]) -> torch.device:
    """GPU by local rank, or CPU if this process sees no CUDA device."""
    if rank is None:
        warnings.warn("No rank provided, defaulting to GPU 0")
        rank = 0
    return cuda_device_for_local_rank(rank)


def _gpu_probe(prefix: str = "GPU"):
    """Best-effort CUDA probe for logs (does not crash if NVML is unavailable)."""
    try:
        print(
            f"{prefix} probe: torch.cuda.is_available()={torch.cuda.is_available()} | "
            f"device_count={torch.cuda.device_count()} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}"
        )
        if torch.cuda.is_available():
            cur = torch.cuda.current_device()
            print(
                f"{prefix} probe: current_device={cur} name={torch.cuda.get_device_name(cur)}"
            )
    except Exception as e:
        print(f"{prefix} probe: (non-fatal) {e}")

def _to_jsonable(obj):
    # dataclasses → dict
    if is_dataclass(obj):
        obj = asdict(obj)

    # basic types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # torch tensors / numpy arrays
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
            # 0-dim tensors / numpy scalars
            try:
                return obj.item()
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    # common torch meta
    if str(type(obj)).startswith("<class 'torch.") or str(type(obj)).startswith("<class 'numpy."):
        return str(obj)

    # fallback: __dict__ or string
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


def start_gpu_memory_logger(rank: int, log_dir: str, interval_sec: float = 5.0):
    """
    Periodically log GPU memory usage for every rank.
    Works on ROCm through torch.cuda APIs.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"rank{int(rank)}_gpu_memory.log")
    stop_event = threading.Event()

    def _logger():
        header_needed = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a", buffering=1) as f:
            if header_needed:
                f.write(
                    "timestamp,rank,device,allocated_mb,reserved_mb,"
                    "max_allocated_mb,max_reserved_mb\n"
                )

            while not stop_event.is_set():
                try:
                    if torch.cuda.is_available():
                        device_idx = torch.cuda.current_device()
                        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
                        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
                        max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)

                        f.write(
                            f"{datetime.now().isoformat()},{rank},{device_idx},"
                            f"{allocated:.2f},{reserved:.2f},"
                            f"{max_allocated:.2f},{max_reserved:.2f}\n"
                        )
                    else:
                        f.write(f"{datetime.now().isoformat()},{rank},NA,NA,NA,NA,NA\n")
                except Exception as e:
                    f.write(f"{datetime.now().isoformat()},{rank},ERROR,{str(e)},,,\n")

                stop_event.wait(interval_sec)

    t = threading.Thread(target=_logger, daemon=True)
    t.start()
    return stop_event, log_path


def log_gpu_memory_snapshot(rank: int, log_path: str, tag: str):
    """
    Write one on-demand GPU memory snapshot for this rank.
    """
    if not torch.cuda.is_available():
        return

    try:
        device_idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2)

        with open(log_path, "a", buffering=1) as f:
            f.write(
                f"{datetime.now().isoformat()},{rank},{device_idx},"
                f"{allocated:.2f},{reserved:.2f},"
                f"{max_allocated:.2f},{max_reserved:.2f},tag={tag}\n"
            )
    except Exception:
        pass


def main():
    # ---------- args & frozen config ----------
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-json", required=True, help="Path to engine_frozen.json written by Engine")
    args = p.parse_args()

    with open(args.cfg_json, "r") as f:
        raw = json.load(f)

    cfg = OmegaConf.create(raw["cfg"])          # EngineConfig-like dict (resolved)
    hydra_out_dir = raw["hydra_output_dir"]
    ckpt_dir = raw.get("slurm_checkpoint_dir") or os.path.join(hydra_out_dir, "engine", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    if communication_mode(cfg) == "hybrid":
        from src.omnifed.hybrid.slurm_hybrid_runner import run_hybrid_training

        run_hybrid_training(cfg, hydra_out_dir, ckpt_dir)
        raise SystemExit(0)

    # ---------- Slurm sizing & env ----------
    rank  = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

    # os.environ.setdefault("MASTER_ADDR", _first_host_from_nodelist())
    # os.environ.setdefault("MASTER_PORT", str(cfg.topology.local_comm.master_port))
    # os.environ.setdefault("RANK", str(rank))
    # os.environ.setdefault("WORLD_SIZE", str(world))

    master_addr = _first_host_from_nodelist()
    master_port = str(cfg.topology.local_comm.master_port)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)

    # Patch communicator config so it does not keep localhost/127.0.0.1
    if hasattr(cfg.topology, "local_comm") and cfg.topology.local_comm is not None:
        OmegaConf.set_struct(cfg.topology.local_comm, False)
        if "master_port" in cfg.topology.local_comm:
            cfg.topology.local_comm["master_addr"] = master_addr
            cfg.topology.local_comm["master_port"] = master_port
        elif "master_addr" in cfg.topology.local_comm:
            cfg.topology.local_comm["master_addr"] = master_addr

    if hasattr(cfg.topology, "global_comm") and cfg.topology.global_comm is not None:
        OmegaConf.set_struct(cfg.topology.global_comm, False)
        if "master_port" in cfg.topology.global_comm:
            cfg.topology.global_comm["master_addr"] = master_addr
            cfg.topology.global_comm["master_port"] = master_port
        elif "master_addr" in cfg.topology.global_comm:
            cfg.topology.global_comm["master_addr"] = master_addr

    lc = cfg.topology.local_comm
    print(
        f"[main] patched communicator config: "
        f"local_comm.master_addr={OmegaConf.select(lc, 'master_addr', default=master_addr)} "
        f"local_comm.master_port={OmegaConf.select(lc, 'master_port', default=master_port)}",
        flush=True,
    )

    print(f"[main]  rank={rank} world={world} MASTER_ADDR={os.environ['MASTER_ADDR']} PORT={os.environ['MASTER_PORT']}", flush=True)
    _gpu_probe(prefix=f"[rank{rank}]")

    # ---------- Build topology & pick this node ----------
    topology = instantiate(cfg.topology, _recursive_=False)
    topology.setup(
        default_algorithm_cfg=cfg.algorithm,
        default_model_cfg=cfg.model,
        default_datamodule_cfg=cfg.datamodule,
    )
    node_configs = list(topology)
    if world != len(node_configs):
        print(f"[slurm_worker][FATAL] SLURM tasks ({world}) != topology nodes ({len(node_configs)})", flush=True)
        os._exit(1)
    node_cfg = node_configs[rank]

    # ---------- Paths (match Ray Node behavior) ----------
    node_name = node_cfg.name
    log_dir_base = getattr(node_cfg, "log_dir_base", None) or hydra_out_dir
    node_log_dir = os.path.join(log_dir_base, node_name)
    os.makedirs(node_log_dir, exist_ok=True)

    # ---------- Instantiate communicator/model/datamodule/algorithm ----------
    local_comm  = instantiate(node_cfg.local_comm)  # default _recursive_=True
    global_comm = instantiate(node_cfg.global_comm) if getattr(node_cfg, "global_comm", None) else None

    # Multi-node Slurm: rank 0 listens early; clients wait on a shared Lustre marker
    # before connecting (task launch order across nodes is nondeterministic).
    if rank == 0 and _uses_grpc_centralized_server(local_comm) and local_comm.is_server:
        local_comm.setup()
        ready_path = _write_grpc_server_ready_marker(
            hydra_out_dir, master_addr=master_addr, master_port=master_port
        )
        print(
            f"[slurm_worker] rank 0: gRPC server started early (before model load); "
            f"ready marker -> {ready_path}",
            flush=True,
        )

    model       = instantiate(cfg.model)
    datamodule  = instantiate(cfg.datamodule)
    algorithm   = instantiate(node_cfg.algorithm, log_dir=node_log_dir)

    # ---------- Device selection ----------
    devices = resolve_slurm_devices(
        node_cfg, rank=rank, local_rank=local_rank, local_comm=local_comm
    )
    device = devices.model_device
    print(
        f"[slurm_worker] rank={rank} model_device={device} "
        f"agg_device={devices.agg_device}",
        flush=True,
    )
    if hasattr(local_comm, "set_agg_device"):
        local_comm.set_agg_device(devices.agg_device)
    original_device = next(model.parameters()).device
    model = model.to(device, non_blocking=True)

    mem_logger = start_gpu_memory_logger(
        rank=rank,
        log_dir=os.path.join(hydra_out_dir, "engine"),
        interval_sec=10.0,
    )

    if mem_logger is not None:
        mem_stop_event, mem_log_path = mem_logger
        print(
            f"[main] rank {rank} GPU memory log -> {mem_log_path}",
            flush=True,
        )
        log_gpu_memory_snapshot(rank, mem_log_path, "after_model_to_device")
    else:
        mem_stop_event = None
        mem_log_path = ""

    # ---------- Init process group via communicator ----------
    if (
        rank != 0
        and _uses_grpc_centralized_server(local_comm)
        and not local_comm.is_server
    ):
        _wait_for_grpc_server_ready_marker(hydra_out_dir, rank=rank)
    local_comm.setup()
    if global_comm:
        global_comm.setup()
    print("[main]  communicator initialized.", flush=True)

    # ---------- Device selection ----------
    # If communicator backend is NCCL, we must put tensors on CUDA before collectives.
    # backend = getattr(local_comm, "backend", "gloo").lower()
    # use_cuda = (backend == "nccl") and torch.cuda.is_available()
    # device = _resolve_device_auto(local_rank) if use_cuda else torch.device("cpu")
    # original_device = next(model.parameters()).device
    # model = model.to(device, non_blocking=True)

    # backend = getattr(local_comm, "backend", "gloo")
    # device = _resolve_device_auto(local_rank if backend == "nccl" else None)
    # if backend == "nccl" and device.type == "cuda":
    #     torch.cuda.set_device(device.index or 0)
    #     model = model.to(device, non_blocking=True)

    # ---------- DataModule: setup (if it exists) ----------
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    # ---------- Broadcast initial model (model is on CUDA when using NCCL) ----------
    if global_comm:
        model = global_comm.broadcast(model)
    model = local_comm.broadcast(model)

    # ---------- Compute group maxima (put tiny tensors on CUDA for NCCL) ----------
    #ctrl_dev = device if backend == "nccl" and device.type == "cuda" else torch.device("cpu")
    local_iters_per_epoch = _safe_len_train(datamodule)

    group_max = local_comm.aggregate(
        dict(
            iters_per_epoch=torch.tensor(local_iters_per_epoch, dtype=torch.int, device=device),
            epochs_per_round=torch.tensor(getattr(algorithm, "max_epochs_per_round", 1), dtype=torch.int, device=device),
        ),
        AggregationOp.MAX,
    )

    # Read them back on CPU safely
    iters_per_epoch  = int(group_max["iters_per_epoch"].detach().cpu().item())
    epochs_per_round = int(group_max["epochs_per_round"].detach().cpu().item())
    total_rounds     = int(cfg.global_rounds)

    # ---------- Hand off to algorithm.setup(...) exactly like Node ----------
    algorithm.setup(
        local_comm,
        global_comm,
        model,
        datamodule,
        iters_per_epoch,
        epochs_per_round,
        total_rounds,
    )

    # After algorithm.setup(): logger may call progress_info_str during gRPC metrics.
    if hasattr(local_comm, "set_logger"):
        local_comm.set_logger(algorithm)
    if global_comm and hasattr(global_comm, "set_logger"):
        global_comm.set_logger(algorithm)

    if classic_aggregate_payload_from_cfg(cfg) == "gradients":
        from src.omnifed.classic.classic_grad_slurm import install_classic_grad_slurm_sync
        from src.omnifed.summary.per_iteration import install_iteration_recorder

        print(
            f"[slurm_worker] classic grad track: {format_classic_grad_policy(cfg)}",
            flush=True,
        )
        install_iteration_recorder(
            cfg,
            algorithm,
            rank=rank,
            log_dir=os.path.join(hydra_out_dir, "engine"),
            comm_backend=_classic_comm_backend_name(local_comm),
            aggregate_payload="gradients",
        )
        install_classic_grad_slurm_sync(algorithm, local_comm=local_comm)
    elif classic_aggregate_payload_from_cfg(cfg) == "params":
        from src.omnifed.classic.classic_param_slurm import install_classic_param_slurm_sync
        from src.omnifed.summary.per_iteration import install_iteration_recorder

        print(
            "[slurm_worker] classic param track: aggregate_payload='params' (batch_end sync)",
            flush=True,
        )
        install_iteration_recorder(
            cfg,
            algorithm,
            rank=rank,
            log_dir=os.path.join(hydra_out_dir, "engine"),
            comm_backend=_classic_comm_backend_name(local_comm),
            aggregate_payload="params",
        )
        install_classic_param_slurm_sync(algorithm, local_comm=local_comm)

    # Ensure the algorithm's model lives on our chosen device
    # try:
    #     alg_dev = next(algorithm.local_model.parameters()).device
    # except Exception:
    #     alg_dev = torch.device("cpu")

    # if backend == "nccl" and device.type == "cuda" and alg_dev.type != "cuda":
    #     algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)
    #     alg_dev = device

    algorithm.local_model = algorithm.local_model.to(device, non_blocking=True)

    print(f"[main]  training on device: {device}")

    # ---------- Train rounds like Node.run_experiment ----------
    try:
        for r in range(algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
    finally:
        log_gpu_memory_snapshot(rank, mem_log_path, "finally_before_restore")

        if mem_stop_event is not None:
            mem_stop_event.set()
        # Best-effort teardown
        # Restore original device placement
        algorithm.local_model = algorithm.local_model.to(original_device)
        print(
            f"Model restored to original device: {original_device}",
            flush=True,
        )
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # ---------- Persist per-rank results ----------
    results = algorithm.get_experiment_data()
    algorithm.close_metrics()
    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    out_path = os.path.join(node_results_dir, f"node_{rank:03d}_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    out_path_json = os.path.join(node_results_dir, f"node_{rank:03d}_results.json")
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, ensure_ascii=False, indent=2)
    print(f"[slurm_worker] wrote JSON results -> {out_path_json}", flush=True)

    client_ranks = [r for r in range(world) if r != 0]
    if client_ranks and rank == min(client_ranks):
        from src.omnifed.summary.slurm_per_round import write_slurm_per_round_summary_for_run

        write_slurm_per_round_summary_for_run(
            cfg,
            hydra_out_dir,
            world_size=world,
            rpc_server_rank=0,
            rank_writer=rank,
        )

    print(f"[main]  wrote results -> {out_path}", flush=True)
    print(f"[main]  rank={rank} finished.", flush=True)


if __name__ == "__main__":
    main()
