# src/omnifed/slurm_worker.py
from __future__ import annotations
import argparse, json, os, signal, time, subprocess, pickle
from typing import Any, Dict, Optional
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

# Pull the same pieces used by Node
from src.omnifed.communicator import AggregationOp
from src.omnifed.utils import print  # pretty-print wrapper used elsewhere

def _first_host_from_nodelist() -> str:
    out = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
        text=True
    )
    return out.strip().splitlines()[0]

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

    # ---------- Slurm sizing & env ----------
    rank  = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("MASTER_ADDR", _first_host_from_nodelist())
    os.environ.setdefault("MASTER_PORT", str(cfg.topology.local_comm.master_port))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))

    print(f"[slurm_worker] rank={rank} world={world} MASTER_ADDR={os.environ['MASTER_ADDR']} PORT={os.environ['MASTER_PORT']}", flush=True)

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
    model       = instantiate(cfg.model)
    datamodule  = instantiate(cfg.datamodule)
    algorithm   = instantiate(node_cfg.algorithm, log_dir=node_log_dir)

    # ---------- Init process group via communicator ----------
    local_comm.setup()
    if global_comm:
        global_comm.setup()
    print("[slurm_worker] communicator initialized.", flush=True)

    # ---------- DataModule: be tolerant (setup() or prebuilt loaders) ----------
    if hasattr(datamodule, "setup"):
        datamodule.setup()

    # ---------- Broadcast initial model like Node ----------
    if global_comm:
        model = global_comm.broadcast(model)
    model = local_comm.broadcast(model)

    # ---------- Discover per-round sizes the same way Node does ----------
    local_iters_per_epoch = _safe_len_train(datamodule)
    group_max = local_comm.aggregate(
        dict(
            iters_per_epoch=torch.tensor(local_iters_per_epoch, dtype=torch.int),
            epochs_per_round=torch.tensor(getattr(algorithm, "max_epochs_per_round", 1), dtype=torch.int),
        ),
        AggregationOp.MAX,
    )
    iters_per_epoch  = int(group_max["iters_per_epoch"].item())
    epochs_per_round = int(group_max["epochs_per_round"].item())
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

    # ---------- Train rounds like Node.run_experiment ----------
    # Ensure the model lives on the device the algorithm expects
    original_device = next(algorithm.local_model.parameters()).device
    try:
        for r in range(algorithm.max_rounds):
            algorithm.round_exec(r, algorithm.max_rounds)
    finally:
        # Restore to original device and attempt PG teardown
        algorithm.local_model = algorithm.local_model.to(original_device)
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # ---------- Persist per-rank results (Engine expects a list later) ----------
    results = algorithm.get_experiment_data()
    node_results_dir = os.path.join(hydra_out_dir, "engine", "node_results")
    os.makedirs(node_results_dir, exist_ok=True)
    out_path = os.path.join(node_results_dir, f"node_{rank:03d}_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[slurm_worker] wrote results -> {out_path}", flush=True)
    print(f"[slurm_worker] rank={rank} finished.", flush=True)

if __name__ == "__main__":
    main()
