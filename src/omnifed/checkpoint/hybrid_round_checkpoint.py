"""
Round-end checkpoints for hybrid Slurm FedAvg (one save after each completed global round).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

MANIFEST_FILENAME = "manifest.json"
ROUND_DIR_FMT = "round_{:03d}"
MODEL_SHARD_FMT = "model_rank_{:03d}.pt"


def resolve_experiment_checkpoint_dir(cfg: Any) -> Optional[str]:
    """
    ``slurm.checkpoint_dir`` + ``slurm.experiment_id`` when both set; else ``checkpoint_dir`` alone.
    """
    parent = OmegaConf.select(cfg, "slurm.checkpoint_dir", default=None)
    exp_id = OmegaConf.select(cfg, "slurm.experiment_id", default=None)
    if parent is None or str(parent).strip() == "":
        return None
    parent = os.path.abspath(str(parent))
    if exp_id is not None and str(exp_id).strip() != "":
        return os.path.join(parent, str(exp_id).strip())
    return parent


def should_resume(cfg: Any) -> bool:
    raw = OmegaConf.select(cfg, "slurm.resume", default=False)
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1", "yes")
    return bool(raw)


def is_training_complete(exp_dir: str) -> bool:
    """True when ``manifest.json`` exists and ``status`` is ``complete``."""
    manifest = load_manifest(exp_dir)
    return manifest is not None and manifest.get("status") == "complete"


def experiment_dir_from_parent(parent: str, experiment_id: str) -> str:
    return os.path.join(os.path.abspath(parent), str(experiment_id).strip())


def manifest_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, MANIFEST_FILENAME)


def round_dir(exp_dir: str, round_idx: int) -> str:
    return os.path.join(exp_dir, ROUND_DIR_FMT.format(int(round_idx)))


def model_shard_path(exp_dir: str, round_idx: int, rank: int) -> str:
    return os.path.join(round_dir(exp_dir, round_idx), MODEL_SHARD_FMT.format(int(rank)))


def load_manifest(exp_dir: str) -> Optional[dict[str, Any]]:
    path = manifest_path(exp_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_torch_save(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def resume_start_round(cfg: Any, exp_dir: str) -> int:
    """
    Return first ``round_idx`` for ``round_exec`` (0 if fresh or no manifest).
    """
    if not should_resume(cfg):
        return 0
    manifest = load_manifest(exp_dir)
    if manifest is None:
        return 0
    return int(manifest.get("next_round_idx", 0))


def load_round_model_state(
    model: torch.nn.Module,
    exp_dir: str,
    round_idx: int,
    rank: int,
) -> bool:
    """Load post-round model weights for ``round_idx`` into ``model``. Returns True if loaded."""
    path = model_shard_path(exp_dir, round_idx, rank)
    if not os.path.isfile(path):
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    if state is None:
        return False
    model.load_state_dict(state, strict=True)
    return True


def save_round_checkpoint(
    *,
    exp_dir: str,
    round_idx: int,
    rank: int,
    model: torch.nn.Module,
    target_global_rounds: int,
    is_manifest_writer: bool,
    experiment_id: Optional[str] = None,
    preset_name: Optional[str] = None,
    topology_num_clients: Optional[int] = None,
    aggregate_payload: Optional[str] = None,
) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    shard_path = model_shard_path(exp_dir, round_idx, rank)
    _atomic_torch_save(
        shard_path,
        {"model": model.state_dict(), "round_idx": int(round_idx), "rank": int(rank)},
    )

    if not is_manifest_writer:
        return

    last = int(round_idx)
    next_idx = last + 1
    target = int(target_global_rounds)
    status = "complete" if next_idx >= target else "in_progress"
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id or os.path.basename(exp_dir.rstrip("/")),
        "preset": preset_name,
        "topology_num_clients": topology_num_clients,
        "target_global_rounds": target,
        "last_completed_round": last,
        "next_round_idx": next_idx,
        "status": status,
        "latest_round_dir": ROUND_DIR_FMT.format(last),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if aggregate_payload is not None:
        manifest["aggregate_payload"] = str(aggregate_payload)
    prev = load_manifest(exp_dir)
    if prev:
        manifest["created_at"] = prev.get("created_at", manifest["updated_at"])
    else:
        manifest["created_at"] = manifest["updated_at"]
    _atomic_write_json(manifest_path(exp_dir), manifest)
