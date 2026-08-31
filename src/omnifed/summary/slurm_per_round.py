"""
Universal Slurm post-run per-round summary.

Single execution entry for classic centralized and hybrid training pipelines.
Polls ``metrics_full.csv`` / ``node_results``, builds mode-specific tables, writes artifacts.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Optional, Sequence, Tuple

from src.omnifed.summary.pipeline import summary_mode_from_cfg
from src.omnifed.summary.classic_per_round import build_classic_per_round_tables
from src.omnifed.summary.hybrid_per_round import build_hybrid_per_round_tables
from src.omnifed.summary.metrics_full import (
    enrich_payloads_from_metrics_full,
    merge_json_payloads_into,
    payloads_from_metrics_full_for_trainers,
    trainers_metrics_full_ready,
    warn_if_summary_missing_eval,
)

__all__ = [
    "write_slurm_per_round_summary",
    "write_slurm_per_round_summary_for_run",
    "write_classic_slurm_per_round_summary",
    "write_hybrid_slurm_per_round_summary",
]

_SUMMARY_MODES = ("classic", "hybrid")


def _poll_settings(mode: str, world_size: int) -> Tuple[float, float]:
    default_timeout = str(max(90.0, 5.0 * float(world_size)))
    if mode == "classic":
        timeout_s = float(
            os.environ.get(
                "OMNIFED_CLASSIC_SUMMARY_POLL_SEC",
                os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_SEC", default_timeout),
            )
        )
        poll_gap = float(
            os.environ.get(
                "OMNIFED_CLASSIC_SUMMARY_POLL_GAP_SEC",
                os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC", "0.5"),
            )
        )
    else:
        timeout_s = float(
            os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_SEC", default_timeout)
        )
        poll_gap = float(os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC", "0.5"))
    return timeout_s, poll_gap


def _poll_timeout_env_name(mode: str) -> str:
    if mode == "classic":
        return "OMNIFED_CLASSIC_SUMMARY_POLL_SEC"
    return "OMNIFED_HYBRID_SUMMARY_POLL_SEC"


def _trainer_ranks(world_size: int, rpc_server_rank: int) -> Tuple[int, ...]:
    return tuple(gr for gr in range(int(world_size)) if gr != int(rpc_server_rank))


def _wait_for_summary_inputs(
    *,
    mode: str,
    hydra_out_dir: str,
    trainer_ranks: Sequence[int],
    world_size: int,
    rpc_server_rank: int,
) -> bool:
    rd = os.path.join(hydra_out_dir, "engine", "node_results")
    if not os.path.isdir(rd):
        print(
            f"[{mode}] summary: missing {rd} — pass the Hydra run directory "
            "(outputs/DATE/your_job_name/).",
            flush=True,
        )
        return False

    timeout_s, poll_gap = _poll_settings(mode, world_size)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        njson = len(glob.glob(os.path.join(rd, "node_*_results.json")))
        nfull = trainers_metrics_full_ready(
            hydra_out_dir,
            trainer_ranks=trainer_ranks,
            rpc_server_rank=int(rpc_server_rank),
            world_size=int(world_size),
        )
        if njson >= int(world_size) or nfull >= len(trainer_ranks):
            return True
        time.sleep(poll_gap)

    njson = len(glob.glob(os.path.join(rd, "node_*_results.json")))
    nfull = trainers_metrics_full_ready(
        hydra_out_dir,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=int(rpc_server_rank),
        world_size=int(world_size),
    )
    print(
        f"[{mode}] summary: timeout ({njson}/{world_size} JSON, "
        f"{nfull}/{len(trainer_ranks)} metrics_full). "
        f"Raise {_poll_timeout_env_name(mode)}.",
        flush=True,
    )
    return False


def _write_summary_artifacts(
    *,
    mode: str,
    hydra_out_dir: str,
    md_txt: str,
    csv_txt: str,
) -> str:
    eng = os.path.join(hydra_out_dir, "engine")
    txt_path = os.path.join(eng, f"{mode}_per_round_summary.txt")
    csv_path_engine = os.path.join(eng, f"{mode}_per_round_summary.csv")
    csv_path_run = os.path.join(hydra_out_dir, f"{mode}_per_round_summary.csv")

    os.makedirs(eng, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(md_txt)

    for cp in (csv_path_engine, csv_path_run):
        with open(cp, "w", encoding="utf-8", newline="") as fc:
            fc.write(csv_txt)

    print(
        f"[{mode}] per-round summary: engine/{mode}_per_round_summary.{{txt,csv}} "
        f"+ {mode}_per_round_summary.csv (run root)",
        flush=True,
    )
    print(md_txt.strip(), flush=True)
    return txt_path


def write_slurm_per_round_summary(
    hydra_out_dir: str,
    *,
    mode: str,
    world_size: int,
    rpc_server_rank: int = 0,
    rank_writer: int,
    topo: Any = None,
) -> Optional[str]:
    """Poll trainer metrics and write per-round summary for ``mode`` (classic or hybrid)."""
    _ = rank_writer
    mode_norm = str(mode).lower()
    if mode_norm not in _SUMMARY_MODES:
        raise ValueError(f"summary mode must be one of {_SUMMARY_MODES}, got {mode!r}")
    if mode_norm == "hybrid" and topo is None:
        raise ValueError("topo is required when mode='hybrid'")

    trainer_ranks = _trainer_ranks(world_size, rpc_server_rank)
    if not trainer_ranks:
        print(f"[{mode_norm}] summary: no client trainer ranks", flush=True)
        return None

    if not _wait_for_summary_inputs(
        mode=mode_norm,
        hydra_out_dir=hydra_out_dir,
        trainer_ranks=trainer_ranks,
        world_size=world_size,
        rpc_server_rank=rpc_server_rank,
    ):
        return None

    contexts = ("sync", "eval", "train") if mode_norm == "classic" else ("sync", "eval")
    payloads = payloads_from_metrics_full_for_trainers(
        hydra_out_dir,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=int(rpc_server_rank),
        world_size=int(world_size),
        contexts=contexts,
    )
    merge_json_payloads_into(payloads, hydra_out_dir)
    enrich_payloads_from_metrics_full(
        hydra_out_dir,
        payloads,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=int(rpc_server_rank),
        world_size=int(world_size),
        contexts=contexts,
    )

    if mode_norm == "classic":
        md_txt, csv_txt = build_classic_per_round_tables(
            payloads=payloads,
            trainer_ranks=trainer_ranks,
        )
    else:
        md_txt, csv_txt = build_hybrid_per_round_tables(
            topo=topo,
            payloads=payloads,
            trainer_ranks=trainer_ranks,
        )

    warn_if_summary_missing_eval(
        mode=mode_norm,
        hydra_out_dir=hydra_out_dir,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=int(rpc_server_rank),
        world_size=int(world_size),
        csv_txt=csv_txt,
    )

    return _write_summary_artifacts(
        mode=mode_norm,
        hydra_out_dir=hydra_out_dir,
        md_txt=md_txt,
        csv_txt=csv_txt,
    )


def write_slurm_per_round_summary_for_run(
    cfg: Any,
    hydra_out_dir: str,
    *,
    world_size: int,
    rpc_server_rank: int = 0,
    rank_writer: int,
    topo: Any = None,
) -> Optional[str]:
    """Detect pipeline from ``cfg`` and write the matching per-round summary."""
    mode = summary_mode_from_cfg(cfg)
    if mode == "hybrid" and topo is None:
        raise ValueError("topo is required when engine.communication_mode='hybrid'")
    return write_slurm_per_round_summary(
        hydra_out_dir,
        mode=mode,
        world_size=world_size,
        rpc_server_rank=rpc_server_rank,
        rank_writer=rank_writer,
        topo=topo,
    )


def write_classic_slurm_per_round_summary(
    hydra_out_dir: str,
    *,
    world_size: int,
    rpc_server_rank: int = 0,
    rank_writer: int,
) -> Optional[str]:
    return write_slurm_per_round_summary(
        hydra_out_dir,
        mode="classic",
        world_size=world_size,
        rpc_server_rank=rpc_server_rank,
        rank_writer=rank_writer,
    )


def write_hybrid_slurm_per_round_summary(
    hydra_out_dir: str,
    *,
    topo: Any,
    world_size: int,
    rpc_server_rank: int,
    rank_writer: int,
) -> Optional[str]:
    return write_slurm_per_round_summary(
        hydra_out_dir,
        mode="hybrid",
        world_size=world_size,
        rpc_server_rank=rpc_server_rank,
        rank_writer=rank_writer,
        topo=topo,
    )
