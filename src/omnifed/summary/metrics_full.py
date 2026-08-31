"""Shared metrics_full.csv loading and per-round aggregation helpers."""

from __future__ import annotations

import csv
import glob
import io
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from src.omnifed.hybrid.topology_roles import hybrid_rank_to_centralized_node_index

_METRICS_FULL_META_COLS = ("global_step", "round_idx", "epoch_idx", "batch_idx")


def load_node_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_context_records_from_metrics_full_csv(
    csv_path: str, *, agg_context: str
) -> List[Dict[str, Any]]:
    """Rebuild wide context rows from long-format ``metrics_full.csv`` (always complete)."""
    groups: Dict[tuple, Dict[str, Any]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("agg_ctx") != agg_context:
                    continue
                meta: Dict[str, Any] = {}
                key_parts: List[tuple[str, str]] = []
                for col in _METRICS_FULL_META_COLS:
                    val = row.get(col)
                    if val is None or val == "":
                        continue
                    meta[col] = val
                    key_parts.append((col, str(val)))
                gkey = tuple(key_parts)
                if gkey not in groups:
                    groups[gkey] = dict(meta)
                metric_key = row.get("metric_key") or ""
                if not metric_key:
                    continue
                raw_val = row.get("metric_val")
                try:
                    groups[gkey][metric_key] = float(raw_val)
                except (TypeError, ValueError):
                    groups[gkey][metric_key] = raw_val
    except OSError:
        return []

    def _sort_key(rec: Dict[str, Any]) -> int:
        try:
            return int(float(rec.get("global_step", 0)))
        except (TypeError, ValueError):
            return 0

    return sorted(groups.values(), key=_sort_key)


def payloads_from_metrics_full_for_trainers(
    hydra_out_dir: str,
    *,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
    contexts: Sequence[str] = ("sync", "eval"),
) -> Dict[int, Dict[str, Any]]:
    """Build per-rank payload dicts directly from ``Node0.*/metrics_full.csv``."""
    num_clients = len(trainer_ranks)
    payloads: Dict[int, Dict[str, Any]] = {}
    for gr in trainer_ranks:
        try:
            node_idx = hybrid_rank_to_centralized_node_index(
                int(gr),
                rpc_server_rank=int(rpc_server_rank),
                world_size=int(world_size),
                num_clients=int(num_clients),
            )
        except ValueError:
            continue
        full_path = os.path.join(hydra_out_dir, f"Node0.{node_idx}", "metrics_full.csv")
        if not os.path.isfile(full_path):
            continue
        pl: Dict[str, Any] = {"rank": int(gr)}
        for ctx in contexts:
            records = load_context_records_from_metrics_full_csv(full_path, agg_context=ctx)
            if records:
                pl[ctx] = records
        if len(pl) > 1:
            payloads[int(gr)] = pl
    return payloads


def merge_json_payloads_into(
    payloads: Dict[int, Dict[str, Any]],
    hydra_out_dir: str,
) -> None:
    """Attach JSON-only fields; never overwrite sync/eval already loaded from metrics_full."""
    rd = os.path.join(hydra_out_dir, "engine", "node_results")
    for p in sorted(glob.glob(os.path.join(rd, "node_*_results.json"))):
        try:
            gr = int(os.path.basename(p).replace("node_", "").replace("_results.json", ""))
        except ValueError:
            continue
        pl = load_node_json(p)
        if not isinstance(pl, dict) or pl.get("role") == "hybrid_grpc_server":
            continue
        existing = payloads.get(gr)
        if existing is None:
            payloads[gr] = pl
            continue
        for key, val in pl.items():
            if key in ("sync", "eval"):
                if key not in existing or not existing.get(key):
                    existing[key] = val
                continue
            existing[key] = val


def trainers_metrics_full_ready(
    hydra_out_dir: str,
    *,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
) -> int:
    """Count trainers with a ``metrics_full.csv`` on disk."""
    num_clients = len(trainer_ranks)
    ready = 0
    for gr in trainer_ranks:
        try:
            node_idx = hybrid_rank_to_centralized_node_index(
                int(gr),
                rpc_server_rank=int(rpc_server_rank),
                world_size=int(world_size),
                num_clients=int(num_clients),
            )
        except ValueError:
            continue
        if os.path.isfile(os.path.join(hydra_out_dir, f"Node0.{node_idx}", "metrics_full.csv")):
            ready += 1
    return ready


def load_sync_records_from_metrics_full_csv(csv_path: str) -> List[Dict[str, Any]]:
    return load_context_records_from_metrics_full_csv(csv_path, agg_context="sync")


def enrich_payloads_from_metrics_full(
    hydra_out_dir: str,
    payloads: Dict[int, Dict[str, Any]],
    *,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
    contexts: Sequence[str] = ("sync", "eval"),
) -> None:
    """Prefer on-disk ``Node0.*/metrics_full.csv`` for metric contexts."""
    num_clients = len(trainer_ranks)
    for gr, pl in payloads.items():
        try:
            node_idx = hybrid_rank_to_centralized_node_index(
                int(gr),
                rpc_server_rank=int(rpc_server_rank),
                world_size=int(world_size),
                num_clients=int(num_clients),
            )
        except ValueError:
            continue
        full_path = os.path.join(hydra_out_dir, f"Node0.{node_idx}", "metrics_full.csv")
        if not os.path.isfile(full_path):
            continue
        for ctx in contexts:
            records = load_context_records_from_metrics_full_csv(full_path, agg_context=ctx)
            if records:
                pl[ctx] = records


def enrich_payloads_sync_from_metrics_full(
    hydra_out_dir: str,
    payloads: Dict[int, Dict[str, Any]],
    *,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
) -> None:
    enrich_payloads_from_metrics_full(
        hydra_out_dir,
        payloads,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=rpc_server_rank,
        world_size=world_size,
        contexts=("sync",),
    )


def enrich_payloads_eval_from_metrics_full(
    hydra_out_dir: str,
    payloads: Dict[int, Dict[str, Any]],
    *,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
) -> None:
    enrich_payloads_from_metrics_full(
        hydra_out_dir,
        payloads,
        trainer_ranks=trainer_ranks,
        rpc_server_rank=rpc_server_rank,
        world_size=world_size,
        contexts=("eval",),
    )


def metrics_full_has_eval_loss(hydra_out_dir: str, node_idx: int) -> bool:
    full_path = os.path.join(hydra_out_dir, f"Node0.{node_idx}", "metrics_full.csv")
    if not os.path.isfile(full_path):
        return False
    try:
        with open(full_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("agg_ctx") == "eval" and row.get("metric_key") == "eval/loss":
                    return True
    except OSError:
        return False
    return False


def summary_csv_has_eval_trainers(csv_txt: str) -> bool:
    buf = io.StringIO(csv_txt)
    reader = csv.DictReader(buf)
    for row in reader:
        try:
            if int(str(row.get("n_eval_trainers", "0")).strip()) > 0:
                return True
        except ValueError:
            continue
    return False


def warn_if_summary_missing_eval(
    *,
    mode: str,
    hydra_out_dir: str,
    trainer_ranks: Sequence[int],
    rpc_server_rank: int,
    world_size: int,
    csv_txt: str,
) -> None:
    """Emit a loud warning when eval ran on disk but the summary table has no loss."""
    if summary_csv_has_eval_trainers(csv_txt):
        return

    num_clients = len(trainer_ranks)
    on_disk = 0
    for gr in trainer_ranks:
        try:
            node_idx = hybrid_rank_to_centralized_node_index(
                int(gr),
                rpc_server_rank=int(rpc_server_rank),
                world_size=int(world_size),
                num_clients=int(num_clients),
            )
        except ValueError:
            continue
        if metrics_full_has_eval_loss(hydra_out_dir, node_idx):
            on_disk += 1

    if on_disk == 0:
        return

    print(
        f"[{mode}] summary WARN: eval_loss_avg is empty but "
        f"metrics_full.csv on {on_disk}/{len(trainer_ranks)} trainers contains eval/loss. "
        "Update summary metrics_full enrich and regenerate, or rsync latest OmniFed_VT.",
        flush=True,
    )


def get_f(row: Dict[str, Any], key: str) -> Optional[float]:
    if key not in row or row[key] is None or row[key] == "":
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def row_round_idx(row: Dict[str, Any]) -> Optional[int]:
    """Parse ``round_idx`` from a metric row; skip missing / NaN (malformed CSV rows)."""
    raw = row.get("round_idx")
    if raw is None or raw == "":
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def sync_metric_seconds(row: Optional[Dict[str, Any]], base: str) -> Optional[float]:
    """Resolve ``sync/*`` timings as stored by MetricLogger (``sync/global_agg_time`` …)."""
    if not row:
        return None
    v = get_f(row, base)
    if v is not None:
        return v
    return get_f(row, f"sync/{base}")


def sync_round_indices(sync_list: List[Dict[str, Any]]) -> set[int]:
    """All outer ``round_idx`` values present in a trainer's ``sync`` list."""
    out: set[int] = set()
    for row in sync_list or []:
        if not isinstance(row, dict):
            continue
        ridx = row_round_idx(row)
        if ridx is not None:
            out.add(ridx)
    return out


def sync_rows_for_round(sync_list: List[Dict[str, Any]], round_idx: int) -> List[Dict[str, Any]]:
    """Every flushed ``sync`` row for one outer round (custom freq ⇒ multiple rows)."""
    rows: List[Dict[str, Any]] = []
    for row in sync_list or []:
        if not isinstance(row, dict):
            continue
        if row_round_idx(row) == round_idx:
            rows.append(row)
    return rows


def sync_metric_max_for_round(
    sync_list: List[Dict[str, Any]], round_idx: int, base: str
) -> Optional[float]:
    vals: List[float] = []
    for row in sync_rows_for_round(sync_list, round_idx):
        v = sync_metric_seconds(row, base)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return max(vals)


def sync_metric_mean_for_round(
    sync_list: List[Dict[str, Any]], round_idx: int, base: str
) -> Optional[float]:
    """Mean timing across all ``sync`` rows for ``round_idx`` (batch-level sync)."""
    vals: List[float] = []
    for row in sync_rows_for_round(sync_list, round_idx):
        v = sync_metric_seconds(row, base)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def train_loss_avg_for_round(
    train_list: List[Dict[str, Any]], round_idx: int
) -> Optional[float]:
    """Mean batch training loss for ``round_idx`` (``train/loss`` from MetricLogger)."""
    vals: List[float] = []
    for row in train_list or []:
        if not isinstance(row, dict):
            continue
        if row_round_idx(row) != round_idx:
            continue
        v = row.get("train/loss")
        if v is None or v == "":
            v = row.get("loss")
        if v is None or v == "":
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def eval_accuracy_avg_for_round(
    eval_list: List[Dict[str, Any]], round_idx: int
) -> Optional[float]:
    candidates: List[float] = []
    for row in eval_list or []:
        if not isinstance(row, dict):
            continue
        if row_round_idx(row) != round_idx:
            continue
        for k, v in row.items():
            if not isinstance(k, str) or "accuracy" not in k.lower():
                continue
            if v is None or v == "":
                continue
            try:
                candidates.append(float(v))
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None
    return sum(candidates) / len(candidates)


def eval_loss_avg_for_round(
    eval_list: List[Dict[str, Any]], round_idx: int
) -> Optional[float]:
    """Mean eval loss for ``round_idx`` (``eval/loss`` from MetricLogger)."""
    vals: List[float] = []
    for row in eval_list or []:
        if not isinstance(row, dict):
            continue
        if row_round_idx(row) != round_idx:
            continue
        v = row.get("eval/loss")
        if v is None or v == "":
            v = row.get("loss")
        if v is None or v == "":
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def format_ms(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{1000.0 * v:.2f}"
