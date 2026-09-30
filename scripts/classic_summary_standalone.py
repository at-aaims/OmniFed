#!/usr/bin/env python3
"""
Build classic centralized per-round summary from an existing Hydra run directory.

No OmniFed imports — stdlib only. Safe to copy to Frontier without rsync:

  python3 classic_summary_standalone.py /path/to/test_pi_..._grpc_grad_a

Reads Node0.1 … Node0.6/metrics_full.csv (primary). Falls back to parsing
slurm-*.out for End sync/eval context lines if metrics_full is missing.
"""

from __future__ import annotations

import csv
import glob
import io
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

_META_COLS = ("global_step", "round_idx", "epoch_idx", "batch_idx")


def _f(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        x = float(val)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _metric_seconds(row: Dict[str, Any], base: str) -> Optional[float]:
    v = _f(row.get(base))
    if v is not None:
        return v
    return _f(row.get(f"sync/{base}"))


def _load_metrics_full_context(csv_path: str, agg_context: str) -> List[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, Any]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("agg_ctx") != agg_context:
                    continue
                key_parts: List[tuple[str, str]] = []
                meta: Dict[str, Any] = {}
                for col in _META_COLS:
                    val = row.get(col)
                    if val is None or val == "":
                        continue
                    meta[col] = val
                    key_parts.append((col, str(val)))
                gkey = tuple(key_parts)
                if gkey not in groups:
                    groups[gkey] = dict(meta)
                mk = row.get("metric_key") or ""
                if mk:
                    groups[gkey][mk] = row.get("metric_val")
    except OSError:
        return []
    return list(groups.values())


def _round_idx(row: Dict[str, Any]) -> Optional[int]:
    raw = row.get("round_idx")
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _max_sync_metric(rows: List[Dict[str, Any]], round_idx: int, base: str) -> Optional[float]:
    vals: List[float] = []
    for row in rows:
        if _round_idx(row) != round_idx:
            continue
        v = _metric_seconds(row, base)
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def _eval_losses(rows: List[Dict[str, Any]], round_idx: int) -> List[float]:
    out: List[float] = []
    for row in rows:
        if _round_idx(row) != round_idx:
            continue
        v = row.get("eval/loss", row.get("loss"))
        fv = _f(v)
        if fv is not None:
            out.append(fv)
    return out


def _format_ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    return f"{1000.0 * seconds:.2f}"


def _build_from_metrics_full(run_dir: str) -> Optional[Tuple[str, str]]:
    sync_by_node: List[List[Dict[str, Any]]] = []
    eval_by_node: List[List[Dict[str, Any]]] = []
    for i in range(1, 7):
        full = os.path.join(run_dir, f"Node0.{i}", "metrics_full.csv")
        if not os.path.isfile(full):
            continue
        sync_by_node.append(_load_metrics_full_context(full, "sync"))
        eval_by_node.append(_load_metrics_full_context(full, "eval"))

    if not sync_by_node and not eval_by_node:
        return None

    rounds: set[int] = set()
    for sync_rows in sync_by_node:
        for row in sync_rows:
            r = _round_idx(row)
            if r is not None:
                rounds.add(r)
    for eval_rows in eval_by_node:
        for row in eval_rows:
            r = _round_idx(row)
            if r is not None:
                rounds.add(r)

    hdr = [
        "round_idx",
        "grpc_agg_max_ms",
        "grad_apply_max_ms",
        "eval_loss_avg",
        "n_eval_trainers",
    ]
    md = [
        "",
        "### Classic per-round summary (from Node0.*/metrics_full.csv)",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join([":---"] * len(hdr)) + "|",
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr)

    for ridx in sorted(rounds):
        agg_vals: List[float] = []
        apply_vals: List[float] = []
        for sync_rows in sync_by_node:
            xa = _max_sync_metric(sync_rows, ridx, "local_agg_time")
            xg = _max_sync_metric(sync_rows, ridx, "grad_apply_time")
            if xa is not None:
                agg_vals.append(xa)
            if xg is not None:
                apply_vals.append(xg)
        losses: List[float] = []
        for eval_rows in eval_by_node:
            losses.extend(_eval_losses(eval_rows, ridx))
        loss_avg = sum(losses) / len(losses) if losses else None
        loss_txt = f"{loss_avg:.6f}" if loss_avg is not None else "—"

        row = [
            ridx,
            _format_ms(max(agg_vals) if agg_vals else None),
            _format_ms(max(apply_vals) if apply_vals else None),
            loss_txt,
            str(len(losses)),
        ]
        w.writerow(row)
        md.append("| " + " | ".join(str(c) for c in row) + " |")

    return "\n".join(md) + "\n", buf.getvalue()


_END_CTX_RE = re.compile(
    r"End (?P<ctx>sync|eval) context @ .*?\(ROUND_IDX=(?P<round>\d+).*?\|\s*(?P<metrics>.+)$"
)
_KV_RE = re.compile(r"([^=;]+)=([^;]+)")


def _parse_metric_blob(blob: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in _KV_RE.finditer(blob):
        key = m.group(1).strip()
        fv = _f(m.group(2).strip())
        if fv is not None:
            out[key] = fv
    return out


def _build_from_slurm_log(run_dir: str) -> Optional[Tuple[str, str]]:
    logs = sorted(glob.glob(os.path.join(run_dir, "slurm-*.out")))
    if not logs:
        return None

    sync_agg: DefaultDict[int, List[float]] = defaultdict(list)
    sync_apply: DefaultDict[int, List[float]] = defaultdict(list)
    eval_loss: DefaultDict[int, List[float]] = defaultdict(list)

    for log_path in logs:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                m = _END_CTX_RE.search(line)
                if not m:
                    continue
                ridx = int(m.group("round"))
                metrics = _parse_metric_blob(m.group("metrics"))
                if m.group("ctx") == "sync":
                    for key in ("local_agg_time", "sync/local_agg_time"):
                        if key in metrics:
                            sync_agg[ridx].append(metrics[key])
                            break
                    for key in ("grad_apply_time", "sync/grad_apply_time"):
                        if key in metrics:
                            sync_apply[ridx].append(metrics[key])
                            break
                else:
                    for key in ("eval/loss", "loss"):
                        if key in metrics:
                            eval_loss[ridx].append(metrics[key])
                            break

    rounds = sorted(set(sync_agg) | set(sync_apply) | set(eval_loss))
    if not rounds:
        return None

    hdr = [
        "round_idx",
        "grpc_agg_max_ms",
        "grad_apply_max_ms",
        "eval_loss_avg",
        "n_eval_trainers",
    ]
    md = [
        "",
        f"### Classic per-round summary (from {os.path.basename(logs[0])} — log fallback)",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join([":---"] * len(hdr)) + "|",
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(hdr)

    for ridx in rounds:
        losses = eval_loss.get(ridx, [])
        loss_avg = sum(losses) / len(losses) if losses else None
        loss_txt = f"{loss_avg:.6f}" if loss_avg is not None else "—"
        row = [
            ridx,
            _format_ms(max(sync_agg[ridx]) if sync_agg[ridx] else None),
            _format_ms(max(sync_apply[ridx]) if sync_apply[ridx] else None),
            loss_txt,
            str(len(losses)),
        ]
        w.writerow(row)
        md.append("| " + " | ".join(str(c) for c in row) + " |")

    return "\n".join(md) + "\n", buf.getvalue()


def main() -> None:
    run_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    result = _build_from_metrics_full(run_dir)
    source = "metrics_full.csv"
    if result is None:
        result = _build_from_slurm_log(run_dir)
        source = "slurm-*.out"
    if result is None:
        print(
            f"No summary data found under {run_dir}\n"
            "Expected Node0.1-6/metrics_full.csv or slurm-*.out with "
            "'End sync/eval context' lines.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    md, csv_text = result
    eng = os.path.join(run_dir, "engine")
    os.makedirs(eng, exist_ok=True)
    for path in (
        os.path.join(eng, "classic_per_round_summary.csv"),
        os.path.join(run_dir, "classic_per_round_summary.csv"),
    ):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
    with open(os.path.join(eng, "classic_per_round_summary.txt"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote classic_per_round_summary.csv ({source})", flush=True)
    print(md.strip(), flush=True)


if __name__ == "__main__":
    main()
