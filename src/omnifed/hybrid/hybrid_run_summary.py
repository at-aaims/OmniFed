"""
Post-run hybrid Slurm summary: per-global-round table from ``node_*_results.json``.

Reads ``sync/local_agg_time``, ``sync/global_agg_time``, ``sync/local_bcast_time`` with
metadata ``round_idx``. Optional ``eval/*accuracy*`` columns when evaluation logs them.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["write_hybrid_slurm_per_round_summary"]


def _fac_leader_ranks(topo: Any) -> List[int]:
    leaders: List[int] = []
    for fac in topo.facilities:
        members = [int(m) for m in fac.mpi.members]
        if not members:
            raise ValueError("facility with empty members in topology")
        leaders.append(members[0])
    return leaders


def _fac_member_ranks(topo: Any) -> List[List[int]]:
    return [[int(m) for m in fac.mpi.members] for fac in topo.facilities]


def _load_node_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sync_rows_by_round(sync_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_r: Dict[int, Dict[str, Any]] = {}
    for row in sync_list or []:
        if not isinstance(row, dict):
            continue
        raw = row.get("round_idx")
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        by_r[int(raw)] = row
    return by_r


def _get_f(row: Dict[str, Any], key: str) -> Optional[float]:
    if key not in row or row[key] is None or row[key] == "":
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _sync_metric_seconds(row: Optional[Dict[str, Any]], base: str) -> Optional[float]:
    """Resolve ``sync/*`` timings as stored by MetricLogger (``sync/global_agg_time`` …)."""
    if not row:
        return None
    v = _get_f(row, base)
    if v is not None:
        return v
    return _get_f(row, f"sync/{base}")
def _eval_accuracy_avg_for_round(eval_list: List[Dict[str, Any]], round_idx: int) -> Optional[float]:
    candidates: List[float] = []
    for row in eval_list or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("round_idx", -1)) != round_idx:
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


def _eval_loss_avg_for_round(eval_list: List[Dict[str, Any]], round_idx: int) -> Optional[float]:
    """Mean ``eval/loss`` over eval rows matching ``round_idx`` (FedAvg MNIST rollup)."""
    vals: List[float] = []
    for row in eval_list or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("round_idx", -1)) != round_idx:
            continue
        v = row.get("eval/loss")
        if v is None or v == "":
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_ms(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{1000.0 * v:.2f}"


def _build_tables(
    *,
    topo: Any,
    payloads: Dict[int, Dict[str, Any]],
    trainer_ranks: Sequence[int],
) -> Tuple[str, str]:
    """Markdown (table only) + CSV body as strings."""
    leaders = _fac_leader_ranks(topo)
    members_by_f = _fac_member_ranks(topo)
    n_f = len(leaders)

    all_rounds: set[int] = set()
    for gr in trainer_ranks:
        pl = payloads.get(gr) or {}
        all_rounds.update(_sync_rows_by_round(pl.get("sync") or []).keys())
    rounds_sorted = sorted(all_rounds)

    hdr_g = [f"gRPC_F{i+1}_ms" for i in range(n_f)]
    hdr_la = [f"local_agg_F{i+1}_max_ms" for i in range(n_f)]
    hdr_lb = [f"local_bcast_F{i+1}_max_ms" for i in range(n_f)]
    hdr_tail = ["accuracy_avg", "n_acc_trainers", "eval_loss_avg", "n_eval_trainers"]

    tbl_header = "| round_idx | " + " | ".join([*hdr_g, *hdr_la, *hdr_lb, *hdr_tail]) + " |"
    ncols = 1 + len(hdr_g) + len(hdr_la) + len(hdr_lb) + len(hdr_tail)
    sep_row = "|" + "|".join([":---"] * ncols) + "|"

    md_rows = [tbl_header, sep_row]

    csv_header = ["round_idx", *hdr_g, *hdr_la, *hdr_lb, *hdr_tail]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(csv_header)

    for ridx in rounds_sorted:
        grpc = []
        for lr in leaders:
            sr = _sync_rows_by_round((payloads.get(lr) or {}).get("sync") or []).get(ridx)
            grpc.append(None if sr is None else _sync_metric_seconds(sr, "global_agg_time"))

        la_m: List[Optional[float]] = []
        lb_m: List[Optional[float]] = []
        for members in members_by_f:
            la_v: List[float] = []
            lb_v: List[float] = []
            for gr in members:
                sr = _sync_rows_by_round(
                    (payloads.get(gr) or {}).get("sync") or []
                ).get(ridx)
                if not sr:
                    continue
                xa = _sync_metric_seconds(sr, "local_agg_time")
                xb = _sync_metric_seconds(sr, "local_bcast_time")
                if xa is not None:
                    la_v.append(xa)
                if xb is not None:
                    lb_v.append(xb)
            la_m.append(max(la_v) if la_v else None)
            lb_m.append(max(lb_v) if lb_v else None)

        acc_list: List[float] = []
        for gr in trainer_ranks:
            a = _eval_accuracy_avg_for_round((payloads.get(gr) or {}).get("eval") or [], ridx)
            if a is not None:
                acc_list.append(a)
        acc_txt = (
            f"{sum(acc_list) / len(acc_list):.4f}"
            if acc_list
            else "—"
        )

        loss_list: List[float] = []
        for gr in trainer_ranks:
            lo = _eval_loss_avg_for_round((payloads.get(gr) or {}).get("eval") or [], ridx)
            if lo is not None:
                loss_list.append(lo)
        loss_txt = (
            f"{sum(loss_list) / len(loss_list):.6f}"
            if loss_list
            else "—"
        )

        md_cells = (
            [*[_format_ms(x) for x in grpc], *[_format_ms(x) for x in la_m], *[_format_ms(x) for x in lb_m],
             acc_txt, str(len(acc_list)), loss_txt, str(len(loss_list))]
        )
        md_rows.append("| " + str(ridx) + " | " + " | ".join(md_cells) + " |")

        csv_row = [ridx]
        csv_row.extend(_format_ms(x) for x in grpc)
        csv_row.extend(_format_ms(x) for x in la_m)
        csv_row.extend(_format_ms(x) for x in lb_m)
        csv_row.extend([acc_txt, str(len(acc_list)), loss_txt, str(len(loss_list))])
        w.writerow(csv_row)

    intro = "\n".join(
        [
            "",
            "### Hybrid per-global-round summary (from JSON node_results)",
            "",
            "**gRPC_*:** `sync/global_agg_time` seconds on facility leader (**ms** printed). ",
            "**local_agg_* / local_bcast_*:** **max** over ranks in facility (**ms**). ",
            "**accuracy_avg:** mean of per-trainer *accuracy* scalars logged in **`eval`** for that round ",
            "(keys containing `accuracy`, case‑insensitive). Requires eval to record accuracy.",
            "**eval_loss_avg:** mean across trainers of per-trainer average **`eval/loss`** (multiple eval ",
            "rows sharing the same **round_idx** are averaged inside each trainer first).",
            "",
        ]
    )
    md = intro + "\n".join(md_rows) + "\n"

    csv_text = buf.getvalue()

    return md, csv_text


def write_hybrid_slurm_per_round_summary(
    hydra_out_dir: str,
    *,
    topo: Any,
    world_size: int,
    rpc_server_rank: int,
    rank_writer: int,
) -> Optional[str]:
    """Poll all ``node_*_results.json``. Call exactly once from ``rank_writer``.
    Saves ``engine/hybrid_per_round_summary.{txt,csv}`` and echoes Markdown to stdout.
    """
    _ = rank_writer

    trainer_ranks = tuple(
        gr for gr in range(int(world_size)) if gr != int(rpc_server_rank)
    )

    rd = os.path.join(hydra_out_dir, "engine", "node_results")
    if not os.path.isdir(rd):
        print(
            f"[hybrid] summary: missing {rd} — pass the Hydra **run directory** that "
            "contains **engine/node_results** (e.g. outputs/DATE/your_job_name/). "
            "Do not pass the repo root.",
            flush=True,
        )
        return None
    timeout_s = float(
        os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_SEC", str(max(90.0, 5.0 * float(world_size))))
    )
    poll_gap = float(os.environ.get("OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC", "0.5"))

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        njson = len(glob.glob(os.path.join(rd, "node_*_results.json")))
        if njson >= int(world_size):
            break
        time.sleep(poll_gap)
    else:
        njson = len(glob.glob(os.path.join(rd, "node_*_results.json")))
        print(
            f"[hybrid] summary: timeout ({njson}/{world_size} JSON files). Raise OMNIFED_HYBRID_SUMMARY_POLL_SEC.",
            flush=True,
        )
        return None

    payloads: Dict[int, Dict[str, Any]] = {}
    for p in sorted(glob.glob(os.path.join(rd, "node_*_results.json"))):
        try:
            gr = int(os.path.basename(p).replace("node_", "").replace("_results.json", ""))
        except ValueError:
            continue
        pl = _load_node_json(p)
        if isinstance(pl, dict) and pl.get("role") != "hybrid_grpc_server":
            payloads[int(gr)] = pl

    md_txt, csv_txt = _build_tables(topo=topo, payloads=payloads, trainer_ranks=trainer_ranks)

    eng = os.path.join(hydra_out_dir, "engine")
    txt_path = os.path.join(eng, "hybrid_per_round_summary.txt")
    csv_path_engine = os.path.join(eng, "hybrid_per_round_summary.csv")
    # Run-root copy for easy GitHub previews / uploads alongside slurm-*.out
    csv_path_run = os.path.join(hydra_out_dir, "hybrid_per_round_summary.csv")

    os.makedirs(eng, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(md_txt)

    for cp in (csv_path_engine, csv_path_run):
        with open(cp, "w", encoding="utf-8", newline="") as fc:
            fc.write(csv_txt)

    print(
        "[hybrid] per-round summary: engine/hybrid_per_round_summary.{txt,csv} "
        "+ hybrid_per_round_summary.csv (run root, for commits / GitHub)",
        flush=True,
    )
    print(md_txt.strip(), flush=True)

    return txt_path
