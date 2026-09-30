"""Classic centralized per-round summary table builder."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.omnifed.summary.metrics_full import (
    eval_loss_avg_for_round,
    format_ms,
    sync_metric_max_for_round,
    sync_metric_mean_for_round,
    sync_round_indices,
    train_loss_avg_for_round,
)


def build_classic_per_round_tables(
    *,
    payloads: Dict[int, Dict[str, Any]],
    trainer_ranks: Sequence[int],
) -> Tuple[str, str]:
    """Markdown table + CSV body (flat topology, no facility columns)."""
    all_rounds: set[int] = set()
    for gr in trainer_ranks:
        pl = payloads.get(gr) or {}
        all_rounds.update(sync_round_indices(pl.get("sync") or []))
    rounds_sorted = sorted(all_rounds)

    hdr_tail = [
        "grpc_agg_mean_ms",
        "grpc_agg_max_ms",
        "grad_apply_max_ms",
        "eval_loss_avg",
        "n_eval_trainers",
        "train_loss_avg",
    ]
    tbl_header = "| round_idx | " + " | ".join(hdr_tail) + " |"
    ncols = 1 + len(hdr_tail)
    sep_row = "|" + "|".join([":---"] * ncols) + "|"
    md_rows = [tbl_header, sep_row]

    csv_header = ["round_idx", *hdr_tail]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(csv_header)

    for ridx in rounds_sorted:
        agg_mean_vals: List[float] = []
        agg_max_vals: List[float] = []
        apply_vals: List[float] = []
        for gr in trainer_ranks:
            sync_list = (payloads.get(gr) or {}).get("sync") or []
            xa_mean = sync_metric_mean_for_round(sync_list, ridx, "local_agg_time")
            xa_max = sync_metric_max_for_round(sync_list, ridx, "local_agg_time")
            xg = sync_metric_max_for_round(sync_list, ridx, "grad_apply_time")
            if xa_mean is not None:
                agg_mean_vals.append(xa_mean)
            if xa_max is not None:
                agg_max_vals.append(xa_max)
            if xg is not None:
                apply_vals.append(xg)
        grpc_mean_ms = max(agg_mean_vals) if agg_mean_vals else None
        grpc_max_ms = max(agg_max_vals) if agg_max_vals else None
        grad_apply_ms = max(apply_vals) if apply_vals else None

        loss_list: List[float] = []
        for gr in trainer_ranks:
            lo = eval_loss_avg_for_round((payloads.get(gr) or {}).get("eval") or [], ridx)
            if lo is not None:
                loss_list.append(lo)
        loss_txt = f"{sum(loss_list) / len(loss_list):.6f}" if loss_list else "—"

        train_loss_list: List[float] = []
        for gr in trainer_ranks:
            tl = train_loss_avg_for_round((payloads.get(gr) or {}).get("train") or [], ridx)
            if tl is not None:
                train_loss_list.append(tl)
        train_loss_txt = (
            f"{sum(train_loss_list) / len(train_loss_list):.6f}"
            if train_loss_list
            else "—"
        )

        md_cells = [
            format_ms(grpc_mean_ms),
            format_ms(grpc_max_ms),
            format_ms(grad_apply_ms),
            loss_txt,
            str(len(loss_list)),
            train_loss_txt,
        ]
        md_rows.append("| " + str(ridx) + " | " + " | ".join(md_cells) + " |")

        w.writerow(
            [
                ridx,
                format_ms(grpc_mean_ms),
                format_ms(grpc_max_ms),
                format_ms(grad_apply_ms),
                loss_txt,
                str(len(loss_list)),
                train_loss_txt,
            ]
        )

    intro = "\n".join(
        [
            "",
            "### Classic centralized per-global-round summary (from Node0 metrics_full + node_results)",
            "",
            "**Data source:** ``Node0.*/metrics_full.csv`` (primary); ``node_*_results.json`` (secondary). ",
            "**grpc_agg_mean_ms:** **max** across clients of per-trainer **mean** `sync/local_agg_time` for that round (**ms**). ",
            "Same timing buckets apply to gRPC and TorchDist (column name is historical). ",
            "**grpc_agg_max_ms:** **max** `sync/local_agg_time` across all client batch syncs in that round (**ms**). ",
            "**grad_apply_max_ms:** **max** `sync/grad_apply_time` across client trainers (**ms**). ",
            "**eval_loss_avg:** mean across trainers of eval loss at **round end** (`eval/loss`). ",
            "**train_loss_avg:** mean across trainers of batch training loss for that round (`train/loss`). ",
            "",
        ]
    )
    return intro + "\n".join(md_rows) + "\n", buf.getvalue()
