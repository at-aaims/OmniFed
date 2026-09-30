"""Hybrid per-round summary table builder."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.omnifed.summary.metrics_full import (
    eval_accuracy_avg_for_round,
    eval_loss_avg_for_round,
    format_ms,
    sync_metric_max_for_round,
    sync_round_indices,
)


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


def build_hybrid_per_round_tables(
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
        all_rounds.update(sync_round_indices(pl.get("sync") or []))
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
            sync_list = (payloads.get(lr) or {}).get("sync") or []
            grpc.append(sync_metric_max_for_round(sync_list, ridx, "global_agg_time"))

        la_m: List[Optional[float]] = []
        lb_m: List[Optional[float]] = []
        for members in members_by_f:
            la_v: List[float] = []
            lb_v: List[float] = []
            for gr in members:
                sync_list = (payloads.get(gr) or {}).get("sync") or []
                xa = sync_metric_max_for_round(sync_list, ridx, "local_agg_time")
                xb = sync_metric_max_for_round(sync_list, ridx, "local_bcast_time")
                if xa is not None:
                    la_v.append(xa)
                if xb is not None:
                    lb_v.append(xb)
            la_m.append(max(la_v) if la_v else None)
            lb_m.append(max(lb_v) if lb_v else None)

        acc_list: List[float] = []
        for gr in trainer_ranks:
            a = eval_accuracy_avg_for_round((payloads.get(gr) or {}).get("eval") or [], ridx)
            if a is not None:
                acc_list.append(a)
        acc_txt = f"{sum(acc_list) / len(acc_list):.4f}" if acc_list else "—"

        loss_list: List[float] = []
        for gr in trainer_ranks:
            lo = eval_loss_avg_for_round((payloads.get(gr) or {}).get("eval") or [], ridx)
            if lo is not None:
                loss_list.append(lo)
        loss_txt = f"{sum(loss_list) / len(loss_list):.6f}" if loss_list else "—"

        md_cells = (
            [* [format_ms(x) for x in grpc], * [format_ms(x) for x in la_m], * [format_ms(x) for x in lb_m],
             acc_txt, str(len(acc_list)), loss_txt, str(len(loss_list))]
        )
        md_rows.append("| " + str(ridx) + " | " + " | ".join(md_cells) + " |")

        csv_row = [ridx]
        csv_row.extend(format_ms(x) for x in grpc)
        csv_row.extend(format_ms(x) for x in la_m)
        csv_row.extend(format_ms(x) for x in lb_m)
        csv_row.extend([acc_txt, str(len(acc_list)), loss_txt, str(len(loss_list))])
        w.writerow(csv_row)

    intro = "\n".join(
        [
            "",
            "### Hybrid per-global-round summary (from Node0 metrics_full + node_results)",
            "",
            "**Data source:** ``Node0.*/metrics_full.csv`` (primary); ``node_*_results.json`` (secondary). ",
            "**gRPC_*:** **max** `sync/global_agg_time` on facility leader across all ``sync`` ",
            "rows in that outer round (**ms**). ",
            "**local_agg_* / local_bcast_*:** **max** over ranks in facility, each rank taking ",
            "the **max** across local/global sync events in that outer round (**ms**). ",
            "**accuracy_avg:** mean of per-trainer *accuracy* scalars logged in **`eval`** for that round ",
            "(keys containing `accuracy`, case‑insensitive). Requires eval to record accuracy.",
            "**eval_loss_avg:** mean across trainers of per-trainer average **`eval/loss`** (multiple eval ",
            "rows sharing the same **round_idx** are averaged inside each trainer first).",
            "",
        ]
    )
    md = intro + "\n".join(md_rows) + "\n"
    return md, buf.getvalue()
