#!/usr/bin/env python3
"""Regenerate classic_per_round_summary.* from an existing Hydra run directory."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.omnifed.summary.slurm_per_round import write_classic_slurm_per_round_summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build classic centralized per-round summary CSV from Node0 metrics."
    )
    p.add_argument(
        "hydra_out_dir",
        help="Hydra run dir (contains Node0.* and engine/node_results/)",
    )
    p.add_argument("--world-size", type=int, default=7)
    p.add_argument("--rpc-server-rank", type=int, default=0)
    args = p.parse_args()

    out = write_classic_slurm_per_round_summary(
        os.path.abspath(args.hydra_out_dir),
        world_size=int(args.world_size),
        rpc_server_rank=int(args.rpc_server_rank),
        rank_writer=1,
    )
    if out is None:
        raise SystemExit(1)
    print(f"Wrote summary under {out}", flush=True)


if __name__ == "__main__":
    main()
