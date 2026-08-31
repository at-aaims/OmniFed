#!/usr/bin/env python3
"""
Export per-rank per-iteration CSV + GPU memory logs from a classic Slurm run.

Usage (on Frontier, from run directory or parent):

  python3 scripts/export_classic_per_rank_logs.py \\
    outputs/2026-08-06_12-14-36/test_pi_centralized_sync_grad_cifar10_resnet18_torchdist

Writes:
  engine/classic_per_rank_notion.txt   — paste into Notion (Rank0 … Rank6 sections)
  engine/classic_per_rank_logs.zip     — all rank CSV + GPU log files

Stdlib only — safe to run without full OmniFed imports.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


def _find_engine_dir(run_dir: Path) -> Path:
    eng = run_dir / "engine"
    if eng.is_dir():
        return eng
    raise SystemExit(f"No engine/ directory under {run_dir}")


def _detect_ranks(engine: Path) -> list[int]:
    ranks: set[int] = set()
    for pattern in (
        "rank*_classic_per_iteration_summary.csv",
        "rank*_gpu_memory.log",
    ):
        for path in engine.glob(pattern):
            match = re.match(r"rank(\d+)_", path.name)
            if match:
                ranks.add(int(match.group(1)))
    return sorted(ranks)


def _read_text(path: Path, *, max_lines: int | None) -> str:
    if not path.is_file():
        return f"(missing: {path.name})\n"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines is not None and len(lines) > max_lines:
        keep_tail = max(1, max_lines // 5)
        keep_head = max(1, max_lines - keep_tail - 1)
        omitted = len(lines) - keep_head - keep_tail
        lines = (
            lines[:keep_head]
            + [f"... truncated {omitted} lines ..."]
            + lines[-keep_tail:]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_notion_text(
    engine: Path,
    ranks: list[int],
    *,
    max_iter_lines: int | None,
    max_gpu_lines: int | None,
) -> str:
    parts: list[str] = []
    parts.append(f"Run: {engine.parent.name}\nEngine: {engine}\n")

    for rank in ranks:
        iter_path = engine / f"rank{rank}_classic_per_iteration_summary.csv"
        gpu_path = engine / f"rank{rank}_gpu_memory.log"

        iter_body = _read_text(iter_path, max_lines=max_iter_lines)
        gpu_body = _read_text(gpu_path, max_lines=max_gpu_lines)

        block = []
        block.append(f"Rank{rank}\n")
        block.append("=" * 40 + "\n\n")
        block.append("Per iteration summary\n---------------------\n\n")
        block.append("```csv\n")
        block.append(iter_body if iter_body.strip() else "(empty)\n")
        block.append("```\n\n")
        block.append("GPU logs\n--------\n\n")
        block.append("```csv\n")
        block.append(gpu_body if gpu_body.strip() else "(empty)\n")
        block.append("```\n\n")
        parts.append("".join(block))

    return "\n".join(parts).rstrip() + "\n"


def write_zip(engine: Path, ranks: list[int], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rank in ranks:
            for name in (
                f"rank{rank}_classic_per_iteration_summary.csv",
                f"rank{rank}_gpu_memory.log",
            ):
                path = engine / name
                if path.is_file():
                    zf.write(path, arcname=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export per-rank per-iteration CSV + GPU logs for Notion."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Hydra run dir (contains engine/ and slurm-*.out)",
    )
    parser.add_argument(
        "--max-iter-lines",
        type=int,
        default=None,
        help="Truncate per-iteration CSV (keeps head + tail). Default: full file.",
    )
    parser.add_argument(
        "--max-gpu-lines",
        type=int,
        default=200,
        help="Truncate GPU log lines (default 200; full iter CSV, trimmed GPU).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip writing classic_per_rank_logs.zip",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Notion text path (default: engine/classic_per_rank_notion.txt)",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    engine = _find_engine_dir(run_dir)
    ranks = _detect_ranks(engine)
    if not ranks:
        raise SystemExit(f"No rank*_classic_per_iteration_summary.csv or gpu logs in {engine}")

    notion_path = args.output or (engine / "classic_per_rank_notion.txt")
    notion_text = build_notion_text(
        engine,
        ranks,
        max_iter_lines=args.max_iter_lines,
        max_gpu_lines=args.max_gpu_lines,
    )
    notion_path.write_text(notion_text, encoding="utf-8")

    print(f"Wrote {notion_path} ({len(ranks)} ranks: {ranks})", flush=True)

    if not args.no_zip:
        zip_path = engine / "classic_per_rank_logs.zip"
        write_zip(engine, ranks, zip_path)
        print(f"Wrote {zip_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
