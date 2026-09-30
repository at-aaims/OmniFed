from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def get_initial_model_paths(
    cfg: Any,
) -> tuple[Path, Path]:
    checkpoint_root = Path(
        cfg.torchtitan.subclusters.checkpoint_root
    )

    job_id = os.environ["SLURM_JOB_ID"]

    initial_path = (
        checkpoint_root
        / f"job_{job_id}"
        / "initial"
        / "model.pt"
    )

    ready_path = initial_path.with_suffix(".ready")

    return initial_path, ready_path


def wait_for_file(
    path: str | Path,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 1.0,
) -> None:
    target = Path(path)
    deadline = time.monotonic() + timeout_seconds

    while not target.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {target}"
            )

        time.sleep(poll_seconds)