from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class FrozenRunConfig:
    cfg: DictConfig
    hydra_output_dir: str
    slurm_checkpoint_dir: str
    source_path: Path


def load_frozen_run_config(
    path: str | Path,
) -> FrozenRunConfig:
    source_path = Path(path)

    with source_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw: dict[str, Any] = json.load(file)

    cfg = OmegaConf.create(raw["cfg"])
    hydra_output_dir = str(
        raw["hydra_output_dir"]
    )

    slurm_checkpoint_dir = str(
        raw.get("slurm_checkpoint_dir")
        or Path(hydra_output_dir)
        / "engine"
        / "ckpt"
    )

    return FrozenRunConfig(
        cfg=cfg,
        hydra_output_dir=hydra_output_dir,
        slurm_checkpoint_dir=slurm_checkpoint_dir,
        source_path=source_path,
    )


def parse_frozen_config_argument() -> Path:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg-json",
        required=True,
        help="Path to Engine's frozen configuration",
    )

    arguments = parser.parse_args()

    return Path(arguments.cfg_json)