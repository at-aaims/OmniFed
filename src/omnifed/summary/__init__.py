"""Universal Slurm summary generation (classic centralized and hybrid)."""

from src.omnifed.summary.per_iteration import (
    accumulate_iter_comm,
    install_iteration_recorder,
)
from src.omnifed.summary.pipeline import summary_mode_from_cfg
from src.omnifed.summary.slurm_per_round import (
    write_slurm_per_round_summary,
    write_slurm_per_round_summary_for_run,
)

__all__ = [
    "accumulate_iter_comm",
    "install_iteration_recorder",
    "summary_mode_from_cfg",
    "write_slurm_per_round_summary",
    "write_slurm_per_round_summary_for_run",
]
