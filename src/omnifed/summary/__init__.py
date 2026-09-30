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
from src.omnifed.summary.startup import (
    emit_model_startup,
    instantiate_model_timed,
    move_model_to_device_timed,
)

__all__ = [
    "accumulate_iter_comm",
    "emit_model_startup",
    "install_iteration_recorder",
    "instantiate_model_timed",
    "move_model_to_device_timed",
    "summary_mode_from_cfg",
    "write_slurm_per_round_summary",
    "write_slurm_per_round_summary_for_run",
]
