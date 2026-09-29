"""Launch backends: Ray vs Slurm.

``engine.mode=slurm`` uses ``execution.slurm``: ``slurm_launcher`` /
``slurm_worker`` for 1-GPU clients, ``torchtitan_launcher`` /
``torchtitan_worker`` when ``torchtitan.module`` is set. Topology still
chooses hops.
"""

from .shared import uses_torchtitan, validate_execution_mode

__all__ = ["uses_torchtitan", "validate_execution_mode"]
