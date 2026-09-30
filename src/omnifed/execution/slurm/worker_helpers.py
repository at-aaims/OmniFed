# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.

"""1-GPU Slurm worker helpers (round-end eval). Not a separate pipeline."""

from __future__ import annotations

import types

from src.omnifed.utils import print


def install_round_end_eval(algorithm, *, local_comm=None) -> None:
    """Optional round-end eval if ``datamodule.eval`` is set."""
    del local_comm
    base_round_end = getattr(algorithm, "_round_end", lambda self: None)

    def _round_end_with_eval(self) -> None:
        if getattr(self.datamodule, "eval", None) is not None:
            print(
                f"Round-end evaluation @ {self.progress_info_str}",
                flush=True,
            )
            self._BaseAlgorithm__eval_epoch(self.local_model)
        base_round_end()

    algorithm._round_end = types.MethodType(_round_end_with_eval, algorithm)


__all__ = ["install_round_end_eval"]
