# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any

import rich.repr
import torch

from .fedavg import FedAvg


@rich.repr.auto
class FedAvgLLM(FedAvg):
    """FedAvg with HF causal LM batches (tensor dict forward + HF loss)."""

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.local_model.parameters(), lr=float(local_lr))

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise TypeError(
                "FedAvgLLM expects batches as dicts with transformer keys "
                f"(got {type(batch).__name__})."
            )
        out = self.local_model(**batch)
        loss = getattr(out, "loss", None)
        if loss is None:
            raise RuntimeError(
                "Causal LM forward did not produce ``loss``. "
                "Ensure batches include ``labels`` (masked LM / CLM)."
            )
        return loss

    def _infer_batch_size(self, batch: Any) -> int:
        if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor):
            return int(batch["input_ids"].shape[0])
        return super()._infer_batch_size(batch)
