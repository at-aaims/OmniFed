"""Lightweight LM datamodule utilities (collate stacking)."""

from __future__ import annotations

import torch

from src.omnifed.data.lm_datamodule import _collate_stack_dict


def test_collate_stack_dict_lm_batch() -> None:
    a = {
        "input_ids": torch.ones(8, dtype=torch.long),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "labels": torch.ones(8, dtype=torch.long),
    }
    b = {
        "input_ids": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "labels": torch.zeros(8, dtype=torch.long),
    }
    stacked = _collate_stack_dict([a, b])
    assert stacked["input_ids"].shape == (2, 8)
