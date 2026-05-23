"""Hugging Face causal LM factories for Hydra ``instantiate``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def load_llama_from_pretrained_checkpoint(
    pretrained_model_name_or_path: str,
    *,
    local_files_only: bool = True,
    torch_dtype_str: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> Any:
    """
    Load ``LlamaForCausalLM`` from a Hub-style directory on disk (Frontier offline).

    ``torch_dtype_str``: ``float32``, ``float16``, ``bfloat16``, or ``none`` /
    omitted for model default weights.
    """
    from transformers import LlamaForCausalLM

    kwargs: Dict[str, Any] = dict(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        local_files_only=bool(local_files_only),
    )

    td = torch_dtype_str
    if td is None or str(td).lower() in ("none", "null"):
        td = ""
    td = td.strip().lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if td and td not in ("default", ""):
        if td not in dtype_map:
            raise ValueError(
                f"Unknown torch_dtype_str={torch_dtype_str!r}; "
                f"expected one of {sorted(set(dtype_map))}."
            )
        kwargs["torch_dtype"] = dtype_map[td]

    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)

    return LlamaForCausalLM.from_pretrained(**kwargs)
