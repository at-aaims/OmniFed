"""Causal LM data (HF ``datasets.load_from_disk`` + tokenizer) for OmniFed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.omnifed.data.datamodule import DataModule
from src.omnifed.data.federated_shards import (
    FEDERATED_CLIENT_INDEX_ENV,
    resolve_federated_client_index,
)


def _collate_stack_dict(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("empty batch")
    out: Dict[str, torch.Tensor] = {}
    for k in samples[0]:
        out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


class _TextTokMapDataset(Dataset):  # type: ignore[type-arg]
    """Random-access HF ``Dataset`` row ``text`` field → tensors (no padding collation)."""

    def __init__(self, hf_split: Any, tokenizer: Any, max_length: int) -> None:
        self._rows = hf_split
        self._tok = tokenizer
        self._max_length = int(max_length)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._rows[int(idx)]
        text = row.get("text", "")
        enc = self._tok(
            text,
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        item = {
            k: enc[k].squeeze(0)
            for k in ("input_ids", "attention_mask")
            if k in enc
        }
        item["labels"] = item["input_ids"].clone()
        return item


def build_c4_lm_datamodule(
    dataset_path: str,
    tokenizer_path: str,
    *,
    num_federated_clients: int,
    train_split: str = "train",
    eval_split: str = "validation",
    max_length: int = 1024,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    num_workers: int = 0,
    shard_train: bool = True,
    shard_eval: bool = False,
    federated_client_id_env_var: str = FEDERATED_CLIENT_INDEX_ENV,
    max_train_batches: Optional[int] = None,
    include_eval: bool = True,
) -> DataModule:
    """
    Train/eval ``DataLoader`` pairs for C4-like on-disk corpus (HF ``DatasetDict``).

    Shards the **train** split across FL trainers when ``shard_train`` and
    ``num_federated_clients > 1``. Clerk (index -1) has ``train=None``.

    ``max_train_batches`` (smoke): after sharding, keep at most
    ``max_train_batches * train_batch_size`` train rows so ``len(train)``
    is the cap. ``include_eval=False`` skips the val loader.
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ddict = load_from_disk(str(dataset_path))
    train_ds = ddict[train_split]
    eval_ds = ddict[eval_split] if include_eval else None

    n_cli = int(num_federated_clients)
    if n_cli < 1:
        raise ValueError(f"num_federated_clients must be >= 1, got {n_cli}")

    client_idx = resolve_federated_client_index(
        n_cli, env_var=federated_client_id_env_var
    )

    if client_idx is not None:
        if shard_train and n_cli > 1:
            train_ds = train_ds.shard(num_shards=n_cli, index=client_idx)
        if shard_eval and n_cli > 1 and eval_ds is not None:
            eval_ds = eval_ds.shard(num_shards=n_cli, index=client_idx)
        if max_train_batches is not None:
            cap_rows = int(max_train_batches) * int(train_batch_size)
            if cap_rows < 0:
                raise ValueError(
                    f"max_train_batches must be >= 0, got {max_train_batches}"
                )
            take = min(len(train_ds), cap_rows)
            train_ds = train_ds.select(range(take))

    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    collate = _collate_stack_dict
    pin_memory = torch.cuda.is_available()
    persistent = int(num_workers) > 0

    train_loader = None
    if client_idx is not None:
        train_map = _TextTokMapDataset(train_ds, tok, max_length)
        train_loader = DataLoader(
            train_map,
            batch_size=int(train_batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            persistent_workers=persistent,
            collate_fn=collate,
        )
    eval_loader = None
    if include_eval:
        eval_map = _TextTokMapDataset(eval_ds, tok, max_length)
        eval_loader = DataLoader(
            eval_map,
            batch_size=int(eval_batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=pin_memory,
            persistent_workers=persistent,
            collate_fn=collate,
        )
    return DataModule(train=train_loader, eval=eval_loader)
