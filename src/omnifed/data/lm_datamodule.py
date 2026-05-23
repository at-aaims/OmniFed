"""Causal LM data (HF ``datasets.load_from_disk`` + tokenizer) for OmniFed."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from src.omnifed.data.datamodule import DataModule


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
    federated_client_id_env_var: str = "OMNIFED_FEDERATED_CLIENT_INDEX",
) -> DataModule:
    """
    Train/eval ``DataLoader`` pairs for C4-like on-disk corpus (HF ``DatasetDict``).

    Shards the **train** split across FL clients when ``shard_train`` and
    ``num_federated_clients > 1``. Client id is read from
    ``federated_client_id_env_var`` (set by ``slurm_hybrid_runner`` for hybrid jobs).
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ddict = load_from_disk(str(dataset_path))
    train_ds = ddict[train_split]
    eval_ds = ddict[eval_split]

    n_cli = int(num_federated_clients)
    if n_cli < 1:
        raise ValueError(f"num_federated_clients must be >= 1, got {n_cli}")

    raw_id = os.environ.get(federated_client_id_env_var)
    if raw_id is None or raw_id == "":
        client_idx = 0
    else:
        client_idx = int(raw_id)
    if not (0 <= client_idx < n_cli):
        raise ValueError(
            f"{federated_client_id_env_var}={raw_id!r} out of range for "
            f"num_federated_clients={n_cli}"
        )

    if shard_train and n_cli > 1:
        train_ds = train_ds.shard(num_shards=n_cli, index=client_idx)
    if shard_eval and n_cli > 1:
        eval_ds = eval_ds.shard(num_shards=n_cli, index=client_idx)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    train_map = _TextTokMapDataset(train_ds, tok, max_length)
    eval_map = _TextTokMapDataset(eval_ds, tok, max_length)

    collate = _collate_stack_dict
    pin_memory = torch.cuda.is_available()
    persistent = int(num_workers) > 0

    train_loader = DataLoader(
        train_map,
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate,
    )
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
