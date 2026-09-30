# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hydra entry for vision DataModules with unique train shards (CIFAR, etc.)."""

from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import DataLoader

from src.omnifed.data.datamodule import DataModule
from src.omnifed.data.federated_shards import (
    rebuild_dataloader,
    resolve_federated_client_index,
    resolve_num_federated_clients,
    unique_train_subset,
)
from src.omnifed.utils import print


def build_federated_vision_datamodule(
    train: Optional[DataLoader] = None,
    eval: Optional[DataLoader] = None,
    num_federated_clients: Optional[int] = None,
    shard_train: bool = True,
    shard_eval: bool = False,
    **_: Any,
) -> DataModule:
    """
    Wrap Hydra-built loaders: shard train across trainers; server has train=None.

    Eval stays full unless ``shard_eval`` (default off). Same module for classic
    and hierarchical; runners set ``OMNIFED_FEDERATED_CLIENT_INDEX`` /
    ``OMNIFED_NUM_FEDERATED_CLIENTS``.
    """
    n_cli = resolve_num_federated_clients(num_federated_clients)
    client_idx = resolve_federated_client_index(n_cli)

    if shard_train and train is not None:
        if client_idx is None:
            print(
                f"[datamodule] server: train=None (num_federated_clients={n_cli})",
                flush=True,
            )
            train = None
        elif n_cli > 1:
            n_full = len(train.dataset)
            sharded = unique_train_subset(train.dataset, n_cli, client_idx)
            train = rebuild_dataloader(train, sharded)
            print(
                f"[datamodule] train shard client_idx={client_idx}/{n_cli} "
                f"len={len(sharded)}/{n_full}",
                flush=True,
            )

    if shard_eval and eval is not None and client_idx is not None and n_cli > 1:
        eval = rebuild_dataloader(
            eval, unique_train_subset(eval.dataset, n_cli, client_idx)
        )

    return DataModule(train=train, eval=eval)
