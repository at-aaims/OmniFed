# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated Hydra alias. Llama uses ``algorithm: fedavg`` + ``optimizer: adamw``."""

import rich.repr

from .fedavg import FedAvg


@rich.repr.auto
class FedAvgLLM(FedAvg):
    """Same loop as FedAvg. Kept so old ``algorithm: fedavg_llm`` yamls still instantiate."""
