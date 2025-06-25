# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import defaultdict
from typing import Any, Dict

import rich.repr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, SumMetric

from ..communicator.BaseCommunicator import Communicator
from .BaseAlgorithm import Algorithm

# ======================================================================================


@rich.repr.auto
class FedAvg(Algorithm):
    """
    Federated Averaging algorithm with local SGD training and weighted model averaging.
    """

    def __init__(
        self,
        comm: Communicator,
        model: nn.Module,
        local_epochs: int,
        lr: float,
    ):
        print(f"{self.__class__.__name__} init...")
        self.comm: Communicator = comm
        self.model: nn.Module = model

        # ---
        self.local_epochs: int = local_epochs
        self.lr: float = lr

        # ---
        self.opt: torch.optim.Optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
        )
        self.criterion: nn.Module = nn.CrossEntropyLoss()

    def on_round_start(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Broadcast model parameters at the beginning of each round.
        """
        print(f"on_round_start: round_num={round_num}, metrics={metrics}", flush=True)
        # Server broadcasts model parameters to all clients
        self.comm.broadcast(
            self.model,
            # src=0, # Default
        )
        return metrics

    def train_round(
        self,
        round_num: int,
        dataloader: DataLoader,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform local training for the current round.

        TODO: Abstract some of this away into the base class to enable the override-only-what-is-needed paradigm in subclasses.
        """
        print(f"train_round: round_num={round_num}, metrics={metrics}", flush=True)

        self.model.train()
        _device = next(self.model.parameters()).device

        # Round-level metrics
        _metrics_mean = defaultdict(lambda: MeanMetric().to(_device))
        _metrics_sum = defaultdict(lambda: SumMetric().to(_device))

        # Train for specified number of local epochs
        # epoch_iter = tqdm(range(self.local_epochs), desc=f"Round {round_num + 1}", unit="epoch")
        epoch_iter = range(self.local_epochs)
        for epoch_idx in epoch_iter:
            epoch_start_time = time.time()
            data_iter = iter(dataloader)

            for batch_idx in range(len(dataloader)):
                _t_batch_start = time.time()

                # Data loading phase
                _t_data_start = time.time()
                x, y = next(data_iter)
                x, y = x.to(_device), y.to(_device, non_blocking=True)
                _t_data = time.time() - _t_data_start

                # Compute phase
                _t_comp_start = time.time()

                # Track samples processed (cumulative across all epochs)
                batch_size = x.size(0)
                _metrics_sum["data/samples"].update(batch_size)
                _metrics_sum["data/batches"].update(1)

                # Forward pass
                self.opt.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                # sample-weighted loss
                _metrics_mean["loss/train"].update(loss.detach(), batch_size)

                # Backward pass
                loss.backward()

                # Calculate gradient norm
                _metrics_mean["train/grad_norm"].update(self.get_grad_norm(self.model))

                self.opt.step()

                _t_compute = time.time() - _t_comp_start
                _t_batch = time.time() - _t_batch_start

                # Update timing metrics
                _metrics_mean["time/step_data"].update(_t_data)
                _metrics_mean["time/step_compute"].update(_t_compute)
                _metrics_mean["time/step"].update(_t_batch)

            # Track epoch timing
            epoch_time = time.time() - epoch_start_time
            _metrics_mean["time/epoch"].update(epoch_time)
            _metrics_sum["time/round"].update(epoch_time)

            epoch_metrics = {
                **{
                    f"{k}_avg": round(v.compute().item(), 2)
                    for k, v in _metrics_mean.items()
                },
                **{
                    f"{k}_sum": round(v.compute().item(), 2)
                    for k, v in _metrics_sum.items()
                },
            }
            print(f"Epoch {epoch_idx + 1}/{self.local_epochs} | {epoch_metrics}")

        # Update the metrics dictionary with computed values
        metrics.update(
            {
                "round_num": round_num + 1,
                **{k: v.compute().item() for k, v in _metrics_mean.items()},
                **{k: v.compute().item() for k, v in _metrics_sum.items()},
            }
        )

        return metrics

    def on_round_end(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform model aggregation and additional logging after local training.
        """
        print(f"on_round_end: round_num={round_num}, metrics={metrics}", flush=True)

        # Log pre-aggregation model norm
        metrics["agg/param_norm_pre"] = self.get_param_norm(self.model)

        try:
            num_samples = metrics["data/samples"]
        except KeyError:
            raise KeyError(
                "Expected 'data/samples' in metrics, but it was not found. "
                "Available keys: " + ", ".join(metrics.keys())
            )

        # Use communicator's weighted averaging with sample count
        total_samples = self.comm.aggregate(
            msg=torch.Tensor([num_samples]),
            compute_mean=False,
        )
        weight_scaling = num_samples / total_samples.item()

        for _, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param.data *= weight_scaling

        self.model = self.comm.aggregate(
            msg=self.model,
            communicate_params=True,
            compute_mean=False,
        )

        # Log post-aggregation model norm
        metrics["agg/param_norm_post"] = self.get_param_norm(self.model)

        return metrics
