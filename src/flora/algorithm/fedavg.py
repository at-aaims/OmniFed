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

from typing import Any, Dict

import rich.repr
import torch
from torch import nn
from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    calibration_error,
    f1_score,
    matthews_corrcoef,
    precision,
    recall,
)

from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedAvg(BaseAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.

    FedAvg performs standard federated learning by averaging model parameters across clients after local training rounds.
    Only model parameters are aggregated; all clients synchronize with the global model at the start of each round.

    [FedAvg](https://arxiv.org/abs/1602.05629) | H. Brendan McMahan | 2016-02-17
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Forward pass and compute the cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        return loss


# ======================================================================================


@rich.repr.auto
class FedAvgCustom(FedAvg):
    """
    FedAvg with MultiStepLR learning rate scheduling and comprehensive metrics.

    Extends the base FedAvg algorithm with:
    - SGD optimizer with momentum (0.9) and weight decay (1e-4)
    - MultiStepLR scheduler that reduces learning rate by 10x at epochs 100 and 150
    - Comprehensive evaluation metrics including accuracy, precision, recall, F1, AUROC, etc.
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Configure SGD optimizer with momentum and weight decay, plus MultiStepLR scheduler.

        Args:
            local_lr: Initial learning rate for local training

        Returns:
            Configured SGD optimizer with scheduler attached as self.scheduler
        """
        optimizer = torch.optim.SGD(
            self.local_model.parameters(), lr=local_lr, momentum=0.9, weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )

        return optimizer

    def _train_epoch_end(self) -> None:
        """
        Step the learning rate scheduler at the end of each training epoch.

        This ensures the learning rate is reduced at the specified milestone epochs.
        Also logs the current learning rate for monitoring.
        """
        self.scheduler.step()

        # Log current learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        self.log_metric("learning_rate", current_lr)

    def _eval_batch(self, batch: Any) -> Dict[str, float]:
        """
        Execute evaluation with comprehensive classification metrics.

        Computes a wide range of metrics to assess model performance:
        - Essential metrics: Accuracy, Precision, Recall, F1-Score
        - Advanced metrics: AUROC, Average Precision, Calibration Error, Matthews Correlation
        - Confidence: Average prediction confidence

        Args:
            batch: Evaluation batch containing inputs and targets

        Returns:
            Dictionary containing all computed metrics
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Determine number of classes and task type
        num_classes = outputs.size(1)
        task = "binary" if num_classes == 2 else "multiclass"

        # Compute loss
        loss = nn.functional.cross_entropy(outputs, targets)

        # Essential metrics
        acc = accuracy(preds, targets, task=task, num_classes=num_classes)
        prec = precision(
            preds,
            targets,
            task=task,
            num_classes=num_classes,
        )
        rec = recall(
            preds,
            targets,
            task=task,
            num_classes=num_classes,
        )
        f1 = f1_score(
            preds,
            targets,
            task=task,
            num_classes=num_classes,
        )

        # Advanced metrics
        auc = auroc(probs, targets, task=task, num_classes=num_classes)
        ap = average_precision(probs, targets, task=task, num_classes=num_classes)
        cal_err = calibration_error(probs, targets, task=task, num_classes=num_classes)
        mcc = matthews_corrcoef(preds, targets, task=task, num_classes=num_classes)

        # Confidence metric
        confidence = torch.max(probs, dim=1)[0].mean()

        return {
            "loss": loss.item(),
            "accuracy": acc.item(),
            "precision": prec.item(),
            "recall": rec.item(),
            "f1_score": f1.item(),
            "auroc": auc.item(),
            "average_precision": ap.item(),
            "calibration_error": cal_err.item(),
            "matthews_corrcoef": mcc.item(),
            "avg_confidence": confidence.item(),
        }
