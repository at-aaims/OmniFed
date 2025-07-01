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

import copy
from typing import Any, Dict, List, Optional

import torch
from torch import nn


def get_param_norm(model: nn.Module) -> float:
    """
    Calculate L2 norm of model parameters for monitoring.

    Useful for tracking parameter magnitude during training.

    Args:
        model: Model to compute parameter norm for
    """
    param_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm += p.data.norm(2).item() ** 2
    param_norm = param_norm**0.5
    return param_norm


def get_grad_norm(model: nn.Module) -> float:
    """
    Calculate L2 norm of parameter gradients for monitoring.

    Useful for gradient clipping decisions and training diagnostics.

    Args:
        model: Model to compute gradient norm for
    """
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm**0.5
    return grad_norm


def clip_grads(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradient norms to prevent exploding gradients.

    Returns the computed gradient norm before clipping.

    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm threshold
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def scale_grads(model: nn.Module, scale_factor: float) -> None:
    """
    Scale all parameter gradients by a constant factor.

    Args:
        model: Model to scale gradients for
        scale_factor: Factor to scale gradients by
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale_factor)


def scale_params(model: nn.Module, scale_factor: float) -> None:
    """
    Scale all trainable model parameters by a constant factor.

    Args:
        model: Model to scale parameters for
        scale_factor: Factor to scale parameters by
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data.mul_(scale_factor)


def weighted_avg_models(
    models: List[nn.Module],
    weights: List[float],
) -> nn.Module:
    """
    Core averaging logic for model parameters.

    Args:
        models: List of models to average
        weights: Corresponding weights for each model

    Returns:
        New model with averaged parameters
    """
    if not models:
        raise ValueError("Cannot average empty list of models")
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    # Clone the first model's structure and zero its parameters
    averaged_model = copy.deepcopy(models[0])

    # Zero out parameters efficiently
    with torch.no_grad():
        for param in averaged_model.parameters():
            param.zero_()

    # Weighted sum of all model parameters
    with torch.no_grad():
        for model, weight in zip(models, weights):
            for avg_param, model_param in zip(
                averaged_model.parameters(), model.parameters()
            ):
                avg_param.add_(model_param, alpha=weight)

    return averaged_model


def weighted_avg_tensors(
    tensor_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted aggregation of tensor dictionaries.

    Args:
        tensor_dicts: List of tensor dictionaries to aggregate
        weights: Optional weights for each dict (default: uniform averaging)

    Returns:
        Aggregated tensor dictionary
    """
    if not tensor_dicts:
        raise ValueError("Cannot aggregate empty list of tensor dicts")

    if weights is None:
        weights = [1.0 / len(tensor_dicts)] * len(tensor_dicts)

    if len(tensor_dicts) != len(weights):
        raise ValueError("Number of tensor dicts must match number of weights")

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    # Get parameter names from first dict and validate consistency
    param_names = set(tensor_dicts[0].keys())
    # Validate remaining dicts against the first one
    for i, tensors in enumerate(tensor_dicts[1:], 1):
        if set(tensors.keys()) != param_names:
            raise ValueError(f"Tensor dict {i} has different parameter names")

    # Weighted sum of tensors
    aggregated = {}
    for name in param_names:
        # Initialize with first weighted tensor
        first_weight, first_tensors = weights[0], tensor_dicts[0]
        aggregated[name] = first_weight * first_tensors[name]

        # Add remaining weighted tensors using alpha argument
        for weight, tensors in zip(weights[1:], tensor_dicts[1:]):
            aggregated[name].add_(tensors[name], alpha=weight)

    return aggregated


def compute_model_delta(
    model1: nn.Module,
    model2: nn.Module,
    requires_grad_only: bool = True,  # Note: Could use filter callback for > flexibility, but this covers main use cases
) -> Dict[str, torch.Tensor]:
    """
    Compute parameter deltas between two models: model1 - model2.

    Args:
        model1: First model
        model2: Second model
        requires_grad_only: If True, only compute deltas for parameters that require gradients

    Returns:
        Dictionary mapping parameter names to delta tensors
    """
    deltas = {}

    model1_params = dict(model1.named_parameters())
    model2_params = dict(model2.named_parameters())

    for name, param1 in model1_params.items():
        if requires_grad_only and not param1.requires_grad:
            continue

        if name not in model2_params:
            raise ValueError(f"Parameter '{name}' not found in second model")

        param2 = model2_params[name]

        if param1.shape != param2.shape:
            raise ValueError(
                f"Parameter '{name}' shape mismatch: {param1.shape} vs {param2.shape}"
            )

        deltas[name] = param1.data - param2.data

    return deltas


def apply_model_delta(
    model: nn.Module, deltas: Dict[str, torch.Tensor], scale: float = 1.0
) -> None:
    """
    Apply parameter deltas to a model: model += scale * deltas.

    Args:
        model: Model to update
        deltas: Dictionary of parameter deltas
        scale: Scaling factor for deltas (default: 1.0)
    """
    model_params = dict(model.named_parameters())

    with torch.no_grad():
        for name, delta in deltas.items():
            # Validate parameter existence and shape compatibility
            if name not in model_params:
                raise ValueError(f"Parameter '{name}' not found in model")

            param = model_params[name]

            if param.shape != delta.shape:
                raise ValueError(
                    f"Parameter '{name}' shape mismatch: {param.shape} vs {delta.shape}"
                )

            param.add_(delta, alpha=scale)


def calculate_batch_size(batch: Any) -> int:
    """
    Extract number of samples from batch for metrics tracking.
    Handles common PyTorch batch formats automatically.

    Override for custom batch structures.
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0)

    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        # Try first element
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return first.size(0)

    if hasattr(batch, "__len__"):
        return len(batch)

    raise ValueError(f"Cannot estimate batch size for type {type(batch)}")
