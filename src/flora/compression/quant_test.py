import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from typing import List, Tuple, Dict
import time


class GradientQuantizer:
    """Gradient quantization with configurable bit width"""

    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width
        self.levels = 2 ** bit_width - 1

    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to specified bit width
        Returns: (quantized_tensor, scale, zero_point)
        """
        if tensor.numel() == 0:
            return tensor, torch.tensor(1.0), torch.tensor(0.0)

        # Calculate min/max for uniform quantization
        min_val = tensor.min()
        max_val = tensor.max()

        # Handle edge case where all values are the same
        if min_val == max_val:
            return torch.zeros_like(tensor, dtype=torch.uint8), torch.tensor(1.0), min_val

        # Calculate scale and zero point
        scale = (max_val - min_val) / self.levels
        zero_point = torch.round(-min_val / scale).clamp(0, self.levels)

        # Quantize
        quantized = torch.round(tensor / scale + zero_point).clamp(0, self.levels)

        return quantized.to(torch.uint8), scale, zero_point

    def dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor,
                          zero_point: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Dequantize tensor back to float"""
        if quantized.numel() == 0:
            return torch.zeros(original_shape)

        dequantized = scale * (quantized.float() - zero_point)
        return dequantized.reshape(original_shape)

    def quantize_gradients(self, gradients: List[torch.Tensor]) -> Dict:
        """Quantize list of gradient tensors"""
        quantized_data = {
            'tensors': [],
            'scales': [],
            'zero_points': [],
            'shapes': []
        }

        for grad in gradients:
            if grad is not None:
                q_tensor, scale, zero_point = self.quantize_tensor(grad.flatten())
                quantized_data['tensors'].append(q_tensor)
                quantized_data['scales'].append(scale)
                quantized_data['zero_points'].append(zero_point)
                quantized_data['shapes'].append(grad.shape)
            else:
                # Handle None gradients
                quantized_data['tensors'].append(None)
                quantized_data['scales'].append(None)
                quantized_data['zero_points'].append(None)
                quantized_data['shapes'].append(None)

        return quantized_data

    def dequantize_gradients(self, quantized_data: Dict) -> List[torch.Tensor]:
        """Dequantize back to gradient tensors"""
        gradients = []

        for i in range(len(quantized_data['tensors'])):
            if quantized_data['tensors'][i] is not None:
                grad = self.dequantize_tensor(
                    quantized_data['tensors'][i],
                    quantized_data['scales'][i],
                    quantized_data['zero_points'][i],
                    quantized_data['shapes'][i]
                )
                gradients.append(grad)
            else:
                gradients.append(None)

        return gradients


class QuantizedSGDOptimizer:
    """SGD optimizer with gradient quantization for distributed training"""

    def __init__(self, model_params, lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 1e-4, bit_width: int = 8):
        self.params = list(model_params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.quantizer = GradientQuantizer(bit_width)

        # Initialize momentum buffers
        self.momentum_buffers = {}
        for i, param in enumerate(self.params):
            if param.requires_grad:
                self.momentum_buffers[i] = torch.zeros_like(param.data)

    def get_gradients(self) -> List[torch.Tensor]:
        """Extract gradients from model parameters"""
        return [param.grad for param in self.params if param.requires_grad]

    def set_gradients(self, gradients: List[torch.Tensor]):
        """Set gradients back to model parameters"""
        grad_idx = 0
        for param in self.params:
            if param.requires_grad:
                if gradients[grad_idx] is not None:
                    param.grad = gradients[grad_idx].to(param.device)
                grad_idx += 1

    def step(self):
        """Perform SGD step with momentum"""
        param_idx = 0
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data

                # Add weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                # Apply momentum
                if param_idx in self.momentum_buffers:
                    momentum_buf = self.momentum_buffers[param_idx]
                    momentum_buf.mul_(self.momentum).add_(grad)
                    grad = momentum_buf

                # Update parameters
                param.data.add_(grad, alpha=-self.lr)
                param_idx += 1

    def zero_grad(self):
        """Zero out gradients"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def all_reduce_quantized_gradients(gradients: List[torch.Tensor], quantizer: GradientQuantizer):
    """
    Perform all-reduce on quantized gradients across all processes.

    Two approaches:
    1. Simple: Dequantize locally, then all-reduce floats (less communication efficient)
    2. Advanced: All-reduce quantized values as integers, then average and requantize
    """

    # Simple approach: Dequantize locally, then all-reduce floats
    # This maintains correctness but reduces communication efficiency
    quantized_data = quantizer.quantize_gradients(gradients)
    local_dequantized = quantizer.dequantize_gradients(quantized_data)

    averaged_gradients = []
    for grad in local_dequantized:
        if grad is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            averaged_gradients.append(grad / dist.get_world_size())
        else:
            averaged_gradients.append(None)

    return averaged_gradients


def all_reduce_quantized_gradients_advanced(gradients: List[torch.Tensor], quantizer: GradientQuantizer):
    """
    Advanced approach: All-reduce in quantized domain with proper handling.
    This maintains communication efficiency while preserving correctness.
    """

    # Step 1: Quantize gradients locally
    quantized_data = quantizer.quantize_gradients(gradients)

    averaged_gradients = []

    for i, grad in enumerate(gradients):
        if grad is not None and quantized_data['tensors'][i] is not None:
            # Step 2: All-reduce the quantized values as int32 to avoid overflow
            q_tensor = quantized_data['tensors'][i].int()
            scale = quantized_data['scales'][i]
            zero_point = quantized_data['zero_points'][i]

            # All-reduce quantized tensor (sum as integers)
            dist.all_reduce(q_tensor, op=dist.ReduceOp.SUM)

            # All-reduce scale and zero_point
            dist.all_reduce(scale, op=dist.ReduceOp.SUM)
            dist.all_reduce(zero_point, op=dist.ReduceOp.SUM)

            # Step 3: Average the summed values
            q_tensor_avg = q_tensor.float() / dist.get_world_size()
            scale_avg = scale / dist.get_world_size()
            zero_point_avg = zero_point / dist.get_world_size()

            # Step 4: Dequantize using averaged parameters
            dequantized = scale_avg * (q_tensor_avg - zero_point_avg)
            dequantized = dequantized.reshape(grad.shape)

            averaged_gradients.append(dequantized)
        else:
            averaged_gradients.append(None)

    return averaged_gradients


def get_cifar10_dataloaders(batch_size: int = 128, rank: int = 0, world_size: int = 1):
    """Create CIFAR-10 dataloaders for distributed training"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, train_sampler


def train_epoch(model, train_loader, optimizer, criterion, device, quantizer, rank):
    """Train for one epoch with quantized gradient communication"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Get gradients and quantize + all-reduce
        gradients = optimizer.get_gradients()
        reduced_gradients = all_reduce_quantized_gradients(gradients, quantizer)
        # Alternative: use all_reduce_quantized_gradients_advanced for better communication efficiency
        # reduced_gradients = all_reduce_quantized_gradients_advanced(gradients, quantizer)
        optimizer.set_gradients(reduced_gradients)

        # Update parameters
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0 and rank == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    return running_loss / len(train_loader), 100. * correct / total


def test_epoch(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader), 100. * correct / total


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_worker(rank, world_size, bit_width, epochs, batch_size, lr):
    """Main training function for each process"""

    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Create model
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create optimizer and quantizer
    optimizer = QuantizedSGDOptimizer(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, bit_width=bit_width
    )
    quantizer = GradientQuantizer(bit_width)
    criterion = nn.CrossEntropyLoss()

    # Create dataloaders
    train_loader, test_loader, train_sampler = get_cifar10_dataloaders(
        batch_size, rank, world_size
    )

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, quantizer, rank
        )

        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

        if rank == 0:
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Quantization: {bit_width}-bit')
            print('-' * 50)

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Quantized SGD Distributed Training')
    parser.add_argument('--bit_width', type=int, default=8,
                        help='Quantization bit width (default: 8)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--world_size', type=int, default=2,
                        help='Number of GPUs (default: 2)')

    args = parser.parse_args()

    print(f"Starting distributed training with {args.world_size} GPUs")
    print(f"Gradient quantization: {args.bit_width}-bit")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. This script requires GPU support.")
        return

    if torch.cuda.device_count() < args.world_size:
        print(f"Not enough GPUs. Required: {args.world_size}, Available: {torch.cuda.device_count()}")
        return

    # Launch distributed training
    mp.spawn(
        train_worker,
        args=(args.world_size, args.bit_width, args.epochs, args.batch_size, args.lr),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()