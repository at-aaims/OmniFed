# FLUX

A modular distributed framework for federated learning research and deployment, built on Ray and Hydra. Supports multiple FL algorithms, network topologies, and scales from local development to HPC clusters and cross-institutional deployments.

## Quick Start

```bash
# Run a basic federated learning experiment
./main.sh --config-name test_fedavg_centralized_torchdist

# Run all test configurations
./run_test_configs.sh
```

## Core Features

- **Multiple FL Algorithms**: FedAvg, FedProx, Scaffold, FedDyn, MOON, FedPer, FedBN, FedMom, FedNova, DiLoCo, Ditto
- **Flexible Topologies**: Centralized, multi-group hierarchical deployments
- **Communication Backends**: gRPC for cross-network, TorchDist for HPC clusters
- **Ray-based Distribution**: Scalable actor-based distributed computing
- **Hydra Configuration**: Modular, composable configuration management

## Architecture

- **Engine**: Ray orchestration and experiment coordination
- **Node**: Distributed Ray actors with lazy initialization
- **BaseAlgorithm**: Extensible FL algorithm framework
- **Topology**: Network structure definitions and node management
- **Communicators**: Message transport with broadcast/aggregate operations
- **DataModule**: Federated data loading and partitioning

## Supported Datasets

MNIST, Fashion-MNIST, CIFAR-10/100, Caltech-101/256 with automatic download and preprocessing.

## Dependencies

Core infrastructure: Ray, Hydra, PyTorch, gRPC

## Configuration

Framework uses Hydra for modular configuration composition:

```bash
# View configuration structure
python main.py --info defaults-tree

# Show final composed configuration
python main.py --cfg job
```
