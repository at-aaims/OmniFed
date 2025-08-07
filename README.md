# FLUX

A federated learning framework built on Ray and Hydra. FLUX scales from local experiments to HPC clusters and cross-institutional scenarios with 11 built-in algorithms and extensible architecture.

## Quick Start

```bash
# Clone and install
git clone <repository-url>

cd FLUX
pip install -r requirements.txt

# Run basic federated learning experiment
./main.sh --config-name test_fedavg_centralized_torchdist
```

This trains a neural network across 3 simulated clients using the MNIST dataset.

## Key Features

**Architecture & Extensibility**
- **Modular Composability**: Mix and match algorithms, topologies, and communication protocols
- **Extensible Interface**: Implement custom algorithms by overriding just two core methods
- **Research-Friendly Design**: Lifecycle hooks inject custom logic at round, epoch, and batch boundaries
- **Flexible Scaling**: Ray-based distributed coordination from local testing to HPC clusters

**Research & Experimentation**
- **Fully Configurable**: Configure algorithms, models, data, and parameters through type-safe Hydra files
- **11 Algorithm Implementations**: Includes FedAvg, SCAFFOLD, MOON, FedProx, DiLoCo, and six others
- **Native PyTorch Support**: Works with existing PyTorch models, optimizers, and training patterns

## Installation

**Requirements**: Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

## Basic Usage

### Run Experiments

**Different deployment types:**

```bash
# Local/HPC (PyTorch distributed)
./main.sh --config-name test_fedavg_centralized_torchdist

# Cross-network (gRPC)
./main.sh --config-name test_fedavg_centralized_grpc

# Hierarchical (multi-level)
./main.sh --config-name test_fedavg_hierarchical
```

**Parameter scaling:**

```bash
# More clients, rounds, epochs
./main.sh --config-name test_fedavg_centralized_torchdist topology.num_clients=10
./main.sh --config-name test_fedavg_centralized_torchdist global_rounds=10
./main.sh --config-name test_fedavg_centralized_torchdist algorithm.max_epochs_per_round=8
```

### Configuration

```bash
# View available configuration groups
python main.py --help

# Show final composed configuration
python main.py --config-name test_fedavg_centralized_torchdist --cfg job

# Override any parameter
./main.sh --config-name test_fedavg_centralized_torchdist global_rounds=10 algorithm.local_lr=0.001
```

## How FLUX Works

FLUX coordinates training through these components:

- **Algorithm**: The FL method (FedAvg, SCAFFOLD, etc.)
- **Topology**: How devices are connected (centralized server, hierarchical groups)
- **Communication**: How messages are sent (PyTorch distributed or gRPC)
- **Data**: Each device's private dataset

FLUX handles device coordination, model aggregation, and distributed execution automatically.

## Project Structure

```
FLUX/
├── src/flora/              # Main framework code
│   ├── algorithm/          # FL algorithms (FedAvg, SCAFFOLD, etc.)
│   ├── communicator/       # Communication protocols (TorchDist, gRPC)
│   ├── topology/           # Network structures (centralized, hierarchical)
│   ├── model/              # Neural network components
│   ├── data/               # Data loading and partitioning
│   ├── utils/              # Logging, metrics, and utilities
│   ├── engine.py           # Ray orchestration and coordination
│   └── node.py             # Federated learning participant actors
├── conf/                   # Hydra configuration files
│   ├── algorithm/          # Algorithm-specific configs
│   ├── datamodule/         # Dataset configurations
│   ├── model/              # Model architecture configs
│   ├── topology/           # Network topology configs
│   └── test_*.yaml         # Example experiment configurations
├── wiki/                   # Documentation and guides
├── outputs/                # Experiment results and logs
├── scripts/                # Utility and test scripts
├── main.py                 # Python entry point
├── main.sh                 # Development script with setup handling
└── requirements.txt        # Dependencies
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Test your changes
4. Submit pull request

## Citation

```bibtex
@inproceedings{flux2025,
  title={FLUX: A Modular Federated Learning Framework},
  author={Authors},
  year={2025}
}
```
