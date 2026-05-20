# OmniFed (Beta)

A federated learning framework built on [Ray](https://ray.io/) and [Hydra](https://hydra.cc/). OmniFed scales from local experiments to HPC clusters and cross-institutional scenarios with 10+ built-in algorithms and extensible architecture.

## Key Features

- **🧩 Modular**: Mix and match 10+ single-file algorithm implementations, topologies, and communication protocols
- **📊 Flexible**: Local, HPC, and cross-network deployments with multiple communication backends
- **⚙️ Extensible**: Custom algorithms, communicators, and topologies with minimal code requirements
- **🔬 Research-Friendly**: Easy experimentation with lifecycle hooks and [PyTorch](https://pytorch.org/) compatibility
- **🚀 Scalable**: [Ray](https://ray.io/)-powered distributed coordination from laptops to HPC clusters

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd OmniFed
pip install -r requirements.txt

# Run basic federated learning experiment
./main.sh --config-name test_fedavg_centralized_torchdist
```

This configures a federated learning experiment with multiple nodes using the CIFAR-10 dataset. You'll see:

- **Setup phase**: Ray cluster initialization, node actor creation, and model broadcasting
- **Training progress**: Loss, accuracy, and other metrics logged per batch/epoch  
- **Communication logs**: Model aggregation and synchronization between nodes
- **Results**: Final metrics saved to timestamped output directory

## Running Experiments

**Different deployment types:**

```bash
# Local/HPC clusters (PyTorch distributed communication)
./main.sh --config-name test_fedavg_centralized_torchdist

# Cross-network deployment (gRPC communication)
./main.sh --config-name test_fedavg_centralized_grpc

# Multi-tier hierarchical setup
./main.sh --config-name test_fedavg_hierarchical
```

**Customize parameters:**

```bash
# Override any parameter
./main.sh --config-name test_fedavg_centralized_torchdist topology.num_clients=10 global_rounds=10 algorithm.max_epochs_per_round=8
```

## Configuration

**Explore available configurations:**

```bash
# See all available experiment configs
python main.py --help
```

**Inspect configurations:**

```bash
# Preview full configuration before running
python main.py --config-name test_fedavg_centralized_torchdist --cfg job

# Show resolved configuration (with interpolations)
python main.py --config-name test_fedavg_centralized_torchdist --cfg job --resolve

# Focus on specific config sections
python main.py --config-name test_fedavg_centralized_torchdist --cfg job --package algorithm
```

**Troubleshooting:**

```bash
# Get detailed Hydra system information
python main.py --info
```

See [Hydra's command line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for more configuration options.

## How OmniFed Works

OmniFed orchestrates federated learning experiments through a modular architecture:

**Core Components:**

- **Algorithm**: Defines the federated learning strategy (e.g., FedAvg for simple averaging, SCAFFOLD for drift correction, MOON for contrastive learning)
- **Topology**: Specifies the network structure and client-server relationships (centralized for star topology, hierarchical for multi-tier deployments)
- **Communicator**: Handles message passing between nodes (PyTorch distributed for HPC clusters, gRPC for cross-network scenarios)
- **DataModule**: Manages data loading, partitioning, and distribution across clients
- **Model**: Any PyTorch nn.Module with seamless integration

**Execution Flow:**

1. **Initialization**: Ray spawns distributed actors based on topology configuration
2. **Local Training**: Each client trains on private data for specified epochs/batches
3. **Model Exchange**: Clients send updates to aggregators via the communicator
4. **Aggregation**: Server combines updates using the algorithm's aggregation strategy
5. **Model Distribution**: Updated global model is broadcast back to clients
6. **Evaluation**: Periodic validation on local and/or global test sets

**Additional Capabilities:**

- **Flexible Scheduling**: Control when aggregation and evaluation occur
- **Metric Tracking**: Built-in logging system for training loss and custom metrics
- **Stateful Algorithms**: Support for momentum, control variates, and personalized models

## Project Structure

```
OmniFed/
├── src/flora/              # Main framework code
│   ├── algorithm/          # Federated learning algorithms
│   │   ├── base.py         # Base algorithm class
│   │   ├── fedavg.py       # FedAvg
│   │   └── ...             # 10 more algorithms (SCAFFOLD, MOON, FedProx, etc.)
│   ├── communicator/       # Communication protocols
│   │   ├── base.py         # Base communicator class
│   │   ├── torchdist.py    # PyTorch distributed backend
│   │   ├── grpc.py         # gRPC backend
│   │   └── ...             # gRPC server/client implementations
│   ├── topology/           # Network structures
│   │   ├── base.py         # Base topology class
│   │   ├── centralized.py  # Centralized topology
│   │   ├── hierarchical.py # Hierarchical topology
│   │   └── ...
│   ├── model/              # Built-in model examples and reusable components
│   │   └── ...
│   ├── data/               # Data loading and partitioning
│   │   ├── datamodule.py   # DataModule class
│   │   └── ...
│   ├── utils/              # Utilities and helpers
│   │   ├── metric_logger.py    # Metrics tracking and logging
│   │   ├── results_display.py  # Results visualization
│   │   └── ...
│   ├── engine.py           # Ray orchestration and coordination
│   └── node.py             # Federated learning participant actors
├── conf/                   # Hydra configuration files
│   ├── algorithm/          # Algorithm-specific configs
│   ├── datamodule/         # Dataset configurations
│   ├── model/              # Model architecture configs
│   ├── topology/           # Network topology configs
│   └── test_*.yaml         # Example experiment configurations
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
@inproceedings{10.1145/3731599.3767397,
author = {Tyagi, Sahil and Cozma, Andrei and Kotevska, Olivera and Wang, Feiyi},
title = {OmniFed: A Modular Framework for Configurable Federated Learning from Edge to HPC},
year = {2025},
isbn = {9798400718717},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3731599.3767397},
doi = {10.1145/3731599.3767397},
abstract = {Federated Learning (FL) is critical for edge and High Performance Computing (HPC) where data is not centralized and privacy is crucial. We present OmniFed, a modular framework designed around decoupling and clear separation of concerns for configuration, orchestration, communication, and training logic. Its architecture supports configuration-driven prototyping and code-level override-what-you-need customization. We also support different topologies, mixed communication protocols within a single deployment, and popular training algorithms. It also offers optional privacy mechanisms including Differential Privacy (DP), Homomorphic Encryption (HE), and Secure Aggregation (SA), as well as compression strategies. These capabilities are exposed through well-defined extension points, allowing users to customize topology and orchestration, learning logic, and privacy/compression plugins, all while preserving the integrity of the core system. We evaluate multiple models and algorithms to measure various performance metrics. By unifying topology configuration, mixed-protocol communication, and pluggable modules in one stack, OmniFed streamlines FL deployment across heterogeneous environments. Github repository is available at https://github.com/at-aaims/OmniFed.},
booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {516–523},
numpages = {8},
keywords = {Federated Learning (FL), Collaborative Learning (CL), Privacy-Preserving Machine Learning (ML), Edge computing, High Performance Computing (HPC), Deep Learning (DL), Compression},
location = {
},
series = {SC Workshops '25}
}
```
