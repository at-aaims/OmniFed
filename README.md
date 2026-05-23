# OmniFed (Beta)

A federated learning framework built on [Ray](https://ray.io/) and [Hydra](https://hydra.cc/). OmniFed scales from local experiments to HPC clusters and cross-institutional scenarios with 10+ built-in algorithms and extensible architecture.

## Key Features

- **🧩 Modular**: Mix and match 10+ single-file algorithm implementations, topologies, and communication protocols
- **📊 Flexible**: Local, HPC, and cross-network deployments with multiple communication backends
- **⚙️ Extensible**: Custom algorithms, communicators, and topologies with minimal code requirements
- **🔬 Research-Friendly**: Easy experimentation with lifecycle hooks and [PyTorch](https://pytorch.org/) compatibility
- **🚀 Scalable**: [Ray](https://ray.io/)-powered distributed coordination from laptops to HPC clusters

## Quick Start

### OmniFed with SLURM

```bash
# Run basic federated learning experiment with SLURM
./main.sh --config-name test_fedavg_centralized_torchdist   engine.mode=slurm   slurm.enabled=true   slurm.partition=debug   slurm.nodes=2   slurm.ntasks_per_node=1   slurm.time=02:00:00   slurm.gres="gpu:1"
```

### Hybrid Slurm (`engine.communication_mode=hybrid`)

Cross-facility gRPC plus per-facility Torch MPI; see **`docs/HYBRID_SLURM_REFERENCE.md`**.

**Hydra presets (Phase C):**

- **`--config-name test_hybrid_engine_contract`** — **`engine.hybrid.topology_config`** → **`conf_hybrid/topology/built_symmetric_2x3.yaml`** (named reproducible lattice).
- **`--config-name test_hybrid_layout_fedavg`** — **same experiment** (**`world_size`** 7, **`topology.num_clients: 6`**) via **`engine.hybrid.layout`** only (Figure‑2 style: lattice next to **`topology`** / **`engine`** blocks — no **`conf_hybrid`** YAML path).

How **`slurm.nodes`** / **`ntasks_per_node`** relate to **`#SBATCH --ntasks`** (**hybrid `world_size`**): **`docs/HYBRID_SLURM_REFERENCE.md`** §**4.3** (**Phase D**).

**Centralized baseline (not hybrid):** For classic **MNIST FedAvg** over a **single** Torch collective world (TorchDist/NCCL, rank-0 server with train dataloader stubbed), use **`--config-name test_fedavg_centralized_torchdist`** — that pulls **`conf/test_fedavg_centralized_torchdist.yaml`**, keeps default **`engine.communication_mode=classic`**, and is **different** from the hybrid presets. Examples:
- **Ray:** `./main.sh --config-name test_fedavg_centralized_torchdist`
- **Slurm:** same `--config-name` with **`engine.mode=slurm`** and **`slurm.*`** knobs (same pattern as **OmniFed with SLURM** above).

Prerequisites:

- Frozen config + **`slurm_worker`** on each task (**`PYTHONPATH`** to repo root; **`PYEXE`** for ROCm stack on compute nodes is often injected via **`engine.py`** `setup_lines` on Frontier).
- **MNIST offline** on OLCF Frontier: compute nodes may not reach the public internet — pre-stage torchvision MNIST under Lustre and pass **`download=false`** and matching **`dataset.root`** for train and eval.

Minimal Frontier-style submit (seven tasks, seven nodes; substitute **`test_hybrid_layout_fedavg`** for the **`--config-name`** line if you prefer **`engine.hybrid.layout`**):

```bash
./main.sh --config-name test_hybrid_engine_contract \
  overwrite=true \
  engine.mode=slurm \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  slurm.account=YOUR_PROJECT \
  slurm.partition=batch \
  slurm.time=00:45:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

Optional knobs (see **`conf/base.yaml`** **`engine.hybrid`**): **`server_shutdown: leader_done`** (default — wait for leader marker files plus a wall-time cap), **`leader_done_poll_sec`**, **`sleep`** fallback, **`server_sec_per_round`**.

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
@inproceedings{omnifed2025,
  title={OmniFed: A Modular Federated Learning Framework},
  author={Authors},
  year={2025}
}
```
