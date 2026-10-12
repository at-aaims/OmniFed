# Running OmniFed with Ray

This guide runs the centralized TorchDist experiment with three clients and one server.

## Backend Selection

The runtime checks available GPU resources across the connected Ray cluster before creating actors.

| Configured backend | Available GPUs | Effective execution |
|---|---|---|
| `gloo` | Any | Gloo with CPU execution |
| `nccl` | Fewer than four | Gloo with CPU execution |
| `nccl` | Four or more | NCCL with one GPU per actor |

The server participates in TorchDist collectives, so this experiment requires four distinct GPUs for NCCL execution.

GPU availability is checked before actor creation. This check does not reserve resources or guarantee successful placement.

## Environment Setup

Activate your environment from the repository directory:

```bash
cd /home/user/omnifed
conda activate omnifed
```

Set the dataset directory:

```bash
export OMNIFED_DATA_DIR="/home/user/omnifed/datasets"
```

For a cluster, use a dataset location accessible to all worker nodes. Each node must have compatible dependencies and access to the OmniFed code.

## Single-Node Execution

Start the experiment using a local Ray instance:

```bash
./main.sh \
  --config-name old_way/test_fedavg_centralized_torchdist \
  engine.mode=ray
```

To explicitly select Gloo and CPU execution:

```bash
./main.sh \
  --config-name old_way/test_fedavg_centralized_torchdist \
  engine.mode=ray \
  topology.local_comm.backend=gloo
```

## Multi-Node Execution

### Start the Head Node

Activate the environment on the head node, then run:

```bash
ray start --head \
  --port=6380 \
  --num-cpus="$(nproc)"
```

Ray normally detects the node's GPUs automatically. To explicitly specify the GPU count, add `--num-gpus=N`, replacing `N` with the actual number of usable GPUs on that node.

### Connect Worker Nodes

Activate the same environment on each worker node, then run:

```bash
ray start \
  --address="<HEAD_NODE_IP>:6380" \
  --num-cpus="$(nproc)"
```

Replace `<HEAD_NODE_IP>` with the head node's IP address reachable from the workers. You may also add `--num-gpus=N` using that worker's actual GPU count.

The nodes must be able to communicate over Ray and TorchDist network connections.

### Check the Cluster

On the head node:

```bash
ray status
```

Confirm that the worker nodes and their resources are registered.

### Run the Experiment

From the repository directory on the head node:

```bash
./main.sh \
  --config-name old_way/test_fedavg_centralized_torchdist \
  engine.mode=ray \
  ray.address=auto
```

The runtime checks cluster-wide GPU availability and chooses the effective backend for all actors. It discovers the node hosting TorchDist rank 0 and uses that node's IP address for rendezvous.

## Expected Startup Messages

When NCCL is requested but fewer than four GPUs are available:

```text
requested=nccl, effective=gloo, required_gpus=4, ...
```

Each actor should report:

```text
backend=gloo, device=cpu, num_gpus=0.0
```

When sufficient GPUs are available:

```text
requested=nccl, effective=nccl, required_gpus=4, ...
```

Each actor should report:

```text
backend=nccl, device=cuda:0, num_gpus=1.0
```

`cuda:0` refers to the first GPU visible inside each actor. Different actors receive different physical GPUs.