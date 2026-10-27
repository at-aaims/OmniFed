# Common

```bash
module load miniforge3/23.11.0-0
conda activate /lustre/orion/gen150/proj-shared/shared_titan/omnifed_env
cd /lustre/orion/gen150/scratch/shruti2395/OmniFed_VT
export ENV_TARBALL="/lustre/orion/gen150/proj-shared/shared_titan/omnifed_env.tar.gz"
export OMNIFED_DATA_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data"
export PYTHONPATH="/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT"
```

# 1 GPU client, 7 nodes, 1 GPU each node, centralized, gRPC

```bash
export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/Llama-3.2-400M-Amharic"
export COMMON='overwrite=true slurm.account=gen150 slurm.partition=batch slurm.time=01:59:00 slurm.exclusive=true slurm.gres=null slurm.ntasks_per_node=1'
./main.sh --config-name experiment/single_level_c4_llama400_fedavg_grpc $COMMON
```

# Decentralized, TorchDist, 6 clients

```bash
export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/Llama-3.2-400M-Amharic"
export COMMON='overwrite=true slurm.account=gen150 slurm.partition=batch slurm.time=01:59:00 slurm.exclusive=true slurm.gres=null slurm.ntasks_per_node=1'
./main.sh --config-name experiment/single_level_c4_llama400_fedavg_torchdist $COMMON
```

# Decentralized, TorchDist, 6 GPUs on 1 node, 6 clients

```bash
export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/Llama-3.2-400M-Amharic"
export COMMON='overwrite=true slurm.account=gen150 slurm.partition=batch slurm.time=01:59:00 slurm.exclusive=true slurm.gres=null slurm.ntasks_per_node=6'
./main.sh --config-name experiment/single_level_c4_llama400_fedavg_torchdist \
  $COMMON \
  slurm.nodes=1 slurm.gpus_per_node=6
```

# TorchTitan, centralized

```bash
export PYTHONPATH="/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT:/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed"
export TORCHTITAN_ROOT="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed"
export TORCHTITAN_HF_ASSETS="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed/assets/hf/Llama-3.1-8B"
export TORCHTITAN_DATASET="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed/tests/assets/c4_test"
export OMNIFED_TORCHTITAN_SCRATCH="/lustre/orion/gen150/scratch/shruti2395/omnifed_titan"
export COMMON='overwrite=true slurm.account=gen150 slurm.partition=batch slurm.time=01:59:00 slurm.exclusive=true slurm.gres=null slurm.cpus_per_task=6'
./main.sh --config-name experiment/single_level_torchtitan_fedavg $COMMON
```

# TorchTitan, decentralized

```bash
export PYTHONPATH="/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT:/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed"
export TORCHTITAN_ROOT="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed"
export TORCHTITAN_HF_ASSETS="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed/assets/hf/Llama-3.1-8B"
export TORCHTITAN_DATASET="/lustre/orion/gen150/proj-shared/shared_titan/torchtitan-omnifed/tests/assets/c4_test"
export OMNIFED_TORCHTITAN_SCRATCH="/lustre/orion/gen150/scratch/shruti2395/omnifed_titan"
export COMMON='overwrite=true slurm.account=gen150 slurm.partition=batch slurm.time=01:59:00 slurm.exclusive=true slurm.gres=null slurm.cpus_per_task=6'
./main.sh --config-name experiment/single_level_torchtitan_fedavg \
  $COMMON \
  topology=decentralized topology.num_clients=1 topology.local_comm=null
```
