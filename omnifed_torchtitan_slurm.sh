#!/bin/bash
#SBATCH --job-name=omnifed_titan_grpc
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --signal=B:USR1@180
#SBATCH -C nvme
#SBATCH --exclusive
#SBATCH --account=GEN150
#SBATCH --partition=batch
#SBATCH -o /autofs/nccs-svm1_home2/sabiha/omnifed-titan/outputs/2026-07-11/test_fedavg_centralized_torchdist/slurm-%j.out
#SBATCH -e /autofs/nccs-svm1_home2/sabiha/omnifed-titan/outputs/2026-07-11/test_fedavg_centralized_torchdist/slurm-%j.err
#SBATCH --chdir=/autofs/nccs-svm1_home2/sabiha/omnifed-titan
#SBATCH --gpus-per-node=8
set -euo pipefail
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
export OMNIFED_DATA_DIR="/ccs/home/sabiha/OmniFed/datasets"
echo "[setup] OMNIFED_DATA_DIR=$OMNIFED_DATA_DIR"

export ENV_TARBALL="/ccs/home/sabiha/omnifed.tar.gz"
export ENV_ROOT="/mnt/bb/${USER}/omnifed_env"
export ENV_BASE="/mnt/bb/${USER}"
export ENV_COPY="/mnt/bb/${USER}/omnifed.tar.gz"
export MIOPEN_USER_DB_PATH="/mnt/bb/${USER}/miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR="/mnt/bb/${USER}/miopen-cache"
export MIOPEN_FIND_MODE=1
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'mkdir -p "$MIOPEN_USER_DB_PATH"'
echo "[setup] MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"

echo "[setup] hostname=$(hostname)"
echo "[setup] ENV_TARBALL=$ENV_TARBALL"
echo "[setup] ENV_BASE=$ENV_BASE"
echo "[setup] ENV_ROOT=$ENV_ROOT"
echo "[setup] ENV_COPY=$ENV_COPY"
ls -lh "$ENV_TARBALL"

echo "[setup] Broadcasting packed env to compute nodes"
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'mkdir -p "$ENV_BASE"'
sbcast -pf "$ENV_TARBALL" "$ENV_COPY"
echo "[setup] sbcast finished"
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'ls -lh "$ENV_COPY"'

echo "[setup] starting unpack..."
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'rm -rf "$ENV_ROOT" && mkdir -p "$ENV_ROOT" && tar -xzf "$ENV_COPY" -C "$ENV_ROOT"'
echo "[setup] unpack finished"
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'ls -lh "$ENV_ROOT/bin/python"'
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'ls -lh "$ENV_ROOT/bin/activate"'

if [ ! -f "$ENV_ROOT/bin/python" ]; then echo "[setup] ERROR: python missing at $ENV_ROOT/bin/python"; exit 1; fi
if [ ! -f "$ENV_ROOT/bin/activate" ]; then echo "[setup] ERROR: activate missing at $ENV_ROOT/bin/activate"; exit 1; fi

echo "[setup] running conda-unpack..."
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc 'source "$ENV_ROOT/bin/activate" && conda-unpack'
echo "[setup] conda-unpack finished"

export PYEXE="$ENV_ROOT/bin/python"
echo "[setup] PYEXE=$PYEXE"
"$PYEXE" --version

export PYTHONPATH="/autofs/nccs-svm1_home2/sabiha/omnifed-titan:${PYTHONPATH:-}"
export PYEXE="${PYEXE:-python}"
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SERVER_HOST="${HOSTS[0]}"
export CFG_JSON="/autofs/nccs-svm1_home2/sabiha/omnifed-titan/outputs/engine_frozen.json"
CHECKPOINT_ROOT="/lustre/orion/gen150/scratch/sabiha/omnifed_titan/job_${SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_ROOT"

srun --exclusive --nodes=1 --ntasks=1 --nodelist="$SERVER_HOST" env OMNIFED_ROLE=server FEDERATED_RANK=0 CHECKPOINT_ROOT="$CHECKPOINT_ROOT" bash -lc 'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi; unset ROCR_VISIBLE_DEVICES; export HF_HOME="/mnt/bb/sabiha/hf_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"; export TRANSFORMERS_CACHE="${HF_HOME}/transformers"; mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"; echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; echo "HF_HOME=$HF_HOME"; echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"; exec "$PYEXE" -u -m src.omnifed.slurm_worker --cfg-json "$CFG_JSON"' &
CLIENT_0_NODES=$(IFS=,; echo "${HOSTS[*]:1:2}")
CLIENT_0_MASTER="${HOSTS[1]}"
srun --exclusive --nodes=2 --ntasks=16 --ntasks-per-node=8 --nodelist="$CLIENT_0_NODES" env OMNIFED_ROLE=client CLIENT_ID=0 FEDERATED_RANK=1 CLIENT_LEADER_RANK=0 CLIENT_MASTER_ADDR="$CLIENT_0_MASTER" CLIENT_MASTER_PORT=29600 SERVER_ADDR="$SERVER_HOST" CLIENT_CHECKPOINT_ROOT="$CHECKPOINT_ROOT/client_0" bash -lc 'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi; unset ROCR_VISIBLE_DEVICES; export HF_HOME="/mnt/bb/sabiha/hf_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"; export TRANSFORMERS_CACHE="${HF_HOME}/transformers"; mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"; echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; echo "HF_HOME=$HF_HOME"; echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"; exec "$PYEXE" -u -m src.omnifed.slurm_worker --cfg-json "$CFG_JSON"' &
CLIENT_1_NODES=$(IFS=,; echo "${HOSTS[*]:3:2}")
CLIENT_1_MASTER="${HOSTS[3]}"
srun --exclusive --nodes=2 --ntasks=16 --ntasks-per-node=8 --nodelist="$CLIENT_1_NODES" env OMNIFED_ROLE=client CLIENT_ID=1 FEDERATED_RANK=2 CLIENT_LEADER_RANK=0 CLIENT_MASTER_ADDR="$CLIENT_1_MASTER" CLIENT_MASTER_PORT=29601 SERVER_ADDR="$SERVER_HOST" CLIENT_CHECKPOINT_ROOT="$CHECKPOINT_ROOT/client_1" bash -lc 'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi; unset ROCR_VISIBLE_DEVICES; export HF_HOME="/mnt/bb/sabiha/hf_cache/${SLURM_JOB_ID}/rank_${SLURM_PROCID}"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"; export TRANSFORMERS_CACHE="${HF_HOME}/transformers"; mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"; echo "[worker] hostname=$(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"; echo "HF_HOME=$HF_HOME"; echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"; exec "$PYEXE" -u -m src.omnifed.slurm_worker --cfg-json "$CFG_JSON"' &
wait
echo "All Torchtitan subclusters completed"
