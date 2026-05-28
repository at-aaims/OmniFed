#!/bin/bash
#SBATCH -A gen150
#SBATCH -J ddp_test
#SBATCH -p batch
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH -o /lustre/orion/gen150/scratch/shruti2395/llama_test/logs/ddp-%j.o
#SBATCH -e /lustre/orion/gen150/scratch/shruti2395/llama_test/logs/ddp-%j.e

set -eo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
source activate pytorch_rocm

export MASTER_PORT=29500
scontrol show hostnames "$SLURM_NODELIST" > job.nodes.list
export MASTER_ADDR=$(head -n1 job.nodes.list)
echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_NODELIST=$SLURM_NODELIST"

cd /lustre/orion/gen150/scratch/shruti2395/llama_test
srun --export=ALL -u bash -lc '
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  fi
  unset ROCR_VISIBLE_DEVICES
  echo "[rank $SLURM_PROCID] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"
  /ccs/home/shruti2395/.conda/envs/pytorch_rocm/bin/python -u ddp_test.py
'
