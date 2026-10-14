from __future__ import annotations


def build_frontier_setup_lines() -> list[str]:
    return [
        "module load PrgEnv-gnu/8.6.0",
        "module load rocm/6.4.1",
        "module load rccl-net-plugin",
        "module load craype-accel-amd-gfx90a",
        "module list",
        # "export NCCL_DEBUG=INFO",
        # "export NCCL_DEBUG_SUBSYS=INIT,NET",
        "",
        (
            'export ENV_TARBALL='
            '"/ccs/home/${USER}/omnifed.tar.gz"'
        ),
        (
            'export ENV_ROOT='
            '"/mnt/bb/${USER}/omnifed_env"'
        ),
        (
            'export ENV_BASE='
            '"/mnt/bb/${USER}"'
        ),
        (
            'export ENV_COPY='
            '"/mnt/bb/${USER}/omnifed.tar.gz"'
        ),
        (
            'export MIOPEN_USER_DB_PATH='
            '"/mnt/bb/${USER}/miopen-cache"'
        ),
        (
            'export MIOPEN_CUSTOM_CACHE_DIR='
            '"/mnt/bb/${USER}/miopen-cache"'
        ),
        "export MIOPEN_FIND_MODE=1",
        "",
        (
            'srun -N "$SLURM_JOB_NUM_NODES" '
            '-n "$SLURM_JOB_NUM_NODES" '
            "--ntasks-per-node=1 bash -lc "
            "'mkdir -p \"$MIOPEN_USER_DB_PATH\"'"
        ),
        (
            'echo "[setup] MIOPEN_USER_DB_PATH='
            '$MIOPEN_USER_DB_PATH"'
        ),
        "",
        'echo "[setup] hostname=$(hostname)"',
        'echo "[setup] ENV_TARBALL=$ENV_TARBALL"',
        'echo "[setup] ENV_ROOT=$ENV_ROOT"',
        'echo "[setup] ENV_COPY=$ENV_COPY"',
        'ls -lh "$ENV_TARBALL"',
        "",
        (
            'srun -N "$SLURM_JOB_NUM_NODES" '
            '-n "$SLURM_JOB_NUM_NODES" '
            "--ntasks-per-node=1 bash -lc "
            "'mkdir -p \"$ENV_BASE\"'"
        ),
        'sbcast -pf "$ENV_TARBALL" "$ENV_COPY"',
        "",
        (
            'srun -N "$SLURM_JOB_NUM_NODES" '
            '-n "$SLURM_JOB_NUM_NODES" '
            "--ntasks-per-node=1 bash -lc "
            "'rm -rf \"$ENV_ROOT\" "
            "&& mkdir -p \"$ENV_ROOT\" "
            "&& tar -xzf \"$ENV_COPY\" "
            "-C \"$ENV_ROOT\"'"
        ),
        "",
        (
            'if [ ! -f "$ENV_ROOT/bin/python" ]; '
            'then echo "[setup] ERROR: python missing"; '
            "exit 1; fi"
        ),
        (
            'if [ ! -f "$ENV_ROOT/bin/activate" ]; '
            'then echo "[setup] ERROR: activate missing"; '
            "exit 1; fi"
        ),
        "",
        (
            'srun -N "$SLURM_JOB_NUM_NODES" '
            '-n "$SLURM_JOB_NUM_NODES" '
            "--ntasks-per-node=1 bash -lc "
            "'source \"$ENV_ROOT/bin/activate\" "
            "&& conda-unpack'"
        ),
        "",
        'export PYEXE="$ENV_ROOT/bin/python"',
        'echo "[setup] PYEXE=$PYEXE"',
        '"$PYEXE" --version',
    ]