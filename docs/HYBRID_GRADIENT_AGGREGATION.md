# Hybrid Gradient Aggregation (Slurm Pipeline)

**Branch:** `feature/hybrid-omnifed-on-upstream`  
**Scope:** `src/omnifed/hybrid/` only — no Flora runtime path  
**Related:** [HYBRID_QSGD_IMPLEMENTATION_STEPS.md](./HYBRID_QSGD_IMPLEMENTATION_STEPS.md) (QSGD compression on global gRPC)

---

## Goal

Add a **single Hydra knob** so hybrid Slurm FedAvg can aggregate either:

| Mode | `engine.hybrid.aggregate_payload` | FedAvg variant | Typical use |
|------|-----------------------------------|----------------|-------------|
| **Parameters** (default) | `params` | Weight averaging after local training | Current production behavior |
| **Gradients** | `gradients` | FedSGD-style grad averaging | QSGD papers, PR #86 parity |

The same setting applies to **both** communication hops:

1. **Local MPI** (within each facility) — `TorchMPIAdapter`
2. **Global gRPC** (facility leaders ↔ central PS) — `GrpcLeaderCommunicator`

**No mixed mode** (e.g. params locally + grads globally). Both hops always use the same payload type.

---

## Why gradients?

QSGD was designed for **gradient** communication. The initial hybrid QSGD port compressed **sample-weighted parameters** on the global hop; that can be numerically unstable at low bit widths. Gradient aggregation is the intended path for compression experiments and alignment with collaborator PR #86.

Parameter mode remains the **default** so existing configs and Frontier jobs behave unchanged.

---

## Configuration

### Hydra knob (`conf/base.yaml`)

```yaml
engine:
  hybrid:
    aggregate_payload: params   # params | gradients
```

### CLI override

```bash
engine.hybrid.aggregate_payload=gradients
```

### Internal mapping

| Config value | `communicate_params` | Aggregates |
|--------------|----------------------|------------|
| `params` | `True` | `param.data` |
| `gradients` | `False` | `param.grad` |

Helper module: `src/omnifed/hybrid/hybrid_aggregate_config.py`

```python
from src.omnifed.hybrid.hybrid_aggregate_config import (
    hybrid_aggregate_payload_from_cfg,
    hybrid_communicate_params_from_cfg,
)
```

---

## End-to-end sync flow

Aggregation runs at **`round_end`** by default (after all local epochs in a global round), same as dense hybrid ResNet runs.

### Parameter mode (`params`) — unchanged

```
Local train (backward + optimizer.step each batch)
    ↓
MPI: sample-weighted all-reduce on param.data (within facility)
    ↓
gRPC: leaders send weighted param.data → PS FedAvg → pull averaged weights
    ↓
MPI broadcast param.data (within facility)
```

### Gradient mode (`gradients`)

```
Local train (backward only; accumulate grads over the round)
    ↓
MPI: sample-weighted all-reduce on param.grad (within facility)
    ↓
gRPC: leaders send weighted param.grad → PS average → pull averaged grad into param.grad
    ↓
Leaders only: optimizer.step()  (apply global averaged grad to param.data)
    ↓
MPI broadcast param.data (within facility; non-leaders wait for broadcast)
    ↓
zero_grad (all ranks)
```

### MPI sub-steps (both modes)

Within each facility, local aggregation is always:

1. All-reduce **total sample count** (scalar SUM)
2. **`scale_params`** or **`scale_grads`** by `local_samples / group_total`
3. All-reduce **SUM** on `param.data` or `param.grad` (weighted average)

Broadcast always distributes **`param.data`** (weights), never grads.

---

## Training loop (gradient mode)

Normal FedAvg calls `zero_grad → backward → optimizer.step()` **every batch**, which clears gradients before sync.

Gradient mode **defers** the optimizer step until after aggregation:

| Phase | Params mode | Gradients mode |
|-------|-------------|----------------|
| Round start | — | `zero_grad` once |
| Each batch | zero_grad, backward, **step** | **backward only** (grads accumulate) |
| After sync | — | **leaders:** `optimizer.step()`; then broadcast weights |
| After sync | optimizer reset (base) | `zero_grad` |

Implemented in `install_hybrid_slurm_sync()` (`src/omnifed/hybrid/hybrid_slurm_sync.py`) by patching:

- `_round_start` — zero grads at round start
- `_train_batch` — backward only, log `grad_norm`
- `_optimizer_step` — no-op during local training
- `_aggregate_within_group` — `scale_grads` + MPI grad all-reduce
- `__sync_comm` — apply averaged grad on leaders, then broadcast, then clear grads

Training helpers: `src/omnifed/hybrid/hybrid_grad_training.py`

---

## Implementation phases (completed)

### Phase 0 — Config knob

| Item | Location |
|------|----------|
| Hydra default | `conf/base.yaml` → `engine.hybrid.aggregate_payload: params` |
| Parser / helpers | `src/omnifed/hybrid/hybrid_aggregate_config.py` |
| Startup log | `slurm_hybrid_runner.py` prints `aggregate_payload=...` |

**Tests:** `tests/test_hybrid_aggregate_config.py`

---

### Phase 1 — Global gRPC wiring

Wire `communicate_params` from config through the gRPC stack (server previously always used params).

| File | Change |
|------|--------|
| `grpc_leader_comm.py` | Read cfg; pass `communicate_params` to `GrpcCommunicator.aggregate()` |
| `communicator/global_grpc.py` | Constructor + `CentralServerServicer(..., communicate_params=...)` |
| `communicator/global_grpc_server.py` | Log `Aggregate payload: params\|gradients` |
| `slurm_hybrid_runner.py` | Daemon PS passes `communicate_params` |

**Safety:** If `communicate_params=False` but `param.grad` is missing at gRPC send time, raise a clear `RuntimeError` (training path not ready).

---

### Phase 2 — Local MPI wiring

| File | Change |
|------|--------|
| `algorithm/utils.py` | New `scale_grads()` (mirror of `scale_params`) |
| `torch_mpi_adapter.py` | `communicate_params` flag → all-reduce `p.data` or `p.grad` |
| `hybrid_slurm_sync.py` | Grad overrides for `_aggregate_within_group` |
| `slurm_hybrid_runner.py` | Same flag for `TorchMPIAdapter` + `install_hybrid_slurm_sync()` |

---

### Phase 3 — Training correctness

| File | Change |
|------|--------|
| `hybrid_grad_training.py` | `apply_optimizer_grads`, `clear_model_grads`, grad validation |
| `hybrid_slurm_sync.py` | Deferred step, leader-only apply, post-broadcast grad clear |
| `checkpoint/hybrid_round_checkpoint.py` | Manifest field `aggregate_payload` |
| `slurm_hybrid_runner.py` | Save payload in manifest; **fail resume** if mode changed |

**Tests:** `tests/test_hybrid_grad_training.py` (+ config + checkpoint tests → **14 passed** locally)

---

## QSGD + gradients

QSGD compression (`engine.hybrid.global_compression`) applies on the **global gRPC hop only**. Local MPI stays **dense** in both modes.

Recommended experiment matrix after gradient path is validated:

```bash
# Dense grad baseline
engine.hybrid.aggregate_payload=gradients
engine.hybrid.global_compression.enabled=false

# QSGD on gradients (global hop)
engine.hybrid.aggregate_payload=gradients
engine.hybrid.global_compression.enabled=true
engine.hybrid.global_compression.scheme=qsgd
engine.hybrid.global_compression.bit_width=4   # s=4 → 16 levels
```

QSGD wire format (norm in `meta_tensor`, signed levels in `values_data`, `width`/`level` for int8/int32 decode) is unchanged — see `global_grpc_compression.py` and `HYBRID_QSGD_IMPLEMENTATION_STEPS.md`.

**Do not** enable `aggregate_payload=gradients` until Phases 0–3 are deployed on Frontier (rsync + unit tests).

---

## Checkpointing

Round checkpoints store **model weights** only (same as before). The manifest now records aggregation mode:

```json
{
  "aggregate_payload": "gradients",
  "last_completed_round": 2,
  "status": "in_progress"
}
```

**Resume rule:** `engine.hybrid.aggregate_payload` must match the saved manifest. Switching `params ↔ gradients` mid-experiment requires a new `slurm.experiment_id`.

---

## Key files (reference)

| Area | Files |
|------|-------|
| Config | `conf/base.yaml`, `hybrid_aggregate_config.py` |
| Training | `hybrid_slurm_sync.py`, `hybrid_grad_training.py`, `algorithm/utils.py` (`scale_grads`) |
| Local MPI | `torch_mpi_adapter.py` |
| Global gRPC | `grpc_leader_comm.py`, `communicator/global_grpc.py`, `global_grpc_server.py`, `global_grpc_client.py` |
| Runner | `slurm_hybrid_runner.py` |
| Checkpoint | `checkpoint/hybrid_round_checkpoint.py` |
| Tests | `tests/test_hybrid_aggregate_config.py`, `tests/test_hybrid_grad_training.py` |

---

## Local verification

```bash
conda activate omnifed_flora
cd /path/to/OmniFed_VT
export PYTHONPATH="${PWD}"

python -m pytest \
  tests/test_hybrid_aggregate_config.py \
  tests/test_hybrid_grad_training.py \
  tests/test_hybrid_round_checkpoint.py \
  -v
```

Expected: **14 passed**.

Hydra compose check (use absolute config dir):

```bash
python -c "
from pathlib import Path
from hydra import compose, initialize_config_dir
from src.omnifed.hybrid.hybrid_aggregate_config import hybrid_aggregate_payload_from_cfg
conf_dir = str(Path('.').resolve() / 'conf')
with initialize_config_dir(version_base=None, config_dir=conf_dir):
    cfg = compose(
        config_name='test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd',
        overrides=['engine.hybrid.aggregate_payload=gradients'],
    )
print(hybrid_aggregate_payload_from_cfg(cfg))
"
# → gradients
```

On Frontier, also run `tests/test_hybrid_global_grpc_compression.py` after regenerating proto with `protoc` in `pytorch_rocm`.

---

## Frontier submit examples

### Parameter mode (default — existing scripts)

```bash
QSGD_BIT_WIDTH=4 ./test_scripts/frontier_hybrid_cifar10_resnet18_7_qsgd.sh slurm.time=02:00:00
```

### Gradient mode + QSGD (after rsync)

```bash
./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=5 \
  engine.hybrid.aggregate_payload=gradients \
  engine.hybrid.global_compression.enabled=true \
  engine.hybrid.global_compression.scheme=qsgd \
  engine.hybrid.global_compression.bit_width=4 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root="${CIFAR_ROOT}" \
  datamodule.eval.dataset.root="${CIFAR_ROOT}" \
  slurm.account=gen150 \
  slurm.partition=batch \
  slurm.time=02:00:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null \
  slurm.exclusive=true
```

### Post-run checks

1. **Slurm:** `COMPLETED`, exit `0:0`
2. **Config:** `grep aggregate_payload outputs/.../.hydra/config.yaml` → `gradients`
3. **Logs:** `Aggregate payload: gradients`, `Compression: True/False`
4. **CSV:** 5 rows in `hybrid_per_round_summary.csv`; finite `eval_loss_avg`
5. **Compare:** grad + QSGD vs dense param baseline (4918094) and grad dense vs param dense

---

## Log lines to expect

**Trainer rank (startup):**

```text
[hybrid] rank=1/7 ... aggregate_payload='gradients'
```

**gRPC server rank:**

```text
[hybrid] rank=0 gRPC server aggregate_payload='gradients'
Compression: True, Aggregate payload: gradients, Updates' Accumulation: True
Client client_1 initialized (QSGD), connecting to ...
```

**Gradient sync (leaders):**

```text
scaled N/N grads | scale_factor=...
[hybrid] grad_apply ...
```

---

## Progress log

| Date | Phase | Status |
|------|-------|--------|
| 2026-06 | 0 — Config knob | Done |
| 2026-06 | 1 — gRPC `communicate_params` | Done |
| 2026-06 | 2 — Local MPI grad all-reduce | Done |
| 2026-06 | 3 — Deferred optimizer + checkpoint payload | Done |
| TBD | Frontier grad dense + grad QSGD matrix | Pending rsync / submit |

---

## Out of scope (v1)

- Mixed payload (params local + grads global)
- Gradient aggregation on local MPI with compression (local hop stays dense)
- Flora / centralized `communicator/` path changes
- Error feedback for QSGD on parameters or gradients
