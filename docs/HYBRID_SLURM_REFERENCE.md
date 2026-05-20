# Hybrid communication + Slurm + OmniFed — reference

This document records **what we set out to do**, **the phased steps**, **what is implemented in the codebase**, **what you validated on Frontier**, and **what remains** (Phase B steps 8–9). Keep it next to `conf_hybrid/` and the hybrid modules under `src/omnifed/hybrid/`.

---

## 1. Overall aim

**Goal:** Run federated learning on OLCF Frontier (and similar Slurm sites) where:

- **Within each facility:** PyTorch collective communication uses **Torch MPI** (`TorchMPICommunicator` / Flora), one process group per facility.
- **Across facilities:** A **gRPC** “leader” path aggregates models (FedAvg-style global step).
- **Orchestration:** `engine.mode=slurm` uses the **Engine** to freeze config, submit **Slurm**, and launch **`slurm_worker`** on each task; the **same algorithm loop** (`algorithm.round_exec`) drives training as in the classic single-world TorchDist path.

**Non-goals in early phases:** Full Engine + datasets in Phase A (smoke only).

---

## 2. Phase A — Slurm hybrid smoke (no full Engine)

| Step | Description | Status | Code / artifacts |
|------|-------------|--------|-------------------|
| **1** | Topology builder: inputs `dedicated_rpc_server`, `num_facilities`, `mpi_ranks_per_facility` (int or list); output resolved dict (`world_size`, `rpc`, facilities, members, leaders, `client_ranks`). Unit test vs hand-written 7-rank / 2×3 layout. | Done | `src/omnifed/hybrid/topology_builder.py`, tests |
| **2** | Hydra layout under `conf_hybrid` so default topologies do not need hand-written rank lists. | Done | `conf_hybrid/`, e.g. `conf_hybrid/topology/built_symmetric_2x3.yaml` |
| **3** | Minimal hybrid smoke: read resolved cfg + `SLURM_PROCID` as global rank; device/backend; facility Torch MPI; one collective; one gRPC round per topology roles; exit 0. No dataset. | Done | `src/omnifed/hybrid/hybrid_comm_smoke.py` |
| **4** | Slurm batch script: 7 nodes, 1 task/node, 1 GPU/task; hostnames / env from `scontrol show hostnames`; `srun` smoke entrypoint; document env vars. | Done | `test_scripts/slurm_frontier/hybrid_comm_smoke.slurm`, `run_hybrid_smoke_one_task.sh` |
| **5** | Run on Frontier; confirm clean exit and logs (Torch groups + gRPC). | Done | e.g. job **4576044**; later **4576474** (gloo), **4576606** (NCCL) |

### Phase A — NCCL / RCCL readiness (smoke only)

- **`hybrid_comm_smoke.py`:** `gloo` → CPU; `nccl` → `LOCAL_RANK` / `SLURM_LOCALID`, `torch.cuda.set_device`, model on GPU before `TorchMPICommunicator` where applicable. gRPC **server** rank keeps model on **CPU**.
- **gRPC client:** averaged weights respect parameter device (`grpc_client.py` / related Flora path) so leaders stay on GPU after a gRPC round.
- **Slurm wrapper:** `ROCR_VISIBLE_DEVICES` → `HIP_VISIBLE_DEVICES` in `run_hybrid_smoke_one_task.sh`.

### Phase A — Frontier note: Gloo `errno 97` (Address family not supported by protocol)

Often **non-fatal**: c10d/Gloo probing IPv6 vs IPv4 or hostname resolution. If the job **completes** and collectives succeed, treat as noise unless you need quieter logs. Mitigations (try one at a time): short hostnames from `scontrol show hostnames`, force IPv4 rendezvous addresses, or site-specific NCCL/Gloo env (e.g. `NCCL_SOCKET_FAMILY=AF_INET` for NCCL).

---

## 3. Phase B — Full Engine integration

| Step | Description | Status | Code / notes |
|------|-------------|--------|--------------|
| **6** | **Engine config contract:** `engine.communication_mode`: `classic` \| `hybrid`; `engine.hybrid.topology_config` (YAML name under `conf_hybrid/topology/`). Hybrid only with `engine.mode=slurm`. Slurm `--ntasks` = hybrid `world_size` and must match `len(topology)` (e.g. centralized `num_clients = world_size - 1` when one rank is RPC-only). | Done | `conf/base.yaml`, `src/omnifed/engine_communication.py`, `src/omnifed/engine.py`, `tests/test_engine_communication.py`, `conf/test_hybrid_engine_contract.yaml` |
| **7** | **`slurm_worker` hybrid branch:** load hybrid topology; per-facility Torch MPI; gRPC by role; **`install_hybrid_slurm_sync`** for FedAvg global step without invalid torch scalar reduce across gRPC; **`round_exec`** loop; results under `engine/node_results/`. | Done | `src/omnifed/slurm_worker.py` (hybrid → `run_hybrid_training`), `src/omnifed/hybrid/slurm_hybrid_runner.py`, `hybrid_slurm_sync.py`, `torch_mpi_adapter.py`, `grpc_leader_comm.py`, `comm_bridge.py`, `slurm_hostlist.py` |
| **8** | **Hardening:** single trainer path; **avoid hardcoding** gRPC server `id==0` where topology says otherwise; **datamodule / client id** mapping for **non-contiguous** global ranks. | **Remaining** | See §6 |
| **9** | **Docs / examples:** `main.sh` + README-style recipe for `engine.mode=slurm` + hybrid FedAvg (canonical Frontier invocation). | **Remaining** | `main.sh` today runs `main.py` + protoc; hybrid example overrides should be documented |

### Phase B Step 6 — Frozen config + Slurm

- Engine writes **`engine_frozen.json`** (shared path under Hydra outputs) and submits **`sbatch`** via `SlurmOnlyLauncher` (`src/omnifed/slurm_launcher.py`).
- Inside the allocation, each task runs:  
  `python -m src.omnifed.slurm_worker --cfg-json <path>`.

### Phase B Step 7 — Entry point (implemented)

```text
slurm_worker.py (excerpt)
  if communication_mode(cfg) == "hybrid":
      run_hybrid_training(cfg, hydra_out_dir, ckpt_dir)
      raise SystemExit(0)
```

Trainer path for hybrid is **`src/omnifed/hybrid/slurm_hybrid_runner.py`** (`run_hybrid_training`).

### Phase B Step 7 — What we implemented (summary)

Step 7 replaces the Phase B Step 6 **stub** (hybrid branch exited immediately) with **real hybrid training** on each Slurm task:

| Piece | Role |
|-------|------|
| **Frozen config** | Same as classic: Engine writes **`engine_frozen.json`**; each rank runs **`slurm_worker --cfg-json …`**. |
| **Branch** | `communication_mode(cfg) == "hybrid"` → **`run_hybrid_training`** then `SystemExit(0)` (classic TorchDist path not used). |
| **Topology** | Load **`conf_hybrid/topology/<engine.hybrid.topology_config>`**; apply env overrides; **patch `rpc` / facility `mpi` addresses** from **`SLURM_JOB_NODELIST`** (ordered hostnames). |
| **RPC server rank** | Rank **`topo.rpc.server_rank`** (default 0): CPU model, Flora **`GrpcCommunicator`** daemon, **sleep** (`server_run_extra_sec` + `rounds × server_sec_per_round`), shutdown; writes a small **stub JSON** in `node_results`. |
| **Other ranks** | Per-facility **`TorchMPICommunicator`** + **`TorchMPIAdapter`**; **gRPC client** for **facility leaders** via **`GrpcLeaderCommunicator`**; **`HybridCommBridge`** passes facility sample counts into the gRPC global step. |
| **FedAvg sync** | **`install_hybrid_slurm_sync`**: custom **`__sync_comm`** so **global gRPC aggregation** does not run the invalid **torch scalar SUM** across leaders; **`_aggregate_across_groups`** uses **`comm.aggregate(model, SUM)`** for leaders. |
| **Training loop** | Same as elsewhere: **`algorithm.setup(...)`** then **`for r in range(max_rounds): algorithm.round_exec(r, max_rounds)`**. |
| **Outputs** | **`engine/node_results/node_<SLURM_PROCID>_results.{pkl,json}`** under the Hydra run’s output tree. |

**Tests (local):** e.g. `pytest tests -k "hybrid or engine_communication"` (includes layout / engine contract tests).

### Phase B Step 7 — Verify on Frontier (commands)

Run these **in order** from the repo root on a Frontier login node (adjust paths, account, partition, and `<JOBID>` / `<DATE>`).

**1 — Tree contains Step 7**

```bash
cd /lustre/orion/gen150/scratch/shruti2395/OmniFed_VT
test -f src/omnifed/hybrid/slurm_hybrid_runner.py && grep -q run_hybrid_training src/omnifed/slurm_worker.py && echo OK
```

**2 — Environment**

```bash
module load miniforge3/23.11.0-0
conda activate pytorch_rocm
export PYEXE=/ccs/home/shruti2395/.conda/envs/pytorch_rocm/bin/python
export PYTHONPATH="$PWD"
```

**3 — Config is hybrid + 7-rank topology**

```bash
grep -A2 'communication_mode:' conf/test_hybrid_engine_contract.yaml
grep topology_config conf/test_hybrid_engine_contract.yaml
```

Expect **`hybrid`** and **`built_symmetric_2x3.yaml`** (`world_size` 7; `num_clients` 6 in that example).

**4 — Submit Engine (canonical 7 nodes × 1 task)**

```bash
./main.sh --config-name test_hybrid_engine_contract \
  overwrite=true \
  engine.mode=slurm \
  slurm.account=gen150 \
  slurm.partition=batch \
  slurm.time=00:45:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

Save **`Submitted batch job <JOBID>`**.

**5 — Confirm generated `sbatch` has `--ntasks=7`**

In the login-node log printed by Engine, the block **“Generated sbatch”** must include **`#SBATCH --ntasks=7`**, and you should see:

`[Engine] communication_mode=hybrid: Slurm --ntasks=7 ...`

**6 — Wait for job**

```bash
squeue -u shruti2395
```

**7 — Exit code**

```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
```

Target: **`COMPLETED`** and **`0:0`**.

**8 — Stdout shows hybrid training (not Step 6 stub)**

```bash
OUT=/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT/outputs/<DATE>/test_hybrid_engine_contract
grep -E '\[hybrid\]|run_hybrid|communication_mode' "$OUT/slurm-<JOBID>.out" | head -40
```

**9 — Seven result files (`node_000` … `node_006`)**

```bash
ls -1 "$OUT/engine/node_results/"
```

**10 — Spot-check JSON**

```bash
head -30 "$OUT/engine/node_results/node_001_results.json"
```

**Queue note:** `squeue` may show **`PD (Priority)`** for a long time on shared partitions; that is normal.

### Config example (FedAvg + MNIST + hybrid 7 ranks)

- **`conf/test_hybrid_engine_contract.yaml`** — sets `communication_mode: hybrid`, `topology_config: built_symmetric_2x3.yaml`, `num_clients: 6`, `global_rounds: 1`.  
  **Note:** The file header still mentions an old “stub” exit; Step 7 **replaced** that with real training—treat the header as stale until someone edits the comment.

### Rank-0 gRPC server lifetime (tunable)

- **`conf/base.yaml`** → `engine.hybrid.server_run_extra_sec`, `server_sec_per_round`: rank that runs the **Flora `GrpcCommunicator` daemon** sleeps (`slurm_hybrid_runner._run_grpc_server_only`) so workers can finish rounds. If jobs **hang or disconnect** near the end, increase these or (future work) replace sleep with explicit round synchronization.

---

## 4. Codebase map (quick)

| Topic | Location |
|-------|----------|
| Mode + Slurm task count | `src/omnifed/engine_communication.py` |
| Slurm submit + `setup_lines` / `PYEXE` | `src/omnifed/engine.py`, `src/omnifed/slurm_launcher.py` |
| Classic vs hybrid worker entry | `src/omnifed/slurm_worker.py` |
| Hybrid training loop | `src/omnifed/hybrid/slurm_hybrid_runner.py` |
| FedAvg sync patch (gRPC global step) | `src/omnifed/hybrid/hybrid_slurm_sync.py` |
| Host list → topology addresses | `src/omnifed/hybrid/slurm_hostlist.py` |
| Smoke test module | `src/omnifed/hybrid/hybrid_comm_smoke.py` |
| Frontier smoke batch | `test_scripts/slurm_frontier/hybrid_comm_smoke.slurm` |

---

## 5. Frontier operations cheat sheet

### 5.1 Sync repo

From your laptop (adjust paths user/host):

```bash
rsync -avz \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' --exclude 'outputs' \
  ./OmniFed_VT/ \
  USER@loginNN.frontier.olcf.ornl.gov:/lustre/orion/gen150/scratch/USER/OmniFed_VT/
```

Always run jobs from the **same tree** you synced (NCCL smoke + Step 7 depend on recent `grpc_client` + hybrid runner code).

### 5.2 Environment (login node)

```bash
module load miniforge3/23.11.0-0   # or your site recipe
conda activate pytorch_rocm
export PYEXE=/ccs/home/USER/.conda/envs/pytorch_rocm/bin/python
export PYTHONPATH=/path/to/OmniFed_VT
export OMNIFED_REPO=/path/to/OmniFed_VT
```

For **GPU** training on compute nodes, the **Engine-generated** batch script typically loads `PrgEnv-gnu`, `rocm`, `craype-accel-amd-gfx90a`, sets **`PYEXE`**, **`OMNIFED_DATA_DIR`**, etc. (see `engine.py` `setup_lines`—**edit paths** for your account and scratch).

### 5.3 Phase A smoke (no Engine)

```bash
cd "$OMNIFED_REPO"
export PYEXE=... PYTHONPATH="$PWD" OMNIFED_REPO="$PWD"
# gloo (default in script)
sbatch test_scripts/slurm_frontier/hybrid_comm_smoke.slurm
# NCCL / RCCL
export HYBRID_SMOKE_BACKEND=nccl
sbatch test_scripts/slurm_frontier/hybrid_comm_smoke.slurm
```

### 5.4 Phase B — Engine + hybrid (`test_hybrid_engine_contract`)

Canonical **7 tasks, 7 nodes, 1 GPU per node** (matches `built_symmetric_2x3`):

```bash
cd "$OMNIFED_REPO"
./main.sh --config-name test_hybrid_engine_contract \
  overwrite=true \
  engine.mode=slurm \
  slurm.account=gen150 \
  slurm.partition=batch \
  slurm.time=00:30:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

**Alternative** (fewer nodes, more tasks per node): e.g. `slurm.nodes=1`, `slurm.ntasks_per_node=8` still yields **7 tasks** if Engine sets `#SBATCH --ntasks=7`—verify in the generated log:  
`[Engine] communication_mode=hybrid: Slurm --ntasks=7 ...`

**After submit:** inspect `outputs/<date>/test_hybrid_engine_contract/slurm-<jobid>.out` and `.err`, and under that run’s Hydra output, **`engine/node_results/node_*_results.{json,pkl}`**.

**Full Step-by-step verification (pre-checks, `grep`, `sacct`, result listing):** see **§3 — “Phase B Step 7 — Verify on Frontier (commands)”**.

### 5.5 Centralized classic baseline (your successful pattern)

Example (15 clients, 2×8 tasks): use `test_fedavg_centralized_torchdist` with `topology.num_clients=15`, `nccl`, and your scratch dataset paths (`download=false` when data is pre-staged).

---

## 6. Remaining work (Phase B steps 8–9)

### Step 8 — Hardening (do when logs expose wrong assumptions)

1. **gRPC server identity**  
   Full training path still constructs the Flora daemon with **`id=0`** in `_run_grpc_server_only` (`slurm_hybrid_runner.py`). Topology may use **`rpc.server_rank`** not equal to 0 in other layouts—reconcile **`id`** and **`server_rank`** with Flora’s expectations.

2. **Client / datamodule id mapping**  
   Hybrid **global ranks** need a **consistent map** to FL “client” indices and datamodule overrides (`topology.overrides`) when ranks are **non-contiguous** or RPC-only ranks do not train. Validate with a topology larger than 7 after Step 7 is stable.

3. **RPC server shutdown**  
   Replace or augment **sleep-based** server lifetime with round-aware shutdown if long runs become flaky.

### Step 9 — Documentation and examples

1. Update **`conf/test_hybrid_engine_contract.yaml`** header (remove “stub” wording).  
2. Add a **Frontier hybrid** section to the main **README** (or this doc) with the exact `./main.sh ...` line and prerequisites (`OMNIFED_DATA_DIR`, `PYEXE`).  
3. Optionally extend **`main.sh`** comments with a copy-paste hybrid block (no behavioral change required).

---

## 7. Frontier validation pointer

**Step 7 cluster proof** is the numbered checklist in **§3 — “Phase B Step 7 — Verify on Frontier (commands)”**. The short recipe is also in **§5.4**.

**Acceptance:** `sacct` **0:0**; **`[hybrid]`** lines in **`slurm-<jobid>.out`**; **`node_000`–`node_006`** under **`engine/node_results/`**; no systematic gRPC connection or MPI init failures (if failures occur, tune **`server_run_extra_sec`** / **`server_sec_per_round`** in `conf/base.yaml` or fix host binding).

If that passes, treat Step 7 as **validated on Frontier** and move to **§6 (Steps 8–9)** as needed.

---

## 8. Revision history (human-maintained)

| Date | Note |
|------|------|
| 2026-05 | Document created: aligns Phase A/B steps with repo; notes Frontier jobs 4576044 / 4576474 / 4576606 / contract run 4576327 style; remaining Steps 8–9. |
| 2026-05 | Added §3 Step 7 “what we implemented” table + Frontier verification commands (1–10); §7 points to that checklist. |

---

*End of reference.*
