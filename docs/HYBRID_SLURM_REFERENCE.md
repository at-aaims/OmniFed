# Hybrid communication + Slurm + OmniFed — reference

This document records **what we set out to do**, **the phased steps**, **what is implemented in the codebase**, **what was validated on Frontier**, and optional **experiment follow-ups**. Keep it next to `conf_hybrid/` and `src/omnifed/hybrid/`.

For **Figure 2‑style user knobs, schema sketch, and Phases A–F backlog** (**requirements Phase A**), see **`docs/HYBRID_USER_KNOBS_AND_ROADMAP.md`** (distinct from §2 historical “Phase A smoke” below). For **Slurm `nodes`/`ntasks` vs hybrid `world_size`**, see §**4.3** below (**Phase D**).

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
| **7** | **`slurm_worker` hybrid branch:** load hybrid topology; per-facility Torch MPI; gRPC by role; **`install_hybrid_slurm_sync`** for FedAvg global step without invalid torch scalar reduce across gRPC; **`round_exec`** loop; results under `engine/node_results/`. | **Done + validated on Frontier** (May 2026, job **4625686**; see §3.1) | `src/omnifed/slurm_worker.py`, `src/omnifed/hybrid/slurm_hybrid_runner.py`, `hybrid_slurm_sync.py`, `torch_mpi_adapter.py`, `grpc_leader_comm.py`, `comm_bridge.py`, `slurm_hostlist.py` |
| **8** | **Hardening** (Flora **`id==0`** server contract; **`hybrid_rank_to_centralized_node_index`**; **`leader_done`** RPC shutdown vs **`sleep`**). | **Done** | `topology_roles.py`, `slurm_hybrid_runner.py`, `grpc_communicator.py` (comment); **`conf/base.yaml`** `server_shutdown`, `leader_done_poll_sec`. |
| **9** | **Docs / examples** (README Hybrid Slurm; YAML header; `main.sh` comments). | **Done** | `README.md`, `conf/test_hybrid_engine_contract.yaml`, `main.sh` |

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

### Phase B Step 7 — Validated on OLCF Frontier (May 2026)

| Job ID | Result | Notes |
|--------|--------|--------|
| **4625007** | `FAILED` (MNIST download) | Compute nodes could not reach the public internet; torchvision tried to download **`train-images-idx3-ubyte.gz`** and timed out. Only **`node_000_results.json`** appeared (RPC server stub). **Fix:** pre-stage MNIST on Lustre and pass **`datamodule.*.dataset.download=false`** plus **`root=.../torchvision-mnist`** (same pattern as classic `test_fedavg_centralized_torchdist` on Frontier). |
| **4625686** | **`COMPLETED` / `0:0`** | **`[run_hybrid_training]`** lines for ranks **0–6**; workers on **`cuda:0`** with **`fac1` / `fac2`**; **`node_000`** (JSON stub) + **`node_001`–`node_006`** (JSON + PKL). **`slurm-4625686.err`:** non-fatal Gloo **`errno: 97`** warnings; optional **`UserWarning`** from **`algorithm/base.py`** (`global_agg` / `local_bcast` buffer tracking)—job still succeeded. |

**Repository on GitHub fork:** **[dshruti20/OmniFed](https://github.com/dshruti20/OmniFed)** (renamed from `OmniFed_VT`); feature branch **`feature/hybridcommu-slurm-engine`** carries this work.

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

Either preset is valid (**§4.3**); pick one and **`grep`** it:

```bash
# File-preset sibling
grep -A2 'communication_mode:' conf/test_hybrid_engine_contract.yaml
grep topology_config conf/test_hybrid_engine_contract.yaml

# Or layout-first (Phase C) — no topology_config line
grep -A2 'communication_mode:' conf/test_hybrid_layout_fedavg.yaml
grep -q 'layout:' conf/test_hybrid_layout_fedavg.yaml && grep 'num_clients:' conf/test_hybrid_layout_fedavg.yaml
```

Expect **`hybrid`**, **`world_size` 7 semantics** (**`topology_config`** → **`built_symmetric_2x3`** *or* inline **`layout`: 2 facilities × 3 MPI + **`dedicated_rpc_server`**), and **`topology.num_clients: 6`**.

**4 — Submit Engine (canonical 7 nodes × 1 task)**

On Frontier, **always** disable MNIST download and point **`root`** at scratch data already visible on compute nodes (see job **4625007** if you omit this):

```bash
./main.sh --config-name test_hybrid_engine_contract \
  overwrite=true \
  engine.mode=slurm \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
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

(Adjust **`scratch`** path and account for your project.)

Save **`Submitted batch job <JOBID>`**. Omitting **`datamodule.*` overrides** is only for **login or internet-capable** runs; **Frontier compute nodes** usually require the block above.

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
OUT=/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT/outputs/<DATE>/<CONFIG_NAME>
grep -E '\[hybrid\]|\[run_hybrid_training\]|run_hybrid|communication_mode' "$OUT/slurm-<JOBID>.out" | head -40
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
- **`conf/test_hybrid_layout_fedavg.yaml`** — same hybrid lattice via **`engine.hybrid.layout`** only (**Phase C**; **§4.3** table).

### Rank-0 gRPC server lifetime (tunable)

- **`conf/base.yaml`** → **`engine.hybrid`**:
  - **`server_run_extra_sec`**, **`server_sec_per_round`**: ceiling for how long the RPC rank waits (**used as max wall time**, especially if **`leader_done`** waits).
  - **`server_shutdown`**: **`leader_done`** (write **`engine/hybrid_grpc_leader_done/rank_<leader>.done`** after training, default) or **`sleep`** (legacy fixed nap).
  - **`leader_done_poll_sec`**: polling interval while waiting for markers.

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

### 4.1 Training loop & sync cadence (FedAvg hybrid)

Detailed walkthrough (**minibatches, epochs, `local_agg` → leader gRPC → `local_bcast`**, vs **`algorithm.schedules.aggregation`** / **`round_end`**) lives in **`docs/HYBRID_TRAINING_AND_SYNC.md`**. In short:

- One **`BaseAlgorithm.__sync()`** invocation (when **`schedules.aggregation`** says so — typically **`round_end`**) executes **facility reduce → gRPC merge at facility leaders → facility broadcast** **back-to-back**; **non-leader** ranks **skip** the gRPC block but still do **facility** reduce + **receive** broadcast.
- **Frequency** is **not** a single Flora **`comm_freq`** in OmniFed; it comes from **`conf/algorithm/schedules/aggregation/*.yaml`**, plus **`global_rounds`** and **`algorithm.max_epochs_per_round`**.
- **Output artifacts** (**`Node0.*/`**, **`engine/node_results/`**, CSV **`sync/`** timing keys): **`docs/README_HYDRA_RUN_OUTPUTS.md`**.

### 4.2 Runtime hybrid topology (`engine.hybrid.layout`)

Instead of a **`conf_hybrid/topology/*.yaml`** preset, **`conf/base.yaml`** supports **`engine.hybrid.layout`** with keyword args matching **`build_hybrid_topology`** (**`num_facilities`**, **`mpi_ranks_per_facility`**, **`dedicated_rpc_server`**, **`rpc_port`**, etc.). **`Slurm --ntasks`** and **`run_hybrid_training`** use **`layout`** when both **`layout`** and **`topology_config`** are set (**`layout` wins**); otherwise **`topology_config`** (e.g. **`built_symmetric_2x3.yaml`**) behaves as before. If **both** are set they must imply the **same** **`world_size`** (otherwise Engine validation fails — **`validate_hybrid_slurm_topology_alignment`**). Optional **`engine.hybrid.training`** is merged onto the hybrid cfg for parity with **`conf_hybrid`** YAML.

### 4.3 Slurm allocation ↔ hybrid **`world_size`** (roadmap **Phase D**)

This section is the **user story for `slurm.*` knobs** alongside **`engine.hybrid.layout`** or **`topology_config`**. Implementation: **`resolve_slurm_ntasks`** (**`engine_communication.py`**) → **`SlurmConfig.ntasks`** → **`#SBATCH --ntasks=W`** (**`slurm_launcher.py`**).

**Rules (today’s hybrid Slurm milestone):**

1. **`W = hybrid world_size`** (from **`layout`** or **`conf_hybrid/topology/<topology_config>`** — **§4.2**, Phase **B** alignment with **`topology.num_clients + 1`**).
2. **One Slurm MPI task ↔ one hybrid global rank.** Inside the job, **`SLURM_NTASKS`** must equal **`W`** (**Phase B** validates on workers).
3. **Capacity:** Slurm expects enough **slots**: in practice **`slurm.nodes × slurm.ntasks_per_node ≥ W`** (Frontier schedulers reserve by node). The Engine does **not** magic extra nodes beyond one step: **`slurm.nodes`** is replaced by **`max(slurm.nodes, ceil(W / ntasks_per_node))`** before submit so under-sized **`nodes`** are bumped **upward** (**`engine.py`**). If **`nodes`** was already sufficient, nothing changes.
4. **Simplest Frontier pattern:** **`ntasks_per_node=1`** and **`nodes=W`** (**7 × 7** for the symmetric 2×3 + RPC presets). Packed nodes (**fewer hosts, **`ntasks_per_node>1`**) remain **experimental** until **Phase E** documents **`LOCAL_RANK`** / GPU binding rigorously (**§5.4** “Alternative” smoke only).

**Checklist before submit**

| Question | OK if |
|---------|-------|
| What is **`W`**? | Login-node log **`[Engine] communication_mode=hybrid: Slurm --ntasks=W …`** |
| Enough nodes? **`nodes`** | **`ceil(W / slurm.ntasks_per_node)`** (Engine may bump **`nodes`**; watch for **`[Engine] slurm.nodes raised …`** in the log (**`engine.py`**)) |
| Does **`topology.num_clients`** match? | **`num_clients == W − 1`** |

**Concrete presets (**`W=7`**)** — interchangeable Slurm/datamodule block; only **`--config-name`** differs (**§5.4**):

| Hydra **`--config-name`** | Hybrid lattice source |
|---------------------------|------------------------|
| **`test_hybrid_engine_contract`** | **`engine.hybrid.topology_config`** → **`built_symmetric_2x3.yaml`** |
| **`test_hybrid_layout_fedavg`** | **`engine.hybrid.layout`** only (**Phase C**) |

Both produce the same **`#SBATCH --ntasks=7`** and the same **`run_hybrid_training`** roles; **`outputs/<date>/<config_name>/`** differs by preset name only.

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

### 5.4 Phase B — Engine + hybrid (Phase D: **§4.3** allocation story)

Canonical **7 tasks, 7 nodes, 1 GPU per node** (**`built_symmetric_2x3`** lattice ↔ **`engine.hybrid.layout`** in **`test_hybrid_layout_fedavg`**). **Frontier:** include **offline MNIST** overrides (same idea as classic FedAvg on compute nodes).

**File‑preset sibling** (`topology_config:` → **`built_symmetric_2x3.yaml`**):

```bash
cd "$OMNIFED_REPO"
./main.sh --config-name test_hybrid_engine_contract \
  overwrite=true \
  engine.mode=slurm \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
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

**Layout‑first preset** (same **`--ntasks=7`**; Hydra **`outputs/`** subdirectory uses **`test_hybrid_layout_fedavg`**): swap **`test_hybrid_layout_fedavg`** as **`--config-name`** with the identical Slurm/datamodule block.

**Alternative** (fewer nodes, more tasks per node): e.g. `slurm.nodes=1`, `slurm.ntasks_per_node=8` still yields **7 tasks** if Engine sets `#SBATCH --ntasks=7`—verify in the generated log:  
`[Engine] communication_mode=hybrid: Slurm --ntasks=7 ...`

**After submit:** inspect **`outputs/<date>/<config_name>/slurm-<jobid>.out`** and `.err`, and **`engine/node_results/node_*_results.{json,pkl}`**.

**Full Step-by-step verification (pre-checks, `grep`, `sacct`, result listing):** see **§3 — “Phase B Step 7 — Verify on Frontier (commands)”**.

### 5.5 Centralized classic baseline (your successful pattern)

Example (15 clients, 2×8 tasks): use `test_fedavg_centralized_torchdist` with `topology.num_clients=15`, `nccl`, and your scratch dataset paths (`download=false` when data is pre-staged).

---

## 6. Phase B Steps 8–9 (implemented + follow-ups)

### Step 8 — Hardening (closed in-repo)

| Item | What we did |
|------|--------------|
| Flora gRPC PS `id == 0` vs `rpc.server_rank` | Flora’s **`GrpcCommunicator`** only builds the daemon server branch when **`id == 0`** (communicator role, not **`SLURM_PROCID`**). Hybrid still runs that process on **`topology.rpc.server_rank`**; constructor **`id`** stays **`0`**. Comments in **`grpc_communicator.py`** and **`_run_grpc_server_only`** document this contract. Changing **`rpc.server_rank` ≠ 0** does **not** require **`id≠0`** at the Flora layer until Flora is refactored. |
| **Client / `topology.overrides` mapping** | **`hybrid_rank_to_centralized_node_index`** in **`topology_roles.py`**: **`rpc_server_rank` → centralized index 0**; training ranks (**all others**) map to **`1 … num_clients`** in ascending hybrid rank order. **`run_hybrid_training`** selects **`node_cfg = node_cfgs[mapped]`** instead of **`node_cfgs[SLURM_PROCID]`**. Unit tests cover **`rpc.server_rank`** 0 and 3. Larger topologies inherit the same rule. |
| **RPC shutdown** | Default **`engine.hybrid.server_shutdown: leader_done`**: leaders write **`engine/hybrid_grpc_leader_done/rank_<r>.done`** after **`round_exec`** finishes; daemon rank clears stale markers at start then polls (**`leader_done_poll_sec`**) up to **`server_run_extra_sec + rounds × server_sec_per_round`** and calls **`grpc_shutdown`**. Fallback: **`sleep`** restores previous fixed-nap semantics. Configure in **`conf/base.yaml`**. |

### Step 9 — Documentation (done)

- **`README.md`**: Hybrid Slurm subsection (`engine.communication_mode=hybrid`, Frontier MNIST, example `main.sh` line).
- **`conf/test_hybrid_engine_contract.yaml`**: Updated header / example (no stub wording).
- **`main.sh`**: Commented hybrid one-liner pattern.

### Optional follow-ups (not required)

- Stress-test **`hybrid_rank_to_centralized_node_index`** on very large asymmetric layouts (`world_size ≫ 11`).
- If **`global_agg` / `local_bcast` “buffers unchanged”** warnings correlate with poor metrics, trace **`algorithm/base.py`** tracking vs hybrid sync.
- Add **`test_hybrid_engine_contract_frontier.yaml`** with scratch **`dataset.root`** + **`download=false`** baked in.


---

## 7. Frontier validation pointer (**Phase D** operations)

**Step 7 cluster proof** is the numbered checklist in **§3 — “Phase B Step 7 — Verify on Frontier (commands)”**. The short recipe is also in **§5.4**. **Allocation rules** (**`--ntasks`**, **`nodes`**, **`ntasks_per_node`**) are summarized in **§4.3** (**Phase D**). **Recorded success:** **§3.1** (job **4625686**).

**Acceptance:** `sacct` **0:0**; **`[hybrid]`** markers in **`slurm-<jobid>.out`**; **`node_000`–`node_006`** under **`engine/node_results/`**; **`download=false`** + scratch **`root`** on Frontier (job **4625007** showed public MNIST timeouts).

**Phase D cluster parity:** after **Phase C**, run **§7b** once with **`--config-name test_hybrid_layout_fedavg`** (same Slurm block as **`test_hybrid_engine_contract`**) so **layout-first YAML** is exercised on OLCF — expect the same **`--ntasks`** line and seven **`node_*`** files; only the Hydra output folder name changes.

If that passes, treat Step 7 as **validated on Frontier**.

### 7b — Re-check Steps **8–9** on Frontier (after `rsync`)

Do this when you’ve **pushed or rsync’d** the repo that includes **`leader_done`** shutdown, **`hybrid_rank_to_centralized_node_index`**, and doc/YAML updates.

1. **Sync** your local tree to Frontier (same excludes as before — at least avoid clobbering **`outputs/`** if you want old logs).
2. **Run** the **same** hybrid job that worked before — **§5.4** (**`test_hybrid_engine_contract`** *or*, for Phase C parity, **`--config-name test_hybrid_layout_fedavg`** with the same offline MNIST + **`slurm.nodes=7`**, **`ntasks_per_node=1`** block).
3. **Expect** `sacct` **`COMPLETED`** / **`0:0`** and seven **`engine/node_results/`** files as in Step 7.
4. **Step 8 extras (quick):** In **`slurm-*.out`** for the **RPC rank**, look for **`shutdown_mode='leader_done'`** and either **`all gRPC leader markers present`** or **`leader-done wait timed out`** (timeout still means the run can succeed if wall cap was enough — prefer the **markers** line for a clean validation). After the job, **`ls engine/hybrid_grpc_leader_done/`** should show **`.done` files for each gRPC leader rank** (not for every training rank).
5. **Step 9:** Confirm **`README`** / **`test_hybrid_engine_contract`** header on Frontier match your laptop (optional `diff` or `head`).

---

## **Next steps (experiment tuning)**

Hygiene from **Steps 8–9** above is landed. Practical next explorations:

1. **FedAvg aggregation schedule / “communication frequency”** — tighten via **`algorithm.schedules.aggregation`** (e.g. trigger every **N** local steps or **`batch_end`**) consistent with OmniFed YAML; correlate with Flora gRPC **round-number** semantics and MNIST staleness metrics.
2. **Two global rounds, seven local steps analogy** — if you literal-mean **`global_rounds=2`** and additional gRPC-visible rounds, bump **`global_rounds`** plus tune **`schedules`** so **`round_exec`** + hybrid **`GrpcLeaderCommunicator`** align with what you plot as “GRPC rounds.”
3. **Optional Frontier defaults** — new YAML config with MNIST **`root`**/`download=false` wired for OLCF scratch.

---

## 8. Revision history

| Date | Note |
|------|------|
| 2026-05 | Document created: aligns Phase A/B steps with repo; notes Frontier jobs 4576044 / 4576474 / 4576606 / contract run 4576327 style; remaining Steps 8–9. |
| 2026-05 | Frontier Step 7 validated: job **4625686** (`COMPLETED`); job **4625007** MNIST download failure documented; **§5.4** / Step 4 use offline MNIST; fork URL **[dshruti20/OmniFed](https://github.com/dshruti20/OmniFed)**; **§7** adds explicit **Next step** (Step 8 → 9). |
| 2026-05 | **Steps 8–9** landed: Flora **`id==0`** doc + **`leader_done`** shutdown (default); **`hybrid_rank_to_centralized_node_index`** + tests; README / YAML / **`main.sh`**; **§6** rewritten from “remaining” → “implemented + follow-ups.” |
| 2026-05 | **§7b** added: minimal Frontier checklist to re-verify Steps **8–9** after `git pull` / **`rsync`**. |
| 2026-05 | **§4.1** + **`docs/HYBRID_TRAINING_AND_SYNC.md`**: FedAvg hybrid training/sync ordering and schedule vs **`comm_freq`**. |
| 2026-05 | **`docs/README_HYDRA_RUN_OUTPUTS.md`**: catalog of **`outputs/…`** artifacts + **`sync/*_time`** metrics. §4.1 cross-link. |
| 2026-05 | **`engine.hybrid.layout`**: runtime **`build_hybrid_topology`** (**§4.2**) supersedes **`topology_config`** when set (both must imply same **`world_size`** — **`validate_hybrid_slurm_topology_alignment`**). |
| 2026-05 | **`docs/HYBRID_USER_KNOBS_AND_ROADMAP.md`**: Phase A requirements + roadmap (**user knobs / schema sketch / Phases A–F**); intro links here vs historical §2 Phase A smoke. |
| 2026-05 | **`conf/test_hybrid_layout_fedavg`**: Phase C layout-first preset + **`§5.4`** / **`§7b`** pointers; **`tests/test_hybrid_phase_c_preset`**. |
| 2026-05 | **Phase D:** **`HYBRID_SLURM_REFERENCE`** §**4.3** (Slurm **`nodes`/`ntasks`** ↔ hybrid **`W`**); §**3** pre-checks + §**7** parity note; **`[Engine] slurm.nodes raised …`** (**`engine.py`**). |

---

*End of reference.*
