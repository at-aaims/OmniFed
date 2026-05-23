# Running OmniFed experiments on OLCF Frontier (engine → hybrid pipeline)

This guide is written for **the principal collaborator** who will **reproduce and present** OmniFed experiments to a PI. It follows the **actual development path**: validate the **centralized SLURM + Engine** stack, prove **hybrid communications** incrementally, then run **full Engine hybrid FedAvg**, and optionally the **Llama‑150M + C4** hybrid extension.

Deep references live under **`./archive/hybrid-engine-pipeline/`** (especially `HYBRID_SLURM_REFERENCE.md`, `HYBRID_USER_KNOBS_AND_ROADMAP.md`, `HYBRID_TRAINING_AND_SYNC.md`, `README_HYDRA_RUN_OUTPUTS.md`). This file stays **linear and command-oriented**.

---

## Conventions

| Symbol | Meaning |
|--------|---------|
| **`YOUR_USER`** | Frontier username |
| **`YOUR_PROJECT`** | Slurm charge account (example: `gen150`) |
| **`OMNIFED_REPO`** | Repo root on Lustre (example: `/lustre/orion/gen150/scratch/YOUR_USER/OmniFed_VT`) |

**Critical OLCF constraint:** Frontier **compute nodes cannot download** public datasets. Pre-stage MNIST on Lustre and pass `datamodule.*.dataset.download=false` plus `root=…`. Same idea for LM data (C4 tokenizer/weights cached on scratch — see Llama+C4 roadmap).

---

## 0. One-time: sync codebase and shell environment

### 0.a Sync from laptop to Frontier (optional)

```bash
rsync -avz \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  /path/to/local/OmniFed_VT/ \
  YOUR_USER@loginNN.frontier.olcf.ornl.gov:${OMNIFED_REPO}/
```

Or work directly from a clone on Lustre and `git pull`.

### 0.b Login node environment (every session)

```bash
module load miniforge3/23.11.0-0
conda activate pytorch_rocm   # or site-approved ROCm PyTorch env

cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"
export PYEXE="${CONDA_PREFIX}/bin/python"
```

Ensure `main.sh` stays executable (`chmod +x main.sh`). **`main.sh` runs `grpc_tools.protoc`** on `communicator/grpc.proto` before **`main.py`** — required for Flora/gRPC builds.

---

## 1. Baseline — Engine + **centralized** TorchDistrib SLURM (not hybrid)

**Purpose:** Prove the **classic** path: Hydra → Engine freezes config → **`sbatch`** → each task runs **`slurm_worker`** with **TorchDistrib** FedAvg (`engine.communication_mode` **not** `hybrid`).

**Hydra preset:** `conf/test_fedavg_centralized_torchdist.yaml` (FedAvg + simple CNN + MNIST pattern used across the stack).

Typical Frontier overrides (adapt client count / topology for larger demos):

```bash
./main.sh --config-name test_fedavg_centralized_torchdist \
  overwrite=true \
  engine.mode=slurm \
  topology.num_clients=<N_CLIENTS> \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  slurm.account=YOUR_PROJECT \
  slurm.partition=batch \
  slurm.time=01:00:00 \
  slurm.nodes=<MATCH_ALLOCATION> \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

**Success:** `sacct` exit `0:0`; per-rank outputs under Hydra **`outputs/<date>/<config>/`**. For **`CentralizedTopology`**, Slurm must allocate **one task per logical participant** (**typically `topology.num_clients + 1`** ranks — server slot plus trainers). Override **`topology.num_clients`** and **`slurm.nodes` / `ntasks_per_node`** together so **`nodes × ntasks_per_node`** matches that world size (**not** the hybrid lattice — classic TorchDistrib path; see repo **`README.md`**).

This step establishes that **Frontier SLURM + MNIST staging + Engine submission** works before layering hybrid.

---

## 2. Intermediate — Phase A hybrid **communication smoke** (no full Engine loop)

**Purpose:** Isolate **facility Torch MPI collectives + gRPC round** without FedAvg **`round_exec`** or datasets. Validates rank roles, ROCm/NCCL path, hostlist patching pattern.

From repo root:

```bash
cd "${OMNIFED_REPO}"
export PYEXE PYTHONPATH OMNIFED_REPO
# gloo-friendly default in batch script — for GPU/NCCL:
export HYBRID_SMOKE_BACKEND=nccl
sbatch test_scripts/slurm_frontier/hybrid_comm_smoke.slurm
```

Inspect job output; confirm clean exit record (historical Frontier jobs cited in **`archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §2).

---

## 3. Full Engine hybrid — **`test_hybrid_engine_contract`** (file topology preset)

**Purpose:** **`engine.communication_mode: hybrid`** with **`topology_config` → conf_hybrid `built_symmetric_2x3`** (**world_size = 7**, **six trainers + dedicated RPC**). FedAvg **`round_exec`** on each rank; **`run_hybrid_training`** in **`slurm_hybrid_runner.py`**.

**Invariant:** **`topology.num_clients + 1 = hybrid_world_size`** (here **6 + 1 = 7**). Slurm must launch **seven tasks**: simplest pattern **`nodes=7`**, **`ntasks_per_node=1`**.

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

**Verification (PI-facing checklist):**

1. Login log line: **`[Engine] communication_mode=hybrid: Slurm --ntasks=7`** (or your resolved `W`).
2. Generated script: **`#SBATCH --ntasks=7`**.
3. `sacct -j <JOBID> …` → **`COMPLETED`**, **`0:0`**.
4. `outputs/<date>/test_hybrid_engine_contract/engine/node_results/` → **`node_000` … `node_006`** (JSON; RPC stub on `node_000`).
5. Optional: `hybrid_per_round_summary.csv` at run root (hybrid summary writer).

**Lessons already learned:** Job **4625007** failed when MNIST tried to download on compute; **4625686** succeeded with offline MNIST — see **`archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §3.1.

---

## 4. Layout-first hybrid preset — **`test_hybrid_layout_fedavg`** (Phase C parity)

**Purpose:** Same **7-rank lattice** as §3, but topology is declared inline via **`engine.hybrid.layout`** (no separate `conf_hybrid` topology file). Proves **Figure‑2-style** YAML ergonomics and shared validators (**`validate_hybrid_slurm_topology_alignment`**, **`hybrid_world_size_from_cfg`**).

Use the **same Slurm + MNIST block** as §3; only change `--config-name`:

```bash
./main.sh --config-name test_hybrid_layout_fedavg \
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

**Acceptance:** Identical **`--ntasks`** semantics as §3; Hydra output folder name reflects **`test_hybrid_layout_fedavg`**.

**Operational note:** **`archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §4.3 — Engine may **`[Engine] slurm.nodes raised …`** if **`nodes × ntasks_per_node`** was below **`world_size`**; **`#SBATCH --ntasks=W`** stays pinned to hybrid **`W`**.

---

## 5. Larger-scale hybrid (MNIST narrative for PI — e.g. 129 ranks)

Validated pattern on this branch: **`topology.num_clients=128`**, **`2 × 64` facilities + RPC**, **`slurm.nodes=129`**, **`ntasks_per_node=1`**. Override **`engine.hybrid.layout.mpi_ranks_per_facility`**, **`training.dataset_total_clients`**, and **`slurm`** consistently.

**Golden rule:** **`SLURM_NTASKS == hybrid_world_size`** and **`topology.num_clients == world_size − 1`**.

Consult **`archive/hybrid-engine-pipeline/HYBRID_USER_KNOBS_AND_ROADMAP.md`** §**2–3** before changing layout integers.

---

## 6. Optional extension — Llama‑150M + C4 (**hybrid LM**)

**Purpose:** Same hybrid Engine path; swap **algorithm → FedAvgLLM**, **model → HF Llama**, **datamodule → C4 `load_from_disk`** with federated shards keyed by **`OMNIFED_FEDERATED_CLIENT_INDEX`**.

Offline preparation (**login node only** — Hub access): C4 subset **`save_to_disk`**, Mistral tokenizer cache, Llama **`snapshot_download`**, plus exports **`OMNIFED_C4_DISK`**, **`OMNIFED_TOKENIZER_DIR`**, **`OMNIFED_LLAMA_WEIGHTS`**.

**Authoritative command sequence:** **`archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`** — section **Frontier procedure (login → data → submit)** and anchor **`hybrid-lm-job-7-nodes-gen150`**.

Abbreviated submit (after env set):

```bash
./main.sh --config-name test_hybrid_layout_fedavg_llama150m \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=2 \
  slurm.account=YOUR_PROJECT \
  slurm.partition=batch \
  slurm.time=01:30:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

**Important:** **`datamodule.num_federated_clients`** must equal **`topology.num_clients`** **as literals** — do **not** interpolate **`${topology.num_clients}`** in **`c4_lm_federated_disk.yaml`** (`engine_frozen.json` freeze can fail — see roadmap §J).

---

## 7. Reading results for PI slides

| Artifact | Meaning |
|---------|---------|
| **`outputs/<date>/<config>/slurm-<JOBID>.out`** | Merged stdout — `[hybrid]`, **`run_hybrid_training`**, RPC shutdown |
| **`engine/node_results/node_*_results.json`** | Per-rank rollup — **`sync/local_agg_time`**, **`sync/global_agg_time`**, **`sync/local_bcast_time`** |
| **`hybrid_per_round_summary.csv`** | Per-round **`gRPC_F*`** columns (leader timings — often asymmetric; see **`README_HYDRA_RUN_OUTPUTS.md`**) |
| **`Node0.*/metrics_*.csv`** | Fine-grained training/eval/sync metrics |

FedAvg synchronization order (**facility reduce → leader gRPC → facility broadcast**) is documented in **`archive/hybrid-engine-pipeline/HYBRID_TRAINING_AND_SYNC.md`**.

---

## 8. Suggested PI narrative (one paragraph)

We first validated **centralized federated learning on Frontier SLURM** through the OmniFed Engine. We then exercised **standalone hybrid communication** (Torch MPI per institution + Flora gRPC for global aggregation), integrated that path into **`slurm_worker`** via **`run_hybrid_training`**, and reproduced **seven-node FedAvg** with offline MNIST using both **file-based** and **layout-first** Hydra presets. Scaling follows the same invariants (**`num_clients + 1 = world_size`**). An optional **Llama + C4** configuration reuses the hybrid engine with **disk-only** data and **FedAvgLLM**.

---

## See also

- `archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md` — numbered Frontier verification, job history, §7b re-checks  
- `archive/hybrid-engine-pipeline/README_TEST_HYBRID_ENGINE_CONTRACT.md` — preset touch map  
- `archive/hybrid-engine-pipeline/CHAT_HANDOFF_HYBRID.md` — roadmap phases A–D vs E/F  
- `archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md` — LM data prep + Slurm command block  
