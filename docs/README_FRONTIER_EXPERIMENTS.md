# Running on OLCF Frontier (Engine + hybrid pipeline)

**Scope of this section:** running the Engine on OLCF Frontier (validation of the SLURM path) and the hybrid communication pipeline (TorchMPI per facility + Flora gRPC across facilities) end-to-end through the Engine, including the Llama-150M + C4 hybrid LM extension.

**Suggested placement** in the combined OmniFed handbook: a top-level chapter such as **“Running on OLCF Frontier (Engine + hybrid pipeline)”**, after generic local / centralized run instructions and before any compression-specific material.

SLURM support inside the Engine itself—freeze config on the login node, emit `sbatch`, drive `slurm_worker` across tasks—was added by **Sabiha**. Everything below is simply the order **we used** on Frontier once that path existed: centralized baseline first, then hybrid smoke, then full Engine hybrid, then LM.

**Order we walked through:**

1. Set up a Frontier login session (modules, conda env, env vars, paths).
2. Stage datasets on Lustre (Frontier compute nodes have no internet).
3. Run the centralized SLURM Engine baseline (proves Engine + SLURM before hybrid).
4. Run the Phase-A hybrid communication smoke (TorchMPI + gRPC, **no** Engine FedAvg loop).
5. Run the full Engine hybrid FedAvg experiment (**file-topology** preset).
6. Run the **layout-first** hybrid preset (same lattice, no separate topology YAML).
7. Scale the hybrid run up (e.g. 17 or 129 ranks).
8. Run **ResNet-18 + CIFAR-10** hybrid FedAvg (vision regression sibling to MNIST + `simple_cnn`).
9. Run the Llama-150M + C4 hybrid LM experiment (and scale it the same way in principle).
10. Run the Llama ~400M + C4 hybrid LM experiment (same lattice knobs as 150M).
11. Read and interpret result artifacts.

Long-form references (job numbers, roadmap phases, artefact glossary) stayed in **`./archive/hybrid-engine-pipeline/`** so newcomers are not drowned in prose; this README stays chronological.

Deep references worth opening when debugging: `./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`, `HYBRID_USER_KNOBS_AND_ROADMAP.md`, `HYBRID_TRAINING_AND_SYNC.md`, `README_HYDRA_RUN_OUTPUTS.md`.

---

## 1. Conventions used in the commands below

| Symbol | Meaning |
|--------|---------|
| **`YOUR_USER`** | Your Frontier username. |
| **`YOUR_PROJECT`** | Slurm charge account (example: `gen150`). |
| **`OMNIFED_REPO`** | Repo root on Lustre, e.g. `/lustre/orion/gen150/scratch/YOUR_USER/OmniFed_VT`. |
| **`<JOBID>`** | Slurm job id returned after Engine submit (`sbatch` / launcher log). |
| **`<N_CLIENTS>`** | Number of federated clients (**excludes** the parameter-server / RPC slot in centralized semantics). |
| **`W`** | Hybrid world size **`topology.num_clients + 1`** (one rank per participant, including dedicated RPC rank when configured). |

Frontier compute nodes typically **cannot** reach the public internet. Datasets and model weights **must be pre-staged on Lustre** from the **login** node; Hydra overrides then point `root=` at those paths together with **`download=false`**.

### Slurm **exclusive** mode (`slurm.exclusive=true`)

For **multi-node hybrid** jobs (MNIST, CIFAR+ResNet18, Llama‑150M, Llama ~400M), pass **`slurm.exclusive=true`** on **`./main.sh`**. The Engine emits **`#SBATCH --exclusive`** so each allocated node is not shared with other jobs (helps GPU/MPI stability on large allocations). Default in **`conf/base.yaml`** is **`slurm.exclusive: false`**; Frontier **`batch`** jobs are often whole-node already, but we **recommend exclusive for all production hybrid submits** at 7 / 17 / 129 nodes.

---

## 2. Frontier login setup and staged data

### 2.A Get the code onto Frontier

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

…or work directly from a clone on Lustre and **`git pull`** when the branch moves.

### 2.B Login-node environment

```bash
module load miniforge3/23.11.0-0
conda activate pytorch_rocm   # or the site-approved ROCm PyTorch env

cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"
export PYEXE="${CONDA_PREFIX}/bin/python"
chmod +x main.sh   # once
```

Here **`main.sh`** also drives **`grpc_tools.protoc`** on **`src/omnifed/communicator/grpc.proto`** before **`main.py`**; that compilation step matched what we relied on whenever Flora/gRPC code generation had to agree with runtime.

### 2.C Pre-stage MNIST on Lustre

```bash
mkdir -p /lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist
python -c "
from torchvision import datasets
root = '/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist'
datasets.MNIST(root=root, train=True,  download=True)
datasets.MNIST(root=root, train=False, download=True)
"
```

Every MNIST **`main.sh`** block below assumes **`download=false`** and **`root=`** aiming at this tree.

### 2.D Pre-stage CIFAR-10 on Lustre (ResNet-18 hybrid)

```bash
mkdir -p /lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10
python -c "
from torchvision import datasets
root = '/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10'
datasets.CIFAR10(root=root, train=True,  download=True)
datasets.CIFAR10(root=root, train=False, download=True)
"
```

ResNet-18 hybrid presets use **`conf/test_hybrid_layout_fedavg_cifar10_resnet18.yaml`** ( **`torchvision.models.resnet18`** + **`torchvision.datasets.CIFAR10`** ). Every CIFAR block below uses **`download=false`** and the same **`root=`** for train and eval.

Optional helpers: **`test_scripts/frontier_hybrid_cifar10_resnet18_17.sh`** and **`test_scripts/frontier_hybrid_cifar10_resnet18_129.sh`** (set **`OMNIFED_REPO`**, **`OMNIFED_CIFAR10_ROOT`**, **`SLURM_ACCOUNT`** if paths differ).

### 2.E LM pre-flight (**login node only**; skip entire block if §8 Llama+C4 never runs)

These are **copy-pastes from the Frontier runs we traced** (`./archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`) — kept here so one file is self-contained. Replace **`YOUR_USER`** everywhere (below we use scratch **`gen150`**; adjust project/paths if yours differ).

**(Once per conda env)** LM Python packages:

```bash
pip install "datasets>=2.18.0" "transformers>=4.41.0" "sentencepiece>=0.2.0" "huggingface_hub>=0.23"
```

**(Optional)** keep Hugging Face cache on Lustre instead of **`$HOME`** during **`load_dataset`** / Hub pulls:

```bash
export HF_HOME="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/.hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
```

**C4 — `save_to_disk` subset** (**50 000** train **/ 2 000** val rows; tweak **`TRAIN_ROWS` / `VAL_ROWS`** as needed):

```bash
python <<'PY'
from datasets import load_dataset, DatasetDict

OUT = "/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/allenai_c4"
TRAIN_ROWS = 50_000
VAL_ROWS = 2_000

train = load_dataset(
    "allenai/c4",
    "en",
    split=f"train[:{TRAIN_ROWS}]",
    trust_remote_code=True,
)
validation = load_dataset(
    "allenai/c4",
    "en",
    split=f"validation[:{VAL_ROWS}]",
    trust_remote_code=True,
)
DatasetDict({"train": train, "validation": validation}).save_to_disk(OUT)
print("Saved to:", OUT)
PY
```

Quick **C4** sanity (**`load_from_disk`**):

```bash
python -c "
from datasets import load_from_disk
root = '/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/allenai_c4'
d = load_from_disk(root)
print(d)
print(len(d['train']), len(d['validation']))
print(d['train'].column_names)
"
```

**Mistral tokenizer** → Lustre (**compute stays `local_files_only`**):

```bash
python <<'PY'
from transformers import AutoTokenizer
TOKEN_DIR = "/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/tokenizer_Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tok.save_pretrained(TOKEN_DIR)
print("Saved tokenizer ->", TOKEN_DIR)
PY

python -c "
from transformers import AutoTokenizer
p = '/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/tokenizer_Mistral-7B-v0.1'
AutoTokenizer.from_pretrained(p, local_files_only=True)
print('tok ok')
"
```

**Llama‑150 M weights** — **`snapshot_download`** (repo we used):

```bash
mkdir -p /lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/PrimeIntellect_llama-150m-fresh

python <<'PY'
from huggingface_hub import snapshot_download

out = "/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/PrimeIntellect_llama-150m-fresh"
snapshot_download(
    repo_id="PrimeIntellect/llama-150m-fresh",
    local_dir=out,
)
print("done ->", out)
PY

python -c "
from transformers import LlamaForCausalLM
p = '/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/PrimeIntellect_llama-150m-fresh'
LlamaForCausalLM.from_pretrained(p, local_files_only=True)
print('llama ok')
"
```

(Optional Hub rate limits: **`export HF_TOKEN=...`** in the **same login session** before downloads.)

---

**`OMNIFED_*`** — **same shell session as `./main.sh`** (Hydra embeds resolved paths into **`engine_frozen.json`**):

```bash
export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA_WEIGHTS="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/PrimeIntellect_llama-150m-fresh"
```

For **Llama ~400 M**, add (**only** when using **`test_hybrid_layout_fedavg_llama400m`**; path is **your** offline HF tree **`snapshot_download`**, not interchangeable with 150 M):

```bash
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/<your_llama400m_hf_folder>"
```

## 3. Centralized SLURM baseline (Engine, **not** hybrid)

**Why this step exists.** Before any hybrid experiments, we proved that the classic path survived Frontier: Hydra → Engine freezes **`engine_frozen.json`** → **`sbatch`** → each task **`slurm_worker`** with TorchDistrib FedAvg (**`communication_mode`** still **classic**). That matches the SLURM contract described above (**Sabiha’s Engine plumbing**).

**Hydra preset:** **`conf/test_fedavg_centralized_torchdist.yaml`** (FedAvg + simple CNN + MNIST—the same backbone we swapped back to whenever LM work needed a regression sanity check).

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

**What we verified afterwards.** The Engine-side log echoed **`communication_mode`** and **`--ntasks`**, the regenerated **`omnifed_slurm_only.sh`** at repo root bore **`#SBATCH --ntasks=<W>`** consistent with **`CentralizedTopology`**, **`sacct`** showed **`COMPLETED`** **`0:0`**, per-rank material landed under **`outputs/<date>/<config>/`**.

For TorchDistrib **`CentralizedTopology`**, capacity must match **`topology.num_clients + 1`** (server slot + clients)—we bumped **`topology.num_clients`**, **`slurm.nodes`**, and **`slurm.ntasks_per_node`** together until **nodes × tasks** matched that participant count (**not yet** hybrid’s RPC+dedicated lattice story).

---

## 4. Hybrid communication smoke (**no** Engine FedAvg loop, **no** dataset)

**Why this step exists.** Before we trusted **`run_hybrid_training`**, we isolated the **two primitives**: facility-local TorchMPI collectives plus one cross-site Flora **gRPC** round (“Phase A”). No **`round_exec`**, no dataloader—a deliberate narrow slice we used while shaking out ROCm/NCCL binds and **`slurm_hostlist`** address patching.

```bash
cd "${OMNIFED_REPO}"
export PYEXE PYTHONPATH OMNIFED_REPO

export HYBRID_SMOKE_BACKEND=nccl   # batch script defaults differ; GPU path we smoke-tested

sbatch test_scripts/slurm_frontier/hybrid_comm_smoke.slurm
```

**What “good” looked like:** **`sacct`** **`0:0`**, stdout showing leaders participating in **`gRPC`** while everyone else exercised the local collective. Historical job citations sit in **`./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §2.

---

## 5. Full Engine hybrid FedAvg (**file-topology preset**)

**Why this step exists.** This was our **canonical** hybrid demo: **`communication_mode: hybrid`**, **`topology_config`** resolving to **`conf_hybrid/topology/built_symmetric_2x3.yaml`**. Lattice is **`world_size = 7`**: **6** trainers (**2 × 3** ranks per facility) + **1** dedicated Flora RPC-only rank. **`algorithm.round_exec`** ran on trainers; **`slurm_worker`** bounced into **`run_hybrid_training`** inside **`src/omnifed/hybrid/slurm_hybrid_runner.py`**.

**Invariant we repeated to collaborators / PI slides:** **`topology.num_clients + 1 = hybrid_world_size`**. Here **6 + 1 = 7**, so Slurm launches **seven** tasks—simplest pattern **`slurm.nodes=7`**, **`slurm.ntasks_per_node=1`**.

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
   slurm.gres=null \
   slurm.exclusive=true
```

**How we sanity-checked.**

1. Login log line looked like **`[Engine] communication_mode=hybrid: Slurm --ntasks=7`** (or whichever **`W`** resolved).
2. Generated driver script **`#SBATCH --ntasks=7`**.
3. **`sacct`** → **`COMPLETED`**, **`0:0`**.
4. **`outputs/<date>/test_hybrid_engine_contract/engine/node_results/`** bore **`node_000` … `node_006`**: RPC ranks produced the stub **`node_000.json`**, trainers carried full **`sync`** arrays.
5. Optional: **`hybrid_per_round_summary.csv`** at Hydra root— stitched after the fact by **`hybrid_run_summary.py`**.

Frontier reality check we already absorbed: **`4625007`** failed when MNIST download hit public hosts from compute; **`4625686`** succeeded offline—see **`./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §3.1.

---

## 6. Layout-first hybrid preset (same lattice, no separate **`conf_hybrid`** topology file)

**Why this step exists.** We re-ran the **exact same** seven-global-rank lattice, but spelled it purely under **`engine.hybrid.layout`** in **`conf/test_hybrid_layout_fedavg.yaml`**—matching the “Figure-2 ergonomics” story (lattice beside **`topology`** / **`engine`** blocks rather than invoking a **`topology_config`** file). Validators (**`validate_hybrid_slurm_topology_alignment`**, **`hybrid_world_size_from_cfg`**) stayed on the identical code path—the experiment differed mainly in authoring style.

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
   slurm.gres=null \
   slurm.exclusive=true
```

Acceptance mirrored §5 (**same **`--ntasks`** semantics**) while Hydra’s folder naming reflected **`test_hybrid_layout_fedavg`**.

**Operational note.** If **`slurm.nodes × ntasks_per_node`** started below **`world_size`**, **`engine.py`** printed **`[Engine] slurm.nodes raised …`** and bumped capacity while **`#SBATCH --ntasks=W`** stayed laser-focused on **`W`**. We kept returning to **`./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §**4.3** whenever collaborators asked “why **`nodes`** flipped during submit?”

---

## 7. Larger-scale hybrid (17 or 129 ranks)

We never needed new Python for this tier—pure Hydra choreography. **Same layout knobs** apply to MNIST, CIFAR+ResNet18, and Llama presets.

| Scale | `mpi_ranks_per_facility` | `topology.num_clients` | `dataset_total_clients` | `slurm.nodes` | `W` |
|-------|--------------------------|------------------------|-------------------------|---------------|-----|
| 7 (default in presets) | 3 | 6 | 6 | 7 | 7 |
| 17 | 8 | 16 | 16 | 17 | 17 |
| 129 | 64 | 128 | 128 | 129 | 129 |

Standing rules: **`SLURM_NTASKS == hybrid_world_size`**, **`topology.num_clients == W − 1`**, and for LM runs **`datamodule.num_federated_clients`** must equal **`topology.num_clients`** (literal integers). Appendix narrative: **`./archive/hybrid-engine-pipeline/HYBRID_USER_KNOBS_AND_ROADMAP.md`**.

**MNIST — 129-node example** (add **`engine.hybrid.server_*`** if **`global_rounds > 1`** and the rank‑0 gRPC server dies early—see §7.1):

```bash
./main.sh --config-name test_hybrid_layout_fedavg \
  overwrite=true engine.mode=slurm global_rounds=10 \
  engine.hybrid.layout.mpi_ranks_per_facility=64 \
  topology.num_clients=128 \
  engine.hybrid.training.dataset_total_clients=128 \
  engine.hybrid.server_run_extra_sec=600 \
  engine.hybrid.server_sec_per_round=5000 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/torchvision-mnist \
  slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=04:00:00 \
  slurm.nodes=129 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
  slurm.exclusive=true
```

Confirm login output: **`[Engine] communication_mode=hybrid: Slurm --ntasks=129`**.

### 7.1 gRPC server walltime (MNIST / CIFAR at many rounds)

Preset default **`server_run_extra_sec=120`** + **`server_sec_per_round=300`** ⇒ **`max_wall ≈ 120 + global_rounds×300`**. Long MNIST runs at 129 nodes with **`global_rounds=10`** can shut down rank‑0 before trainers finish (**`Connection refused` on :50051**). Override **`engine.hybrid.server_run_extra_sec=600`** and **`engine.hybrid.server_sec_per_round=5000`** (already set in **`test_hybrid_layout_fedavg_cifar10_resnet18`** and Llama presets).

---

## 7.2 ResNet-18 + CIFAR-10 hybrid FedAvg

**Preset:** **`conf/test_hybrid_layout_fedavg_cifar10_resnet18.yaml`** (composes **`test_fedavg_centralized_torchdist_cifar10_resnet18`** — **not** MNIST/`simple_cnn`). Pre-stage CIFAR: **§2.D**.

**7-node smoke** (preset defaults; add offline paths):

```bash
./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18 \
  overwrite=true engine.mode=slurm global_rounds=2 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=00:45:00 \
  slurm.nodes=7 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
  slurm.exclusive=true
```

**17-node:**

```bash
./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18 \
  overwrite=true engine.mode=slurm global_rounds=2 \
  engine.hybrid.layout.mpi_ranks_per_facility=8 \
  topology.num_clients=16 \
  engine.hybrid.training.dataset_total_clients=16 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=02:00:00 \
  slurm.nodes=17 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
  slurm.exclusive=true
```

**129-node** (extended server budget is in the preset):

```bash
./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18 \
  overwrite=true engine.mode=slurm global_rounds=10 \
  engine.hybrid.layout.mpi_ranks_per_facility=64 \
  topology.num_clients=128 \
  engine.hybrid.training.dataset_total_clients=128 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/cifar10 \
  slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=04:00:00 \
  slurm.nodes=129 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
  slurm.exclusive=true
```

---

## 8. Llama + C4 hybrid LM (**150 M default; ~400 M sibling preset**)

**Why this step exists.** We deliberately **reuse** the hybrid Runner—only Hydra swaps **algorithm / model / datamodule**:

| Piece | Swap |
|--------|------|
| **Algorithm** | **`FedAvgLLM`** — HF dict-style forward, **AdamW**, batch inferred from **`input_ids`** |
| **Model** | Same **`load_llama_from_pretrained_checkpoint`**; weights via **`OMNIFED_LLAMA_WEIGHTS`** (150 M presets) **or** **`OMNIFED_LLAMA400_WEIGHTS`** (**`test_*_llama400m`**) |
| **Datamodule** | **`datasets.load_from_disk`** C4 shards; **`Dataset.shard`** on **train**, driven by **`OMNIFED_FEDERATED_CLIENT_INDEX`** seeded in **`slurm_hybrid_runner.py`** |

Pre-flight (**C4**, tokenizer, checkpoints, **`pip`**) is **§2.E**. Immediately before **`./main.sh`** the **`export OMNIFED_*`** lines **must live in that same shell**. Use **`slurm.exclusive=true`** on every LM hybrid submit (7 / 17 / 129). Scale-up lattice: **§7** table; LM runs also need **`datamodule.num_federated_clients`** matching **`topology.num_clients`**.

**Llama‑150 M — 7-node:**

```bash
cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"
export PYEXE="${CONDA_PREFIX}/bin/python"

export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA_WEIGHTS="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/PrimeIntellect_llama-150m-fresh"

./main.sh --config-name test_hybrid_layout_fedavg_llama150m \
   overwrite=true \
   engine.mode=slurm \
   global_rounds=2 \
   slurm.account=YOUR_PROJECT \
   slurm.partition=batch \
   slurm.time=01:35:00 \
   slurm.nodes=7 \
   slurm.ntasks_per_node=1 \
   slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 \
   slurm.gpus_per_task=1 \
   slurm.gres=null \
   slurm.exclusive=true
```

**Llama‑150 M — 17-node:**

```bash
# Same exports as 7-node, then:
./main.sh --config-name test_hybrid_layout_fedavg_llama150m \
   overwrite=true engine.mode=slurm global_rounds=2 \
   engine.hybrid.layout.mpi_ranks_per_facility=8 \
   topology.num_clients=16 \
   engine.hybrid.training.dataset_total_clients=16 \
   datamodule.num_federated_clients=16 \
   slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=01:59:00 \
   slurm.nodes=17 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
   slurm.exclusive=true
```

**Llama‑150 M — 129-node:**

```bash
# Same exports as 7-node, then:
./main.sh --config-name test_hybrid_layout_fedavg_llama150m \
   overwrite=true engine.mode=slurm global_rounds=2 \
   engine.hybrid.layout.mpi_ranks_per_facility=64 \
   topology.num_clients=128 \
   engine.hybrid.training.dataset_total_clients=128 \
   datamodule.num_federated_clients=128 \
   slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=01:59:00 \
   slurm.nodes=129 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
   slurm.exclusive=true
```

**Llama ~400 M — 7-node** (**`OMNIFED_LLAMA400_WEIGHTS`** only — do not repoint **`OMNIFED_LLAMA_WEIGHTS`**):

```bash
cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"
export PYEXE="${CONDA_PREFIX}/bin/python"

export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/YOUR_USER/omnifed_data/<your_llama400m_hf_folder>"

./main.sh --config-name test_hybrid_layout_fedavg_llama400m \
   overwrite=true \
   engine.mode=slurm \
   global_rounds=2 \
   slurm.account=YOUR_PROJECT \
   slurm.partition=batch \
   slurm.time=01:35:00 \
   slurm.nodes=7 \
   slurm.ntasks_per_node=1 \
   slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 \
   slurm.gpus_per_task=1 \
   slurm.gres=null \
   slurm.exclusive=true
```

**Llama ~400 M — 17-node** (validated on Frontier with **`Llama-3.2-400M-Amharic`** weights):

```bash
# Same OMNIFED_* exports as 7-node, then:
./main.sh --config-name test_hybrid_layout_fedavg_llama400m \
   overwrite=true engine.mode=slurm global_rounds=2 \
   engine.hybrid.layout.mpi_ranks_per_facility=8 \
   topology.num_clients=16 \
   engine.hybrid.training.dataset_total_clients=16 \
   datamodule.num_federated_clients=16 \
   slurm.job_name=omnifed_llama400_hybrid_17 \
   slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=01:59:00 \
   slurm.nodes=17 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
   slurm.exclusive=true
```

**Llama ~400 M — 129-node:**

```bash
# Same OMNIFED_* exports as 7-node, then:
./main.sh --config-name test_hybrid_layout_fedavg_llama400m \
   overwrite=true engine.mode=slurm global_rounds=2 \
   engine.hybrid.layout.mpi_ranks_per_facility=64 \
   topology.num_clients=128 \
   engine.hybrid.training.dataset_total_clients=128 \
   datamodule.num_federated_clients=128 \
   slurm.job_name=omnifed_llama400_hybrid_129 \
   slurm.account=YOUR_PROJECT slurm.partition=batch slurm.time=01:59:00 \
   slurm.nodes=129 slurm.ntasks_per_node=1 slurm.cpus_per_task=4 \
   slurm.gpus_per_node=1 slurm.gpus_per_task=1 slurm.gres=null \
   slurm.exclusive=true
```

**Llama ~400 M** runs are slower than 150 M; under a tight partition cap use **`global_rounds=1`** for smoke tests. Presets already set **`server_run_extra_sec=600`** and **`server_sec_per_round=5000`** for rank‑0 gRPC lifetime.

### 8.1 Hybrid checkpointing & auto-chain (2h `batch` walltime)

Full design: **`docs/CHECKPOINTING_HYBRID_SLURM_PLAN.md`** (§11 Frontier copy-paste).

**Round-end checkpoints** on Lustre: `slurm.checkpoint_dir` + `slurm.experiment_id` → `…/omnifed_checkpoints/<experiment_id>/manifest.json` and `round_XXX/` shards.

**Manual resume:** Job 1 `slurm.resume=false`, Job 2+ `slurm.resume=true` (validated: `llama150_7_ckpt_v1` on 7-node Llama‑150M).

**Auto-chain to N global rounds** (login node — use **`screen`** or **`tmux`** so the wrapper keeps running):

```bash
export EXP_ID=llama400_7_autochain_v1
./scripts/frontier_llama400m_7node_autochain.sh
```

Wrapper submits, waits for **COMPLETED**, resubmits with **`slurm.resume=true`** until `manifest.json` shows `"status": "complete"`. Emits **`#SBATCH -d singleton`** via `slurm.dependency_singleton=true`.

**Structural footnote (after hitting `InterpolationKeyError`):** **`datamodule.num_federated_clients`** must match **`topology.num_clients`** as **literals** in each preset / CLI overrides (`conf/datamodule/c4_lm_federated_disk.yaml` uses **`???`** until the preset fills it). We avoided **`${topology.num_clients}`** in that datamodule because **`OmegaConf.to_container`** during **`engine_frozen.json`** could fail. **`./archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`** §**J** captures the remediation.

**Flora gRPC / LM (`RESOURCE_EXHAUSTED`):** Hybrid FedAvg **`SendUpdate`** ships a protobuf with full **`float32`** weights. The old **100 MiB** **`grpc.max_*_message_length`** cap is CN-scale only; Llama‑class runs hit **`CLIENT: Sent message larger than max (... vs 104857600)`** during **`global_agg`**. OmniFed raises the shared **`GRPC_MAX_MESSAGE_BYTES`** in **`src/flora/communicator/grpc_limits.py`** (daemon **and** stub must stay aligned).

**grpcio sizing:** **`2147483648`** (literally **2 GiB**) overflows grpcio **signed int32** channel args with **`OverflowError`**; the constant caps at **`INT32_MAX`** (**`2147483647`**).


---

## 9. Reading the result artefacts (PI-facing)

| Artefact | What we looked for | Typical use |
|----------|---------------------|-------------|
| **`outputs/<date>/<config>/slurm-<JOBID>.out`** | Merged stdout—**`[hybrid]`**, **`run_hybrid_training`** breadcrumbs, Flora shutdown chatter | Immediate post-job skim |
| **`engine/node_results/node_*_results.json`** | Per-rank rollups (**`sync/local_agg_time`**, **`sync/global_agg_time`**, **`sync/local_bcast_time`**) | Explaining intra- vs cross-facility timing |
| **`hybrid_per_round_summary.csv`** | **`gRPC_F*`** ms columns per facility leader (**often asymmetric**—PS ordering, **not** a summarizer defect) | Spreadsheet-ready cross-facility story |
| **`Node0.*/metrics_*.csv`** | Detailed train/eval/sync streams | Accuracy / loss slides |

FedAvg synchronization order—facility reduce → leader Flora step → facility broadcast—mirrors **`./archive/hybrid-engine-pipeline/HYBRID_TRAINING_AND_SYNC.md`**, reflected in **`sync/`** timing keys saved per rank. Broader glossary: **`./archive/hybrid-engine-pipeline/README_HYDRA_RUN_OUTPUTS.md`**.

---

## Closing narrative (**how we pitched the arc**)

Centralized TorchDistrib on Frontier anchored that **Sabiha’s SLURM engine path** behaved. Hybrid smoke guaranteed **facility collectives + gRPC** behaved before layering FedAvg. **`test_hybrid_engine_contract`** and **`test_hybrid_layout_fedavg`** then showed the Engine loop running **either** **`topology_config` file** OR inline **`layout`**, same lattice. Larger hybrids proved **only Hydra knobs** scaled us to **17** and **129** ranks once **`W`** / **`topology.num_clients`** / **`training.dataset_total_clients`** (and LM **`num_federated_clients`**) marched in lock-step. **ResNet-18 + CIFAR-10** swapped vision blocks without touching the hybrid runner; **Llama 150M / ~400M + C4** did the same for LM blocks, with **`slurm.exclusive=true`** recommended on all multi-node submits.
