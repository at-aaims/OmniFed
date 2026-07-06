# Roadmap — Llama‑150M + C4 in OmniFed hybrid FedAvg

**Purpose:** Lay out phased steps to run **Llama‑150M** on **C4** (offline / `load_from_disk`) through the **existing** OmniFed hybrid Slurm pipeline (FedAvg, `local_agg` → `global_agg` → `local_bcast`, validators, presets).

**Explicitly deferred:** OmniFed roadmap phases **E** (`ntasks_per_node` > 1, multi‑GPU nodes) and **F** (pluggable `inner_comm` / `outer_comm`). First LLM tests stay **one task × one GPU per node**.

**Behavioral reference (not copy-paste paths):** `frontier_example/train_llama150m.py` — HF `LlamaForCausalLM`, tokenizer cache, **`datasets.load_from_disk`** layout for **`allenai/c4`**-style dumps (`allenai/c4` → directory name **`allenai_c4`**).

**Regression requirement:** Ability to switch back to **current CNN + MNIST** (already validated at 129 nodes) **without changing** Slurm/engine/hybrid semantics — swap **model / datamodule / training‑side code** via config once implemented (**`conf/test_hybrid_layout_fedavg`** + **`test_fedavg_centralized_torchdist`** chain uses **`simple_cnn` + `mnist`**, not CIFAR).

---

## Status snapshot (maintenance)

| Block | Status | Notes |
|-------|--------|--------|
| **Phase 0** — experiment matrix | **Completed** | Topologies and rollout order locked below. |
| **Phase 1** — requirements map | **Completed** | Mapping and gaps documented; no code landed yet. |
| **Phase 2** — implementation | **Core complete** | New preset **`test_hybrid_layout_fedavg_llama150m`**, **`FedAvgLLM`**, C4 **`build_c4_lm_datamodule`**, HF loader, hybrid env **`OMNIFED_FEDERATED_CLIENT_INDEX`**. Populate tokenizer/model Lustre paths + run Phase 3. |
| **Phase 3** — staged runs | **Runbook documented** | End-to-end commands: **[Frontier procedure](#frontier-procedure-login--data--submit-gen150-example)**. |
| **Phase 4** — acceptance | **Pending** | Per checklist after each scale. |

---

## Fixed decisions

1. **Scale order:** **7 nodes** → **17 nodes** → **129 nodes** (same hybrid/FedAvg engine path; LM integration first proven small).
2. **C4 preparation:** **`save_to_disk`** layout **plan A** (bounded **`train[:N]` / `validation[:M]`** slice) executed on Frontier login node — sufficient for funnel testing; **large / full C4** deferred until longer training requires it (**not immediate after A**).
3. **Starter topology (7 ranks):** Use preset defaults **`test_hybrid_layout_fedavg`**: **`topology.num_clients = 6`**, **`2 × 3`** facilities + dedicated RPC ⇒ **`world_size = 7`**, **`slurm.nodes = 7`**, **`ntasks_per_node = 1`**.
4. **Topology (17 ranks):** **`topology.num_clients = 16`**, **`2 × 8`** (`mpi_ranks_per_facility=8`), **`dataset_total_clients = 16`**, **`slurm.nodes = 17`**.
5. **Topology (129 ranks — proven MNIST):** **`topology.num_clients = 128`**, **`mpi_ranks_per_facility = 64`** ( **`2 × 64`** ), **`dataset_total_clients = 128`**, **`slurm.nodes = 129`**, **`ntasks_per_node = 1`** (matches validated hybrid FedAvg MNIST launches).
6. **Optimizer:** **`FedAvgLLM`** uses **AdamW** (LM); classic **`FedAvg`** remains **SGD** for CNN baselines.
7. **FedAvg semantics:** Parameter tensor averaging unchanged; LM wiring uses causal LM **`dict`** batches (**`labels` = clone of **`input_ids`**) plus **`Dataset.shard`** for **train** with hybrid-provided **`OMNIFED_FEDERATED_CLIENT_INDEX`**.

---

## Data paths — Frontier scratch (`shruti2395`, `gen150`)

**Hybrid Slurm LM:** every trainer receives **`OMNIFED_FEDERATED_CLIENT_INDEX`** (`0 … num_clients−1`) and **`OMNIFED_CENTRALIZED_NODE_INDEX`** from **`slurm_hybrid_runner`** before **`instantiate(cfg.datamodule)`**. MNIST hybrids ignore unless a reader consumes these env vars.

**Scratch root:** `/lustre/orion/gen150/scratch/shruti2395/`  
**Repo checkout (example):** `/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT`  

| Role | Path | Notes |
|------|------|--------|
| **MNIST (torchvision)** | `/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist` | Used with **`datamodule.*.dataset.download=false`** overrides in hybrid Slurm command. |
| **C4 (HF `datasets` on disk)** | `/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4` | Prepared with **plan A** (`DatasetDict(train, validation)`, name matches **`allenai/c4` → `allenai_c4`** pattern from `train_llama150m.py`). **Subset** sizing chosen at login prep time — revisit doc when row counts frozen. |
| **HF caches (recommended during prep)** | `/lustre/orion/gen150/scratch/shruti2395/omnifed_data/.hf_home` (and **`HF_DATASETS_CACHE`** under it) | Keeps downloads off `$HOME`; set when **`load_dataset`** / tokenizer model cache prep runs on login. |
| **LLM tokenizer / weights caches** | Set via env (**required on compute**) | **`OMNIFED_TOKENIZER_DIR`** — Mistral (**`mistralai/Mistral-7B-v0.1`**) tokenizer prepared with **`AutoTokenizer(..., local_files_only=True)`** on login. **`OMNIFED_LLAMA_WEIGHTS`** — **150 M** presets (`llama150m_hf_disk`). **`OMNIFED_LLAMA400_WEIGHTS`** — **~400 M tier** presets (`llama400m_hf_disk`, **`test_*_llama400m`**); **separate** offline tree (**[F-400m](#f-400m-weights)**). Overrides in **`conf/model/*.yaml`** (`${oc.env:...}`). |

**Example C4 prep (login slice):** `TRAIN_ROWS=50_000`, `VAL_ROWS=2_000` → **`allenai_c4`** under **`omnifed_data/`**.

**Compute nodes:** No Hugging Face downloads on compute — materialize C4, tokenizer, and weights on Lustre **on the login node** first.

---

<a id="frontier-procedure-login--data--submit-gen150-example"></a>
## Frontier procedure (login → data → submit) — **`gen150` example**

Paths match **`shruti2395`** / **`/lustre/orion/gen150/scratch/`** — adjust **`USER`**, **`PROJECT`**, **`PYEXE`**, and conda install location (**`/ccs/home/...`** vs **`autofs/...`**) if your account layout differs.

### A. Sync codebase (development machine → Frontier)

Skip if you edit directly on Lustre; otherwise typical **`rsync`**:

```bash
rsync -avz \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  ~/OmniFed_VT/ \
  shruti2395@login03.frontier.olcf.ornl.gov:/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT/
```

### B. Frontier login node — conda, repo, **`PYTHONPATH`**

```bash
module load miniforge3/23.11.0-0    # OLCF-standard Miniforge module
conda activate pytorch_rocm

cd /lustre/orion/gen150/scratch/shruti2395/OmniFed_VT
export PYTHONPATH="/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT"
export PYEXE="/ccs/home/shruti2395/.conda/envs/pytorch_rocm/bin/python"
```

(Optional) keep HF Hub cache downloads on scratch instead of **`$HOME`**:

```bash
export HF_HOME="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/.hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
```

### C. LM Python dependencies (**once per env**)

```bash
pip install "datasets>=2.18.0" "transformers>=4.41.0" "sentencepiece>=0.2.0" "huggingface_hub>=0.23"
```

### D. C4 **`save_to_disk`** (login; Hub allowed here — plan A subset)

Produces **`allenai_c4`** with **`train`** + **`validation`** (example: **50 000 / 2 000** rows — tune **`TRAIN_ROWS` / `VAL_ROWS`**):

```bash
python <<'PY'
from datasets import load_dataset, DatasetDict

OUT = "/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4"
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

Sanity:

```bash
python -c "
from datasets import load_from_disk
d = load_from_disk('/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4')
print(d)
print(len(d['train']), len(d['validation']))
print(d['train'].column_names)
"
```

### E. Mistral tokenizer → Lustre (login; **`local_files_only`** on compute)

```bash
python <<'PY'
from transformers import AutoTokenizer
TOKEN_DIR = "/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1"
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tok.save_pretrained(TOKEN_DIR)
print("Saved tokenizer ->", TOKEN_DIR)
PY

python -c "
from transformers import AutoTokenizer
p = '/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1'
AutoTokenizer.from_pretrained(p, local_files_only=True)
print('tok ok')
"
```

### F. Llama‑150M weights → Lustre (login **`snapshot_download`**)

```bash
mkdir -p /lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh

python <<'PY'
from huggingface_hub import snapshot_download

out = "/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh"
snapshot_download(
    repo_id="PrimeIntellect/llama-150m-fresh",
    local_dir=out,
)
print("done ->", out)
PY

ls "/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh/config.json"

export LLAMA="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh"
python -c "
from transformers import LlamaForCausalLM
p = '/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh'
LlamaForCausalLM.from_pretrained(p, local_files_only=True)
print('llama ok')
"
```

(Optional) set **`HF_TOKEN`** on login for higher Hub rate limits or **gated** checkpoints: `export HF_TOKEN=hf_…` (must **accept license** on the model page before download). **`unset HF_TOKEN`** if you see bogus **`401 Unauthorized`** errors.

<a id="f-400m-weights"></a>
### F-400m. Llama **~400 M** tier — weights on Lustre (**`test_*_llama400m`** presets)

**Goal:** Populate a **directory** next to **`PrimeIntellect_llama-150m-fresh`** and point **`OMNIFED_LLAMA400_WEIGHTS`** at it. Presets (**`conf/test_hybrid_layout_fedavg_llama400m.yaml`**, **`conf/model/llama400m_hf_disk.yaml`**) load **`LlamaForCausalLM`** with **`local_files_only=true`** (**`compute`**: no Hub).

**Important**

1. **Replace the Hub id.** Use a **real** Hub id in the form **`namespace/model-name`** (e.g. **`PrimeIntellect/…`**, **`meta-llama/…`**) — **not** the placeholder string **`namespace/repo-name-you-chose-on-the-hub`**. A typo or placeholder yields **`401` / Repository Not Found** from the Hub API.
2. **Pick a Llama-compatible repo** whose **`config.json`** corresponds to **`LlamaForCausalLM`** (**`model_type`** is **`llama`**). Confirm the repo exists on **`https://huggingface.co`** before downloading.
3. **Flora protobuf size.** Default federation ships **full float32 state** over gRPC (**`GRPC_MAX_MESSAGE_BYTES`** ≈ signed **`INT32_MAX`**, ~2 GiB). **~150 M–~500 M** dense Llama-scale weights typically fit in that ceiling; **~1 B fp32** can hit **`RESOURCE_EXHAUSTED`**. Larger models ⇒ different dtype / chunked protocol (outside this roadmap).
4. **PrimeIntellect** ships **`PrimeIntellect/llama-150m-fresh`** (you already mirror) and **`PrimeIntellect/llama-1b-fresh`** (~1 B — **risky** for default fp32 Flora). There is **no** bundled “exact 400 M” PrimeIntellect mirror in these examples — **pick a Hub id** (~350–450 M **`llama`** if available) **or** copy an internal snapshot tree into **`OUT`**.

Directory name is arbitrary (example **`PrimeIntellect_llama-400m-fresh`**); contents must match the chosen checkpoint.

```bash
# Frontier login node (conda with huggingface_hub installed — same env as §C acceptable)
SCRATCH=/lustre/orion/gen150/scratch/shruti2395
OUT="${SCRATCH}/omnifed_data/PrimeIntellect_llama-400m-fresh"
export REPO_ID="YourOrg/your-real-llama-hub-repo"    # <-- MUST be a live Hub model id

mkdir -p "$OUT"
export OUTDIR="$OUT"

# Optional gated models (Meta Llama): export HF_TOKEN=hf_... after accepting license on hub
# If download fails with 401: fix token OR unset HF_TOKEN if it is stale/wrong.

python <<'PY'
import os
from huggingface_hub import snapshot_download
out = os.environ["OUTDIR"]
repo = os.environ["REPO_ID"]
snapshot_download(repo_id=repo, local_dir=out)
print("done ->", out)
PY
```

**Alternative (CLI)**

```bash
huggingface-cli download "$REPO_ID" --local-dir "$OUT" --local-dir-use-symlinks False
```

**Verify `config.json`, weight files, and `LlamaForCausalLM` load**

```bash
ls -la "${OUT}/config.json"
ls "${OUT}"/*.safetensors 2>/dev/null || ls "${OUT}"/pytorch_model*.bin 2>/dev/null || true

python <<'PY'
import os
from transformers import LlamaForCausalLM
p = os.environ["OUTDIR"]
LlamaForCausalLM.from_pretrained(p, local_files_only=True, torch_dtype="float32")
print("LlamaForCausalLM load OK:", p)
PY
```

**Compute / Slurm exports** (**same shell as `./main.sh`**) — add **beside** 150 M vars when running **`test_hybrid_layout_fedavg_llama400m`**:

```bash
export OMNIFED_LLAMA400_WEIGHTS="${OUT}"   # or the absolute path you downloaded into
```

**Common failure:** **`OSError`** / **`HFValidationError`** **`Repo id must be…`** means **`OMNIFED_LLAMA400_WEIGHTS`** pointed at **missing** paths or placeholders — **`instantiate`** then fails on rank 0 (**RPC** loads a model skeleton). OmniFed **`load_llama_from_pretrained_checkpoint`** (**`hf_causal_lm.py`**) also raises **`FileNotFoundError`** if **`local_files_only`** and **`OUT`** is **not** a directory.

### G. **`OMNIFED_*`** exports (same shell as **`./main.sh`**)

Hydra expands **`${oc.env:OMNIFED_...}`** when the driver builds config; **`engine_frozen.json`** stores **resolved absolute paths**. Export these immediately before **`./main.sh`**:

```bash
export OMNIFED_C4_DISK="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4"
export OMNIFED_TOKENIZER_DIR="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1"
export OMNIFED_LLAMA_WEIGHTS="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-150m-fresh"
```

**Llama ~400 M presets** (**`./main.sh --config-name test_hybrid_layout_fedavg_llama400m`**, etc. — **same C4 + tokenizer** as 150 M). Set **only** **`OMNIFED_LLAMA400_WEIGHTS`** (**do not** repoint **`OMNIFED_LLAMA_WEIGHTS`** for safety):

```bash
export OMNIFED_LLAMA400_WEIGHTS="/lustre/orion/gen150/scratch/shruti2395/omnifed_data/PrimeIntellect_llama-400m-fresh"
```

<a id="hybrid-lm-job-7-nodes-gen150"></a>

### H. Hybrid LM job — **`world_size`** 7 (**6** trainers + RPC), **`batch`**, **`gen150`**

```bash
./main.sh --config-name test_hybrid_layout_fedavg_llama150m \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=2 \
  slurm.account=gen150 \
  slurm.partition=batch \
  slurm.time=01:30:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null
```

Notes:

- **`main.sh`** invokes **`grpc_tools.protoc`** on `./src/omnifed/communicator/grpc.proto` before **`main.py`**.
- **`slurm.nodes=7`** matches preset **`topology.num_clients=6`** + dedicated RPC ⇒ **`world_size=7`**.
- **NCCL** is still chosen for **`topology.local_comm.backend: nccl`**; **`NCCL_SOCKET_IFNAME`** (or OLCF FABRIC vars) only if multi‑node Torch MPI misbehaves — see **`./HYBRID_SLURM_REFERENCE.md`**.

### I. Queue & logs

```bash
squeue -u shruti2395
# After run starts / finishes:
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Start,Elapsed,AllocNodes

# Stdout/stderr alongside Hydra run dir, e.g.:
# outputs/<YYYY-MM-DD>/test_hybrid_layout_fedavg_llama150m/slurm-<JOBID>.out
```

### J. Larger scale (**must keep counts aligned**)

For **NOT**‑7 presets, override **both**:

- **`topology.num_clients`** (plus matching **`engine.hybrid.layout.mpi_ranks_per_facility`**, **`training.dataset_total_clients`**, **`slurm.nodes`**)  
- **`datamodule.num_federated_clients`** — **literal**, same integer as **`topology.num_clients`** (**do not** use **`${topology.num_clients}`** in **`c4_lm_federated_disk`**; **`OmegaConf.to_container`** for **`engine_frozen.json`** can fail).

Example pattern (values illustrative):  
`topology.num_clients=128 datamodule.num_federated_clients=128 … slurm.nodes=129 … mpi_ranks_per_facility=64 …`

### K. Regression — previous MNIST hybrid (**different preset**)

```bash
./main.sh --config-name test_hybrid_layout_fedavg \
  overwrite=true engine.mode=slurm global_rounds=10 \
  engine.hybrid.layout.mpi_ranks_per_facility=3 \
  topology.num_clients=6 \
  engine.hybrid.training.dataset_total_clients=6 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
  datamodule.eval.dataset.root=/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist \
  slurm.account=gen150 slurm.partition=batch slurm.time=01:30:00 \
  slurm.nodes=7 slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 slurm.gpus_per_node=1 slurm.gpus_per_task=1 \
  slurm.gres=null
```

`**OMNIFED_*`** not required for MNIST (optional to unset).


---

## Phase 0 — Lock the experiment matrix

**Completed.** Contents captured in **Fixed decisions**, **Data paths**, and **[Frontier procedure](#frontier-procedure-login--data--submit-gen150-example)** above.

<details>
<summary>Original checklist (archived wording)</summary>

1. Target topologies for **7**, **17**, **129** ranks documented.  
2. LM preset knobs and offline policy agreed at intent level (**Phase 2** fills Hydra literals).

</details>

---

## Phase 1 — Requirements map (before coding)

**Completed.**

- **OmniFed hook map:** **`instantiate(cfg.model)` / `instantiate(cfg.datamodule)`** in **`run_hybrid_training`**; **`dict`** batch device transfer exists in **`BaseAlgorithm._transfer_batch_to_device`**; **`FedAvg._compute_loss`** and **`_infer_batch_size`** need LM-aware behavior for causal LM (**`input_ids`**, labels / model loss output).  
- **Gaps:** Classification CE vs **`LlamaForCausalLM`** + labels; **`_infer_batch_size`** missing `input_keys` today; eval metrics (**perplexity** vs accuracy); memory and **larger `sync/global_agg_time`** vs CNN (**expected**, not summarizer bugs).  
- **Data:** **`load_from_disk`** root + client-consistent sharding for federation (parity with **`topology.num_clients`**), not blindly **`SLURM_PROCID`** DDP sizing from the standalone Frontier script.

**Phase 2+** executes the wiring; this doc stays the charter.

---

## Current step

Follow **[Frontier procedure (login → data → submit)](#frontier-procedure-login--data--submit-gen150-example)**. After codebase changes (**Hydra **`num_federated_clients`** / presets**), **`rsync`** OmniFed checkout to Frontier before **`./main.sh`**.

---

## Phase 2 — file inventory *(new vs touched)*

### New files

| Path |
|------|
| `src/omnifed/algorithm/fedavg_llm.py` |
| `src/omnifed/data/lm_datamodule.py` |
| `src/omnifed/model/hf_causal_lm.py` |
| `conf/algorithm/fedavg_llm.yaml` |
| `conf/model/llama150m_hf_disk.yaml` |
| `conf/datamodule/c4_lm_federated_disk.yaml` |
| `conf/test_fedavg_llm_centralized_torchdist.yaml` |
| `conf/test_hybrid_layout_fedavg_llama150m.yaml` |
| `tests/test_lm_collate_utils.py` |

### Modified files

| Path | Change |
|------|--------|
| `src/omnifed/hybrid/slurm_hybrid_runner.py` | Sets **`OMNIFED_FEDERATED_CLIENT_INDEX`** and **`OMNIFED_CENTRALIZED_NODE_INDEX`** before **`instantiate(cfg.datamodule)`** for per-client C4 **train** shards. |
| `src/omnifed/algorithm/__init__.py` | Exports **`FedAvgLLM`**. |
| `requirements.txt` | Adds **`datasets`**, **`transformers`**, **`sentencepiece`**. |
| `./HYBRID_LLAMA150M_C4_ROADMAP.md` | Status, paths, Phase 2 landed, this inventory. |

---

## Phase 2 — Implementation (**landed**)

**Hydra presets**

| Config | Purpose |
|--------|---------|
| **`conf/test_hybrid_layout_fedavg_llama150m.yaml`** | Hybrid **`world_size = 7`** (6 trainers + RPC) — Llama (disk) + C4 (**`load_from_disk`**) + **`FedAvgLLM`**. |
| **`conf/test_hybrid_layout_fedavg_llama400m.yaml`** | Same layout as **150 M**; weights via **`OMNIFED_LLAMA400_WEIGHTS`** + **`conf/model/llama400m_hf_disk.yaml`** (prep: **[F-400m](#f-400m-weights)**). |
| **`conf/test_fedavg_llm_centralized_torchdist.yaml`** | Centralized TorchDist stack (composed by the **150 M** hybrid preset). |
| **`conf/test_fedavg_llm_centralized_torchdist_llama400m.yaml`** | Centralized TorchDist stack (composed by the **400 M** hybrid preset). |
**Code / conf**

| Path | Role |
|------|------|
| **`src/omnifed/algorithm/fedavg_llm.py`** | **`FedAvgLLM`** — HF causal forward + **`AdamW`**, **`input_ids`** batch-size inference. |
| **`src/omnifed/data/lm_datamodule.py`** | **`build_c4_lm_datamodule`** — **`load_from_disk`**, **`train` shard** by federated client id. |
| **`src/omnifed/model/hf_causal_lm.py`** | **`load_llama_from_pretrained_checkpoint`** for Hydra **`instantiate`**. |
| **`conf/model/llama150m_hf_disk.yaml`** | Offline **`LlamaForCausalLM.from_pretrained`** (**`OMNIFED_LLAMA_WEIGHTS`**). |
| **`conf/model/llama400m_hf_disk.yaml`** | Same loader; **`OMNIFED_LLAMA400_WEIGHTS`** (**[F-400m](#f-400m-weights)**). |
| **`conf/datamodule/c4_lm_federated_disk.yaml`** | **`num_federated_clients: ???`** — each LM preset sets a literal (**must equal **`topology.num_clients`**). Omit **`${topology.num_clients}`** so Slurm **`engine_frozen.json`** (**`OmegaConf.to_container`**) succeeds. |
| **`conf/algorithm/fedavg_llm.yaml`** | Algorithm Hydra stub. |

**Launcher:** **[Frontier procedure — section H](#hybrid-lm-job-7-nodes-gen150)** (short form: export **`OMNIFED_*`**, then **`./main.sh --config-name test_hybrid_layout_fedavg_llama150m`** …).


<details>
<summary>Original Phase 2 checklist (tracked)</summary>

~~6~~ Hydra / hybrid LM preset. ~~7~~ Minimal algorithm (`FedAvgLLM`). ~~8~~ Optional single‑node LM smoke — ad hoc.

</details>

---

## Phase 3 — Staged scaling (safest operational order)

9. **7 nodes** — correctness and stability (Phase core landed).  
10. **17 nodes** — IO / skew.  
11. **129 nodes** — full scale vs CNN+MNIST on same topology (**config flip**).

---

## Phase 4 — Acceptance checks (every scale)

12. Topology alignment (**`SLURM_NTASKS`**, **`hybrid_world_size_from_cfg`**, **`validate_hybrid_slurm_topology_alignment`**).

13. **`engine/node_results`** / **`hybrid_per_round_summary`** — **`gRPC_F1` / `gRPC_F2` asymmetry** expected (see **`README_HYDRA_RUN_OUTPUTS.md`**).

14. **Regression gate:** hybrid **MNIST** preset unchanged on **smallest layout**.

---

## Related docs

- `./CHAT_HANDOFF_HYBRID.md` — hybrid roadmap snapshot (A–D done; E/F deferred).
- `./HYBRID_USER_KNOBS_AND_ROADMAP.md` — invariants (**`num_clients + 1 == world_size`**).
- `./HYBRID_SLURM_REFERENCE.md` — Frontier submit, **`layout`**, Steps 8–9.
- `./HYBRID_TRAINING_AND_SYNC.md` — one **`__sync`** block.
- `./README_HYDRA_RUN_OUTPUTS.md` — **`node_results`**, per-round hybrid summary.

---

## Maintenance

- Record **validated Slurm CLI** snippets and **tokenizer / weight dirs** beside **launch sketch** once first jobs pass.
- Bump **status snapshot** as phases complete.
