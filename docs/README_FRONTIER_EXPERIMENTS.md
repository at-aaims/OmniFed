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
7. Scale the hybrid run up (e.g. 129 ranks).
8. Run the Llama-150M + C4 hybrid LM experiment (and scale it the same way in principle).
9. Read and interpret result artifacts.

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

### Pre-stage LM data (only if §8 Llama + C4 will run)

This is a separate, heavier story: C4 **`save_to_disk`**, Mistral tokenizer cache on disk, Llama **`snapshot_download`**. The literal script block we copied from lives in **`./archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`** under **“Frontier procedure (login → data → submit)”**.

After we finished that on the login node, every LM shell session reused three exports (`OMNIFED_C4_DISK`, `OMNIFED_TOKENIZER_DIR`, `OMNIFED_LLAMA_WEIGHTS`; exact subdirectory names match whatever paths we parked under **`omnifed_data/`** on scratch). Skip §8 entirely if this never runs.

---

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
   slurm.gres=null
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
   slurm.gres=null
```

Acceptance mirrored §5 (**same **`--ntasks`** semantics**) while Hydra’s folder naming reflected **`test_hybrid_layout_fedavg`**.

**Operational note.** If **`slurm.nodes × ntasks_per_node`** started below **`world_size`**, **`engine.py`** printed **`[Engine] slurm.nodes raised …`** and bumped capacity while **`#SBATCH --ntasks=W`** stayed laser-focused on **`W`**. We kept returning to **`./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** §**4.3** whenever collaborators asked “why **`nodes`** flipped during submit?”

---

## 7. Larger-scale hybrid demo (e.g. 129 ranks)

We never needed new Python for this tier—pure Hydra choreography. MNIST hybrids on this branch already validated:

- **`topology.num_clients = 128`**
- **`engine.hybrid.layout.mpi_ranks_per_facility = 64`** (**2 × 64** trainers)
- **`slurm.nodes = 129`**, **`slurm.ntasks_per_node = 1`** ⇒ **129** tasks ⇒ **`W = 129`**
- **`engine.hybrid.training.dataset_total_clients = 128`** (stay aligned with **`topology.num_clients`**)

Standing rule **`SLURM_NTASKS == hybrid_world_size`** and **`topology.num_clients == W − 1`**. Appendix-style narrative for knobs sits in **`./archive/hybrid-engine-pipeline/HYBRID_USER_KNOBS_AND_ROADMAP.md`**.

---

## 8. Llama + C4 hybrid LM (**150 M default; ~400 M sibling preset**)

**Why this step exists.** We deliberately **reuse** the hybrid Runner—only Hydra swaps **algorithm / model / datamodule**:

| Piece | Swap |
|--------|------|
| **Algorithm** | **`FedAvgLLM`** — HF dict-style forward, **AdamW**, batch inferred from **`input_ids`** |
| **Model** | Same **`load_llama_from_pretrained_checkpoint`**; weights via **`OMNIFED_LLAMA_WEIGHTS`** (150 M presets) **or** **`OMNIFED_LLAMA400_WEIGHTS`** (**`test_*_llama400m`**) |
| **Datamodule** | **`datasets.load_from_disk`** C4 shards; **`Dataset.shard`** on **train**, driven by **`OMNIFED_FEDERATED_CLIENT_INDEX`** seeded in **`slurm_hybrid_runner.py`** |

Pre-flight mirrored §2 (**login node pulls Hub assets** once). Afterwards each **150 M** submission used **`./main.sh --config-name test_hybrid_layout_fedavg_llama150m`** plus **`OMNIFED_C4_DISK`**, **`OMNIFED_TOKENIZER_DIR`**, **`OMNIFED_LLAMA_WEIGHTS`**.

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

**Llama ~400 M (same stack, isolated weights preset).** We added **`test_hybrid_layout_fedavg_llama400m`** (**`conf/test_hybrid_layout_fedavg_llama400m.yaml`**) and **`OMNIFED_LLAMA400_WEIGHTS`** so checkpoints never overwrite **`OMNIFED_LLAMA_WEIGHTS`** semantics for 150 M. Reuse **`OMNIFED_C4_DISK`** / **`OMNIFED_TOKENIZER_DIR`**; swap **`export OMNIFED_LLAMA400_WEIGHTS=<offline_hub_tree>`**, then **`--config-name test_hybrid_layout_fedavg_llama400m`** with the same Slurm block (expect slower steps ⇒ fewer **`global_rounds`** or more **`slurm.time`**).

**Structural footnote (after hitting `InterpolationKeyError`):** **`datamodule.num_federated_clients`** must match **`topology.num_clients`** as **literals** in each preset / CLI overrides (`conf/datamodule/c4_lm_federated_disk.yaml` uses **`???`** until the preset fills it). We avoided **`${topology.num_clients}`** in that datamodule because **`OmegaConf.to_container`** during **`engine_frozen.json`** could fail. **`./archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`** §**J** captures the remediation.

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

Centralized TorchDistrib on Frontier anchored that **Sabiha’s SLURM engine path** behaved. Hybrid smoke guaranteed **facility collectives + gRPC** behaved before layering FedAvg. **`test_hybrid_engine_contract`** and **`test_hybrid_layout_fedavg`** then showed the Engine loop running **either** **`topology_config` file** OR inline **`layout`**, same lattice. Larger MNIST hybrids proved **only Hydra knobs** scaled us to **129** ranks once **`W`** / **`topology.num_clients`** / **`training.dataset_total_clients`** marched in lock-step. Llama + C4 finally proved “swap **FedAvg**/CNN blocks, keep **`run_hybrid_training`** untouched,” with offline staging as the solitary operational caveat.
