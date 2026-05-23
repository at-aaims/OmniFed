# Hydra experiment outputs — what each file means

Runs place artifacts under **`outputs/<YYYY-MM-DD>/<config_name>/`** (see **`hydra.run.dir`** in **`conf/base.yaml`**). Paths below are relative to **`OmniFed_VT/`** unless noted.

Hybrid Slurm + FedAvg (**`test_hybrid_engine_contract`**) is the main example; many paths also apply to **classic Slurm** / **Ray** with small differences.

---

## 1. Top-level layout (one run folder)

After **`./main.sh --config-name <name> ...`**, you typically see:

| Path / pattern | What it is |
|----------------|------------|
| **`outputs/<date>/<config_name>/`** | **Hydra run directory** (`runtime.output_dir`). Almost everything below lives here for that job. |
| **`main.log`** | Hydra main process log (login node / driver). May be **empty or short** if the parent process **exits after `sbatch`** (**`engine.mode=slurm`**). |
| **`.hydra/`** | Hydra metadata: resolved **`config.yaml`**, **`hydra.yaml`**, **`overrides.yaml`**, etc. Reproduces exact CLI. |
| **`slurm-<JOBID>.out`** | **Merged stdout** from all Slurm tasks (`#SBATCH -o` set in **`slurm_launcher.py`**). Training prints, **`[hybrid]`**, **`ROUND-START`**, RPC shutdown lines. |
| **`hybrid_per_round_summary.csv`** | **Hybrid Slurm only:** same per-round CSV as **`engine/hybrid_per_round_summary.csv`**, written at the **Hydra run root** next to **`slurm-*`** logs — easy to open in Excel or preview on **GitHub** when you push results. |
| **`slurm-<JOBID>.err`** | **Merged stderr** (warnings, NCCL/gRPC noise). |
| **`Node0.<k>/`** | One folder per **centralized topology** node name (**`CentralizedTopology`**: server `Node0.0`, clients `Node0.1` …). Holds **per-node metrics** (CSV + TensorBoard). |
| **`engine/`** | Engine bundle: **per-rank results**, optional **checkpoints**, hybrid **marker** files. |

**Not inside the dated run folder (important for Slurm):**

| Path | What it is |
|------|------------|
| **`outputs/engine_frozen.json`** | **Frozen experiment config** written on the **login node** before **`sbatch`**. **`dirname(dirname(hydra_output_dir))`** + **`engine_frozen.json`** — i.e. under **`outputs/`** for the default Hydra layout. **All tasks decode this file** onto shared storage at job start. **Last submit overwrites** if you reuse the same date/parent layout. |
| **`omnifed_slurm_only.sh`** (repo **root**) | **Generated batch script** (last write wins). Useful for debugging **`sbatch` contents**; do not treat as archival per job unless you copy it aside. |

---

## 2. `engine/` (shared / per-job under the run dir)

| Path | What it is |
|------|------------|
| **`engine/node_results/node_<bbb>_results.json`** | **Structured experiment payload** per **Slurm / Ray rank**: **`train`**, **`eval`**, **`sync`** arrays (summaries). **`node_000`** is usually the **gRPC daemon** (**stub**: `rank`, `role` only). **Clients** (`node_001` …) hold full **`sync`** breakdown. |
| **`engine/node_results/node_<bbb>_results.pkl`** | Pickle counterpart (may be absent on RPC-only stub). |
| **`engine/hybrid_per_round_summary.txt`** | **Hybrid Slurm only:** Markdown table stitched from all **`node_*_results.json`** after training — per **`round_idx`**, **`gRPC_F*`** (**`sync/global_agg_time`** ms on each facility leader), **`local_agg_*_max`** / **`local_bcast_*_max`** (facility maxima), optional **`accuracy_avg`**, **`eval_loss_avg`** (from **`eval/loss`**), plus trainer counts. Also **printed** to **`slurm-*.out`** (single writer: lowest **`topo.rpc.client_ranks`**). |
| **`engine/hybrid_per_round_summary.csv`** | **Duplicate** of run-root **`hybrid_per_round_summary.csv`** (same bytes). Keeps summaries next to **`node_results`**. Env: **`OMNIFED_HYBRID_SUMMARY_POLL_SEC`** (wait for JSON), **`OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC`**. |
| **`engine/hybrid_grpc_leader_done/rank_<r>.done`** | Marker files (**Step 8**): **gRPC facility leaders** write **`leader_done`** when training finishes so the RPC rank can **`grpc_shutdown`**. Presence = clean shutdown path (**not** every training rank). |
| **`engine/ckpt/`** | Checkpoint dir when **`slurm.checkpoint_dir`** is set or default under **`engine/ckpt`** (preemption / resume experiments). |
| **`engine/rank0_gpu_memory.log`** | Optional periodic GPU memory log (**classic `slurm_worker`** rank 0 path); hybrid runs may or may not create it depending on code path. |

### Hybrid per-round summary table (`hybrid_per_round_summary.{txt,csv}`)

**Hybrid Slurm only.** After training, one writer rank (the **minimum** of **`topo.rpc.client_ranks`**) waits until **`engine/node_results/`** has **`world_size`** `node_*_results.json`, then merges every trainer JSON (skipping the **`role: hybrid_grpc_server`** stub) and emits the table. A copy appears at **run root** as **`hybrid_per_round_summary.csv`** for uploads; **`engine/`** holds the **`.csv`** sibling and **`.txt`**. The Markdown block is **also printed** to **`slurm-<JOBID>.out`**.

Rows are keyed by **`round_idx`** (**last **`sync`** row wins** per rank per round).

| Columns | Meaning |
|---------|---------|
| **`gRPC_F*j*_ms`** | **`sync/global_agg_time`** from facility *j*’s **leader** (stored as **seconds**, printed as **ms**). |
| **`local_agg_F*j*_max_ms`** / **`local_bcast_F*j*_max_ms`** | **Facility max** of **`sync/local_agg_time`** / **`sync/local_bcast_time`** over all ranks in facility *j*. |
| **`accuracy_avg`** / **`n_acc_trainers`** | Mean over trainers of **`eval`** keys containing **`accuracy`**. **`—`** if nothing logs accuracy (FedAvg MNIST often only **`eval/loss`**). |
| **`eval_loss_avg`** / **`n_eval_trainers`** | Mean over trainers of per-trainer **average **`eval/loss`**** for that round (many **`eval`** rows per round are averaged inside each trainer first). |

**Why `gRPC_F1_ms` and `gRPC_F2_ms` often look unrelated (short)**  

Each column is **`sync/global_agg_time` on one facility leader** — wall time for **that leader’s** Flora step **`send_update` + pull averaged model**, not half of a symmetric MPI collective. Flora runs a **central parameter server**: one leader usually **returns from `SendUpdate` early** and then spends **`GetUpdatedModel` waiting** (server lock, peer upload, deserialization, averaging), while the **other leader often blocks longer inside its own `SendUpdate`** RPC when its update is the one that completes the round (heavy server-side merge before the RPC returns). So **different leaders routinely report very different durations for the same federated round** even though **`local_agg_*`** stays similar (**same pattern inside each site**).

*Example shape (illustrative, two facilities):*

| round_idx | `gRPC_F1_ms` | `gRPC_F2_ms` | note |
|-----------|---------------|---------------|------|
| 0 | ~127 500 | ~19 | first global step often cold (protobuf build, transports, JIT); largest skew if one leader waits or serializes longest |
| 1+ | ~200–400 | ~18 | steady state: ~10–20 × gap is typical “first vs second updater / waiter” asymmetry |

For a **single number per round**, use **`max(gRPC_F1_ms, gRPC_F2_ms)`** (manual or spreadsheet), not **`mean`** — that matches **`sync/global_agg_time`** semantics (**per‑process** timelines — see **§4** *Per‑rank variability*).

**Environment (writer rank):**

| Variable | Role |
|---------|------|
| **`OMNIFED_HYBRID_SUMMARY_POLL_SEC`** | Max wait for all JSON files (default **`max(90, 5 × world_size)`** → large **`world_size`** can imply **many minutes**). |
| **`OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC`** | Sleep between checks (default **`0.5`**). |

**Offline regeneration (login node, same layout as Frontier)**

Pass the **Hydra run directory** (folder containing **`engine/node_results`**), **not** the repo root. Set **`PYTHONPATH`** to the OmniFed checkout.

```bash
cd /lustre/orion/gen150/scratch/$USER/OmniFed_VT/outputs/2026-05-22/test_hybrid_layout_fedavg   # your run dir
export PYTHONPATH="/lustre/orion/gen150/scratch/$USER/OmniFed_VT"

python -c "
import os
from omegaconf import OmegaConf
from src.omnifed.hybrid.topology_builder import build_hybrid_topology
from src.omnifed.hybrid.hybrid_run_summary import write_hybrid_slurm_per_round_summary
# Topo must match the Slurm job (facilities / members). Example: world_size = 129 = 2×64 + RPC.
topo = OmegaConf.create(build_hybrid_topology(num_facilities=2, mpi_ranks_per_facility=64))
write_hybrid_slurm_per_round_summary(os.getcwd(), topo=topo, world_size=129, rpc_server_rank=0, rank_writer=0)
"
```

If **`engine/node_results`** is absent from **`hydra_out_dir`**, **`write_hybrid_slurm_per_round_summary`** exits with a message instead of waiting for the poll timeout.

---

## 3. `Node0.<k>/` — training / evaluation metrics (CSV + TensorBoard)

Each **training-facing** centralized node logs here (`MetricLogger`), **not** the RPC-only process.

| File | What it is |
|------|------------|
| **`metrics_full.csv`** | **Wide log**: every metric row with **`global_step`**, **`round_idx`**, **`epoch_idx`**, **`batch_idx`**, **`agg_ctx`** (`train` / `eval` / `sync` / …), **`metric_key`**, **`metric_val`**. Good for spreadsheets or **`grep`**. |
| **`metrics_train.csv`** | **Training-only** rollup ( decorator context **`train`**). |
| **`metrics_eval.csv`** | **Eval-only** rollup. |
| **`metrics_sync.csv`** | **`sync`** metric context — includes **`sync/time_total`** and flushed rows tied to synchronization. Filter **`metric_key`** for **`sync/...`** (see §4). |
| **`events.out.tfevents.*`** | **TensorBoard** event files (same **`log_dir`**). `tensorboard --logdir Node0.1` (etc.). |

**Naming:** **`Node0.0`** = centralized **server** slot in **`CentralizedTopology`** (often **`train`** disabled — folder may exist but be sparse). **`Node0.1` … `Node0.N`** = **clients** in numbering order (**not** identical to **`SLURM_PROCID`** when hybrid RPC rank remaps **`node_cfgs`** — see **`hybrid_rank_to_centralized_node_index`**).

---

## 4. Communication timing — where it is logged

FedAvg **`__sync()`** wraps **facility MPI** and **`global_agg` (Flora gRPC)** with **`track_model_operation`** (**`algorithm/base.py`**). Logged metric names:

| Metric suffix | Meaning |
|----------------|--------|
| **`sync/local_agg_time`** | Time for **within-facility** weighted **`SUM`** / **`all_reduce`** (**Torch MPI adapter**). |
| **`sync/global_agg_time`** | Time for **`GrpcLeaderCommunicator.aggregate`** (one **leader ↔ PS** Flora step). **Only meaningful on facility-leader ranks** that run **`global_comm`**. |
| **`sync/local_bcast_time`** | Time for **`local_comm.broadcast`** after global merge (**facility-internal** broadcast). |
| **`sync/time_total`** | Duration of entire **`sync`** *metric context* flush window — often **much larger** than the sum of the three bullets above because **`sync/time_total`** can include **`__sync()`** scaffolding; for **aggregation-only** microseconds, rely on **`local_agg_time`**, **`global_agg_time`**, **`local_bcast_time`**. |
| **`sync/*_param_norm_before` / `_after` / `_delta`** | Scalar **norm of weights** before/after each phase (sanity / size of change — **not** bandwidth). |
| **`sync/*_params_changed`** / **`_buffers_changed`** | **Hash** booleans (**1** / **0**) — did weights or BatchNorm buffers change. |

**Where to read them**

1. **`engine/node_results/node_<bbb>_results.json`** → **`"sync"`** array (rolled up per aggregation event). Easiest for **“how long did gRPC vs local agg take?”** on **leader** JSON. For a **facility‑wise table** merged from **all ranks**, see **§2** (*Hybrid per-round summary table*).
2. **`Node0.<client>/metrics_sync.csv`** and **`metrics_full.csv`** (filter **`agg_ctx`** = **`sync`** or keys **`sync/local_agg_time`**, …). Useful for timelines with **`global_step`**.

For **overlap with evaluation**, **`sync/time_total`** is the duration of the entire **`__sync()`** method (**`MetricLogger.context("sync", duration_key="time_total")`**), including **`__pre_sync`** (optional pre-eval), **`__sync_comm`** (local / global / bcast phases), and **`__post_sync`** (e.g. **post_aggregation eval** — often seconds). Prefer **`sync/local_agg_time`**, **`sync/global_agg_time`**, **`sync/local_bcast_time`** for **communication-only** comparison.

**`slurm-*.out`** may also contain **`LOCAL_AGG`** / **`GLOBAL_AGG`** / **`LOCAL_BCAST`** lines from **`track_model_operation`** (all ranks merged — noisy but confirms ordering). Compare **`metrics_eval.csv`** when **`sync/time_total`** looks unexpectedly large.

### Per-rank variability (why **`node_001`** ≠ **`node_002`** …)

Each **`node_<bbb>_results.json`** is **one Slurm task** (**`SLURM_PROCID` = `bbb`**), often on a **different host**. Timings are **that process’s wall clock**, not a single shared timeline.

| Effect | Cause |
|--------|--------|
| **`train/`** / **`eval/`** times differ | Node-to-node GPU / IO / OS jitter on **seven** machines. |
| **`global_agg_*`** on **two** ranks only | **Facility leaders** run **`GrpcLeaderCommunicator`**; **workers** **omit** **`global_agg`**; **`node_000`** is **RPC-only** (**stub**). |
| **`global_agg_time`** differs **leader vs leader** | One logical global step still ran — check **`sync/global_agg_param_norm_after`** (**should match** across ranks). Asymmetry is often **server-side ordering**, **TCP path length**, or **queueing**; use **`max(leader global_agg_time)`** or profilers for a conservative bound. |
| **`local_bcast`**: **`param_norm_delta` ≈ 0** vs **large** | **Leader** often **unchanged** (source of broadcast); **workers** **gain** new weights ⇒ **nonzero Δ**. |
| **`sync/time_total`** spreads (e.g. **2.2 s** vs **6.8 s**) | Full **`__sync()`** includes **`post_aggregation` eval**; **stragglers** inflate **`time_total`** even when **`local_agg`/`global_agg`/`local_bcast`** sub-timers are small. |

---

## 5. Quick “success” checklist (hybrid Slurm jobs)

| Check | Where |
|--------|--------|
| Scheduler exit **`0:0`** | `sacct -j JOBID` |
| Result files **`node_000`** … **`node_{W−1}`** (**`W =`** hybrid **`world_size`**) | **`engine/node_results/`** |
| Optional per-round rollup (gRPC / local timings, **`eval/loss`** mean) | **`hybrid_per_round_summary.csv`** and **`engine/hybrid_per_round_summary.*`** |
| Hybrid markers (**leaders**) | **`engine/hybrid_grpc_leader_done/`** |
| Shutdown **leader_done** | **`grep shutdown_mode slurm-*.out`** |

---

## 6. Mental model diagram (text)

```text
Login node Hydra dir: outputs/<date>/<config>/
├── .hydra/                 ← exact config snapshot
├── main.log               ← driver (often exits after sbatch)
├── slurm-<id>.{out,err}   ← all ranks interleaved stdout/stderr
├── hybrid_per_round_summary.csv  ← hybrid-only per-round table (duplicate of engine copy)
├── Node0.{0..6}/           ← MetricLogger CSV + TensorBoard per logical node slot
└── engine/
    ├── node_results/       ← per-SLURM_PROCID JSON/PKL summaries (sync timings here)
    ├── hybrid_per_round_summary.{txt,csv}  ← hybrid rollup + Markdown
    └── hybrid_grpc_leader_done/ ← leader_done handshake (hybrid only)

Sibling of date folder scope:
└── outputs/engine_frozen.json   ← replicated to nodes at job start (shared path)
Repo root:
└── omnifed_slurm_only.sh       ← regenerated submit script template
```

---

## 7. Related docs

- **`docs/HYBRID_SLURM_REFERENCE.md`** — Frontier validation, topology, **`sbatch`** notes.  
- **`docs/HYBRID_TRAINING_AND_SYNC.md`** — when **`local_agg` → global gRPC → `local_bcast`** run vs **`round_end`**.  
- **`docs/README_TEST_HYBRID_ENGINE_CONTRACT.md`** — preset CLI + codebase touch map.  
- **`src/omnifed/hybrid/hybrid_run_summary.py`** — implementation of **`hybrid_per_round_summary.*`**.
