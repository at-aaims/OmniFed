# Hybrid Slurm training loop & synchronization (FedAvg + Engine)

Plain-language companion to **`HYBRID_SLURM_REFERENCE.md`**. Applies when **`engine.communication_mode=hybrid`** and **`slurm_worker`** runs **`run_hybrid_training`** (e.g. **`test_hybrid_engine_contract`**).

---

## 1. What runs where

| Role | Typical hybrid rank (`built_symmetric_2×3`) | Training? | Torch MPI subgroup | gRPC global step? |
|------|--------------------------------------------|-----------|---------------------|-------------------|
| Central parameter server (**Flora daemon**) | **Rank 0** | No (**only server**) | *(none)* | **Receives** leader updates (**not** listed as “client” communicator `id`; server uses Flora **`id=0`**). |
| Facility **leader** | **Rank 1** (fac 1), **Rank 2** (fac 2) | Yes | Yes (**local_rank 0** in each facility) | **Yes** — **`GrpcLeaderCommunicator`**. |
| Facility **workers** | e.g. **3, 4** and **5, 6** | Yes | Yes | **No** — **`global_comm` is `None`**; they still join **facility** collectives. |

---

## 2. Local training (minibatches and epochs)

- **Step / minibatch:** one forward + backward over a **DataLoader batch** (config **`datamodule.train.batch_size`**, commonly **128**).  
  Between aggregations wired from **`algorithm.schedules.aggregation`**, there is **no** required **facility** or **gRPC** round after **each** step—only local optimizer updates (subject to **`batch_end` / `epoch_end`/`round_end`** being enabled).
- **Epoch:** one logical pass through the **trainer’s train loader** (`BaseAlgorithm.__train_epoch`), driven by **`group_max_iters_per_epoch`** (synced maximum across ranks in group after startup).
- **FedAvg “fed round”** (`algorithm.round_exec`): for each **`round_idx`**, the code runs **`max_epochs_per_round`** local epochs, then optionally runs **aggregation** (`__sync`) depending on **`schedules.aggregation`**.

Repeat **`global_rounds`** fed rounds at the **`Engine`/cfg** layer.

---

## 3. When **facility** and **gRPC** communication run (one block)

Each time **`BaseAlgorithm.__sync()`** runs and the hybrid patch applies **`hybrid_slurm_sync._hybrid_slurm__sync_comm`**, **one** sync block runs in this **fixed order**:

1. **`local_agg` (facility)** — sample-weight scaling + **`local_comm.aggregate(..., SUM)`**.  
   Under Torch MPI (`torch_mpi_adapter`), weights are **`all_reduce`(SUM)** over **that facility only** (~weighted FedAvg within the institution).
2. **`global_agg` (between facilities)** — **only ranks with `GrpcLeaderCommunicator`** (**facility leaders**) call **`aggregate(model, SUM)`**, which invokes **Flora gRPC** to the central server (**mean-style combine** on server with **`compute_mean=True`** — see **`grpc_leader_comm`**). Workers **skip** this block.
3. **`local_bcast` (facility)** — **`local_comm.broadcast`** so **every** rank in that facility gets the **post–global** model (leaders originate broadcast as **`src=local_rank 0`** in the subgroup).

**Takeaway:** for every **`__sync`** invocation, **`local_agg` → global (leaders only) → `local_bcast`** execute **together** — same count (**one trio per **`__sync`**). There is no extra standalone “gRPC only” pulse in normal hybrid FedAvg vs that block without also going through the schedule that calls **`__sync`**.

Implementation: **`src/omnifed/hybrid/hybrid_slurm_sync.py`**.

---

## 4. What controls frequency (replacement for manual **`comm_freq`**)

Older Flora CLI demos often used a **`comm_freq`**-style knob. **OmniFed + Hydra** uses **`algorithm.schedules.aggregation`** (plus **`global_rounds`** and **`max_epochs_per_round`**):

| Knob | Role |
|------|------|
| **`conf/algorithm/schedules/aggregation/round_end.yaml`** | If **`round_end.enabled: true`** and **`every: 1`**, FedAvg calls **`__sync()`** after **finishing the local epoch loop** **inside** each **`round_exec`** (when **`schedules.aggregation.round_end()`** is true). **`epoch_end` / `batch_end`** default **off** in that file — so **no** per-epoch / per-batch **`__sync`** unless you enable them. |
| **`algorithm.max_epochs_per_round`** | How many **local training epochs** run **before** that **`round_end`** check can fire (for default **`round_exec`**). |
| **`global_rounds`** | How many **`round_exec`** cycles the experiment runs (**Engine** FedAvg outer loop). |

So “how often global + facility merge?” = **once per **`__sync`**, **`__sync` count** is driven primarily by **`round_end`** (and **`global_rounds`**) *not* a single Flora **`comm_freq`** field in OmniFed presets.

---

## 5. Evaluation vs training sync (**separate schedule**)

**`algorithm.schedules.evaluation`** (**e.g. `evaluation/standard.yaml`**) gates **evaluation passes** (**experiment_start**, **post_aggregation**, **experiment_end**). Evaluation is **orthogonal** to the **`local_agg` / `global_agg` / `local_bcast`** trio—you can have eval epochs without triggering extra aggregation blocks unless **`batch_end`** / **`epoch_end`** aggregation is enabled.

---

## 6. Where to read metrics after a job

Under the Hydra run directory: **`Node0.<rank>/metrics_*.csv`** (CSV/trace) and **`engine/node_results/node_*_results.json`** (rolled-up **`train` / `eval` / `sync`** snapshots). **`sync/*`** fields reflect **`local_agg`**, **`global_agg`**, **`local_bcast`** phases when logging is enabled.

---

## See also

- **`docs/HYBRID_SLURM_REFERENCE.md`** — Frontier checks, topology, Step 7–9.  
- **`docs/README_TEST_HYBRID_ENGINE_CONTRACT.md`** — preset + file touch map.
- **`docs/README_HYDRA_RUN_OUTPUTS.md`** — run output folders and **communication timings** (**`sync/*_time`**, **`metrics_*.csv`**).
