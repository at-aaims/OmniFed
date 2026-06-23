# Hybrid Slurm training loop & synchronization (FedAvg + Engine)

Plain-language companion to **`HYBRID_SLURM_REFERENCE.md`**. Applies when **`engine.communication_mode=hybrid`** and **`slurm_worker`** runs **`run_hybrid_training`**. Sections **7–9** summarize **Llama + C4** training, **`global_rounds`**, and **Flora gRPC** size / **`leader_done`** walltime knobs used on Frontier-scale LM hybrids. **Section 10** lists tunable **training**, **aggregation frequency**, **eval**, **hybrid wall**, and **scale** parameters. **Sections 11–12** cover **data sharding vs Slurm rank** and **parallelism / scaling with model size**.

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

## 7. Llama causal LMs + C4 (federated hybrid workload)

Preset pair: **`test_fedavg_llm_centralized_torchdist`** (algorithm + model + datamodule) layered with **`test_hybrid_layout_fedavg_llama150m`** (**`communication_mode=hybrid`** + Slurm/engine knobs).

### Model (**`FedAvgLLM`** + HF disk checkpoint)

- **Algorithm:** **`src.omnifed.algorithm.fedavg_llm.FedAvgLLM`** on **`torch_hf_causal_lm`**: **150 M** uses **`llama150m_hf_disk`** with **`OMNIFED_LLAMA_WEIGHTS`**; **~400 M** uses **`llama400m_hf_disk`** with **`OMNIFED_LLAMA400_WEIGHTS`**. **`algorithm.local_lr`** in the shared LM preset is **`4.0e-4`** (**`test_fedavg_llm_centralized_torchdist`**).
- **Server / topology rank 0:** **`datamodule.train: null`** — centralized **server slot** has **no** train loader; trainer ranks iterate C4 shards.
- **Sync semantics:** same triple as section 3 — local facility **`local_agg`**, **`global_agg`** on leaders via Flora, then **`local_bcast`** — with **full float32** model tensors on gRPC (no Flora compression by default).

### C4‑style corpus (**`build_c4_lm_datamodule`**)

| Item | Detail |
|------|--------|
| **On-disk layout** | Hugging Face **`DatasetDict`** from **`datasets.save_to_disk`** (`train` + `validation`). Typical Frontier path **`${OMNIFED_DATA_DIR:-...}/allenai_c4`**. Env: **`OMNIFED_C4_DISK`**. |
| **Tokenizer** | **`AutoTokenizer`** from **`OMNIFED_TOKENIZER_DIR`** (**`local_files_only=true`** — cache must exist on compute nodes). |
| **Micro-batching** | **`train_batch_size: 1`**, **`eval_batch_size: 1`**; **`max_length: 1024`**, padded sequences. |
| **Per-client train data** | With **`shard_train: true`** and **`num_federated_clients > 1`**, **train** is **`Dataset.shard(num_shards=N, index=client_idx)`** ≈ **1/N** of rows per logical client (**`OMNIFED_FEDERATED_CLIENT_INDEX`** from **`slurm_hybrid_runner`**). Shard sizes can differ by at most ~**one row**. |
| **Eval** | With **`shard_eval: false`** (C4 LM preset default), **all** clients use the **full** **`validation`** split (not partitioned). |

### Preset coupling (**Llama 150 M hybrid**)

Align **`topology.num_clients`**, **`engine.hybrid.training.dataset_total_clients`**, and **`datamodule.num_federated_clients`** as **matching literals** (avoid **`${topology.num_clients}`** inside **`c4_lm_federated_disk`** merges — **`engine_frozen.json`** / **`OmegaConf.to_container`** gotcha; see **`HYBRID_LLAMA150M_C4_ROADMAP.md`** section **J**). Lattice stays **`world_size = 7`** (6 trainers + 1 RPC) for **`built_symmetric_2×3`**.

Operational copy‑paste (**env**, staging): **`README_FRONTIER_EXPERIMENTS.md`**.

---

## 8. **Global rounds** (Engine outer loop + hybrid lifecycle)

| Knob | Role |
|------|------|
| **`global_rounds`** | Count of **`algorithm.round_exec`** cycles at the Engine layer. **`test_hybrid_layout_fedavg_llama150m`** uses **`global_rounds: 2`** for two full cross‑facility **`global_agg`** phases. |
| **`algorithm.max_epochs_per_round`** | Local epochs inside each **`round_exec`** (**`FedAvgLLM`** preset **`1`** ⇒ one pass over each shard per round; **`group_max_iters_per_epoch`** aligns across ranks **after startup**). |
| **Aggregation cadence** | Default **`schedules/aggregation/round_end.yaml`** (**`every: 1`**) ⇒ **one** **`__sync()`** (section 3) at **round end**, unless **`batch_end`** / **`epoch_end`** aggregation is enabled. |
| **`global_rounds` vs RPC rank** | PS rank **`leader_done`** sleep (section 9) must **cover** cumulative training wall across **all** **`global_rounds`**; **`engine/hybrid_grpc_leader_done/`** markers appear only **after** the last **`round_exec`** completes on facility leaders — premature **`server_sec_per_round`** or MNIST‑scale **`max_wall`** shuts Flora before **`global_agg`**. |

---

## 9. Flora **gRPC** payload limits & RPC server **wall time**

### Message size (**`src/flora/communicator/grpc_limits.py`**)

- **Legacy 100 MiB** **`grpc.max_send_message_length`** / **`max_receive_message_length`** (CNN scale) rejects Llama‑class **`SendUpdate`** payloads (**RESOURCE_EXHAUSTED**, e.g. **~859 MiB vs 104 857 600** bytes).
- **`GRPC_MAX_MESSAGE_BYTES`** raises both **daemon** (**`grpc_communicator.py`**) and **`GrpcClient`** limits together. Must be **`2147483647` (`INT32_MAX` signed)** — assigning **2147483648** (**2 GiB** as a byte count) overflows grpcio’s **`ChannelArgs`** conversion and raises **`OverflowError`** when constructing **`grpc.server(...)`**. See **`README_FRONTIER_EXPERIMENTS.md`** (LM / gRPC bullets).
- **More GPUs per facility:** one **facility‑reduced** model per **`global_agg`** upload per leader — **grpc message size does not grow with intra‑facility GPU count.**

### **`leader_done` `max_wall`** (**`slurm_hybrid_runner._run_grpc_server_only`**)

**`engine.hybrid.server_shutdown`** defaults to **`leader_done`** (**`slurm_hybrid_runner`**). Nap budget:

\[
\textbf{nap (max\_wall)} \approx \texttt{server\_run\_extra\_sec} + \texttt{global\_rounds} \times \texttt{server\_sec\_per\_round}
\]

**`test_hybrid_layout_fedavg_llama150m`:** **`server_run_extra_sec: 600`**, **`server_sec_per_round: 5000`**, **`global_rounds: 2`** ⇒ **~10 600 s ceiling** whilst leaders train (LM epochs **≫** MNIST default **~720 s** nap). Tune **`server_sec_per_round`** / **`global_rounds`** / **`slurm.time`** together; **`leader_done_poll_sec`** (default **5 s**) controls how often RPC rank scans marker files (**`OmegaConf`** **`engine.hybrid.*`**).

### Timing artefacts (**cross‑facility step**)

- **JSON rollup:** **`engine/node_results/node_*_results.json`** — **`sync/global_agg_time`**, **`sync/local_agg_time`**, **`sync/local_bcast_time`**.
- **CSV:** **`hybrid_per_round_summary.csv`** — **`gRPC_F1_ms` / `gRPC_F2_ms`** (leaders’ **`global_agg`** in ms; asymmetric ordering vs PS is normal). Example Llama‑150 M hybrid: **`global_agg_time` ~107–133 s** dominated by serialization + WAN + averaging **~sub‑GiB** tensors.

---

## 10. What you can tune — **training** vs **communication**

Use this as a map of terminology and Hydra knobs. Changing **aggregation triggers** increases or decreases **how often** **`__sync()`** runs (each **`__sync`** is always Section 3’s full trio: **`local_agg` → `global_agg` (leaders / gRPC) → `local_bcast`**).

### 10.1 Terminology (**global / local “rounds”**)

| Phrase | In OmniFed / cfg | Typical meaning |
|--------|------------------|----------------|
| **Global round** / **Fed round** | One **`algorithm.round_exec`** cycle (**Engine** advances **`ROUND_IDX`**). Controlled by **`global_rounds`** (total cycles). | One **scheduled** **`__sync()`** batch at **round end** when **`round_end`** is enabled (default presets). |
| **Local training depth per Fed round** | **`algorithm.max_epochs_per_round`** | How many **local epochs** (**passes over each trainer’s loader**) run **inside** **`round_exec`** **before** triggers like **`round_end`** are evaluated. |
| **Local sync frequency** (inside a Fed round) | **`algorithm.schedules.aggregation.*`** (**`epoch_end`**, **`batch_end`**) | Extra **`__sync()`** calls **between** **`round_end`** events — e.g. sync after **every** local epoch (**`epoch_end`**) or every **K** minibatches (**`batch_end`**). Disabled in default **`round_end.yaml`**. |
| **Global / Flora aggregation frequency** | Same as **`__sync()` frequency** | There is **no** standalone Flora pulse in hybrid FedAvg; **`global_agg`** runs whenever **`__sync()`** runs. |

Presets swap or override **`conf/algorithm/schedules/aggregation/*.yaml`** (**`round_end`**, **`epoch_end`**, **`batch_end`**) or set **`enabled`** / **`every`** / **`at`** under **`algorithm.schedules.aggregation`** (YAML comments illustrate **`at: [...]`** for sparse rounds).

### 10.2 Training-side knobs (**optimization & data throughput**)

| Parameter | Typical location | Notes |
|-----------|-----------------|--------|
| **Learning rate** | **`algorithm.local_lr`** | LM presets use **`4e-4`**; sweep if needed. |
| **Batch sizes** | **`datamodule.train_batch_size`**, **`eval_batch_size`** | LM C4 presets use **`1`** (memory); CNN hybrids may use **`128`**. |
| **Sequence length (LM)** | **`datamodule.max_length`** | Trades throughput vs memory. |
| **`num_workers`** | **`datamodule.num_workers`** | DataLoader parallelism (often **`0`** on small GPU jobs). |

These change **compute per step/epoch**. They do not by themselves decide **whether** **`__sync()`** fires — unless wall-clock length affects **`slurm.time`** / **`leader_done`** (Section 9).

### 10.3 When **`__sync()`** runs (**communication frequency**)

| Parameter | Typical location | Effect |
|-----------|-----------------|--------|
| **`global_rounds`** | **`global_rounds`** (top-level cfg) | **Total** **`round_exec`** loops ⇒ caps how many Fed rounds run. |
| **`max_epochs_per_round`** | **`algorithm.max_epochs_per_round`** | Local epochs **within** each **`round_exec`** before **`round_end`** aggregation is evaluated. |
| **`round_end`** trigger | **`algorithm.schedules.aggregation.round_end`** | **`enabled`** / **`every`** / **`at`**: **`__sync()`** after every **N‑th `round_exec`** completion (usually **`every: 1`**). |
| **`epoch_end`** trigger | **`algorithm.schedules.aggregation.epoch_end`** | See **`aggregation/epoch_end.yaml`**: disables **`round_end`**, aggregates **every epoch** ⇒ **many more** Flora + NCCL rounds per **`round_exec`**. |
| **`batch_end`** trigger | **`algorithm.schedules.aggregation.batch_end`** | See **`aggregation/batch_end.yaml`**: **`__sync()`** once per **`every`** minibatches ⇒ **heavy** overhead unless model/data are tiny. |


**Default LM hybrids:** **`max_epochs_per_round=1`** + **`round_end` only** ⇒ **one **`global_agg` per **`global_round`**. Enable **`epoch_end`** or **`batch_end`** only when you **intentionally** average more often across facilities.

### 10.4 Evaluation (**not **`__sync`**, separate schedule**)

| Parameter | Location | Role |
|-----------|----------|------|
| **`algorithm.schedules.evaluation`** | e.g. **`evaluation/standard.yaml`** | **`experiment_start`**, **`post_aggregation`**, **`experiment_end`**, **`every`** — controls **eval passes**; orthogonal to **`local_agg` / `global_agg` / `local_bcast`** unless you enable **`epoch_end`** / **`batch_end`** aggregation as well (Section 5). |

### 10.5 Hybrid infrastructure (**timeouts & payload limits**)

| Parameter | Location | Role |
|-----------|----------|------|
| **`server_run_extra_sec`**, **`server_sec_per_round`**, **`global_rounds`** | **`engine.hybrid.*`** | **`leader_done`** RPC nap budget (**Section 9**) —must cover cumulative training wall. |
| **`leader_done_poll_sec`** | **`engine.hybrid.leader_done_poll_sec`** | PS rank marker poll interval. |
| **`GRPC_MAX_MESSAGE_BYTES`** | **`src/flora/communicator/grpc_limits.py`** | Max serialized Flora message (**`INT32_MAX`** boundary); LM full-weight uploads. |

### 10.6 Scale (**who participates in facility collectives**)

| Parameter | Location | Role |
|-----------|----------|------|
| **`mpi_ranks_per_facility`**, **`num_facilities`**, **`topology.num_clients`**, **`datamodule.num_federated_clients`**, **`engine.hybrid.training.dataset_total_clients`** | Hybrid layout + presets | Larger facility ⇒ **costlier** facility **`local_agg`** / **`local_bcast`** each time **`__sync()`** runs; Flora **leader-only** **`global_agg`** pattern unchanged. Keep client-count literals aligned (**Roadmap**, section **J**). |

---

## 11. Data split logic (who sees which shards)

### 11.1 Train / eval partitioning (**C4 LM path**)

- Implementation: **`src.omnifed.data.lm_datamodule.build_c4_lm_datamodule`**.  
- **Train:** if **`shard_train`** and **`num_federated_clients > 1`**, the HF dataset uses **`train_ds.shard(num_shards=N, index=client_idx)`** with **`N = num_federated_clients`**. Rows are split into **N** disjoint shards (sizes equal within **±1** row).  
- **Eval:** default **`shard_eval: false`** ⇒ **every** logical client sees the **same full** **`validation`** split (no per-client eval partition).

### 11.2 How **`client_idx`** is chosen on hybrid Slurm

- **`OMNIFED_FEDERATED_CLIENT_INDEX`** is set in **`run_hybrid_training`** to **`cen_idx - 1`**, where **`cen_idx`** is the **`CentralizedTopology`** node index (**`0`** = server, **`1 … num_clients`** = trainers).
- **`hybrid_rank_to_centralized_node_index`** (**`topology_roles.py`**) maps **`SLURM_PROCID`** → **`cen_idx`**: the dedicated **RPC rank** maps to **`0`** (server); **trainer ranks** (all other **`SLURM_PROCID`** values) become **`1 … num_clients`** in **ascending hybrid rank order** (excluding **`rpc_server_rank`**).

**Implication:** **each trainer Slurm task** behaves as **one logical federated client** for the datamodule (its own **`Dataset.shard`** index). Tasks in the **same facility** are **different** logical clients—they **do not** subdivide **one** client’s shard across GPUs in this stack; they carry **distinct** shards and then **facility `local_agg`** combines their weighted updates.

---

## 12. “Parallelism” as model size grows (**what this stack does *not* do**)

### 12.1 Model placement (**full replica per trainer rank**)

- Hybrid workers **`instantiate(cfg.model)`** and move to **`device`** (**`slurm_hybrid_runner`**) — **one full copy of the model per trainer process** (standard **data-parallel–style replication**, not **tensor / pipeline / sequence** parallelism).
- There is **no** built-in **FSDP / ZeRO / Megatron**-style sharding in the hybrid Slurm path described here; **all** parameters participate in **`local_agg`**/**`local_bcast`** and leader **`global_agg`** as **dense** tensors.

### 12.2 What scales with bigger models (Llama 150 M → ~400 M → …)

| Concern | How it behaves in this codebase |
|--------|--------------------------------|
| **GPU RAM** | Each rank needs enough memory for **full weights + optimizer state + activations** at **`train_batch_size`**. Larger models ⇒ lower batch / shorter sequences / fewer tricks **unless** you add new parallelism outside this pipeline. |
| **Facility `local_agg` / `local_bcast`** | Collectives operate over **whole** parameter tensors aggregated across **all ranks in the facility subgroup** ⇒ **communication volume ∝ parameter count × facility_size** each **`__sync()`**. |
| **Flora `global_agg` (leaders)** | **Two** leader uploads/downloads of **serialized full state** (see **`grpc_limits`**) per **`__sync()`** in the default **2‑facility** layout — **payload ∝ model size** in **fp32** protobuf form; must stay **below** **`GRPC_MAX_MESSAGE_BYTES`**. |
| **Wall time** | Heavier forward/backward and **larger** MPI/gRPC transfers; **`server_sec_per_round`** and **`slurm.time`** usually need to grow with model class. |

### 12.3 Adding more GPUs (**throughput vs identity**)

- **More ranks** in a facility speeds up **nothing automatically** for a **single** logical client—they are **different** federated identities with **different** data shards (Section 11). You get **more samples per Fed round per facility** in aggregate, plus **higher** facility **`local_agg` / `local_bcast`** cost.
- True **single-client multi-GPU data parallel** inside one federated participant would require a **different** mapping (same **`OMNIFED_FEDERATED_CLIENT_INDEX`**, replicated loader, **`DistributedDataParallel`**, etc.) — **not** the current LM hybrid preset semantics.

---

## See also

- **`./HYBRID_SLURM_REFERENCE.md`** — Frontier checks, topology, Step 7–9.
- **`./README_TEST_HYBRID_ENGINE_CONTRACT.md`** — preset + file touch map.
- **`./README_HYDRA_RUN_OUTPUTS.md`** — run output folders and **communication timings** (**`sync/*_time`**, **`metrics_*.csv`**).
- **`./HYBRID_LLAMA150M_C4_ROADMAP.md`** — Llama + C4 integration (**datamodule literals**, section **J**).
- **`../../README_FRONTIER_EXPERIMENTS.md`** — Frontier env blocks, **`INT32_MAX`** gRPC note, **`leader_done`** / walltime narrative.
