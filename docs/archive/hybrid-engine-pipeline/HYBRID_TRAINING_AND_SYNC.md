# Hybrid Slurm training loop & synchronization (FedAvg + Engine)

Plain-language companion to **`HYBRID_SLURM_REFERENCE.md`**. Applies when **`engine.communication_mode=hybrid`** and **`slurm_worker`** runs **`run_hybrid_training`**. Sections **7–9** summarize **Llama + C4** training, **`global_rounds`**, and **Flora gRPC** size / **`leader_done`** walltime knobs used on Frontier-scale LM hybrids.

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

## See also

- **`./HYBRID_SLURM_REFERENCE.md`** — Frontier checks, topology, Step 7–9.
- **`./README_TEST_HYBRID_ENGINE_CONTRACT.md`** — preset + file touch map.
- **`./README_HYDRA_RUN_OUTPUTS.md`** — run output folders and **communication timings** (**`sync/*_time`**, **`metrics_*.csv`**).
- **`./HYBRID_LLAMA150M_C4_ROADMAP.md`** — Llama + C4 integration (**datamodule literals**, section **J**).
- **`../../README_FRONTIER_EXPERIMENTS.md`** — Frontier env blocks, **`INT32_MAX`** gRPC note, **`leader_done`** / walltime narrative.
