# Hybrid pipeline — implementation artifacts (what we added and why)

This document complements **`README_FRONTIER_EXPERIMENTS.md`**. That file tells a **collaborator or PI-facing runner** *how to launch* jobs. **This** file catalogs **important source files, configs, and scripts**, explains **why they exist**, and shows **where they sit in the execution graph**.

---

## Purpose of these two READMEs (at `docs/` root)

| File | Audience / goal |
|------|------------------|
| **`README_FRONTIER_EXPERIMENTS.md`** | Ordered **Frontier commands** from engine/centralized checks through hybrid smoke, full hybrid MNIST, scale-up, optional LM — for **reproduction and demos**. |
| **`README_PIPELINE_IMPLEMENTATION_ARTIFACTS.md` (this file)** | **Engineering map**: new or touched modules, YAML, shell — **rationale** and **call graph** so maintainers can extend the pipeline without rereading the whole repo. |

---

## End-to-end execution graph (short)

```text
main.sh  →  grpc protoc  →  main.py  →  Engine (login)
                │                              │
                │                              ├─ freeze cfg → outputs/.../engine_frozen.json
                │                              ├─ resolve_slurm_ntasks (hybrid W)
                │                              └─ SlurmOnlyLauncher → sbatch omnifed_slurm_only.sh

Compute:  srun … slurm_worker --cfg-json <frozen>
                │
                ├─ communication_mode != hybrid  →  classic TorchDist FedAvg path
                │
                └─ communication_mode == hybrid   →  run_hybrid_training (slurm_hybrid_runner)
                            │
                            ├─ load topology (layout or conf_hybrid YAML) + patch addresses (slurm_hostlist)
                            ├─ rank 0: Flora gRPC parameter server (optional leader_done shutdown)
                            ├─ others: TorchMPI per facility + GrpcLeader on facility leaders
                            ├─ install_hybrid_slurm_sync → FedAvg __sync_comm = local_agg → global_agg → local_bcast
                            └─ algorithm.round_exec → node_results + hybrid_per_round_summary
```

---

## Shell and entrypoints

| Path | Role |
|------|------|
| **`main.sh`** | Export debug/Hydra env; **compile** `src/omnifed/communicator/grpc.proto`; invoke **`python -u main.py`**. Hybrid **example one-liner** commented in file header. |
| **`main.py`** | Hydra entry; builds **`Engine`**, dispatches experiment / SLURM submit. |
| **`test_scripts/slurm_frontier/hybrid_comm_smoke.slurm`** | Phase A **batch** driver for **`hybrid_comm_smoke`** (multi-node collectives + gRPC without Engine FedAvg). |
| **`test_scripts/slurm_frontier/run_hybrid_smoke_one_task.sh`** | Per-task helper invoked by smoke Slurm script (env, backend, **ROCR→HIP** device visibility). |
| **`omnifed_slurm_only.sh`** | **Generated** at repo root by **`slurm_launcher.py`** (last submit wins); inspect for **`--ntasks`**, modules, **`PYEXE`**. |

---

## Engine, communication mode, SLURM launch

| Path | Role |
|------|------|
| **`src/omnifed/engine.py`** | Login-side orchestration: **freeze** Hydra cfg for workers, **`resolve_slurm_ntasks`** for hybrid **`W`**, may **bump `slurm.nodes`** so capacity ≥ **`W`**; submits batch script. |
| **`src/omnifed/engine_communication.py`** | **`communication_mode`**, **`hybrid_world_size_from_cfg`**, **`validate_hybrid_slurm_topology_alignment`**, **`load_hybrid_cfg_for_engine`** — single place for **`world_size`** and **topology vs SLURM** consistency. |
| **`src/omnifed/slurm_launcher.py`** | Writes **`#SBATCH`**, merges stdout/stderr, **`srun`** → **`slurm_worker`**. |
| **`src/omnifed/slurm_worker.py`** | **Branch:** hybrid → **`run_hybrid_training`** then exit; else classic distributed path. |
| **`conf/base.yaml`** | Defaults for **`engine.hybrid.*`** (**`layout`**, **`topology_config`**, **`server_shutdown`**, **`leader_done_poll_sec`**, etc.) and **`slurm.*`**. |

---

## Hybrid runtime package (`src/omnifed/hybrid/`)

| Path | Role |
|------|------|
| **`slurm_hybrid_runner.py`** | **Main orchestrator:** build/load topology; map **`hybrid_rank_to_centralized_node_index`** → correct **`node_cfgs`**; start RPC server rank vs trainers; wire **`GrpcLeaderCommunicator`**, **`TorchMPIAdapter`**, **`HybridCommBridge`**; **`install_hybrid_slurm_sync`**; run **`algorithm.round_exec`**; write **`node_results`**; **`leader_done`** files for clean PS shutdown; sets **`OMNIFED_FEDERATED_CLIENT_INDEX`** / **`OMNIFED_CENTRALIZED_NODE_INDEX`** before **`instantiate(cfg.datamodule)`** (used by LM C4 sharding). |
| **`topology_builder.py`** | **`build_hybrid_topology`** — resolves **`world_size`**, facilities, RPC client ranks, communicators metadata. |
| **`hydra_loader.py`** | Compose **`conf_hybrid`** topology YAML and merge into resolved dict for Engine/worker. |
| **`topology_roles.py`** | Facility membership helpers; **`hybrid_rank_to_centralized_node_index`** (RPC server → centralized slot 0; trainers → 1…N). |
| **`slurm_hostlist.py`** | Patch **`rpc` / facility MPI** listen/connect addresses from **`SLURM_JOB_NODELIST`** ordering. |
| **`addr_env.py`** | Optional address/port overrides for multi-job debugging. |
| **`torch_mpi_adapter.py`** | Adapts Flora **`TorchMPICommunicator`** to **`BaseCommunicator`** for **`local_comm`**. |
| **`grpc_leader_comm.py`** | Facility **leaders** use Flora **`GrpcCommunicator`** as **`global_comm`** for **`aggregate`**. |
| **`comm_bridge.py`** | Passes **sample counts** from facility reduce into weighted **global** gRPC step. |
| **`hybrid_slurm_sync.py`** | Monkey-patches **`BaseAlgorithm._BaseAlgorithm__sync_comm`**: **`local_agg` → `global_agg` (leaders only) → `local_bcast`**. |
| **`hybrid_comm_smoke.py`** | Phase A **smoke** — no dataset; one collective + one gRPC round per roles. |
| **`hybrid_run_summary.py`** | After training, **one writer rank** waits for all **`node_*_results.json`**, builds **`hybrid_per_round_summary.{txt,csv}`** and run-root CSV copy (used for quick PI tables). |

---

## Flora / legacy communicators (selected)

| Path | Role |
|------|------|
| **`src/flora/communicator/grpc_communicator.py`** | Central **PS** daemon (**`id==0`** contract) and **client** aggregates; hybrid documents **`rpc.server_rank`** vs Flora **`id`**. |
| **`src/flora/communicator/torch_mpi.py`** | Process-group backend for **intra-facility** collectives. |

---

## Hydra: hybrid presets and topology files

| Path | Role |
|------|------|
| **`conf/test_hybrid_engine_contract.yaml`** | Hybrid Slurm preset using **`engine.hybrid.topology_config` → built_symmetric_2x3**; **`topology.num_clients: 6`**. |
| **`conf/test_hybrid_layout_fedavg.yaml`** | **Layout-first** preset — same 7-rank lattice via **`engine.hybrid.layout`** only (**Phase C**). |
| **`conf_hybrid/topology/built_symmetric_2x3.yaml`** | Named **2×3 + dedicated RPC** topology (**`world_size=7`**). |
| **`conf_hybrid/base.yaml`**, **`conf_hybrid/runtime/default.yaml`** | **`conf_hybrid`** package defaults (mostly parity / seeds; training uses main **`cfg`**). |
| **`tests/test_hybrid_phase_c_preset.py`** | Asserts **layout vs file** topology produce **same validation** / **`world_size`**. |

---

## LM extension (FedAvg + Hugging Face causal LM + C4 disk)

| Path | Role |
|------|------|
| **`src/omnifed/algorithm/fedavg_llm.py`** | **`FedAvgLLM`** — HF **`dict`** forward, **AdamW**, batch size from **`input_ids`**; pairs with federated LM datamodule. |
| **`src/omnifed/algorithm/__init__.py`** | Exports **`FedAvgLLM`**. |
| **`src/omnifed/data/lm_datamodule.py`** | **`build_c4_lm_datamodule`** — **`datasets.load_from_disk`**, **`Dataset.shard`** on **train** by **`OMNIFED_FEDERATED_CLIENT_INDEX`**. |
| **`src/omnifed/model/hf_causal_lm.py`** | **`load_llama_from_pretrained_checkpoint`** for Hydra **`instantiate`**. |
| **`conf/algorithm/fedavg_llm.yaml`** | Algorithm group default for **`FedAvgLLM`**. |
| **`conf/model/llama150m_hf_disk.yaml`** | Offline **`LlamaForCausalLM.from_pretrained`** via **`OMNIFED_LLAMA_WEIGHTS`**. |
| **`conf/datamodule/c4_lm_federated_disk.yaml`** | C4 disk root via **`OMNIFED_C4_DISK`**; **`num_federated_clients`** must be set **literally** per job (match **`topology.num_clients`**). |
| **`conf/test_fedavg_llm_centralized_torchdist.yaml`** | **Centralized** LM stack (TorchDist) — parent of hybrid LM preset. |
| **`conf/test_hybrid_layout_fedavg_llama150m.yaml`** | **Hybrid Slurm** + same lattice as **`test_hybrid_layout_fedavg`** + composes LM centralized bundle. |
| **`tests/test_lm_collate_utils.py`** | Unit tests for LM collate / batch helpers used by the LM datamodule path. |
| **`requirements.txt`** | Adds **`datasets`**, **`transformers`**, **`sentencepiece`** (and peers) for LM path. |

**Integration note:** LM presets **reuse** the same **`run_hybrid_training`** loop; only **model / algorithm / datamodule** groups change. Sharding depends on env vars set in **`slurm_hybrid_runner.py`**.

---

## Documentation map (**`archive/hybrid-engine-pipeline/`**)

| Path | Contents |
|------|----------|
| **`archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** | Master operations reference — Phase A/B steps, Frontier commands, **`--ntasks`** story, validated job IDs. |
| **`archive/hybrid-engine-pipeline/HYBRID_USER_KNOBS_AND_ROADMAP.md`** | User knobs schema, roadmap phases **B–F**, invariants. |
| **`archive/hybrid-engine-pipeline/HYBRID_TRAINING_AND_SYNC.md`** | **`__sync`** ordering (**`local_agg` → global → `local_bcast`**), **`round_exec`**, **`global_rounds`**, vs **`algorithm.schedules.aggregation`**. |
| **`archive/hybrid-engine-pipeline/README_HYDRA_RUN_OUTPUTS.md`** | Artifact catalog — **`node_results`**, **`hybrid_per_round_summary`**, timing columns. |
| **`archive/hybrid-engine-pipeline/README_TEST_HYBRID_ENGINE_CONTRACT.md`** | Extended preset touch map (duplicate summary in **`All_files_touched.md`**). |
| **`archive/hybrid-engine-pipeline/CHAT_HANDOFF_HYBRID.md`** | Snapshot for new sessions — phases done vs deferred (**E**, **F**). |
| **`archive/hybrid-engine-pipeline/HYBRID_LLAMA150M_C4_ROADMAP.md`** | LM data prep, **`OMNIFED_*`**, scaling table, **Frontier procedure** script block. |
| **`archive/hybrid-engine-pipeline/All_files_touched.md`** | Legacy mirror of contract README touch list. |

---

## Deferred engineering (for context)

From **`archive/hybrid-engine-pipeline/CHAT_HANDOFF_HYBRID.md`** / **`archive/hybrid-engine-pipeline/HYBRID_USER_KNOBS_AND_ROADMAP.md`**:

- **Phase E:** **`ntasks_per_node > 1`**, **`LOCAL_RANK`**, multi-GPU binding — **not** part of first LM milestone.  
- **Phase F:** Pluggable **`inner_comm` / `outer_comm`** — communicators today are **declarative** in YAML with **fixed** wiring.

---

## Maintenance

When you add a new **preset** or **communicator**, update:

1. This file’s tables (short description + integration point).  
2. **`README_FRONTIER_EXPERIMENTS.md`** if the PI-facing command sequence changes.  
3. The deep reference in **`archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** if Frontier validation or invariants change.  
