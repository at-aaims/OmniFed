# Hybrid pipeline — what we wired in and where it landed

**Scope of this section:** companion to **`README_FRONTIER_EXPERIMENTS.md`**. That file walks through OLCF Frontier in the chronological order **we experimented**; **this one** inventories the **YAML / Python / shell** pieces introduced or touched so the hybrid path could ride the same Engine **`slurm_worker`** door as TorchDistrib—not a second codebase, **not** a fork of Flora, just branching behavior once **`communication_mode`** flips.

**Suggested placement:** right after **`README_FRONTIER_EXPERIMENTS.md`** inside `docs/` (or as a sibling chapter in a combined handbook) so collaborators who already reproduced jobs can pivot to implementation detail without drowning in **`./archive/hybrid-engine-pipeline/`**.

**Relationship between the two readmes.**

- **`README_FRONTIER_EXPERIMENTS`** answers “**what ran**,” with enough commands to rerun the ladder (central → smoke → contract → layout → scale → LM).

- **`README_PIPELINE_IMPLEMENTATION_ARTIFACTS`** (this file) answers “**what files implement that ladder**”—the Engine freeze path, validators, communicator adapters, patched FedAvg **`__sync`**, hybrid summary stitching, LM sharding hooks.

Older write-ups (**`HYBRID_SLURM_REFERENCE`**, knobs roadmap, artefact glossary) deliberately live under **`./archive/hybrid-engine-pipeline/`**. We skim them here **by filename** rather than rewriting their prose.

---

## How execution actually flowed (Condensed snapshot)

Once **`main.sh`** regenerated protobufs and spawned **`Engine`**, the login-node story always paired **`engine_frozen.json`** with **`sbatch`**. Hybrid **only** rewired compute-side **`slurm_worker`** when **`communication_mode`** said so—no alternate batch template.

```text
main.sh  →  grpc protoc  →  main.py  →  Engine (login)
                │                              │
                │                              ├─ freeze cfg → outputs/.../engine_frozen.json
                │                              ├─ resolve_slurm_ntasks → hybrid W
                │                              └─ SlurmOnlyLauncher → sbatch omnifed_slurm_only.sh

Compute:  srun … slurm_worker --cfg-json <frozen>
                │
                ├─ communication_mode != hybrid  →  classic TorchDist FedAvg path
                │
                └─ communication_mode == hybrid   →  run_hybrid_training (slurm_hybrid_runner)
                            │
                            ├─ load topology (layout or conf_hybrid YAML) + patch addresses (slurm_hostlist)
                            ├─ dedicated RPC rank: Flora gRPC parameter server (shutdown via leader_done)
                            ├─ trainers: TorchMPI per facility + gRPC-only on facility leaders
                            ├─ install_hybrid_slurm_sync patches FedAvg: local_agg → global_agg → local_bcast
                            └─ algorithm.round_exec → engine/node_results + hybrid_per_round_summary
```

Validators (**`validate_hybrid_slurm_topology_alignment`**, **`hybrid_world_size_from_cfg`**) deliberately sit beside **`resolve_slurm_ntasks`** because **we burnt time** chasing **`SLURM_NTASKS != W`** and **`topology.num_clients + 1 != world_size`**; keeping those checks centralized meant login submit and worker bootstrapping shared one story.

---

## Shell + entry scripts

| Artifact | Why it exists / how we used it |
|----------|--------------------------------|
| **`main.sh`** | Standard driver: protobuf compile, **`PYTHONUNBUFFERED`**, **`HYDRA_FULL_ERROR`**, forwarded Hydra overrides; hybrid **`main.sh` comments accumulated** Frontier one-liners as presets stabilized. |
| **`main.py`** | Hydra entry that instantiates **`Engine`**; untouched philosophically—we only leaned on richer **`cfg.engine.hybrid`** + **`cfg.engine.communication_mode`**. |
| **`test_scripts/slurm_frontier/hybrid_comm_smoke.slurm`** | Thin Slurm façade over **`hybrid_comm_smoke.py`** during Phase-A—proof of TorchMPI lanes + Flora without FedAvg scaffolding. |
| **`test_scripts/slurm_frontier/run_hybrid_smoke_one_task.sh`** | Per-task glue (env, ROCm HIP visibility tweaks) we iterated on beside OLCF ROCm quirks. |
| **`omnifed_slurm_only.sh`** (**generated**) | Emitter output from **`slurm_launcher.py`**; **`--ntasks=W`** scrutiny happened here repeatedly because **`nodes`** knobs misled collaborators even when **`W`** matched. |

---

## Engine lane + communication mode (**Sabiha’s SLURM path**, extended here for hybrid)

| Artifact | Narrative hook |
|----------|----------------|
| **`src/omnifed/engine.py`** | Login orchestration inherited from SLURM Engine work: **`engine_frozen.json`**, launcher emission, optional **`nodes` bump**. Hybrid threaded **`resolve_slurm_ntasks`** so **`#SBATCH --ntasks=W`** tracked **`layout` / `topology_config`**. |
| **`src/omnifed/engine_communication.py`** | Where **`communication_mode`** was interpreted and **`hybrid_world_size_from_cfg`** / **`validate_hybrid_slurm_topology_alignment`** / **`load_hybrid_cfg_for_engine`** landed—replacing scattered ad hoc checks. |
| **`src/omnifed/slurm_launcher.py`** | Still emits **`sbatch`** / **`srun`** into **`python -m src.omnifed.slurm_worker`**; template tweaks (**merged logs**, Frontier **`setup_lines`**, etc.) grew beside OLCF quirks. |
| **`src/omnifed/slurm_worker.py`** | Branches on **`communication_mode`**: hybrid delegates to **`run_hybrid_training`** and exits early; centralized path unchanged. |
| **`conf/base.yaml`** | Collected **`engine.hybrid.*`** defaults (**`leader_done_poll_sec`**, **`server_shutdown`**, …) once **`leader_done`** replaced naive sleep-only PS shutdown notes on Frontier. |

---

## Hybrid runtime package (**`src/omnifed/hybrid/`**)

Most of this surfaced while stitching Flora’s **`TorchMPICommunicator`** lanes to OmniFed **`BaseAlgorithm.__sync`** without breaking FedAvg semantics.

| Module | Narrative hook |
|--------|----------------|
| **`slurm_hybrid_runner.py`** | The spine: topology resolution, **`hybrid_rank_to_centralized_node_index`** (RPC rank maps to OmniFed **`Node0.0`** conventions), communicator wiring per rank, **`install_hybrid_slurm_sync`**, **`algorithm.round_exec`**. **`OMNIFED_FEDERATED_CLIENT_INDEX`** / **`OMNIFED_CENTRALIZED_NODE_INDEX`** landed ahead of **`instantiate(cfg.datamodule)`** so C4 federation stayed reproducible **without slicing datasets by hand per rank in YAML**. |
| **`topology_builder.py`** | **`build_hybrid_topology`** so symmetric / asymmetric **`mpi_ranks_per_facility`** lists matched Figure-2-esque YAML without hand-written rank enums. |
| **`hydra_loader.py`** | Merged **`conf_hybrid`** assets when **`topology_config`** pointed into that package. |
| **`topology_roles.py`** | Role queries + **`hybrid_rank_to_centralized_node_index`**—critical once centralized CFG slots diverged from raw **`SLURM_PROCID`**. |
| **`slurm_hostlist.py`** **`addr_env.py`** | Materialized sane RPC / intra-facility **`MASTER_*`** tuples from Frontier hostlists after we chased connection failures tied to bogus defaults. |
| **`torch_mpi_adapter.py`** **`grpc_leader_comm.py`** **`comm_bridge.py`** | Translator layer bridging Flora communicators → OmniFed aggregator expectations; **`comm_bridge`** carried sample weights so Flora’s global **`SUM`** behaved like federated averages. |
| **`hybrid_slurm_sync.py`** | Monkey-patch replacing vanilla **`torch.distributed`** all-reduce choreography with **`local_agg`** → Flora **`global_agg` (leaders)** → **`local_bcast`** in one inseparable **`__sync`**. |
| **`hybrid_comm_smoke.py`** | Deliberately tiny harness used before **`run_hybrid_training`** matured. |
| **`hybrid_run_summary.py`** | Post-training stitcher generating **`hybrid_per_round_summary.(csv|txt)`** plus run-root **`hybrid_per_round_summary.csv`** for PI-friendly spreadsheets—grown after we struggled to diff **`sync/global_agg_time`** across leaders manually in JSON shards. |

---

## Flora intersections we leaned on (**no fork**—comments only where behaviour surprised us)

| Path | Narrative hook |
|------|----------------|
| **`src/flora/communicator/grpc_communicator.py`** | Flora’s **`GrpcCommunicator`** still expects **`id == 0`** for daemon wiring even when OmniFed’s RPC-only rank is elsewhere (**`topology.rpc.server_rank`**); we documented rather than forking Flora. |
| **`src/flora/communicator/torch_mpi.py`** | Facility intra-communicators—unchanged mechanically, reused through **`torch_mpi_adapter`**. |

---

## Hydra surfaces we leaned on (**hybrid presets + topology artefacts**)

| File | Narrative hook |
|------|----------------|
| **`conf/test_hybrid_engine_contract.yaml`** | First end-to-end hybrid preset pinning **`topology_config`** to **`built_symmetric_2x3`**. |
| **`conf/test_hybrid_layout_fedavg.yaml`** | Same **`W`** as contract preset but spelled entirely via **`engine.hybrid.layout`**, proving ergonomics validators covered both YAML styles. |
| **`conf_hybrid/topology/built_symmetric_2x3.yaml`** | Named reproducible **`2 × 3 + RPC`** lattice we referenced in Frontier validation jobs. |
| **`conf_hybrid/base.yaml`** **`runtime/default.yaml`** | Minor defaults / seeds—primary training toggles stayed in main **`cfg`**. |
| **`tests/test_hybrid_phase_c_preset.py`** | Regression ensuring **`compose`** parity between **`layout`** vs **`topology_config` interpretations. |

LM-specific Hydra tails (**`fedavg_llm`**, **`llama150m_hf_disk`**, **`c4_lm_federated_disk`**, **`test_hybrid_layout_fedavg_llama150m`**) sit in the LM table below—we split them consciously so MNIST regressions stayed **`test_hybrid_*`** only.

---

## LM extension (**FedAvgLLM**, C4 disk, Llama weights)

FedAvg tensors still averaged exactly as CNN runs; causal LM swapped forward/loss shaping + adam + sharded **`Dataset`**.

| Artefact | Narrative hook |
|----------|----------------|
| **`src/omnifed/algorithm/fedavg_llm.py`**, **`conf/algorithm/fedavg_llm.yaml`** | Thin FedAvg descendant accepting HF **`dict`** batches + AdamW—kept separate from **`fedavg.py`** on purpose. |
| **`src/omnifed/data/lm_datamodule.py`**, **`conf/datamodule/c4_lm_federated_disk.yaml`**, **`tests/test_lm_collate_utils.py`** | **`build_c4_lm_datamodule`** plus collate tests; federation keyed via **`OMNIFED_*`** env set in **`slurm_hybrid_runner.py`** before **`instantiate(cfg.datamodule)`**. |
| **`src/omnifed/model/hf_causal_lm.py`** **`conf/model/llama150m_hf_disk.yaml`** | HF **`from_pretrained`** path wired for offline Frontier trees via **`OMNIFED_LLAMA_WEIGHTS`**. |
| **`conf/test_fedavg_llm_centralized_torchdist.yaml`**, **`conf/test_hybrid_layout_fedavg_llama150m.yaml`** | Central TorchDistrib LM stack + hybrid Slurm composition mirroring **`test_hybrid_layout_fedavg`**’s lattice. |
| **`requirements.txt`** | Declared **`datasets`**, **`transformers`**, **`sentencepiece`**, **`huggingface_hub`** so Frontier login prep matched compute constraints. |

---

## Archive pointer (**everything else stays verbose on purpose**) 

| `./archive/hybrid-engine-pipeline/...` | Why we kept it bulky |
|-------------------------|-----------------------|
| **`HYBRID_SLURM_REFERENCE.md`** | Chronological Frontier job diary + numbered verification steps—we never wanted to shrink that evidence trail. |
| **`HYBRID_USER_KNOBS_AND_ROADMAP.md`** | Figure-2 schema sketch / roadmap phases **E**/**F**. |
| **`HYBRID_TRAINING_AND_SYNC.md`** | Deep dive on **`__sync`** sequencing vs **`schedules/aggregation`**. |
| **`README_HYDRA_RUN_OUTPUTS.md`** | Exhaustive glossary of **`node_results`**, CSV columns, asymmetric **`gRPC_F*`** timings. |
| **`README_TEST_HYBRID_ENGINE_CONTRACT.md`** **`All_files_touched.md`** | Touch maps for audits. |
| **`CHAT_HANDOFF_HYBRID.md`** | Rolling “where we paused” capsule for collaborators / future threads. |
| **`HYBRID_LLAMA150M_C4_ROADMAP.md`** | Offline LM prep playbook + **`num_federated_clients`** freeze footnote §**J**. |

---

## Deferred engineering (**documented—not implemented here**)

We explicitly parked **multi-task-per-node** (**Phase E**) richer communicator selectors (**Phase F**) outside this hybrid milestone; **`HYBRID_USER_KNOBS_AND_ROADMAP`** still tracks intent so PRs stay scoped.

---

## Maintenance ethos (**same voice we asked collaborators to adopt**)

Whenever a new preset crosses **`run_hybrid_training`**, mirror updates in **`README_FRONTIER_EXPERIMENTS`** (Frontier steps) plus this artefacts map (code/config tables). Frontier validation prose still belongs in **`./archive/hybrid-engine-pipeline/HYBRID_SLURM_REFERENCE.md`** until a job cleanly passes at the newest topology size.
