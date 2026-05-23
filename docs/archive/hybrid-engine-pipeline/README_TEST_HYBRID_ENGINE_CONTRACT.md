# `test_hybrid_engine_contract`: hybrid Slurm run — reference command & file touch map

This README is repo-relative to **`OmniFed_VT/`** (workspace root except **MNIST/TorchVision data**, which typically lives on Lustre paths you pass via CLI overrides).

**Phase C — layout-first sibling:** **`conf/test_hybrid_layout_fedavg.yaml`** — **`engine.hybrid.layout`** only (**same **`world_size` / **`topology.num_clients`** as below** — no **`conf_hybrid`** topology file).

## Command shape (Frontier-style example)

**Preset:** `--config-name test_hybrid_engine_contract` sets **`engine.mode: slurm`**, **`engine.communication_mode: hybrid`**, **`engine.hybrid.topology_config`** → **`conf_hybrid/topology/built_symmetric_2x3.yaml`** (implicit **world size = 7**), and **`topology.num_clients: 6`** (must match **`num_clients + 1 == world_size`** for centralized mapping).

**Slurm sizing:** **`slurm.nodes=7`**, **`slurm.ntasks_per_node=1`** ⇒ **seven tasks**, one task per allocated node (**7 × 1**), matching **`--ntasks = 7`**.

Minimal pattern (adapt account, partitions, lustre MNIST roots):

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

**Outside the repo:** the **lustre dataset roots** above (offline MNIST for compute nodes).

---

## Files involved (touch map)

| Path | Role (short) |
|------|----------------|
| **`main.sh`** | Sets env vars, runs **gRPC protobuf** codegen, invokes **`python -u main.py "$@"`** with Hydra overrides. |
| **`main.py`** | Hydra entry; constructs **`Engine`** and runs experiment / Slurm submission. |
| **`src/omnifed/communicator/grpc.proto`** | **OmniFed** RPC schema; compiled by **`main.sh`** → generated `*_pb2*.py` next to it. |
| **`conf/test_hybrid_engine_contract.yaml`** | Top preset (**`topology_config`** → **`built_symmetric_2x3.yaml`**): **`engine`** hybrid + **`topology.num_clients`** + **`global_rounds`**. |
| **`conf/test_hybrid_layout_fedavg.yaml`** | Phase C preset (**`engine.hybrid.layout`** only; same lattice as **`built_symmetric_2x3`**; **`topology_config`** null). |
| **`tests/test_hybrid_phase_c_preset.py`** | Hydra **`compose`** parity vs file preset (**`validate_hybrid_slurm_topology_alignment`**, **`load_hybrid_cfg_for_engine`**). |
| **`conf/test_fedavg_centralized_torchdist.yaml`** | Parent preset: **`CentralizedTopology`**, FedAvg schedules, **`max_epochs_per_round`**, **`datamodule` batch sizes**, rank-0 **train=null**. |
| **`conf/base.yaml`** | Framework defaults (**`slurm.*`**, **`engine.*`** including **`engine.hybrid.*`**). |
| **`conf/topology/centralized.yaml`** | **`CentralizedTopology`** **`_target_`** + inherits topology base. |
| **`conf/topology/base.yaml`** | Topology Hydra scaffold. |
| **`conf/algorithm/fedavg.yaml`** | **`FedAvg`** **`_target_`** + algorithm base defaults. |
| **`conf/algorithm/base.yaml`** | Algorithm template; pulls **`algorithm/schedules`**. |
| **`conf/algorithm/schedules/base.yaml`** | Aggregation + evaluation schedule groups. |
| **`conf/algorithm/schedules/aggregation/round_end.yaml`** | **`round_end`** aggregation trigger (fed sync cadence vs batch/epoch). |
| **`conf/algorithm/schedules/evaluation/standard.yaml`** | Experiment start/end + post-sync eval gates. |
| **`conf/model/simple_cnn.yaml`** | MNIST **CNN** head / channels for **`instantiate(cfg.model)`**. |
| **`conf/model/base.yaml`** | Model scaffold. |
| **`conf/datamodule/mnist.yaml`** | torchvision **MNIST** train/eval dataset targets + transforms (**`download` / `root`** often overridden). |
| **`conf/datamodule/base.yaml`** | **`DataModule`** + **dataloader** defaults. |
| **`conf/datamodule/dataloader.yaml`** | **`DataLoader`** template (**`batch_size`**, **`shuffle`**, …). |
| **`conf_hybrid/topology/built_symmetric_2x3.yaml`** | Hybrid layout: **2×3 MPI + RPC** ⇒ **world_size 7**. |
| **`conf_hybrid/base.yaml`** | **`conf_hybrid`** Hydra root (topology + runtime defaults). |
| **`conf_hybrid/runtime/default.yaml`** | Merged **conf_hybrid** runtime seeds / legacy Flora-style keys (**training path mostly uses OmniFed **`cfg`**, not these values**). |
| **`src/omnifed/engine.py`** | On login node (**no `SLURM_JOB_ID`**): freezes **`engine_frozen.json`**, **`resolve_slurm_ntasks`**, writes **`SlurmOnlyLauncher`** script. |
| **`src/omnifed/engine_communication.py`** | **`communication_mode`**, **`resolve_slurm_ntasks`**, hybrid topology YAML pointer. |
| **`src/omnifed/slurm_launcher.py`** | Emits **`omnifed_slurm_only.sh`**, **`sbatch`/`srun`** → **`python -m src.omnifed.slurm_worker`**. |
| **`omnifed_slurm_only.sh`** *(generated at repo root)* | Submitted Slurm bash driver. |
| **`outputs/…/engine_frozen.json`** *(generated)* | Resolved **`cfg`** snapshot + **`hydra_output_dir`** (+ checkpoint hint); replicated to nodes. |
| **`src/omnifed/slurm_worker.py`** | **`hybrid`** branch → **`run_hybrid_training`**. |
| **`src/omnifed/hybrid/slurm_hybrid_runner.py`** | Hybrid orchestration (RPC daemon rank, Torch MPI ranks, Flora **gRPC** leaders, **`round_exec`** loop, results + **`leader_done`** markers). |
| **`src/omnifed/hybrid/hydra_loader.py`** | Compose **`conf_hybrid`** (**`topology=…`**), merge built topology dict. |
| **`src/omnifed/hybrid/topology_builder.py`** | Resolve **`world_size`**, **`rpc.client_ranks`**, facilities + members from **`layout`**. |
| **`src/omnifed/hybrid/topology_roles.py`** | Facility membership, **`hybrid_rank_to_centralized_node_index`**. |
| **`src/omnifed/hybrid/slurm_hostlist.py`** | Patch RPC / MPI **`MASTER_ADDR`** from **`SLURM_NODELIST`**. |
| **`src/omnifed/hybrid/addr_env.py`** | Address / port overrides for hybrid jobs. |
| **`src/omnifed/hybrid/torch_mpi_adapter.py`** | **`TorchMPICommunicator`** **`→ BaseCommunicator`** (**`broadcast`**, **`aggregate`** ⇒ **`torch.distributed`**). |
| **`src/omnifed/hybrid/comm_bridge.py`** | Carries **`last_group_total_samples`** for weighted **global** Flora step. |
| **`src/omnifed/hybrid/grpc_leader_comm.py`** | **`global_comm`** = **Flora client** (**`GrpcCommunicator`**) **`aggregate`** to PS. |
| **`src/omnifed/hybrid/hybrid_slurm_sync.py`** | Patched **`__sync_comm`**: intra-facility **MPI** reduce + **leader** **gRPC** + **facility broadcast**. |
| **`src/flora/communicator/grpc_communicator.py`** | **Flora PS / daemon** (**`daemon_server`** on rank 0) and **client** aggregates. |
| **`src/flora/communicator/torch_mpi.py`** | **Flora Torch MPI** communicator per facility. |
| **`src/omnifed/algorithm/fedavg.py`** | **FedAvg** loss / optimizer (**`instantiate`** from **`algorithm` cfg**). |

### Generated / runtime artifacts (under Hydra `outputs/`)

| Path pattern | Role |
|----------------|------|
| **`outputs/<date>/<config_name>/test_hybrid_* /engine/node_results/node_*`** | Per-task **FedAvg summary** payloads (covers **`test_hybrid_engine_contract`** and **`test_hybrid_layout_fedavg`**). |
| **`outputs/<date>/<config_name>/test_hybrid_*/engine/hybrid_grpc_leader_done/`** | **`leader_done`** shutdown handshake files. |
| **`outputs/<date>/<config_name>/test_hybrid_*/Node0.*`** | **`CentralizedTopology`** **`Node0.rank`** dirs — **metrics CSVs** / TB per logical node slot. |

---

## See also

- **`README.md`** (project root) — hybrid Slurm + centralized baseline bullets.  
- **`./HYBRID_SLURM_REFERENCE.md`** — Frontier checklist, validation, revision history.
- **`./README_HYDRA_RUN_OUTPUTS.md`** — **run output catalog** (**`outputs/…`**, metrics CSVs, **`sync/*_time`** for comm).
- **`./HYBRID_TRAINING_AND_SYNC.md`** — training loop: epochs/steps vs **`local_agg` / gRPC / `local_bcast`**, and **`round_end`** schedule (replacing Flora **`comm_freq`** in OmniFed).
