# Hybrid Slurm — user knobs, schema sketch & roadmap (Phase A)

This document is **Phase A**: **requirements + configuration schema at a conceptual level** only (no implementation mandate here). Implementation status is summarized so **new chats** can align quickly.

See also **`./HYBRID_SLURM_REFERENCE.md`** (operations / validation / jobs), **`./HYBRID_TRAINING_AND_SYNC.md`** (FedAvg sync cadence), **`./README_HYDRA_RUN_OUTPUTS.md`** (artifacts).

---

## 1. North-star UX (paper-style)

Users should ultimately describe an experiment similarly to Figure 2 patterns in the OmniFed narrative:

| Theme | Example intent |
|-------|----------------|
| Algorithm | FedAvg vs FedProx, LR, **`global_rounds`**, **`max_epochs_per_round`**, aggregation **`schedules`** |
| Topology | Who trains, how many **clients**, legacy **centralized** node list semantics |
| Model / data | **`model/`**, **`datamodule/`** Hydra packs |
| **Hybrid extension** | **How many institutions (facilities)**, **how many trainers per facility**, **allocation on Slurm (nodes × tasks × GPUs)** |
| Communication | **Within-facility** collective vs **cross-facility** global aggregation (today: **TorchDist/NCCL** vs **Flora gRPC**; later: swappable backends) |

**Phase A deliberately does not** require one single YAML to match Figure 2 exactly today; it states the **intent** so Phases B–F can converge the **Engine + hybrid** path toward that ergonomics **without ambiguity**.

---

## 2. User-facing knobs (desired contract)

Definitions below **name** knobs the product should expose. Some already exist (`topology.num_clients`, `engine.hybrid.*`); others are **targets** for later phases.

### 2.1 Distributed layout (semantic)

| Knob | Description | Constraints / notes |
|------|--------------|---------------------|
| **Facilities (`F`)** | Number of disjoint **local** Torch process groups (“sites”). | **Testing focus:** **F = 2**; roadmap **up to 4**. |
| **Trainers per facility** | Either one integer (**symmetric**) or a **length‑F list** (**asymmetric**). | Matches **`mpi_ranks_per_facility`** in **`build_hybrid_topology`**. |
| **Dedicated RPC server** | One global rank is **only** the central Flora gRPC parameter server (**no FedAvg trainer** on that rank). | **`dedicated_rpc_server: true`** is the supported path today. Implies **`world_size = 1 + Σᵢ ranks_in_facility_i`**. |

### 2.2 Centralized OmniFed topology (FedAvg semantics)

| Knob | Description |
|------|--------------|
| **`topology.num_clients`** | Count of **training** centralized slots (**not** counting the FL “server” slot in **`CentralizedTopology`**). |

**Consistency rule:** for the current hybrid × centralized pairing, **`len(CentralizedTopology) = num_clients + 1`** must equal **hybrid `world_size`** (one slot per hybrid global rank).

### 2.3 Slurm / HPC allocation (Frontier)

| Knob | Meaning (first milestone) |
|------|----------------------------|
| **Total tasks** | Must equal **`world_size`** (one **MPI**/hybrid rank per task unless future multi-rank compaction is defined). |
| **Nodes × `ntasks_per_node`** | **`nodes × ntasks_per_node ≥ total tasks`** commonly; **`--ntasks`** is pinned to **`world_size`**. |

**First milestone (explicit):** **1 GPU ↔ 1 Slurm task ↔ 1 host** wherever possible (**7 nodes, `ntasks_per_node=1`**, seven tasks).

**Implementation story (Phase D):** how **`#SBATCH --ntasks=W`**, **`nodes`**, and **`ntasks_per_node`** line up with hybrid **`layout`** / **`topology_config`** is documented in **`./HYBRID_SLURM_REFERENCE.md`** §**4.3** (Engine may bump **`slurm.nodes`** when the requested count is too small for **`W`**).

**Later milestone:** **multiple tasks per node** (e.g. **8 GPUs ⇒ 8 tasks** on fewer nodes) — **`LOCAL_RANK`/device binding** and Slurm coherence must remain well-defined (**Phase E**, not Phase A).

### 2.4 Communicators (intent today vs later)

| Layer | Desired default (Frontier) | Phase F |
|-------|----------------------------|---------|
| **Inner / intra-facility** | **Torch distributed, NCCL (ROCm stack)** (“MPI-style” process group via existing adapters). | **Configurable target** (`_target_` or enum) — alternate backends stubbed/disabled until implemented. |
| **Outer / cross-facility global step** | **Flora gRPC** to a central PS (facility **leaders** as clients today). | **Configurable target** — not limited to gRPC long term. |

**Implemented today (declarative only):** resolved hybrid YAML topologies include **`communicators`** (defaults **`torch_mpi`** for intra‑facility collectives and **`grpc`** for the Flora global aggregation hop). Optionally override under **`topology.layout.communicators`** in **`conf_hybrid/topology/*.yaml`** or **`engine.hybrid.layout.communicators`**; values are merged onto **`topology_builder.DEFAULT_HYBRID_COMMUNICATORS`** and do **not** change communicator wiring yet (post‑merge Phase F hooks).

---

## 3. Schema sketch (where knobs live conceptually)

This is **not** a mandate to reshape all YAML in one step; it is the **target mental model**.

```yaml
# --- Conceptual (Phase A sketch; may map to multiple files today) ---

defaults:
  - topology: centralized
  - algorithm: fedavg
  - model: ...
  - datamodule: ...

topology:
  num_clients: <world_size_minus_one>

engine:
  mode: slurm
  communication_mode: hybrid
  hybrid:
    # Either file preset OR runtime layout — see §4 / HYBRID_SLURM_REFERENCE §4.2.
    topology_config: <preset.yaml under conf_hybrid/topology/>
    layout: { ... }   # runtime build_hybrid_topology kwargs when set

    # Optional labels merged onto topology_builder.DEFAULT_HYBRID_COMMUNICATORS (documentation / future Phase F hooks).
    # layout:
    #   communicators: { intra_facility: torch_mpi, global_aggregation: grpc }

    # Future (Phase F): instantiate compressor / alternate backends here (targets TBD).
    # inner_comm: { _target_: ... }
    # outer_comm: { _target_: ... }

algorithm: { ... }

global_rounds: ...

slurm:
  nodes: ...
  ntasks_per_node: ...
  # ... Frontier account/partition/gpu fields
```

**Cross-reference — implemented knobs today:** **`conf/base.yaml`** **`engine.hybrid.layout`**, **`engine.hybrid.training`**, **`engine.hybrid.topology_config`**, **`topology.num_clients`**, **`slurm.*`**.

---

## 4. Preset YAML vs `engine.hybrid.layout` (runtime)

- **File preset (`topology_config`):** Checked-in **`conf_hybrid/topology/*.yaml`**; good for reproducible names (“always use **`built_symmetric_2x3`**”).
- **`engine.hybrid.layout`:** Same **`build_hybrid_topology`** keywords **without** a new file (**Figure‑2 ergonomics — Phase C ongoing**); if **`layout`** and **`topology_config`** are both present, **`Phase B`** requires they imply the same **`world_size`** (see **`engine_communication.validate_hybrid_slurm_topology_alignment`**).

**Precedence:** if **`layout`** is a **non-empty** mapping with **`num_facilities`** and **`mpi_ranks_per_facility`**, **`layout` overrides `topology_config`** for hybrid **`world_size`** and **`run_hybrid_training`**.

---

## 5. Phase backlog (engineering; Phase A = this doc)

| Phase | Description | Status (high level as of Phase A write-up) |
|-------|--------------|--------------------------------------------|
| **A** | This document: knobs, schema sketch, communicator intent, phased roadmap | **Delivered (`./HYBRID_USER_KNOBS_AND_ROADMAP.md`)** |
| **B** | Single source for **`world_size`** + **alignment checks** (**`topology.num_clients + 1`**, **`SLURM_NTASKS`**, **`len(topology)`**; **`layout`** vs **`topology_config`** must agree if both set) | **Implemented** (**`engine_communication.hybrid_world_size_from_cfg`**, **`validate_hybrid_slurm_topology_alignment`**; hybrid worker + Engine **`resolve_slurm_ntasks`** use the same gates) |
| **C** | Hydra presets / ergonomics aligning with Figure 2 narrative | **Core sample landed** (**`conf/test_hybrid_layout_fedavg.yaml`**, README + **`tests/test_hybrid_phase_c_preset.py`**); fuller multi-model/dataset bundles still **TBD** |
| **D** | Slurm user story + docs for **allocation ↔ `ntasks`** (both hybrid presets; Frontier §**4.3**/§**7**) | **Delivered** (**`HYBRID_SLURM_REFERENCE.md`** §**4.3**; **`[Engine] slurm.nodes raised …`** log when auto-bumped — **`engine.py`**) |
| **E** | Multi‑GPU-per-node (**`ntasks_per_node` > 1**) | **Not started** |
| **F** | **Outer / inner communicator** configurability beyond gRPC + NCCL | **Not started** |

---

## 6. Operational invariants (do not casually break)

1. **SLURM task count (`SLURM_NTASKS`)** equals hybrid **world_size** (`topology.world_size` after **`build_hybrid_topology`** resolves).
2. **Centralized node count:** `num_clients + 1` equals **world_size** for **`test_hybrid_engine_contract`** and **`test_hybrid_layout_fedavg`** (one logical node per hybrid global rank).
3. **Facility leaders** (typically global ranks **`1 … F`** with **`dedicated_rpc_server`**) participate in **`GrpcLeaderCommunicator`**; **workers** do not; **rank 0** is RPC-only (**Flora `id==0`** — **`HYBRID_SLURM_REFERENCE.md`** §6).

---

## 7. Recommended next engineering step (after Phase A)

Pick **exactly one** downstream checkpoint (avoid mixed scope):

1. ~~**B** / **C** / **D:** validation, layout-first preset, Slurm ↔ **`world_size`** docs~~ **Delivered** (see §**5** backlog).
2. **E (engineering):** multi-task-per-node + **`LOCAL_RANK`** / GPU binding story for hybrid (not started).
3. **F (engineering):** communicator selectors in YAML (**not started**).

**Optional Frontier:** run **`HYBRID_SLURM_REFERENCE.md`** §**7b** with **`test_hybrid_layout_fedavg`** when you want OLCF proof of the layout-first preset (same **`--ntasks`** as **`test_hybrid_engine_contract`**).

---

## 8. Appendix — OLCF Frontier paths & resync (**personal snapshot; edit if account changes**)

| Item | Path / value |
|------|----------------|
| **Frontier user** | **`shruti2395`** |
| **Repo on Lustre** | **`/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT`** |
| **TorchVision MNIST (offline)** | **`/lustre/orion/gen150/scratch/shruti2395/omnifed_data/torchvision-mnist`** |
| **Data root env (see `engine.py` `setup_lines`)** | **`/lustre/orion/gen150/scratch/shruti2395/omnifed_data`** (`OMNIFED_DATA_DIR`) |
| **Slurm project (example)** | **`gen150`** |
| **Hydra run dir** | under repo: **`outputs/<date>/<config_name>/`** |

**Resync local → Frontier** (run on your **laptop** / build machine; fix `loginNN` and local path):

```bash
rsync -avz \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  /home/shruti/OmniFed_VT/ \
  shruti2395@loginNN.frontier.olcf.ornl.gov:/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT/
```

(`outputs/` excluded so old Slurm logs stay on Frontier; drop `--exclude 'outputs/'` only if you want a full mirror.)

**Frontier → refresh with git** (alternative when clone lives on Lustre):

```bash
ssh shruti2395@loginNN.frontier.olcf.ornl.gov \
  'cd /lustre/orion/gen150/scratch/shruti2395/OmniFed_VT && git fetch && git pull'
```

---

## See also

- **`./HYBRID_SLURM_REFERENCE.md`**
- **`./CHAT_HANDOFF_HYBRID.md`** — paste-ready hybrid Slurm roadmap + Cursor “please read” list (edit § “Today’s ask” per chat).
- **`./HYBRID_TRAINING_AND_SYNC.md`**
- **`./README_HYDRA_RUN_OUTPUTS.md`**

---

*Phase A end — schema & roadmap only.*
