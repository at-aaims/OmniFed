# Handoff — OmniFed_VT hybrid Slurm & Figure‑2 direction

**Repo path (local example):** `/home/shruti/OmniFed_VT`  
**Remote validation:** OLCF Frontier; hybrid jobs commonly **7 nodes × 1 task × 1 GPU**, `engine.mode=slurm`, `engine.communication_mode=hybrid`, offline MNIST on Lustre (`download=false` + `dataset.root`). ROCm / NCCL stack as in **`HYBRID_SLURM_REFERENCE.md`** §5.

**Purpose of this file:** Give the **next engineer or chat** enough context **without opening the whole thread**. Update when roadmap milestones move. **Snapshot:** **Phases A–D** core hybrid roadmap items are **done** (validators, layout-first preset, **§4.3** Slurm ↔ **`W`** story — **`HYBRID_SLURM_REFERENCE.md`**). **Next:** **Phase E/F** or optional Frontier **§7b** with **`test_hybrid_layout_fedavg`**. **Do not confuse** historical “Engine Phase B steps” (**Reference §3**) with roadmap **`B`** in **§4** below.

---

## 1. Naming: two different “Phase B” ideas

Avoid mixing these:

| Label | Meaning | Status |
|-------|---------|--------|
| **Historical “Phase B — Engine hybrid”** in **`HYBRID_SLURM_REFERENCE.md`** (Steps **6–9**) | Freeze config, Slurm launch, **`slurm_worker` → `run_hybrid_training`**, gRPC PS + Torch MPI facilities, **`leader_done`**, rank→centralized mapping, Frontier validation | **Done and documented** |
| **Roadmap engineering Phase B** in **`HYBRID_USER_KNOBS_AND_ROADMAP.md`** | **Single authoritative `world_size`** resolution plus **early validation** (**`topology.num_clients + 1`**, **`SLURM_NTASKS`**, **`len(topology)`**, **`layout`/`topology_config` agreement**) | **Done** (**`validate_hybrid_slurm_topology_alignment`**, **`engine_communication.hybrid_world_size_from_cfg`**) |

Anything in **§1** distinguishes historical Reference “Steps 6–9” from roadmap engineering phases below. **`layout`** + **`resolve_slurm_ntasks`** remain; roadmap **Phase B** validators are **implemented** (**`HYBRID_USER_KNOBS`** §5, **`engine_communication`**).

---

## 2. North star (why we’re not stopping at `built_symmetric_2x3`)

The validated path today assumes **one named preset** (**`conf_hybrid/topology/built_symmetric_2x3.yaml`**) → **`world_size = 7`**, fixed **outer = gRPC** (facility leaders ↔ central PS) and **inner = Torch distributed / NCCL‑style** per facility.

The longer-term UX is **Figure‑2‑like Hydra configs**: swap **algorithm**, **topology flavor**, **model**, **dataset**, **aggregation schedules**, **hyperparameters**, and—in the hybrid case—**how many facilities**, **trainers per facility**, **allocation (nodes × tasks × GPUs)** and eventually **explicit inner/outer communicator selectors**. **Phase A** captured that intent in **`docs/HYBRID_USER_KNOBS_AND_ROADMAP.md`** without requiring one mega‑YAML rewrite in a single PR.

---

## 3. What is already landed (assume it stays unless a maintainer says otherwise)

- **Smoke → Engine path:** Topology builder (**`topology_builder.py`**), **`conf_hybrid/`** presets, **`hybrid_comm_smoke`**, Slurm scripts, then **Engine** hybrid branch with **`run_hybrid_training`** (**`slurm_hybrid_runner.py`** etc.).
- **Training semantics:** **`docs/HYBRID_TRAINING_AND_SYNC.md`** — each **`__sync`**: **`local_agg` → `global_agg` (leaders only) → `local_bcast`**; frequency from **`algorithm.schedules.aggregation`**, not Flora **`comm_freq`** alone.
- **Frontier checklist:** **`docs/HYBRID_SLURM_REFERENCE.md`** — jobs, **`main.sh`** overrides, offline MNIST, **`--ntasks=world_size`** verification, §7 / §7b re‑checks.
- **Slurm ↔ hybrid ranks (Phase D):** **`HYBRID_SLURM_REFERENCE`** §**4.3** (**`nodes`**, **`ntasks_per_node`**, **`#SBATCH --ntasks=W`**); **`[Engine] slurm.nodes raised …`** when **`engine.py`** auto-bumps **`nodes`**.
- **Phase C layout-first preset:** **`conf/test_hybrid_layout_fedavg.yaml`** (**`tests/test_hybrid_phase_c_preset.py`**) mirrors **`built_symmetric_2x3`** without **`topology_config`**.
- **Validation (roadmap Phase B):** **`hybrid_world_size_from_cfg`** and **`validate_hybrid_slurm_topology_alignment`** (**`engine_communication.py`**) — **`resolve_slurm_ntasks`** (Engine submit) and **`run_hybrid_training`** (workers) share the same rules: **`topology.num_clients + 1 == world_size`**, **`len(topology)`** match, **`SLURM_NTASKS`** match inside the job, **`layout`** must not contradict **`topology_config`** if both are present.
- **Touch map for the canonical preset:** **`docs/All_files_touched.md`** (Hydra preset → engine → launcher → hybrid modules).

---

## 4. Roadmap backlog (engineering phases — after Phase A doc)

Synced with **`HYBRID_USER_KNOBS_AND_ROADMAP.md`** §5. **Important:** **`B` below = roadmap Phase B**, not the historical Reference “Steps 6–9” label.

| Phase | Intent | Status (handoff snapshot) |
|-------|--------|----------------------------|
| **A** | Requirements, schema sketch, phased backlog | **Delivered** |
| **B** | Single **`world_size`** resolution + **`SLURM_NTASKS`**, **`topology.num_clients`**, and duplicate **`layout`/`topology_config`** guards | **Done** (**`validate_hybrid_slurm_topology_alignment`**, **`hybrid_world_size_from_cfg`** — see **`engine_communication.py`**) |
| **C** | Hydra presets / ergonomics toward Figure‑2 narrated experiments | **Core preset** (**`conf/test_hybrid_layout_fedavg.yaml`**) + docs/tests; fuller bundles **TBD** |
| **D** | Slurm **`nodes`/`ntasks`** story vs hybrid **`W`** + Frontier §**7** pointer | **Delivered** (**`HYBRID_SLURM_REFERENCE`** §**4.3**); optional §**7b** cluster smoke for **`test_hybrid_layout_fedavg`** |
| **E** | **`ntasks_per_node` > 1**, **`LOCAL_RANK`**, multi‑GPU‑per‑node | Not started |
| **F** | Pluggable **`inner_comm` / `outer_comm`** in YAML beyond today’s fixed pair | Not started |

**Working agreement:** Advance **one phase (or explicit sub‑slice)** at a time; **ask before large or cross‑cutting code drops.**

---

## 5. Before you write hybrid code — read order

Strongly suggested for anyone picking this up cold:

1. **`docs/HYBRID_USER_KNOBS_AND_ROADMAP.md`** — intent, invariants (**`num_clients + 1 == world_size`**), communicator roadmap, Appendix §8 (Frontier **`rsync`** / paths if present).
2. **`docs/HYBRID_SLURM_REFERENCE.md`** — what was built, Frontier commands, **`layout`** vs **`topology_config`**, Steps 8–9 behavior.
3. **`docs/HYBRID_TRAINING_AND_SYNC.md`** — what one sync block does.
4. **`docs/All_files_touched.md`** — which YAML and Python paths participate in **`test_hybrid_engine_contract`**.
5. **`docs/README_HYDRA_RUN_OUTPUTS.md`** — where **`node_results`**, **`sync/*_time`**, metrics land.

Then open implementation **as needed**:

- **`src/omnifed/hybrid/hydra_loader.py`**, **`src/omnifed/engine_communication.py`**, **`conf/base.yaml`**, **`conf/test_hybrid_engine_contract.yaml`**, **`src/omnifed/hybrid/slurm_hybrid_runner.py`**.

---

## 6. Operational sharp edges (Frontier)

- Compute nodes usually **cannot download MNIST**; use **`download=false`** and a **shared Lustre `root`**.
- **`SLURM_NTASKS`** must equal hybrid **`world_size`** — mismatches surface at **submit** (**`resolve_slurm_ntasks`**) or first **worker** line (**`run_hybrid_training`**) via **`validate_hybrid_slurm_topology_alignment`** with an explicit message.
- Gloo **`errno 97`** messages may appear non‑fatally (**`HYBRID_SLURM_REFERENCE`**).

---

## 7. Today’s ask for the new agent *(edit each session)*

Replace this paragraph when starting a Cursor session — e.g. “Read handoff only,” “Start Phase E,” or “Design communicator YAML (**Phase F**).”

**Current default:** **Phases A–D** hybrid roadmap core is written up; next engineering choices are **E** (multi-GPU/node) / **F** (communicator knobs) or ad-hoc experiments.

---

## 8. Maintenance

After **major roadmap milestones** ship, revise **§4** / **§1** statuses and refresh **§7** with the next milestone.
