# Archive — hybrid pipeline + Engine (full reference)

**Purpose:** This folder holds **detailed** and **historical** documentation for the hybrid Slurm + Engine work. It is **not** the first stop for new users.

**Start here instead:** [`../README.md`](../README.md) → then **`README_FRONTIER_EXPERIMENTS.md`** and **`README_PIPELINE_IMPLEMENTATION_ARTIFACTS.md`** at the **`docs/`** root.

---

## Files in this archive

| File | Role (short) |
|------|----------------|
| **`HYBRID_SLURM_REFERENCE.md`** | Master reference: Phase A/B steps, Frontier jobs, `layout` vs `topology_config`, Slurm `--ntasks` vs `world_size` (§4.3). |
| **`HYBRID_USER_KNOBS_AND_ROADMAP.md`** | User-facing knobs, schema sketch, engineering phases B–F backlog. |
| **`HYBRID_TRAINING_AND_SYNC.md`** | FedAvg **`__sync`**: `local_agg` → global gRPC → `local_bcast`; schedule vs `comm_freq`. |
| **`README_HYDRA_RUN_OUTPUTS.md`** | What lives under `outputs/…` — `node_results`, `hybrid_per_round_summary`, timing keys. |
| **`README_TEST_HYBRID_ENGINE_CONTRACT.md`** | `test_hybrid_engine_contract` CLI + file touch map. |
| **`CHAT_HANDOFF_HYBRID.md`** | Maintainer handoff snapshot (phases done vs deferred). |
| **`HYBRID_LLAMA150M_C4_ROADMAP.md`** | Llama‑150M + C4 offline prep and Frontier procedure. |
| **`All_files_touched.md`** | Legacy touch map (overlap with preset README above). |

Cross-links **between files in this directory** use **relative** paths (`./OTHER.md`) so browsing on GitHub stays consistent.
