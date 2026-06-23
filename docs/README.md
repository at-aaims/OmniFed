# OmniFed hybrid branch — documentation index

**New here?** Start with these two guides at the **`docs/` root (not in a subfolder):**

| Doc | Use when you need to… |
|-----|------------------------|
| **[`README_FRONTIER_EXPERIMENTS.md`](./README_FRONTIER_EXPERIMENTS.md)** | Run experiments on **OLCF Frontier** in the same order we used: centralized Engine → hybrid smoke → full hybrid FedAvg (MNIST), optional LLM path. |
| **[`CHECKPOINTING_HYBRID_SLURM_PLAN.md`](./CHECKPOINTING_HYBRID_SLURM_PLAN.md)** | Round-end checkpoints, manual resume, and **auto-chain** across 2h Slurm jobs (§11 Frontier commands). |
| **[`README_PIPELINE_IMPLEMENTATION_ARTIFACTS.md`](./README_PIPELINE_IMPLEMENTATION_ARTIFACTS.md)** | Understand **which files** implement the hybrid pipeline and **how they connect** (`main.sh` → Engine → `slurm_worker` → `run_hybrid_training`). |
| **[`HYBRID_FLORA_GRPC_COMPRESSION_PLAN.md`](./HYBRID_FLORA_GRPC_COMPRESSION_PLAN.md)** | **Option A:** TopK compression on Flora gRPC (global round only); first target ResNet-18 / 7 nodes / 5 rounds, then Llama. |

Everything else (long-form references, handoff notes, artifact catalogs, Llama+C4 roadmap) lives in **[`archive/hybrid-engine-pipeline/`](./archive/hybrid-engine-pipeline/README.md)** so the top of **`docs/`** stays uncluttered for new users.

---

## Full reference archive

See **[`archive/hybrid-engine-pipeline/README.md`](./archive/hybrid-engine-pipeline/README.md)** for a file list and pointers into:

- Hybrid Slurm operations & Frontier validation (`HYBRID_SLURM_REFERENCE.md`)
- User knobs & roadmap (`HYBRID_USER_KNOBS_AND_ROADMAP.md`)
- Training / sync semantics (`HYBRID_TRAINING_AND_SYNC.md`)
- Hydra run outputs & metrics (`README_HYDRA_RUN_OUTPUTS.md`)
- Preset touch maps, handoff snapshots, Llama+C4 roadmap, etc.

Code and YAML comments that pointed at old `docs/HYBRID_*.md` paths have been updated to **`docs/archive/hybrid-engine-pipeline/...`** where needed.
