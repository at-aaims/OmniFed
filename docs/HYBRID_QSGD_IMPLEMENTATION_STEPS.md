# Hybrid QSGD Implementation Steps

**Branch:** `feature/hybrid-omnifed-on-upstream`  
**Repo (wukong5):** `/home/shruti/OmniFed_VT`  
**Repo (Frontier):** `/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT`  
**Source:** PR #86 — `Ayazdani1997:feature/grpc_compression` (not merged; copy/adapt QSGD only)  
**Local PR #86 ref:** `pr-86-grpc-compression` (after `git fetch upstream pull/86/head:pr-86-grpc-compression`)

**Related:** [HYBRID_GRADIENT_AGGREGATION.md](./HYBRID_GRADIENT_AGGREGATION.md) — single-knob params vs gradients on both MPI and gRPC (Phases 0–3).

---

## Scope (locked)

| Item | Requirement |
|------|-------------|
| Source | QSGD from PR #86 only — cherry-pick/copy, do NOT merge whole PR |
| Target | Hybrid Slurm pipeline on `feature/hybrid-omnifed-on-upstream` |
| Code location | `src/omnifed/hybrid/` only — NO Flora |
| Compression hop | Global gRPC only (facility leaders ↔ central PS) |
| Unchanged | TorchMPI local agg/bcast stays dense |
| Validate | ResNet-18 (CIFAR-10) + Llama on Frontier after rsync |
| Upstream PR | After Frontier smoke passes (supersedes closed #87) |

---

## Progress log (file changes)

Short record of what landed, why, and how it behaves. Updated after each major step.

**Current status:** Phase 0 ✅ · Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 partial · **6 pytest passed** (wukong5 + Frontier).

**Working env:**
```bash
conda activate omnifed_flora
export PYTHONPATH="${PWD}"
```

---

### Phase 0 — Prep ✅ (2026-06-01)

| Check | Result |
|-------|--------|
| Branch | `feature/hybrid-omnifed-on-upstream` |
| PR #86 ref | `pr-86-grpc-compression` @ `51aaef6` |
| No Flora in hybrid | `rg 'src\.flora' src/omnifed/hybrid/` → empty |
| Baseline tests | 4 passed → later 6 passed after Phase 2 |

No files changed.

---

### Phase 1 — QSGD algorithm module ✅

| File | Why | What it does now |
|------|-----|------------------|
| **`src/omnifed/hybrid/compression/qsgd.py`** (new) | Port QSGD from PR #86 into hybrid package (not Flora, not centralized `communicator/`) | `QSGDQuantCompression`: stochastic quantize by vector norm; `bit_width=s` → `2^s` levels; `compress()` → signed integer levels + norm + width + levels; `decompress()` reverses via `norm * level / levels` |
| **`src/omnifed/hybrid/compression/__init__.py`** | Export new symbols | Re-exports `QSGDQuantCompression`, `QSGD_COMPRESSION_NAME` alongside TopK |

**How it fits:** Algorithm only — no gRPC yet. Same math as PR #86; lives under `src/omnifed/hybrid/compression/` for hybrid global hop.

**v1 note:** No error feedback on QSGD yet (PR #86 centralized path also omits EF here).

**Verified:** `python -c "from ...qsgd import QSGDQuantCompression; print('ok')"` · roundtrip `bit_width=4` → 16 levels.

---

### Phase 2 — Global gRPC wire + encode/decode ✅

| File | Why | What it does now |
|------|-----|------------------|
| **`src/omnifed/hybrid/communicator/global_grpc.proto`** | TopK fields alone cannot carry QSGD norm/width/level | Added optional fields on `LayerState`: `meta_tensor`, `meta_tensor_dtype`, `width`, `level`. TopK still uses fields 1–9 only |
| **`global_grpc_pb2.py`**, **`global_grpc_pb2_grpc.py`** | Regenerated from proto | Python stubs include new QSGD fields. **Regenerate in `omnifed_flora`** so protobuf gencode matches runtime |
| **`src/omnifed/hybrid/communicator/global_grpc_compression.py`** | Single place for global-hop compression on the wire | **TopK:** unchanged (sparse values/indices in bytes). **QSGD:** signed levels in `values_data`, norm in `meta_tensor`, `width`/`level` set. **`build_global_compressor`:** `scheme=topk\|qsgd`. **`hybrid_global_compressor_from_cfg`:** reads Hydra `bit_width`. Dense path when `enabled=false` |
| **`global_grpc_client.py`** | Was typed TopK-only | Accepts `TopKCompression` or `QSGDQuantCompression`; logs mode `TopK` / `QSGD` / `dense` |
| **`global_grpc_server.py`** | Same | PS encode/decode works for both schemes |
| **`conf/base.yaml`** | Hydra knob for QSGD | `engine.hybrid.global_compression`: `scheme` (`topk`\|`qsgd`), `compress_ratio`, `bit_width` |
| **`tests/test_hybrid_global_grpc_compression.py`** | Cover QSGD wire path | +2 tests: QSGD LayerState encode/decode; `build_global_compressor(scheme=qsgd)` |

**Unchanged (already correct):** `grpc_leader_comm.py`, `slurm_hybrid_runner.py` — both call `hybrid_global_compressor_from_cfg`; TorchMPI local hop stays dense.

**Flow (global hop only):**
```
Leader → encode_layer_state (QSGD) → gRPC SendUpdate → PS decode → aggregate dense
PS → encode_layer_state → gRPC GetUpdatedModel → Leader decode → local_bcast dense
```

**Verified:** 6 passed · manual check `QSGDQuantCompression 16 (8,)` for `bit_width=4`.

**Regenerate proto (after `.proto` edits):**
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
  ./src/omnifed/hybrid/communicator/global_grpc.proto
```

---

### Phase 3 — Hydra presets ✅

| File | Why | What it does |
|------|-----|--------------|
| **`conf/test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd.yaml`** | ResNet Frontier QSGD runs | `enabled=true`, `scheme=qsgd`, `bit_width=4`, `global_rounds=5`; inherits 7-node lattice from base ResNet hybrid |
| **`conf/test_hybrid_layout_fedavg_llama150m_grpc_qsgd.yaml`** | Llama-150M QSGD | Same QSGD block on Llama 7-node preset |
| **`conf/test_hybrid_layout_fedavg_llama400m_grpc_qsgd.yaml`** | Llama-400M QSGD | Same for 400M preset |
| **`test_scripts/frontier_hybrid_cifar10_resnet18_7_qsgd.sh`** | One-command Frontier submit | 7-node ResNet; `QSGD_BIT_WIDTH` env (default 4) |
| **`test_scripts/frontier_hybrid_llama150m_7_qsgd.sh`** | Llama Frontier submit | 7-node Llama-150M QSGD; requires C4/tokenizer/weights env vars |

**CLI sweep (no new yaml needed):** `engine.hybrid.global_compression.bit_width=2|4|8`

**Logs when QSGD is on:** `Compression: True`, `Client ... initialized (QSGD)`

**Import fix (upstream gap):** `upstream/main` imports `src.omnifed.communicator.compression.quantization` in `grpc_client.py` but PR #86 is not merged. Added **`quantization.py`** + **`lowrank_approximation.py`** under `src/omnifed/communicator/compression/` (from PR #86) so `main.py` / Engine import succeeds. Hybrid QSGD still uses **`src/omnifed/hybrid/`** only at runtime.

---

### Phase 3 — Hydra presets (checklist) ✅

| File | Status |
|------|--------|
| `conf/base.yaml` | ✅ (Phase 2) |
| `conf/test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd.yaml` | ✅ |
| `conf/test_hybrid_layout_fedavg_llama150m_grpc_qsgd.yaml` | ✅ |
| `conf/test_hybrid_layout_fedavg_llama400m_grpc_qsgd.yaml` | ✅ |
| `test_scripts/frontier_hybrid_*_qsgd.sh` | ✅ |

---

### Phase 4 — Unit tests ⏳ (partial)

| File | Status |
|------|--------|
| QSGD tests in `test_hybrid_global_grpc_compression.py` | ✅ Added in Phase 2 |
| `test_no_flora_imports_in_hybrid.py` | ✅ Still passes |
| Full Phase 4 sign-off after presets | Pending |

---

### Suggested commits (you run git)

1. `hybrid: add QSGD compression module (ported from PR #86)` — Phase 1 files  
2. `hybrid: wire QSGD through global gRPC + Hydra bit_width` — Phase 2 files + proto/pb2 + tests  
3. *(later)* `conf: add ResNet/Llama hybrid QSGD presets + Frontier scripts`

---

## Phase 0 — Prep (no code yet)

| Step | Action | Done when | Status |
|------|--------|-----------|--------|
| 0.1 | Confirm branch | `git checkout feature/hybrid-omnifed-on-upstream` | ✅ |
| 0.2 | Confirm PR #86 ref exists | `git log pr-86-grpc-compression -1` | ✅ |
| 0.3 | Baseline: no Flora in hybrid | `rg 'src\.flora' src/omnifed/hybrid/` → empty | ✅ |
| 0.4 | Baseline tests pass | pytest (see Progress log) | ✅ 6 passed |

---

## Phase 1 — Port QSGD algorithm into hybrid package ✅

| Step | Action | Files | Status |
|------|--------|-------|--------|
| 1.1 | Copy `QSGDQuantCompression` from PR #86 | NEW: `src/omnifed/hybrid/compression/qsgd.py` | ✅ |
| 1.2 | Adapt imports to hybrid `Compression` base (`core.py`) | `qsgd.py` | ✅ |
| 1.3 | Add constant e.g. `QSGD_COMPRESSION_NAME = "QSGDQuantCompression"` | `qsgd.py` | ✅ |
| 1.4 | Export from package | `src/omnifed/hybrid/compression/__init__.py` | ✅ |
| 1.5 | Error feedback for v1 | none on QSGD v1 (documented in Progress log) | ✅ |

**Reference files on PR #86 branch:**
- `src/omnifed/communicator/compression/quantization.py`
- `QSGDTensorCodec` in `src/omnifed/communicator/utils.py`

---

## Phase 2 — Wire format (global gRPC proto + encode/decode) ✅

Hybrid uses `LayerState` in `global_grpc.proto` (TopK today). QSGD needs norm + width + level.

| Step | Action | Files | Status |
|------|--------|-------|--------|
| 2.1 | Extend `LayerState` with optional QSGD fields | `global_grpc.proto` | ✅ |
| 2.2 | Regenerate stubs | `global_grpc_pb2.py`, `global_grpc_pb2_grpc.py` | ✅ |
| 2.3 | QSGD encode/decode on LayerState | `global_grpc_compression.py` | ✅ |
| 2.4 | `build_global_compressor` + `scheme: qsgd` + `bit_width` | `global_grpc_compression.py` | ✅ |
| 2.5 | `encode_layer_state` / `decode_layer_tensor` for topk AND qsgd | `global_grpc_compression.py` | ✅ |
| 2.6 | Generic compressor type hints | `global_grpc_client.py`, `global_grpc_server.py` | ✅ |
| 2.7 | Wiring on global hop only | `grpc_leader_comm.py`, `slurm_hybrid_runner.py` | ✅ (no edits needed) |

**Invariant:** Leaders decompress to dense before `local_bcast`; PS aggregates dense after decompress (same as TopK).

**Already wired (no Flora):**
- `grpc_leader_comm.py` → `hybrid_global_compressor_from_cfg`
- `slurm_hybrid_runner.py` → server-side compressor
- `global_grpc_client.py` / `global_grpc_server.py` → encode/decode on SendUpdate / GetUpdatedModel

---

## Phase 3 — Hydra configuration ✅

| Step | Action | Files | Status |
|------|--------|-------|--------|
| 3.1 | Extend `engine.hybrid.global_compression` | `conf/base.yaml` | ✅ |
| 3.2 | ResNet preset with QSGD on | `conf/test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd.yaml` | ✅ |
| 3.3 | Llama preset(s) with QSGD on | `*_llama150m_grpc_qsgd.yaml`, `*_llama400m_grpc_qsgd.yaml` | ✅ |
| 3.4 | Keep TopK preset for comparison | `conf/test_hybrid_layout_fedavg_cifar10_resnet18_grpc_topk.yaml` | ✅ exists |
| 3.5 | Frontier submit scripts | `test_scripts/frontier_hybrid_*_7_qsgd.sh` | ✅ |

**Example Hydra block:**
```yaml
engine:
  hybrid:
    global_compression:
      enabled: true
      scheme: qsgd
      bit_width: 4    # 2^4 = 16 quantization levels
```

---

## Phase 4 — Unit tests (wukong5) ⏳ partial

| Step | Action | Files | Status |
|------|--------|-------|--------|
| 4.1 | QSGD compress/decompress roundtrip | `tests/test_hybrid_global_grpc_compression.py` | ✅ |
| 4.2 | LayerState QSGD encode → decode | same | ✅ |
| 4.3 | Dense path when `enabled=false` | same | ✅ |
| 4.4 | No Flora in hybrid | `tests/test_no_flora_imports_in_hybrid.py` | ✅ |
| 4.5 | Run full pytest set | | ✅ **6 passed** |

```bash
pytest tests/test_hybrid_global_grpc_compression.py \
       tests/test_no_flora_imports_in_hybrid.py -q
```

---

## Phase 5 — Local smoke (optional but recommended)

| Step | Action |
|------|--------|
| 5.1 | Small run: ResNet hybrid, `global_compression.enabled=true`, `scheme=qsgd`, `bit_width=4` |
| 5.2 | Logs show QSGD mode on gRPC client/server (not "dense") |
| 5.3 | At least 1 global round completes without gRPC decode errors |

---

## Phase 6 — Rsync to Frontier

| Step | Action |
|------|--------|
| 6.1 | Commit on `feature/hybrid-omnifed-on-upstream` |
| 6.2 | Rsync wukong5 → login04.frontier.olcf.ornl.gov |
| | Destination: `/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT` |
| 6.3 | On Frontier: `conda activate pytorch_rocm`, rerun pytest |
| 6.4 | Confirm synced commit matches wukong5 |

**Example rsync (from wukong5):**
```bash
rsync -avz --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'outputs' \
  ~/OmniFed_VT/ \
  shruti2395@login04.frontier.olcf.ornl.gov:/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT/
```

---

## Phase 7 — Frontier Slurm job matrix

**Lattice:** 7 nodes first (2×3 + RPC); scale to 17/129 later.

| Step | Model | Config | Purpose |
|------|-------|--------|---------|
| 7.1 | ResNet-18 | dense (`enabled=false`) | Baseline |
| 7.2 | ResNet-18 | TopK `compress_ratio=0.01` | Existing scheme |
| 7.3 | ResNet-18 | QSGD `bit_width=2` (4 levels) | Aggressive |
| 7.4 | ResNet-18 | QSGD `bit_width=4` (16 levels) | Mid |
| 7.5 | ResNet-18 | QSGD `bit_width=8` (256 levels) | High |
| 7.6 | Llama-150M (or 400M) | dense + one QSGD level | LM sanity |
| 7.7 | Llama | second QSGD level if time | Diff comparison |

| Step | Action | Files |
|------|--------|-------|
| 7.8 | Add/adapt submit scripts | `test_scripts/frontier_hybrid_*_qsgd*.sh` |
| 7.9 | Submit jobs; record job IDs | — |

**Existing reference scripts:**
- `test_scripts/frontier_hybrid_cifar10_resnet18_17.sh`
- `test_scripts/frontier_hybrid_cifar10_resnet18_129.sh`

**Metrics to compare:**
- Accuracy / loss
- `sync/global_agg_time` in `hybrid_per_round_summary.csv`
- gRPC message size (if logged)
- Job completes without decode errors

**Hybrid invariant:** `topology.num_clients + 1 == world_size == SLURM_NTASKS`

---

## Phase 8 — Review results

| Step | Action |
|------|--------|
| 8.1 | Compare logs/CSV: dense vs TopK vs QSGD levels |
| 8.2 | Confirm accuracy drop acceptable at chosen bit widths |
| 8.3 | Note recommended QSGD level(s) for PR description |
| 8.4 | Fill **final comparison table** (below) from each run’s `hybrid_per_round_summary.csv` |

---

## Final comparison table (dense vs TopK vs QSGD levels)

After each Frontier job completes, every run writes **`hybrid_per_round_summary.csv`** at the Hydra run root (per-round gRPC / local sync / **`eval_loss_avg`**). Use this section to compare **across compression settings**.

### Per-run matrix (ResNet-18, 7 nodes, 5 global rounds — fill as jobs finish)

| Run label | Scheme | `bit_width` (s) | QSGD levels (2^s) | Slurm job | Output dir | Status |
|-----------|--------|-----------------|-------------------|-----------|------------|--------|
| dense_baseline | dense | — | — | | `outputs/.../test_hybrid_layout_fedavg_cifar10_resnet18/` | |
| topk_1pct | topk | — (ratio=0.01) | — | | | |
| qsgd_s2 | qsgd | 2 | 4 | | | |
| qsgd_s4 | qsgd | 4 | 16 | 4918064 | | pending / running |
| qsgd_s8 | qsgd | 8 | 256 | | | |

**Per-round detail:** open each run’s **`hybrid_per_round_summary.csv`** (columns: `round_idx`, `gRPC_F1_ms`, `gRPC_F2_ms`, `eval_loss_avg`, …).

### Aggregate comparison (auto-generated)

From Frontier login, after all summary CSVs exist:

```bash
cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"

python scripts/compare_hybrid_compression_runs.py \
  --run "dense,scheme=dense,job_id=<JOBID>=/lustre/.../outputs/<date>/test_hybrid_layout_fedavg_cifar10_resnet18/hybrid_per_round_summary.csv" \
  --run "topk,scheme=topk,job_id=<JOBID>=/lustre/.../outputs/<date>/.../hybrid_per_round_summary.csv" \
  --run "qsgd_s2,scheme=qsgd,bit_width=2,job_id=<JOBID>=/lustre/.../outputs/<date>/.../hybrid_per_round_summary.csv" \
  --run "qsgd_s4,scheme=qsgd,bit_width=4,job_id=4918064=/lustre/.../outputs/<date>/.../hybrid_per_round_summary.csv" \
  --run "qsgd_s8,scheme=qsgd,bit_width=8,job_id=<JOBID>=/lustre/.../outputs/<date>/.../hybrid_per_round_summary.csv" \
  -o hybrid_compression_comparison_resnet7.csv \
  --markdown hybrid_compression_comparison_resnet7.md
```

**Simpler form** (metadata optional):

```bash
python scripts/compare_hybrid_compression_runs.py \
  --run "dense=/path/to/dense/run/hybrid_per_round_summary.csv" \
  --run "qsgd_s4,scheme=qsgd,bit_width=4,job_id=4918064=/path/to/qsgd/run/hybrid_per_round_summary.csv" \
  -o hybrid_compression_comparison.csv \
  --markdown hybrid_compression_comparison.md
```

**Output columns:**

| Column | Meaning |
|--------|---------|
| `final_eval_loss` | `eval_loss_avg` on the **last** global round |
| `mean_eval_loss` | Mean over all rounds |
| `mean_gRPC_ms` / `max_gRPC_ms` | Global gRPC time (all facilities, all rounds) |
| `final_accuracy` | If eval logs accuracy (often empty for CIFAR ResNet) |

### Manual summary table (paste after script or by hand)

| run_label | scheme | bit_width | levels | n_rounds | final_eval_loss | mean_eval_loss | mean_gRPC_ms | max_gRPC_ms | job_id |
|-----------|--------|-----------|--------|----------|-----------------|----------------|--------------|-------------|--------|
| dense | dense | — | — | 5 | | | | | |
| topk | topk | — | — | 5 | | | | | |
| qsgd_s2 | qsgd | 2 | 4 | 5 | | | | | |
| qsgd_s4 | qsgd | 4 | 16 | 5 | | | | | 4918064 |
| qsgd_s8 | qsgd | 8 | 256 | 5 | | | | | |

**Interpretation:** Lower **`eval_loss`** is better; **`mean_gRPC_ms`** shows communication cost trade-off vs dense. QSGD should reduce wire size (not yet in CSV — check logs / future metric).

---

## Phase 9 — Push + upstream PR (after Frontier)

| Step | Action |
|------|--------|
| 9.1 | Push branch to fork |
| 9.2 | Open PR to `at-aaims/main` |
| 9.3 | PR body: supersedes closed #87; related to #86; hybrid global-gRPC QSGD port |
| 9.4 | Coordinate with collaborator (Ayazdani) on algorithm alignment |

**Push auth on wukong5 (Cursor bypasses normal git auth):**
```bash
read -s GITHUB_TOKEN
env -u GIT_ASKPASS -u SSH_ASKPASS -u VSCODE_GIT_ASKPASS \
  git -c credential.helper= -c core.askPass= \
  push -u "https://dshruti20:${GITHUB_TOKEN}@github.com/dshruti20/OmniFed.git" \
  feature/hybrid-omnifed-on-upstream:feature/hybrid-omnifed-on-upstream
unset GITHUB_TOKEN
git fetch origin
git branch -u origin/feature/hybrid-omnifed-on-upstream
```

---

## Dependency order

```
Phase 0 Prep
  → Phase 1 qsgd.py
  → Phase 2 proto + global_grpc_compression
  → Phase 3 Hydra presets
  → Phase 4 unit tests
  → Phase 5 local smoke
  → Phase 6 rsync Frontier
  → Phase 7 Slurm jobs
  → Phase 8 review
  → Phase 9 PR
```

---

## Suggested commit breakdown

1. `hybrid: add QSGD compression module (ported from PR #86)`
2. `hybrid: wire QSGD through global gRPC encode/decode + Hydra`
3. `conf: add ResNet/Llama hybrid QSGD presets + Frontier scripts`

---

## Out of scope (this pass)

- Merging PR #86 wholesale
- QSGD on TorchMPI / local facility comm
- Flora imports or `src/flora/compression/*`
- Checkpoint + QSGD residual interaction (unless needed after first smoke)
- PowerSGD / low-rank from PR #86 (QSGD only)

---

## Git / PR context (completed before QSGD work)

| Step | Status |
|------|--------|
| 1–2 | Fetch + log comparison — Done |
| 3 | Backup patch `021747b` — Done |
| 4 | Merge `upstream/main` into fork `main`, push — Done |
| 5 | Branch `feature/hybrid-omnifed-on-upstream` — Done |
| 6 | `git am` migration patch → `b3520c3` — Done |
| 7 | Verify (no conflicts) — Done |
| 8 | Push branch to fork — Done |
| 9 | PR to `at-aaims/main` — Paused until QSGD + Frontier validation |

**Old obsolete branch:** `feature/hybrid-omnifed-migration` @ `021747b` (closed PR #87) — do not use for new PR.

---

## Related docs

- `docs/HYBRID_OMNIFED_MIGRATION_PLAN.md` — Flora → OmniFed hybrid migration
- `docs/HYBRID_FLORA_GRPC_COMPRESSION_PLAN.md` — original TopK plan (v1 TopK only; QSGD is next)
- `docs/README_FRONTIER_EXPERIMENTS.md` — Frontier submit patterns

---

*Last updated: 2026-06-23 — Phase 3 presets + Frontier scripts; ready for QSGD Slurm jobs.*
