# OmniFed as a system — big picture

This document is the **target architecture** and the **stepwise path** to it.  
It is not a promise that every knob exists today. Development stays incremental: prove one pipeline, then extend, **without breaking** already-tested setups.

---

## 1. Final goal

Run **cross-silo federated training of large models**:

- Several **facilities** (silos) train locally.
- A **central server** coordinates aggregation across facilities.
- In production the central server can be **any machine** (HPC login-like node, dedicated CPU box, or a GPU node). Clients must not assume the server trains.

OmniFed’s job is to stay **modular**: swap algorithm, communicator, compression, what is aggregated, how often, and how a client trains internally — **from Hydra** — without rewriting the rest of the stack.

---

## 2. How we get there (order of proof)

| Stage | What we prove | Status |
|--------|----------------|--------|
| **A. Centralized (classic)** | One parameter server + many clients. Grad or param sync. gRPC or TorchDist. Dense / Top-K / QSGD. Small model (e.g. ResNet-18), **1 GPU per node**. | **Done** (Frontier matrix). Must stay working. |
| **B. Hierarchical (hybrid)** | Two (or more) facilities + one central server. Local comm inside a facility; global comm (today gRPC) between facility leaders and the center. | **Partial**. Continue without regressing A. |
| **C. Pack GPUs as clients** | Smaller models: **N GPUs on one node = N clients** (unique `cuda:i`). Server may be another node with 0 or 1 GPU. | Later; device resolver must already allow it. |
| **D. Fat client (TorchTitan)** | One federated **client** = many GPUs (tensor parallel / pipeline parallel). OmniFed talks to a **leader rank**; internal parallelism is the client’s trainer. | Later. Do not bake “one GPU = full model” into the design. |

Rule: **never break A** while building B–D. File moves and refactors must keep the same Hydra names for stage A unless we explicitly migrate presets.

---

## 3. Roles (always distinct)

```
                    Central server  (aggregate only; may have no GPU)
                           ▲
              global communicator (gRPC today; swappable)
                    ┌──────┴──────┐
              Facility 1     Facility 2     …   (silos)
                    │              │
         local communicator   (TorchDist / MPI / …)
              clients (trainers)
                    │
         optional: many GPUs per client (TorchTitan)
```

- **Central server:** does **not** run the training loop. Aggregation + (optional) compress/decompress of the **global** payload.
- **Facility leader:** trains **and** talks to the center (hierarchical). In centralized classic, rank 0 is **only** server (`train: null`).
- **Client:** one unit of federated data + one optimizer step policy. Today that is often 1 GPU. Later it may be a TorchTitan subcluster.

Device of the **model** and device of **gRPC SUM** are not the same question. Protobuf is always **CPU bytes**. SUM/compress may be CPU or GPU **if memory allows**.

---

## 4. Device resolution (must stay one story)

Resolve **per process**, not “one GPU for the whole job.”

For each rank:

1. **Role** — server vs client vs facility leader (rank + topology, not a `client_type` yaml field).
2. **Visible GPUs** — what Slurm/`CUDA_VISIBLE_DEVICES` gave **this** task.
3. **Logical device** — `cpu` or `cuda:i` among **visible** devices.
4. **Fit** — if the working tensor does not fit, fall back (CPU, shard, or fail clearly). Never assume 1 GPU holds the full LLM.

### 4.1 Current HPC pattern (stage A)

- 7 nodes, **1 GPU per node**, 6 clients + 1 gRPC server.
- Each task typically sees **one** GPU as `cuda:0`.
- Same resolver for server and clients (`slurm_worker._resolve_slurm_device`). Rank 0 may still place a **model** on GPU unless `device_hint: cpu`. **gRPC running SUM is CPU today.**

### 4.2 Same-node many clients (stage C)

Example: **2 nodes** — node 0 has 8 GPUs as **8 clients**; node 1 is **gRPC server**.

- Each client process must get a **unique** `cuda:0` … `cuda:7` (or `CUDA_VISIBLE_DEVICES` so each process only sees one GPU as `cuda:0` — both are valid if **consistent**).
- Server node: GPU optional; aggregation must work with **CPU-only**.

### 4.3 Fat client (stage D)

- Do **not** gather the full model onto the OmniFed leader GPU to Top-K.
- Compress **on the shard** (PP: per layer; TP: per-shard or distributed Top-K).
- Device = that shard’s GPU.

**Hydra:** prefer explicit `device_hint` / per-rank overrides when auto is ambiguous. Auto (`cuda` without index) is **not** enough for 8 clients on one node.

---

## 5. What must be Hydra-modular (final product)

Every row is an independent axis. Combinations are configs, not forks of the codebase.

| Axis | Meaning | Today (approx.) | Later |
|------|---------|-----------------|--------|
| **Federated algorithm** | FedAvg, synchronous DP, … | `algorithm:` | same |
| **Payload** | Aggregate **parameters** vs **gradients** | `engine.classic.aggregate_payload` / `engine.hybrid.aggregate_payload` | Keep as config; it is **not** a separate algorithm class, but it **changes the training hook** (step-then-sync vs sync-then-step). |
| **Communicator** | How ranks talk | `topology.local_comm` / `global_comm` (`GrpcCommunicator`, `TorchDistCommunicator`, hybrid global gRPC) | Central transport swappable; production server “can be anything” behind the same API. |
| **Compression** | Dense / Top-K / QSGD / future | Communicator compressor + `engine.hybrid.global_compression` | Plug-in `Compression` types; local vs global **separate knobs**. |
| **Sync frequency** | How often to aggregate | Classic: `algorithm/schedules/aggregation` (e.g. `batch_end`). Hybrid: local vs global schedules (spec exists). | Facility cadence ≠ central cadence, both Hydra. |
| **In-client trainer** | How one client uses GPUs | Single-process PyTorch | TorchTitan: `gpus`, TP size, PP size, leader rank — Hydra under something like `engine.client_trainer` / `torchtitan`. |

**Payload vs algorithm:**  
The algorithm owns the **round** (epochs, loss, optimizer).  
`aggregate_payload: gradients | params` owns **what** `sync_comm` sends and **when** `optimizer.step` runs. Keep that as config so FedAvg-style param avg and FedSGD-style grad avg share one algorithm package.

**Two compression layers (hierarchical):**  
Inner facility compressor ≠ global gRPC compressor. After local `all_reduce`, the leader has a **dense** facility result, then may compress again for the WAN.

---

## 6. Communicators (do not collapse them)

| Path | Pattern | Wire | SUM lives |
|------|---------|------|-----------|
| **TorchDist (facility or classic)** | Peer `all_reduce` / sparse `all_gather` | NCCL/Gloo, no protobuf | Each rank’s tensor (GPU if NCCL) |
| **Classic gRPC** | Clients → rank-0 server → result back | Protobuf **CPU bytes**, **~2 GB / message** | Server running accum (CPU today; GPU only if it **fits**) |
| **Hybrid global gRPC** | Facility leaders → central server | Separate proto (`LayerState`) | Central server |

Chunking is required when a **dense** payload &gt; 2 GB. Top-K/QSGD often shrink the **wire** so chunking is unnecessary; the server may still need **one** full accumulator to SUM into.

Running SUM on the gRPC server: **one** accum, not N copies. Required before LLM-scale client counts.

---

## 7. Compression and devices

- Math follows **`tensor.device`** (client training GPU / shard GPU). Yaml `device: cuda` must not mean “always `cuda:0`.”
- gRPC: compress on GPU if the tensor is there; copy **only the payload** to CPU for protobuf.
- Server: decompress/SUM/recompress on CPU or GPU **after a fit check**; protobuf pack/unpack stays CPU.
- Correct **global** Top-K of a TP-sharded layer is **not** per-shard `topk` unless we accept that approximation.

---

## 8. What we will not do while organizing the repo

- Do not rename/move files in a way that breaks stage-A yamls (`conf/test_pi_centralized_sync_*`) unless those presets are updated in the **same** change.
- Do not mix hybrid-only knobs into classic presets.
- Do not commit from the agent; humans commit.
- Do not require re-running the full Frontier matrix for every internal rename; keep `_target_` and yaml keys stable, or provide a one-to-one mapping.

---

## 9. Near-term device resolver (concrete)

Keep **one** function used by classic Slurm (and later hybrid):

1. Honor `topology.overrides.<rank>.device_hint` if set (`cpu`, `cuda:3`, …).
2. Else if this rank should use GPU: map **local GPU index** from `LOCAL_RANK` / `SLURM_LOCALID` among `torch.cuda.device_count()` → `cuda:{id}`.
3. Else CPU.
4. Optional: `CUDA_VISIBLE_DEVICES` so each task sees one GPU as `cuda:0` (1 GPU/node today). Both encodings are OK if documented per launch script.

That covers:

- 7×1 GPU (today),
- 8 clients on one 8-GPU node + server on another node,
- later TorchTitan (resolver applies to the **OmniFed leader**; Titan owns the other GPUs).

---

## 10. One-sentence system definition

**OmniFed** is a Hydra-configured federated engine: pluggable **algorithm**, **payload** (grad vs param), **communicator**, **compression**, and **sync schedule**, with a **device/role model** that scales from 1 GPU per client to multi-GPU clients and a central server that may have no GPU.
