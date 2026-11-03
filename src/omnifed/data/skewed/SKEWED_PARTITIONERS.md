# Skewed (non-IID) partitioners

This module splits one centralized dataset among FL clients so that the clients' data differ in a controlled way. The
schemes fall into four families, one file each. Every scheme is a `SkewedPartitioner` (`base.py`) and plugs into the
same Hydra datamodule.

| Family | File | What differs across clients | Schemes |
|---|---|---|---|
| Label distribution skew (prior shift) | `label_distribution.py` | `P(y)`: which classes, how many of each | IID, shards, #C=k, Dirichlet, client Dirichlet, similarity, long tail |
| Feature distribution skew (covariate shift) | `feature_distribution.py` | `P(x)`: each client's inputs pass through its own fixed transform | Gaussian noise, rotation, permutation, corruption, style |
| Concept shift | `concept_shift.py` | `P(y \| x)`: the same input carries a different label | label swap, label permutation, label noise |
| Quantity skew | `quantity_skew.py` | how many samples, with IID labels | Dirichlet, log-normal, power law, explicit |

Real cross-silo partitions (each client is a hospital) are a separate topic: see `FLAMBY.md`.

## Contents

- [All schemes at a glance](#all-schemes-at-a-glance)
- [How every partitioner works](#how-every-partitioner-works)
- [Configs in `conf/datamodule/skewed-data-sim/`](#configs-in-confdatamoduleskewed-data-sim)
- [Using a partitioner from Python](#using-a-partitioner-from-python)
- [Label distribution skew](#label-distribution-skew-label_distributionpy)
- [Feature distribution skew](#feature-distribution-skew-feature_distributionpy)
- [Concept shift](#concept-shift-concept_shiftpy)
- [Quantity skew](#quantity-skew-quantity_skewpy)
- [Combining schemes](#combining-schemes)
- [Testing and printing loaders](#testing-and-printing-loaders)

## All schemes at a glance

The short name is what `make_<family>(name, **kwargs)` accepts, for example `make_label_distribution("dirichlet",
alpha=0.1)`. Every config is in `conf/datamodule/skewed-data-sim/`.

| Class | Short name | Config |
|---|---|---|
| `IIDLabelDistribution` | `iid` | `cal256-iid.yaml` |
| `ShardLabelSkew` | `shards` | `cal256-shard-label-skew.yaml` |
| `QuantityLabelSkew` | `quantity` | `cal256-quantity-label-skew.yaml` |
| `DirichletLabelSkew` | `dirichlet` | `cal256-dirichlet-label-skew.yaml` |
| `ClientDirichletLabelSkew` | `client_dirichlet` | `cal256-client-dirichlet-label-skew.yaml` |
| `SimilarityLabelSkew` | `similarity` | `cal256-similarity-label-skew.yaml`, `full-skew-loader.yaml` |
| `LongTailLabelSkew` | `long_tail` | `cal256-long-tail-label-skew.yaml` |
| `GaussianNoiseFeatureSkew` | `gaussian_noise` | `cal256-gaussian-noise-feature-skew.yaml` |
| `RotationFeatureSkew` | `rotation` | `cal256-rotation-feature-skew.yaml` |
| `PermutationFeatureSkew` | `permutation` | `cal256-permutation-feature-skew.yaml` |
| `CorruptionFeatureSkew` | `corruption` | `cal256-corruption-feature-skew.yaml` |
| `StyleFeatureSkew` | `style` | `cal256-style-feature-skew.yaml` |
| `LabelSwapConceptShift` | `label_swap` | `cal256-label-swap-concept-shift.yaml` |
| `LabelPermutationConceptShift` | `label_permutation` | `cal256-label-permutation-concept-shift.yaml` |
| `LabelNoiseConceptShift` | `label_noise` | `cal256-label-noise-concept-shift.yaml` |
| `DirichletQuantitySkew` | `dirichlet` | `cal256-dirichlet-quantity-skew.yaml` |
| `LogNormalQuantitySkew` | `lognormal` | `cal256-lognormal-quantity-skew.yaml` |
| `PowerLawQuantitySkew` | `power_law` | `cal256-power-law-quantity-skew.yaml` |
| `ExplicitQuantitySkew` | `explicit` | `cal256-explicit-quantity-skew.yaml` |

The `make_*` functions live in their family's file. The registries are `LABEL_DISTRIBUTIONS`, `FEATURE_DISTRIBUTIONS`,
`CONCEPT_SHIFTS` and `QUANTITY_SKEWS`.

## How every partitioner works

`SkewedPartitioner` (`base.py`) does everything except the split itself. A scheme only implements how samples are
divided (`_partition`) and, for feature and concept schemes, a per-client transform.

### Arguments every scheme accepts

| Argument | Default | Meaning |
|---|---|---|
| `num_clients` | `None` | Number of clients. `None` reads `$OMNIFED_NUM_FEDERATED_CLIENTS` (set by the runners). |
| `seed` | `None` | Partition seed. `None` uses `$OMNIFED_PARTITION_SEED`, else 0. It must be the same on every rank. |
| `min_samples_per_client` | `1` (10 for Dirichlet label and quantity schemes) | Raise if a client gets fewer samples; random schemes redraw until met. |
| `stateful_loader` | `False` | `True` returns a torchdata `StatefulDataLoader` (resumable via `state_dict`), `False` a torch `DataLoader`. All configs here set `true`. |
| `loader_kwargs` | `None` | Extra DataLoader kwargs that override the template loader's settings. |
| `label_fn` | `None` | `sample -> label`, only for datasets with no label attribute (`targets`, `labels`, ...). |

### Which client a rank gets

Configs never name a client. When the datamodule is built, the partitioner reads:

- `OMNIFED_FEDERATED_CLIENT_INDEX`: `0 .. N-1` for trainers, `-1` for the server;
- `OMNIFED_NUM_FEDERATED_CLIENTS`: the client count.

The Slurm and Ray runners set both. Every rank uses the same seed to compute the same full partition, then keeps only
its own indices. The full partition exists only briefly while it's being computed.

### Determinism

The same seed, client count and dataset always give the same partition. Each rank logs a `partition_id`, a short hash
of the partition, which must be identical on every rank:

```
[datamodule] DirichletLabelSkew partition_id=… seed=0 client_idx=0/4 len=… classes={…} loader=StatefulDataLoader
```

The per-client transforms of the feature and concept schemes (angles, permutations, label maps, noise) also come from
the seed, and are included in `partition_id`.

### What the datamodule returns

`build_skewed_datamodule(train, eval, partitioner)` (the Hydra `_target_`) rebuilds the `train` loader over this
client's samples, keeping the template's batch size, workers and other settings.

- **Trainer ranks** get `train`, a loader over the client's partition. Feature and concept schemes wrap it in
  `ClientTransformDataset`, which applies the client's transform on the fly.
- **`eval`** is the shared, unpartitioned test set, except under concept shift with `eval_follows_concept`, where each
  trainer's eval labels are mapped like its training labels.
- **The server** (`-1`) gets `train=None` and the clean eval set.

### Statistics

`partitioner.summary(dataset)` returns:

- `sizes`, `class_counts`, `classes_per_client`;
- `coverage` (share of samples assigned to some client);
- `hellinger` / `mean_hellinger` (distance of each client's label mix from the global one: 0 means IID);
- `partition_id`, `seed`;
- plus family-specific keys: `client_features` for feature skew, `observed_class_counts` and `relabelled_fraction` for
  concept shift.

### Schemes that need class labels

Label schemes and concept shift (`NEEDS_LABELS = True`) refuse datasets that declare `has_class_labels = False`, such
as FLamby segmentation and survival datasets. IID and quantity schemes work on any dataset.

## Configs in `conf/datamodule/skewed-data-sim/`

The configs use Caltech-256 (257 classes, ImageFolder layout):

```
$OMNIFED_DATA_DIR/Caltech256/{train,test}/<NNN.class-name>/*.jpg
```

### `cal256-common.yaml`

This holds the shared data and loader settings and can't be selected on its own (it has no `partitioner`):

- `_target_: src.omnifed.data.skewed.base.build_skewed_datamodule`;
- the `train` loader template: ImageFolder with augmentation (RandomResizedCrop 224, flip, rotation, colour jitter),
  `ToTensor`, ImageNet `Normalize`, batch 32;
- the `eval` loader: Resize 256, CenterCrop 224, `Normalize`, batch 64, over the full test set;
- loader defaults (shuffle, workers, pin memory) from `conf/datamodule/base.yaml`.

### Per-scheme configs (`cal256-<scheme>.yaml`)

Each one includes the common file and adds a `partitioner:` block:

```yaml
defaults:
  - /datamodule/skewed-data-sim/cal256-common@_here_   # absolute path, merged into this package
  - _self_

partitioner:
  _target_: src.omnifed.data.skewed.label_distribution.DirichletLabelSkew
  alpha: 0.5
  balance: true
  min_samples_per_client: 10
  max_retries: 1000
  stateful_loader: true
  # seed: 0           # default $OMNIFED_PARTITION_SEED, else 0
  # num_clients: 4    # default $OMNIFED_NUM_FEDERATED_CLIENTS
```

The include must be the absolute `/datamodule/...` path with `@_here_`. A relative `- cal256-common` resolves against
`conf/datamodule/`, not this folder, and fails with `MissingConfigException`.

### `full-skew-loader.yaml`

The same as `cal256-similarity-label-skew.yaml`, but fully self-contained: no `defaults:`, every setting written out.
Read it to see everything a scheme config expands to.

### Running and overriding

```bash
python main.py datamodule=skewed-data-sim/cal256-dirichlet-label-skew
python main.py datamodule=skewed-data-sim/cal256-dirichlet-label-skew datamodule.partitioner.alpha=0.1
python main.py datamodule=skewed-data-sim/cal256-rotation-feature-skew '+datamodule.partitioner.split={_target_: src.omnifed.data.skewed.label_distribution.DirichletLabelSkew, alpha: 0.5}'
python main.py datamodule=skewed-data-sim/cal256-iid +datamodule.partitioner.seed=3   # key not in the yaml: add with +
```

To inspect the composed config without running: `python main.py datamodule=skewed-data-sim/<config> --cfg job
--package datamodule`.

### Making a config for another dataset

1. Copy `cal256-common.yaml` and change the `train` / `eval` datasets and transforms. Any dataset works if its labels are
   readable (`targets` / `labels` attribute, or set `label_fn`).
2. Copy a scheme config and point its include at your common file.
3. For feature schemes, set `input_mean` / `input_std` to your `Normalize` values, or leave them unset if you don't
   normalize.

## Using a partitioner from Python

```python
import os
from torchvision import datasets
from src.omnifed.data.skewed.label_distribution import DirichletLabelSkew

os.environ.update(OMNIFED_NUM_FEDERATED_CLIENTS="4", OMNIFED_FEDERATED_CLIENT_INDEX="1")
train = datasets.ImageFolder(f"{os.environ['OMNIFED_DATA_DIR']}/Caltech256/train", transform=...)
p = DirichletLabelSkew(alpha=0.5, stateful_loader=True)

loader = p.client_loader(train, batch_size=32)      # this rank's loader (client 1 of 4)
idx = p.client_indices(train)                       # this rank's sample indices
parts = p.partition(train)                          # every client's indices (analysis only)
print(p.partition_id(train), p.summary(train)["sizes"])
```

Other methods:

| Method | Returns |
|---|---|
| `client_dataset(dataset)` | This rank's `Subset`, or `ClientTransformDataset` for feature / concept schemes. |
| `partition_info(dataset)` | Cached `PartitionResult`: id, seed, classes, class counts, this client's indices. |
| `class_counts(dataset)` | `[num_clients, num_classes]` count table. |
| `client_eval_dataset(eval, client, num_clients, classes)` | Eval set for a trainer (relabelled under concept shift). |

## Label distribution skew (`label_distribution.py`)

`P(y)` differs across clients while `P(x | y)` stays the same. All of these extend `LabelDistribution`.

### `IIDLabelDistribution`

The homogeneous baseline: a uniformly random split into near-equal parts. It takes no arguments beyond the common ones.
It ignores labels, so it works on any dataset.

Config: `cal256-iid.yaml`.

### `ShardLabelSkew`

The pathological shard split from FedAvg (McMahan et al., 2017):

1. Sort the samples by label (order within a class is randomized).
2. Cut them into `num_clients * shards_per_client` equal shards.
3. Give each client `shards_per_client` random shards.

With 2 shards per client, each client sees about 2 classes (a shard can straddle a class boundary).

| Argument | Default | Meaning |
|---|---|---|
| `shards_per_client` | 2 | Shards (≈ classes) per client. |

Config: `cal256-shard-label-skew.yaml` (`shards_per_client: 2`).

### `QuantityLabelSkew` (#C=k)

NIID-Bench's quantity-based label imbalance (Li et al., 2022). Each client gets exactly `classes_per_client` distinct
classes:

- client `i` always gets class `i % C`, so every class is covered when there are at least `C` clients;
- each class's samples are split evenly among the clients that own it;
- classes no client owns are dropped, with a warning.

| Argument | Default | Meaning |
|---|---|---|
| `classes_per_client` | 2 | Classes per client (k). |

Config: `cal256-quantity-label-skew.yaml` (`classes_per_client: 32` for Caltech's 257 classes).

### `DirichletLabelSkew`

Distribution-based label imbalance (Yurochkin et al., 2019; NIID-Bench). For each class `c`, draw
`p_c ~ Dir(alpha · 1_K)` and split that class's samples over the K clients in proportions `p_c`. Smaller `alpha` means
more skew; `alpha → ∞` approaches IID. Common values are 0.1, 0.3, 0.5 and 1.0. The whole draw is repeated until every
client has `min_samples_per_client` samples.

| Argument | Default | Meaning |
|---|---|---|
| `alpha` | 0.5 | Dirichlet concentration (> 0). |
| `balance` | `True` | NIID-Bench trick: while splitting a class, skip clients that already hold ≥ N/K samples, so sizes stay near-equal and label skew isn't confounded with quantity skew. |
| `min_samples_per_client` | 10 | Redraw until every client has at least this many. |
| `max_retries` | 1000 | Give up (`RuntimeError`) after this many redraws. |

Config: `cal256-dirichlet-label-skew.yaml`.

### `ClientDirichletLabelSkew`

A per-client class mixture (Hsu, Qi & Brown, 2019). Each client draws its own class distribution
`q_i ~ Dir(alpha · C · prior)` and then samples `samples_per_client` points from it without replacement.

`DirichletLabelSkew` fixes how each class is spread over clients; this scheme instead fixes each client's size and draws
its label mix. When a client's preferred classes run out, the rest of its samples come from the classes still
available, in proportion to availability. Clients are filled in random order so this isn't tied to the client index.

| Argument | Default | Meaning |
|---|---|---|
| `alpha` | 0.5 | Per-class concentration (comparable to `DirichletLabelSkew`'s `alpha`). |
| `samples_per_client` | `None` | Samples per client; `None` uses all the data (N // K each). |
| `prior` | `"global"` | Centre of the draw: `"global"` (the dataset's class frequencies) or `"uniform"`. |

Config: `cal256-client-dirichlet-label-skew.yaml` (`prior: global`, because Caltech is imbalanced).

### `SimilarityLabelSkew`

The similarity-controlled split from SCAFFOLD (Karimireddy et al., 2020). A fraction `similarity` of the data is split
IID. The rest is sorted by label and cut into contiguous blocks, one per client, with blocks assigned to clients at
random. `similarity=1` is IID; `similarity=0` gives each client about C/K contiguous classes.

| Argument | Default | Meaning |
|---|---|---|
| `similarity` | 0.1 | Share of data split IID, in [0, 1]. |

Configs: `cal256-similarity-label-skew.yaml` and `full-skew-loader.yaml` (both use `similarity: 0.2`).

### `LongTailLabelSkew`

A global long-tail subsample followed by an inner scheme (Cao et al., 2019; CIFAR-LT as used by CReFF and FedLT).

1. The class ranked `r` in `class_order` keeps `n_max · imbalance_factor^(−r/(C−1))` samples, so the head/tail size ratio
   is `imbalance_factor`.
2. The kept subset is partitioned by `base`. Samples that weren't kept go to no client, so `coverage < 1`.

| Argument | Default | Meaning |
|---|---|---|
| `base` | `DirichletLabelSkew(alpha=0.5)` | Inner scheme. It runs with this object's seed and client count; its own seed and loader options are ignored. |
| `imbalance_factor` | 100 | `n_max / n_min` (≥ 1); 1 means no subsampling. |
| `class_order` | `"index"` | `"index"` (class 0 is the head, standard CIFAR-LT) or `"random"` (seeded). |

Config: `cal256-long-tail-label-skew.yaml` (`imbalance_factor: 10`, `base` = Dirichlet `alpha: 0.5`).

## Feature distribution skew (`feature_distribution.py`)

`P(x)` differs while the labelling rule `P(y | x)` is shared: each client sees the same kind of data, altered by its own
fixed transform. All of these extend `FeatureDistribution`.

### Shared design

- **Which samples a client gets** is decided by `split`, any index scheme (default `IIDLabelDistribution`). So feature
  skew combines with label skew, for example rotation on top of a Dirichlet split.
- **What happens to them** is a per-client transform applied on the fly by `ClientTransformDataset`.
- **Transforms act on tensors**, so the dataset transform must include `ToTensor`. If it also normalizes, set
  `input_mean` / `input_std` to the same values. The transform then runs in [0, 1] pixel space: undo Normalize,
  transform, clamp, redo Normalize. Corruption and style need this; without it they assume inputs are already in
  [0, 1]. All `cal256-*-feature-skew.yaml` configs set the ImageNet statistics.
- **Reproducible**: per-client parameters (angles, permutations, style factors) come from the seed and client index;
  per-sample noise from the seed, client index and sample index. Transforms are small picklable classes, so DataLoader
  workers (spawned on macOS) can use them.
- **The eval set is not transformed**: clients train on shifted data and are evaluated on the shared clean test set.

Arguments every feature scheme takes:

| Argument | Default | Meaning |
|---|---|---|
| `split` | `IIDLabelDistribution()` | Index scheme for which samples each client gets. |
| `input_mean`, `input_std` | `None` | The dataset's `Normalize` statistics; set them if the dataset normalizes. |

### `GaussianNoiseFeatureSkew`

Noise-based feature skew (NIID-Bench). Client `i` of K adds Gaussian noise with standard deviation `sigma · i / (K−1)`:
client 0 is clean and the last client gets `sigma`. `sigma` is in pixel units when `input_mean` / `input_std` are set,
and in tensor units otherwise.

| Argument | Default | Meaning |
|---|---|---|
| `sigma` | 0.1 | Noise level of the noisiest client (≥ 0). |
| `fixed_per_sample` | `True` | Noise fixed per (seed, client, sample), which is reproducible and resumable. `False` draws fresh noise on every read, as NIID-Bench does; that isn't reproducible. |

Config: `cal256-gaussian-noise-feature-skew.yaml`.

### `RotationFeatureSkew`

Rotated MNIST/CIFAR style (for example IFCA, Ghosh et al., 2020). Client `i` rotates every image counter-clockwise by
`angles[i % len(angles)]` degrees. Multiples of 90° on square images are exact; other angles use bilinear interpolation,
keep the image size, and fill the corners with black.

| Argument | Default | Meaning |
|---|---|---|
| `angles` | `(0, 90, 180, 270)` | Rotation per client group, in degrees. |

Config: `cal256-rotation-feature-skew.yaml`.

### `PermutationFeatureSkew`

Permuted MNIST style. Each client applies its own fixed random permutation to the image's `patch_size × patch_size`
patches. The same permutation is used for every sample and channel of that client. `patch_size=1` shuffles individual
pixels, the classic setting; larger patches keep local structure, like a jigsaw.

| Argument | Default | Meaning |
|---|---|---|
| `patch_size` | 1 | Side of the square patches; must divide the image size. |
| `include_identity` | `True` | Client 0 keeps the original layout. |

Config: `cal256-permutation-feature-skew.yaml` (`patch_size: 32`, which divides 224).

### `CorruptionFeatureSkew`

ImageNet-C / CIFAR-10-C style (Hendrycks & Dietterich, 2019). Client `i` applies `corruptions[i % len(corruptions)]` at
severity `severity[i % len(severity)]`, from 1 (mild) to 5 (severe), using ImageNet-C's severity parameters
implemented on tensors.

Available corruptions: `none`, `gaussian_noise`, `shot_noise`, `impulse_noise`, `speckle_noise`, `gaussian_blur`,
`contrast`, `brightness`, `pixelate`, `jpeg_compression`.

| Argument | Default | Meaning |
|---|---|---|
| `corruptions` | 8 types (`gaussian_noise` … `jpeg_compression`) | Corruption per client group. |
| `severity` | 3 | One severity for all clients, or a list cycled over clients. |

Config: `cal256-corruption-feature-skew.yaml` (it includes `none` first, so client 0 is clean).

### `StyleFeatureSkew`

Style / colour shift. Each client draws one fixed set of factors applied to all its images:

- brightness, contrast and saturation factors uniform in `[1−x, 1+x]`;
- a hue shift uniform in `[−hue, hue]`;
- grayscale with probability `grayscale_prob`.

| Argument | Default | Meaning |
|---|---|---|
| `brightness`, `contrast`, `saturation` | 0.4 | Spread of the factors, in [0, 1). |
| `hue` | 0.1 | Spread of the hue shift, in [0, 0.5]. |
| `grayscale_prob` | 0.0 | Chance that a client sees grayscale images. |

Config: `cal256-style-feature-skew.yaml`.

## Concept shift (`concept_shift.py`)

The same inputs carry different labels on different clients: `P(y | x)` differs. Inputs are untouched. (Same label but
different inputs is feature skew.) All of these extend `ConceptShift`.

### Shared design

- **Which samples a client gets** is decided by `split` (any index scheme, default `IIDLabelDistribution`).
- **What its labels become** is a per-client target transform, applied on the fly.
- **Label values**: mappings act on the dataset's own label values, so any label type works. Labels the mapping has
  never seen pass through unchanged.
- **Reproducible**: a group's mapping comes from (seed, group); which clients are noisy, and how noisy, from the seed.
- **Evaluation**: with `eval_follows_concept` (default `True` for swap and permutation), each trainer's eval loader is
  relabelled with its group's mapping, which is what clustered and personalized FL report. Label noise defaults to
  clean eval, the noisy-label protocol. The server always keeps the clean eval set.

Arguments every concept-shift scheme takes:

| Argument | Default | Meaning |
|---|---|---|
| `split` | `IIDLabelDistribution()` | Index scheme for which samples each client gets. |
| `eval_follows_concept` | `True` (`False` for label noise) | Relabel each trainer's eval set with its client's mapping. |

Swap and permutation are group schemes (`GroupConceptShift`): client `i` belongs to group `i % num_groups`, and all
clients of a group share one mapping. They also take:

| Argument | Default | Meaning |
|---|---|---|
| `num_groups` | 2 | Number of concepts (≥ 1). The groups are the ground-truth clusters. |
| `include_identity` | `True` | Group 0 keeps the original labels. |

### `LabelSwapConceptShift`

Clustered-FL benchmarks (for example CFL, Sattler et al., 2020). Each group swaps `swap_pairs` random, disjoint pairs of
classes (`a ↔ b`).

| Argument | Default | Meaning |
|---|---|---|
| `swap_pairs` | 1 | Pairs swapped per group; needs `2 · swap_pairs ≤ num_classes`. |

Config: `cal256-label-swap-concept-shift.yaml` (`num_groups: 2`, `swap_pairs: 32`).

### `LabelPermutationConceptShift`

Clustered / multi-task FL (IFCA- and FedEM-style). Each group relabels every class with its own random derangement, so
no class keeps its label. This is the strongest form: no input has the same label in two groups. It takes no extra
arguments.

Config: `cal256-label-permutation-concept-shift.yaml` (`num_groups: 2`).

### `LabelNoiseConceptShift`

Noisy-label FL (for example FedCorr, Xu et al., 2022). `round(noisy_client_fraction · K)` clients, chosen by the seed,
are noisy. Each noisy client flips its labels with its own rate, and each flip is fixed per sample.

| Argument | Default | Meaning |
|---|---|---|
| `noise_type` | `"symmetric"` | `"symmetric"` flips to a random other class; `"pairflip"` flips class k to k+1 (by sorted label order). |
| `noisy_client_fraction` | 0.5 | Share of clients that are noisy, in [0, 1]. |
| `noise_rate` | 0.4 | Flip probability, or `[low, high]`: each noisy client draws its rate uniformly in that range. |
| `eval_follows_concept` | `False` | Evaluate on clean labels. |

Config: `cal256-label-noise-concept-shift.yaml` (`noise_rate: [0.3, 0.5]`).

## Quantity skew (`quantity_skew.py`)

Clients hold different amounts of data, but each client's label mix still matches the global one. Label schemes can
also produce unequal sizes, but only as a side effect; these schemes control size on its own.

Every scheme here works the same way:

1. Draw per-client size proportions.
2. Round them to sizes that sum to the dataset size (largest remainder).
3. Deal a seeded random permutation of the samples out in those sizes.

All the data is used. Random schemes are redrawn until every client has `min_samples_per_client`; deterministic ones
raise right away if a client would fall short. All of these extend `QuantitySkew`, so they also take:

| Argument | Default | Meaning |
|---|---|---|
| `min_samples_per_client` | 10 | Smallest allowed client. |
| `max_retries` | 1000 | Redraws allowed for random schemes. |

### `DirichletQuantitySkew`

From NIID-Bench (Li et al., 2022). Size proportions `q ~ Dir(beta · 1_K)`. Smaller `beta` means more unequal sizes;
`beta → ∞` approaches equal sizes.

| Argument | Default | Meaning |
|---|---|---|
| `beta` | 0.5 | Dirichlet concentration (> 0). NIID-Bench uses 0.5. |

Config: `cal256-dirichlet-quantity-skew.yaml`.

### `LogNormalQuantitySkew`

LEAF-style unbalanced splits (FedLab's `unbalance_sgm`). Client sizes are proportional to `exp(N(0, sigma²))`:
`sigma = 0` gives equal sizes, and larger `sigma` gives a heavier tail of large clients.

| Argument | Default | Meaning |
|---|---|---|
| `sigma` | 1.0 | Standard deviation of the log-sizes (≥ 0). |

Config: `cal256-lognormal-quantity-skew.yaml`.

### `PowerLawQuantitySkew`

The client of rank `r` (0 = largest) gets a share proportional to `(r+1)^(−exponent)`, so the largest client holds
`K^exponent` times the smallest. `exponent = 0` gives equal sizes; 1 is Zipf-like.

| Argument | Default | Meaning |
|---|---|---|
| `exponent` | 1.0 | Power-law exponent (≥ 0). |
| `shuffle_order` | `True` | Assign ranks to clients in seeded random order. `False` makes client 0 the largest and client K−1 the smallest. |

Config: `cal256-power-law-quantity-skew.yaml`.

### `ExplicitQuantitySkew`

Client `i` gets `proportions[i]` of the data, for example to mirror the data volumes of specific sites in a cross-silo
deployment. Proportions are normalized, so `[2, 1, 1]` and `[0.5, 0.25, 0.25]` are the same. There must be exactly one
entry per client.

| Argument | Default | Meaning |
|---|---|---|
| `proportions` | `(0.4, 0.3, 0.2, 0.1)` | One positive weight per client. |

Config: `cal256-explicit-quantity-skew.yaml`, which has 4 proportions and so needs `OMNIFED_NUM_FEDERATED_CLIENTS=4`.

## Combining schemes

Partitioners nest wherever a scheme takes another scheme as an argument:

| Combination | Argument | Example |
|---|---|---|
| Feature skew on a label-skewed split | `split` | rotation over Dirichlet(0.5) |
| Concept shift on a label-skewed split | `split` | label noise over shards |
| Long tail, then any label scheme | `base` | long tail then shards |
| FLamby natural partition, a shared center split by any scheme | `within_center` | see `FLAMBY.md` |

The inner scheme runs with the outer scheme's seed and client count. Its own `seed`, `num_clients` and loader options
are ignored. In yaml:

```yaml
partitioner:
  _target_: src.omnifed.data.skewed.feature_distribution.RotationFeatureSkew
  angles: [0, 90, 180, 270]
  input_mean: [0.485, 0.456, 0.406]
  input_std: [0.229, 0.224, 0.225]
  stateful_loader: true
  split:
    _target_: src.omnifed.data.skewed.label_distribution.DirichletLabelSkew
    alpha: 0.5
```

## Testing and printing loaders

Each scheme has a test file, one test package per family. The tests need Caltech-256 at
`$OMNIFED_DATA_DIR/Caltech256/{train,test}`. Tests that need the data skip without it; the others use synthetic labels.

| Family | Tests |
|---|---|
| Label | `test_label_dist/test_{iid_label_distribution,shard,quantity,dirichlet,client_dirichlet,similarity,long_tail}_*.py` |
| Feature | `test_feature_dist/test_{gaussian_noise,rotation,permutation,corruption,style}_feature_skew.py` |
| Concept | `test_concept_shift/test_{label_swap,label_permutation,label_noise}_concept_shift.py` |
| Quantity | `test_quantity_skew/test_{dirichlet,lognormal,power_law,explicit}_quantity_skew.py` |

```bash
export OMNIFED_DATA_DIR=/path/to/datasets                    # contains Caltech256/{train,test}
python -m pytest src/omnifed/data/skewed/test_label_dist -v -rs      # one family
python -m pytest src/omnifed/data/skewed/test_quantity_skew/test_power_law_quantity_skew.py -v -rs
python -m pytest src/omnifed/data/skewed -v -rs --ignore=src/omnifed/data/skewed/test_flamby   # all four families
```

### What the tests check

Every scheme's test class mixes in a family suite (`SkewSuite` in `test_label_dist/_common.py`, extended by
`FeatureSkewSuite`, `ConceptShiftSuite` and `QuantitySkewSuite`). Each class sets `CONFIG` to its yaml and
`make_partitioner()` to the same scheme built in Python.

`SkewSuite`, the checks every scheme gets:

- the yaml composes (`test_config_composes`);
- the yaml builds the same partitioner as the Python one (`test_matches_hydra_config`);
- a trainer rank gets a `StatefulDataLoader` over exactly its partition (`test_trainer_gets_stateful_loader`);
- the server gets no train loader (`test_server_gets_no_train_loader`);
- clients are disjoint and every rank computes the same partition (`test_clients_disjoint_and_agree`).

The family suites add:

| Suite | Checks |
|---|---|
| `FeatureSkewSuite` | Transforms are deterministic, shape-preserving, picklable and identical on every rank. `split` decides the assignment and `partition_id` covers the transforms. The yaml's `input_mean` / `input_std` match `Normalize`. Client samples are actually transformed. |
| `ConceptShiftSuite` | Target transforms are deterministic, valid, picklable and identical on every rank. Inputs are untouched while labels change. `eval_follows_concept` is honoured. Label types are preserved. |
| `QuantitySkewSuite` | Sizes sum to the dataset and respect the minimum. The split is disjoint and complete. Labels stay IID. The partition is the same on every rank and changes with the seed. |

Individual test files add scheme-specific checks, for example that skew grows as `alpha` shrinks, or that a stateful
loader resumes from `state_dict`.

### Printing one client's loaders

Run any scheme's test file as a module. It builds the datamodule from that scheme's Python partitioner, as a trainer
rank would, and prints:

- the train and eval loaders;
- the `partition_id` and seed;
- client sizes, classes per client, coverage and mean Hellinger distance.

```bash
OMNIFED_FEDERATED_CLIENT_INDEX=2 python -m src.omnifed.data.skewed.test_label_dist.test_dirichlet_label_skew
OMNIFED_FEDERATED_CLIENT_INDEX=0 python -m src.omnifed.data.skewed.test_feature_dist.test_rotation_feature_skew
```

The client count is fixed at 4 (`NUM_CLIENTS` in `test_label_dist/_common.py`). Change `OMNIFED_FEDERATED_CLIENT_INDEX`
to see another client.
