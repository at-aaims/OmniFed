# FLamby datasets in OmniFed

[FLamby](https://github.com/owkin/FLamby) (Ogier du Terrail et al., NeurIPS 2022 Datasets & Benchmarks) is a benchmark of
cross-silo healthcare datasets. In each dataset the data comes from real hospitals, regions or scanners, so the
natural FL clients are those centers. This guide covers the five datasets wired into OmniFed:

| Dataset | Centers | Task | Download | Config |
|---|---|---|---|---|
| `fed_heart_disease` | 4 hospitals | tabular, binary classification | ~40 KB, automatic | `conf/datamodule/flamby/flamby.yaml` (default dataset), `heart-disease-pooled-dirichlet-label.yaml` |
| `fed_tcga_brca` | 6 regions | tabular survival (no class labels) | none (ships with FLamby) | `conf/datamodule/flamby/tcga-brca-natural.yaml` |
| `fed_ixi` | 3 hospitals | 3D brain MRI segmentation | ~444 MB, automatic | `conf/datamodule/flamby/ixi-natural-dirichlet-quantity.yaml` |
| `fed_isic2019` | 6 centers | dermoscopy images, 8 classes | ~9 GB, automatic (~1 GB kept) | `conf/datamodule/flamby/isic2019-natural.yaml` |
| `fed_lidc_idri` | 4 CT scanner makers | 3D CT lung-nodule segmentation | ~133 GB, automatic, hours | `conf/datamodule/flamby/lidc-idri-natural.yaml` |

Camelyon16 and KiTS19 are also in the registry, but they have to be prepared by hand, so this guide doesn't cover them.

## How it fits together

Downloading and loading are two separate steps, done by different modules:

| File | Role                                                                                                                                                       |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `flamby_download.py` | Downloads the data and puts it in `$OMNIFED_DATA_DIR/flamby/<dataset>/` only if it is missing. It also holds the dataset registry (`FLAMBY_DATASETS`).     |
| `flamby_partition.py` | Loads data that is already downloaded and partitions it for this rank. `build_flamby_datamodule` is the Hydra target of the configs. It never downloads.   |
| `flamby_compat.py` | Lets FLamby's 2024 code run unchanged on current libraries: NumPy 2, MONAI ≥ 1.3, albumentations ≥ 1.4.24 / 2.x, pydicom 3, and no `efficientnet-pytorch`. |
| `check_isic.py`, `check_lidc.py` | Build one rank's loaders from the config and check one batch.                                                                                              |
| `test_flamby/test_fed_<dataset>.py` | Unit tests on the real data. They download the dataset if it is missing.                                                                                   |

A dataset counts as downloaded once its ready-marker `$OMNIFED_DATA_DIR/flamby/<dataset>/.omnifed_flamby_ready.json` has
been written. A dataset that is already there is never downloaded again. If several processes start the same download at
once, a file lock makes one of them download while the others wait.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `OMNIFED_DATA_DIR` | download and load | Data root. Datasets live in `<root>/flamby/<dataset>/`. |
| `OMNIFED_FLAMBY_ACCEPT_LICENSE=1` | download | Confirms you have read and accepted the dataset's licence. Required to download. Not needed to load data that is already downloaded. |
| `OMNIFED_NUM_FEDERATED_CLIENTS` | load | Number of FL clients (set by the runners). |
| `OMNIFED_FEDERATED_CLIENT_INDEX` | load | This rank's client: `0 .. N-1`, or `-1` for the server (set by the runners). |
| `OMNIFED_PARTITION_SEED` | load | Partition seed (default 0). The same seed gives the same partition on every rank. |
| `OMNIFED_FLAMBY_DIR` | both | A FLamby checkout to use instead of `third_party/FLamby`. |
| `OMNIFED_FLAMBY_IXI_URL`, `OMNIFED_FLAMBY_ISIC_URL` | download | Override the IXI / ISIC download URL. |
| `OMNIFED_FLAMBY_LIDC_LIMIT`, `OMNIFED_FLAMBY_LIDC_SOURCE`, `OMNIFED_FLAMBY_LIDC_KEEP_DICOM`, `OMNIFED_FLAMBY_WORKERS` | download | LIDC options (see [Fed-LIDC-IDRI](#fed-lidc-idri)). |

## One-time setup

FLamby is used from source, as the git submodule `third_party/FLamby` pinned at commit `edacf54`. Don't `pip install` it:
its `setup.py` installs extra packages even with `--no-deps`. Run these from the repo root:

```bash
git submodule update --init third_party/FLamby        # already added to .gitmodules
pip install pandas scikit-learn matplotlib seaborn pyyaml tqdm wget lifelines   # what FLamby itself imports
export OMNIFED_DATA_DIR=/path/to/datasets
```

Each dataset needs a few more packages (listed below). To see every dataset's licence, size, packages and download
status:

```bash
python -m src.omnifed.data.skewed.flamby_download --list
```

FLamby writes its own records (licence agreements, `dataset_location.yaml`) inside the submodule. `.gitmodules` has
`ignore = untracked` so that these files don't show up as changes.

## Configs and partition schemes

All FLamby configs are in `conf/datamodule/flamby/`:

- **`flamby.yaml`**: the generic config. Its `dataset:` field selects the dataset (default `fed_heart_disease`).
  Loader settings (`batch_size`, `num_workers`, FLamby's `collate_fn`) default to each dataset's FLamby benchmark
  values, and you can override them.
- **`scheme/*.yaml`**: the partition schemes. Pick one with
  `datamodule/flamby/scheme@datamodule.partitioner=<scheme>`.

  | Scheme | What clients get | Needs class labels |
  |---|---|---|
  | `natural` (default) | Real centers. Centers are merged when there are fewer clients and split IID when there are more. | no |
  | `natural-dirichlet-quantity` | Natural, but a shared center is split with Dirichlet sizes. | no |
  | `natural-dirichlet-label` | Natural, but a shared center is split by Dirichlet label proportions. | yes |
  | `pooled-iid` | All centers pooled and dealt out uniformly. | no |
  | `pooled-dirichlet-quantity` | Pooled data, Dirichlet client sizes. | no |
  | `pooled-dirichlet-label` | Pooled data, Dirichlet label proportions. | yes |

  Label schemes refuse datasets without class labels (TCGA-BRCA, IXI, LIDC-IDRI) with a clear error.
- **Per-dataset configs** (`tcga-brca-natural.yaml`, `isic2019-natural.yaml`, `lidc-idri-natural.yaml`,
  `ixi-natural-dirichlet-quantity.yaml`, `heart-disease-pooled-dirichlet-label.yaml`) each fix a dataset and a scheme.
  To combine any dataset with any scheme, copy one:

  ```yaml
  defaults:
    - /datamodule/flamby/flamby@_here_
    - override /datamodule/flamby/scheme@partitioner: pooled-dirichlet-quantity
    - _self_
  dataset: fed_ixi
  partitioner:
    beta: 0.3        # scheme parameters can be changed here too
  ```

Usage:

```bash
python main.py datamodule=flamby/isic2019-natural                                  # a per-dataset config
python main.py datamodule=flamby/flamby datamodule.dataset=fed_ixi                 # the generic config
python main.py datamodule=flamby/flamby datamodule.dataset=fed_heart_disease \
    datamodule/flamby/scheme@datamodule.partitioner=pooled-dirichlet-label         # dataset + scheme
```

**What every rank gets.** The number of clients and this rank's client come only from `OMNIFED_NUM_FEDERATED_CLIENTS` /
`OMNIFED_FEDERATED_CLIENT_INDEX`. A trainer gets:

- `train`: a `StatefulDataLoader` over its own share;
- `eval`: the test data of its own centers with the natural schemes, or the full test set with the pooled schemes.

The server (`-1`) gets `train=None` and the full test set, and doesn't load the train split at all. Every rank logs the
same `partition_id`.

**How natural splits sizes.** Merging is balanced: the largest center goes to the currently smallest client. Splitting is
proportional: every center keeps one client, extra clients are added in proportion to center size, and clients `0..C-1`
always keep centers `0..C-1`. With ISIC (center sizes 9,930 / 3,163 / 2,691 / 1,807 / 655 / 351), 3 clients get
[0] · [1,4,5] · [2,3] = 9,930 · 4,169 · 4,498 training images. With 8 clients, centers 0 and 1 are each split in two.

## Printing loaders for any dataset

`check_isic.py` and `check_lidc.py` (below) are dataset-specific. For any dataset, save this snippet as a file and run it
from the repo root. Use a file rather than `python - <<EOF`: on macOS, DataLoader worker processes re-import the main
script, so they need a real file.

```python
# print_flamby.py — usage: python print_flamby.py fed_ixi [hydra overrides...]
import os, sys
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

def show(b):
    return tuple(b.shape) if hasattr(b, "shape") else type(b).__name__

if __name__ == "__main__":
    dataset, *overrides = sys.argv[1:] or ["fed_heart_disease"]
    with initialize_config_dir(config_dir=os.path.abspath("conf"), version_base=None):
        cfg = compose(config_name="base", overrides=["datamodule=flamby/flamby", f"datamodule.dataset={dataset}", *overrides])
    dm = instantiate(cfg.datamodule)
    for split, loader in (("train", dm.train), ("eval", dm.eval)):
        if loader is None:
            print(split, "None (server)")
            continue
        x, y = next(iter(loader))[:2]
        print(split, type(loader).__name__, len(loader.dataset), "samples | batch", show(x), show(y))
```

```bash
OMNIFED_NUM_FEDERATED_CLIENTS=4 OMNIFED_FEDERATED_CLIENT_INDEX=0 python print_flamby.py fed_heart_disease
```

## Fed-Heart-Disease

4 hospitals (Cleveland, Hungary, Switzerland, Long Beach VA), 13 tabular features, binary label. Licence CC BY 4.0.

**Download.** This runs FLamby's own download script (~40 KB). It needs `pip install wget`.

```bash
OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_heart_disease
```

FLamby's raw `*.data` files end up in `$OMNIFED_DATA_DIR/flamby/fed_heart_disease/`.

**Config.** `flamby.yaml` uses Heart Disease by default, with the `natural` scheme, batch size 4 and no workers.
`heart-disease-pooled-dirichlet-label.yaml` is an example with the pooled Dirichlet label scheme and `alpha: 0.3`.

**Print loaders.** Expect batches of `x (4, 13)` and `y (4, 1)`, with labels 0/1.

```bash
OMNIFED_NUM_FEDERATED_CLIENTS=4 OMNIFED_FEDERATED_CLIENT_INDEX=0 python print_flamby.py fed_heart_disease
```

**Test.**

```bash
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_heart_disease.py -v -rs
```

Besides the shared checks (see [Tests](#tests)), these tests also cover:

- the licence gate;
- that the datamodule never downloads;
- four processes downloading into a fresh folder at once, with only one actually downloading;
- the `flamby_download` command line;
- both example configs.

## Fed-TCGA-BRCA

6 regions, 39 clinical features, survival target `(event, time)`. Licence: GDC open-access data terms. The splits are
fixed: 248 / 156 / 164 / 129 / 129 / 40 training and 63 / 40 / 42 / 33 / 33 / 11 test patients.

**Download.** The data ships inside FLamby (`third_party/FLamby/flamby/datasets/fed_tcga_brca/brca.csv`), so nothing is
written to `$OMNIFED_DATA_DIR`. FLamby asks for the licence on stdin every time TCGA is loaded until its agreement file
exists, so record it once. This needs `pip install lifelines`.

```bash
OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_tcga_brca
```

**Config.** `tcga-brca-natural.yaml`: the `natural` scheme, batch size 8, no workers. TCGA has no class labels, so
only `natural`, `natural-dirichlet-quantity`, `pooled-iid` and `pooled-dirichlet-quantity` work.

**Print loaders.** Expect `x (8, 39)` and `y (8, 2)`.

```bash
OMNIFED_NUM_FEDERATED_CLIENTS=6 OMNIFED_FEDERATED_CLIENT_INDEX=2 python print_flamby.py fed_tcga_brca
```

**Test.**

```bash
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_tcga_brca.py -v -rs
```

## Fed-IXI

3 hospitals (Guys, HH, IOP), IXI-Tiny T1 brain MRI with brain masks (566 subjects). FLamby's split is 249 / 145 / 59
training and 62 / 36 / 15 test subjects. Licence CC BY-SA 3.0 for IXI; acknowledge TorchIO for IXI-Tiny.

**Download.** This uses OmniFed's own code, because FLamby's hard-coded URL now returns 403. It needs
`pip install monai nibabel requests` (any MONAI version; `flamby_compat` restores `AddChannel`).

```bash
OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_ixi
```

It tries these sources in order:

1. a `.zip` you put in `$OMNIFED_DATA_DIR/flamby/fed_ixi/` yourself;
2. `$OMNIFED_FLAMBY_IXI_URL`;
3. Mendeley Data's "Download All" link;
4. TorchIO's IXITiny copy (MD5 checked);
5. FLamby's old URL.

Both archive layouts are accepted: per-subject folders (Mendeley / FLamby) and TorchIO's flat `image/` + `label/`. Only
the T1 scans and masks are kept, and subjects missing from FLamby's split are dropped. The result is FLamby's layout:
`7kd5wj7v7p-3/IXI_sample/<subject>/{T1,label}/` plus `7kd5wj7v7p-3.zip`, which FLamby reads the images from.

If every URL fails, download `ixi_tiny.zip` (https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1) or Mendeley's
"Download All" (https://data.mendeley.com/datasets/7kd5wj7v7p/3) in a browser. Put the zip in the folder and rerun.

**Config.** `ixi-natural-dirichlet-quantity.yaml`: when there are more clients than hospitals, a shared hospital is split
with Dirichlet sizes. For the plain natural scheme use `flamby.yaml` with `datamodule.dataset=fed_ixi`. Both use batch
size 2 and 2 workers.

**Print loaders.** Expect `x (2, 1, 48, 60, 48)` (T1) and `y (2, 2, 48, 60, 48)` (one-hot brain mask). Run it from a
file (the snippet above), not stdin, because of the 2 workers.

```bash
OMNIFED_NUM_FEDERATED_CLIENTS=3 OMNIFED_FEDERATED_CLIENT_INDEX=0 python print_flamby.py fed_ixi
```

**Test.**

```bash
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_ixi.py -v -rs
```

## Fed-ISIC2019

6 centers, dermoscopy images, 8 classes. These are the 23,247 images of FLamby's fixed split of the ISIC 2019 challenge;
centers have 9,930 / 3,163 / 2,691 / 1,807 / 655 / 351 training images. Licence CC BY-NC 4.0 (ISIC 2019) plus the
HAM10000 terms.

**Download.** This uses OmniFed's own code: FLamby's needed `wget` / `unzip` / `rm`, and its URL now serves a "moved"
page. It needs `pip install albumentations` (any version; `efficientnet-pytorch` is only needed for FLamby's baseline
model).

```bash
OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_isic2019
```

1. It downloads `ISIC_2019_Training_Input.zip` (9.1 GB) from `isic-archive.s3.amazonaws.com`. If interrupted, the
   download resumes where it stopped, and "moved" pages are followed.
2. It extracts only the split's 23,247 images.
3. It runs FLamby's `resize_images.py` (shorter side 224 px, colour constancy) through `flamby_compat`, then checks every
   image was resized.
4. It deletes the zip and the full-size images. What remains is `ISIC_2019_Training_Input_preprocessed/`, about 1 GB.

**Config.** `isic2019-natural.yaml`: the `natural` scheme, batch size 64, 4 workers. ISIC has class labels, so every
scheme works.

**Check / print loaders.** `check_isic.py` builds the loaders from `isic2019-natural.yaml`. It checks that images are
3×200×200 (FLamby's augmentations: random crop for train, centre crop for eval) and labels are in 0..7. Expect
`x (64, 3, 200, 200)` and `y (64,)`. Client 0 of 6 gets center 0: 9,930 train / 2,483 test images.

```bash
python -m src.omnifed.data.skewed.check_isic                          # optional Hydra overrides after it
python -m src.omnifed.data.skewed.check_isic datamodule.num_workers=0
```

`check_isic.py` currently sets `OMNIFED_NUM_FEDERATED_CLIENTS=6`, `OMNIFED_FEDERATED_CLIENT_INDEX=0`,
`OMNIFED_FLAMBY_ACCEPT_LICENSE=1` and `OMNIFED_DATA_DIR=<repo>/datasets` at the top of the file. Edit those lines to
check another client or data folder.

**Test.**

```bash
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_isic2019.py -v -rs
```

## Fed-LIDC-IDRI

4 centers (CT scanner manufacturers) and 1,009 chest CT scans with radiologists' lung-nodule annotations. FLamby's split
is 530 / 59 / 164 / 55 training and 131 / 15 / 41 / 14 test scans. FLamby's `metadata.csv` lists all 1,018 LIDC CT series,
but FLamby itself leaves out 9 patients whose slices are irregularly spaced or duplicated; so does OmniFed
(`LIDC_EXCLUDED` in `flamby_download.py`). Licence CC BY 3.0 plus the TCIA Data Usage Policy.

**Download.** This uses OmniFed's own code, because FLamby's relied on TCIA's retired v4 API and wiki. It needs
`pip install nibabel pydicom dicom-numpy networkx scipy scikit-image idc-index`. It takes hours, so run it once on a
machine with good bandwidth (on HPC, a login or data-transfer node).

```bash
# quick try: 16 scans spread over the 4 centers and both splits
OMNIFED_FLAMBY_LIDC_LIMIT=16 OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_lidc_idri
# all 1,009 scans
OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download fed_lidc_idri
```

- **Scans.** The SeriesInstanceUIDs come from FLamby's `metadata.csv`, minus the 9 in `LIDC_EXCLUDED` (patients
  LIDC-IDRI-0085, 0123, 0146, 0267, 0418, 0514, 0572, 0672, 0979). Those 9 are never downloaded, and their folders from
  runs before they were excluded are deleted. The DICOM files come from NCI Imaging Data Commons
  (via `idc-index`), falling back to TCIA's NBIA API for anything missing. Force a source with
  `OMNIFED_FLAMBY_LIDC_SOURCE=idc` or `=nbia`.
- **Annotations.** The radiologists' XML annotations come from
  `https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip`.
- **Conversion.** FLamby's own conversion (`process_raw.convert_to_niftis`, 1 mm spacing) runs in batches, one
  subprocess per scan (`OMNIFED_FLAMBY_WORKERS` sets how many at once). It writes `patient.nii.gz`,
  `mask_consensus.nii.gz` and `mask.nii.gz` into `<dataset>/<SeriesInstanceUID>/`.
- **Disk.** Each scan's DICOM files are deleted once it has converted, so the 133 GB never sits on disk at once. Keep
  them with `OMNIFED_FLAMBY_LIDC_KEEP_DICOM=1`.
- **Resuming.** Rerunning skips scans that are already converted. Failed scans are listed in
  `<dataset>/lidc_failed.txt` and the run ends with an error, so a rerun retries only those.
- **Extending the quick try.** After a run with `OMNIFED_FLAMBY_LIDC_LIMIT`, delete
  `fed_lidc_idri/.omnifed_flamby_ready.json` and rerun without the limit. Already converted scans are kept.

**Config.** `lidc-idri-natural.yaml`: the `natural` scheme, 1 scan per batch, FLamby's `collate_fn`, 2 workers. LIDC has
no class labels, so the label schemes refuse it.

**Check / print loaders.** `check_lidc.py` first reports how many of the 1,009 scans have converted per center and split
(and any failures in `lidc_failed.txt`). After a complete download it should say `total 1009/1009 scans converted`. It
then checks one train and one eval batch: patches of `(patches, 1, 128, 128, 128)`, a same-shaped binary mask, and CT
values in [0, 1]. A train batch has 2 patches per scan; an eval batch has all 27 patches tiling the 384³ volume.
Client 0 of 4 gets center 0: 530 train and 131 test scans.

```bash
OMNIFED_NUM_FEDERATED_CLIENTS=4 OMNIFED_FEDERATED_CLIENT_INDEX=0 python -m src.omnifed.data.skewed.check_lidc
```

Its defaults (4 clients, client 0, `OMNIFED_DATA_DIR=<repo>/datasets`) only apply when the variables are unset. It also
takes Hydra overrides like `check_isic`.

**Test.**

```bash
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_lidc_idri.py -v -rs
```

## Tests

`test_flamby/test_fed_<dataset>.py` tests each dataset on real data, with shared checks in `FlambySuite`
(`test_flamby/_common.py`). Each test class first calls `download_flamby_dataset`, which downloads only if the data is
missing, and then loads only through the datamodule.

The shared checks cover:

- the pooled data matches FLamby's per-center datasets;
- each rank gets its own center as a `StatefulDataLoader`, built from `flamby.yaml`;
- the server keeps only the pooled test set;
- fewer clients merge centers and more clients split them, always covering every sample exactly once;
- every scheme works (label schemes refuse unlabelled datasets);
- partitions are deterministic and change with the seed;
- loaders resume from `state_dict`.

```bash
export OMNIFED_DATA_DIR=/path/to/datasets OMNIFED_FLAMBY_ACCEPT_LICENSE=1
python -m pytest src/omnifed/data/skewed/test_flamby -v -rs                             # all datasets
python -m pytest src/omnifed/data/skewed/test_flamby/test_fed_ixi.py -v -rs             # one dataset
```

`-rs` shows why anything was skipped: licence not accepted, a missing package (with the `pip install` line), or a
dataset that must be prepared by hand (Camelyon16, KiTS19).

Running the whole suite downloads any of these datasets that are missing, including LIDC-IDRI (133 GB, hours). Download
LIDC first, or leave `test_fed_lidc_idri.py` out. The same applies to `flamby_download all`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `PermissionError ... set OMNIFED_FLAMBY_ACCEPT_LICENSE=1` | Downloading needs the licence variable. Read the licence (`--list`) and set it. |
| `FileNotFoundError: FLamby dataset X is not in ...; download it first` | The datamodule never downloads. Run the `flamby_download` command it prints. |
| `[flamby] reset FLamby's dataset_location.yaml` | Expected. FLamby's own download record was stale (OmniFed's ready-marker is the one that counts), so it is reset before downloading. |
| `DataLoader worker ... exited unexpectedly`, `.../<stdin>` not found (macOS) | A script with DataLoader workers was fed through stdin. Run it as a file or module, or add `datamodule.num_workers=0`. |
| `ImportError` from FLamby when loading | A missing dataset package. `--list` shows the `pip install` line for each dataset. |
| LIDC: `lidc_failed.txt` lists scans with `It appears there are missing slices` | The scan's slices are irregularly spaced or duplicated, so FLamby's conversion (dicom-numpy) rejects it. The 9 scans FLamby itself excludes are already skipped (`LIDC_EXCLUDED`). If another scan fails this way, check its slice positions before adding it there. |
| A download URL returns 403, or a page instead of a zip | Data hosts move files. "Moved" pages are followed automatically. Otherwise set `OMNIFED_FLAMBY_IXI_URL` / `OMNIFED_FLAMBY_ISIC_URL`, or put the zip in the dataset folder and rerun. |
