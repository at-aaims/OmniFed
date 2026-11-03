# Copyright (c) 2026, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Download FLamby datasets (Ogier du Terrail et al., NeurIPS 2022 D&B,
https://github.com/owkin/FLamby) into ``$OMNIFED_DATA_DIR/flamby/<name>``,
only when they are not there yet.

This is the only OmniFed code that downloads FLamby data. Run it once (e.g. on
a login / data-transfer node), then point the datamodule
(``conf/datamodule/flamby/flamby.yaml``, :mod:`.flamby_partition`) at the data::

    export OMNIFED_DATA_DIR=/path/to/datasets
    python -m src.omnifed.data.skewed.flamby_download --list          # licences, sizes, status
    OMNIFED_FLAMBY_ACCEPT_LICENSE=1 \\
        python -m src.omnifed.data.skewed.flamby_download fed_heart_disease fed_ixi
    OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download all

or from Python: :func:`download_flamby_dataset`. A dataset that is already
there is left alone. Several processes may call it at once: one downloads under
a file lock, the others wait and reuse the result. A dataset counts as
downloaded once its ready-marker (``.omnifed_flamby_ready.json``) is written,
after FLamby's scripts finished and produced files.

Licences
--------
FLamby does not redistribute data. Each dataset has its own licence (see
``--list``) that you must read and accept; downloads run only with
``OMNIFED_FLAMBY_ACCEPT_LICENSE=1``. FLamby itself asks for the licence on
stdin; with the variable set, this module answers for you.

==================  =======  =============  =========================================
dataset             centers  download       notes
==================  =======  =============  =========================================
fed_heart_disease   4        ~40 KB         automatic
fed_tcga_brca       6        none           bundled with FLamby (only the licence is recorded)
fed_ixi             3        ~444 MB        automatic from Mendeley Data (own code: FLamby's URL is dead)
fed_isic2019        6        ~9 GB          automatic (own code), then FLamby's resize; ~1 GB kept
fed_lidc_idri       4        ~133 GB        automatic (own code: IDC or TCIA), hours; DICOM -> NIfTI
fed_camelyon16      2        ~900 GB raw    manual: put prepared features in the folder
fed_kits19          6        ~30 GB         manual: FLamby's nnU-Net preprocessing
==================  =======  =============  =========================================

FLamby source
-------------
FLamby is used from source, as the git submodule ``third_party/FLamby``
(or ``$OMNIFED_FLAMBY_DIR``), without ``pip install``: its ``setup.py``
pip-installs extra packages even with ``--no-deps``::

    git submodule add https://github.com/owkin/FLamby.git third_party/FLamby
    git -C third_party/FLamby checkout edacf54d5211520583b0133d55ac39b6fda8324b
    pip install pandas scikit-learn matplotlib seaborn pyyaml tqdm wget lifelines

plus each dataset's extra packages (``--list``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "DATA_DIR_ENV",
    "FLAMBY_LICENSE_ENV",
    "FLAMBY_DIR_ENV",
    "FlambySpec",
    "FLAMBY_DATASETS",
    "flamby_data_root",
    "flamby_package_dir",
    "is_flamby_downloaded",
    "flamby_dataset_dir",
    "download_flamby_dataset",
    "record_flamby_agreement",
    "IXI_URL_ENV",
    "ISIC_URL_ENV",
    "LIDC_SOURCE_ENV",
    "LIDC_LIMIT_ENV",
    "LIDC_EXCLUDED",
]

DATA_DIR_ENV = "OMNIFED_DATA_DIR"
FLAMBY_LICENSE_ENV = "OMNIFED_FLAMBY_ACCEPT_LICENSE"
FLAMBY_DIR_ENV = "OMNIFED_FLAMBY_DIR"  # a FLamby checkout (folder containing flamby/)
_REPO_ROOT = Path(__file__).resolve().parents[4]  # skewed/ data/ omnifed/ src/ <repo>
_MARKER = ".omnifed_flamby_ready.json"
_COMPAT_RUNNER = Path(__file__).resolve().with_name("flamby_compat.py")  # runs a script after the patches
IXI_URL_ENV = "OMNIFED_FLAMBY_IXI_URL"  # override the Fed-IXI download URL
ISIC_URL_ENV = "OMNIFED_FLAMBY_ISIC_URL"  # override the ISIC 2019 training-images URL
LIDC_SOURCE_ENV = "OMNIFED_FLAMBY_LIDC_SOURCE"  # auto (default) | idc | nbia
LIDC_LIMIT_ENV = "OMNIFED_FLAMBY_LIDC_LIMIT"  # download only N scans (balanced over centers / splits)
LIDC_KEEP_DICOM_ENV = "OMNIFED_FLAMBY_LIDC_KEEP_DICOM"  # 1: keep the DICOM files after conversion
WORKERS_ENV = "OMNIFED_FLAMBY_WORKERS"  # parallel conversions (default: half the CPUs)
DOWNLOAD_CMD = "OMNIFED_FLAMBY_ACCEPT_LICENSE=1 python -m src.omnifed.data.skewed.flamby_download"


# ════════════════════════════════════════════════════════════════════════════
# Dataset registry
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FlambySpec:
    """One FLamby dataset: where it comes from (download) and how to load it (datamodule)."""

    module: str  # e.g. "flamby.datasets.fed_heart_disease"
    cls: str  # dataset class in that module
    num_centers: int  # fallback if the module has no NUM_CLIENTS
    task: str
    license: str
    size: str
    # Download steps, run in order: (script path relative to the flamby package,
    # arguments with "{out}" = target folder). Several candidate paths can be
    # given with "|" (FLamby spells one folder "dataset_creation_scipts").
    steps: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Python function in this module that downloads instead of FLamby's scripts
    # (used when FLamby's script is broken, e.g. a dead URL); called as fetch(target).
    fetch: str = ""
    finish: str = ""  # function in this module run after the steps, as finish(target)
    bundled: bool = False  # data ships with FLamby: nothing to download
    manual: str = ""  # set when the download cannot be automated
    takes_data_path: bool = True  # the class accepts data_path=...
    extra: str = ""  # pip extra for `pip install -e ".[extra]"`
    pip: str = ""  # packages needed on top of the base ones, to import and download
    batch_size: int = 8  # FLamby's benchmark BATCH_SIZE
    num_workers: int = 0
    collate_fn: str = ""  # FLamby's collate function ("flamby.datasets.x.collate_fn"), if any


_SCRIPTS = "datasets/{name}/dataset_creation_scripts"

FLAMBY_DATASETS: Dict[str, FlambySpec] = {
    "fed_heart_disease": FlambySpec(
        module="flamby.datasets.fed_heart_disease",
        cls="FedHeartDisease",
        num_centers=4,
        task="binary classification (tabular, 13 features)",
        license="CC-BY 4.0 (UCI Heart Disease)",
        size="~40 KB",
        steps=((_SCRIPTS.format(name="fed_heart_disease") + "/download.py", ("--output-folder", "{out}")),),
        extra="heart",
        batch_size=4,
        pip="wget",
    ),
    "fed_tcga_brca": FlambySpec(
        module="flamby.datasets.fed_tcga_brca",
        cls="FedTcgaBrca",
        num_centers=6,
        task="survival analysis (tabular)",
        license="GDC open-access data terms",
        size="~120 KB, bundled with FLamby",
        bundled=True,
        takes_data_path=False,
        extra="tcga",
        batch_size=8,
        pip="lifelines",
    ),
    "fed_ixi": FlambySpec(
        module="flamby.datasets.fed_ixi",
        cls="FedIXITiny",
        num_centers=3,
        task="3D brain segmentation (T1 MRI)",
        license="CC BY-SA 3.0 (IXI)",
        size="~444 MB",
        fetch="_fetch_ixi",  # FLamby's hard-coded S3 cache URL now returns 403
        extra="ixi",
        batch_size=2,
        num_workers=2,
        pip="monai nibabel requests",
    ),
    "fed_isic2019": FlambySpec(
        module="flamby.datasets.fed_isic2019",
        cls="FedIsic2019",
        num_centers=6,
        task="multiclass dermoscopy classification (images)",
        license="CC-BY-NC 4.0 (ISIC 2019) + HAM10000 dataverse terms",
        size="~9 GB download, ~1 GB kept",
        # own download (no wget / unzip needed), then FLamby's resize_images.py
        fetch="_fetch_isic",
        steps=((_SCRIPTS.format(name="fed_isic2019") + "/resize_images.py", ()),),
        finish="_finish_isic",
        extra="isic2019",
        batch_size=64,
        num_workers=4,
        pip="albumentations  (efficientnet-pytorch only for FLamby's baseline model)",
    ),
    "fed_lidc_idri": FlambySpec(
        module="flamby.datasets.fed_lidc_idri",
        cls="FedLidcIdri",
        num_centers=4,
        task="3D lung nodule segmentation (CT)",
        license="CC BY 3.0 + TCIA Data Usage Policy",
        size="~133 GB of DICOM (removed after conversion), hours; 1,009 scans",
        # own download (FLamby's uses TCIA's retired v4 API): DICOM from IDC or
        # TCIA's NBIA API, then FLamby's own DICOM -> NIfTI conversion
        fetch="_fetch_lidc",
        extra="lidc",
        batch_size=1,
        num_workers=2,
        collate_fn="flamby.datasets.fed_lidc_idri.collate_fn",
        pip="nibabel pydicom dicom-numpy networkx scipy scikit-image idc-index",
    ),
    "fed_camelyon16": FlambySpec(
        module="flamby.datasets.fed_camelyon16",
        cls="FedCamelyon16",
        num_centers=2,
        task="whole-slide image classification",
        license="CC0 1.0",
        size="~900 GB raw, ~50 GB of features",
        manual=(
            "needs Google Drive OAuth credentials and GPU tiling: follow "
            "https://owkin.github.io/FLamby/fed_camelyon.html and put the prepared data in "
        ),
        extra="cam16",
        batch_size=16,
        collate_fn="flamby.datasets.fed_camelyon16.collate_fn",
    ),
    "fed_kits19": FlambySpec(
        module="flamby.datasets.fed_kits19",
        cls="FedKits19",
        num_centers=6,
        task="3D kidney tumour segmentation (CT)",
        license="CC-BY-NC-SA 4.0",
        size="~30 GB",
        manual=(
            "needs the kits19 repo and nnU-Net preprocessing: follow "
            "https://owkin.github.io/FLamby/fed_kits19.html (FLamby's own config is used)"
        ),
        takes_data_path=False,
        extra="kits19",
        batch_size=2,
        num_workers=2,
        pip="batchgenerators nibabel nnunet==1.7.0",
    ),
}


def _spec(name: str) -> FlambySpec:
    try:
        return FLAMBY_DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown FLamby dataset {name!r}; choose from {sorted(FLAMBY_DATASETS)}") from None


def _ensure_importable(package: str) -> Optional[Path]:
    """
    Make ``package`` importable; for ``flamby``, fall back to a source checkout
    at ``$OMNIFED_FLAMBY_DIR`` or ``<repo>/third_party/FLamby``. Returns the
    folder added to ``sys.path`` (``None`` if nothing was added).
    """
    if importlib.util.find_spec(package) is not None:
        return None
    if package != "flamby":
        return None
    for root in (os.environ.get(FLAMBY_DIR_ENV), _REPO_ROOT / "third_party" / "FLamby"):
        if root and (Path(root) / "flamby" / "__init__.py").is_file():
            root = str(Path(root).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            importlib.invalidate_caches()
            return Path(root)
    return None



# ════════════════════════════════════════════════════════════════════════════
# Locations, licence, lock
# ════════════════════════════════════════════════════════════════════════════


def flamby_data_root(data_dir: Optional[str] = None) -> Path:
    """``<data_dir>/flamby``, with ``data_dir`` defaulting to ``$OMNIFED_DATA_DIR``."""
    base = data_dir or os.environ.get(DATA_DIR_ENV)
    if not base:
        raise RuntimeError(f"set {DATA_DIR_ENV} (or pass data_dir=) to say where FLamby data lives")
    return Path(base).expanduser() / "flamby"


def _license_accepted() -> bool:
    """Whether ``$OMNIFED_FLAMBY_ACCEPT_LICENSE`` confirms the licences were read and accepted."""
    return os.environ.get(FLAMBY_LICENSE_ENV, "").strip().lower() in ("1", "true", "yes")


def _licence_error(name: str, spec: FlambySpec, what: str) -> PermissionError:
    return PermissionError(
        f"{name} is licensed under {spec.license}. Read and accept the licence, then set "
        f"{FLAMBY_LICENSE_ENV}=1 {what}."
    )


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive inter-process lock (POSIX flock), so only one rank downloads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX or no flock support
            warnings.warn(f"could not lock {path}; avoid downloading from several processes at once")
        try:
            yield
        finally:
            with contextlib.suppress(ImportError, OSError):
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _script_path(flamby_dir: Path, candidates: str) -> Path:
    for rel in candidates.split("|"):
        p = flamby_dir / rel
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"FLamby download script not found ({candidates}) under {flamby_dir}. The scripts "
        "ship with the repo: use the submodule third_party/FLamby (or a full clone)."
    )


def _reset_flamby_config(config: Path, target: Path) -> None:
    """
    Remove FLamby's own download record before a fresh download.

    FLamby's scripts keep the data location and progress in
    ``dataset_location.yaml`` inside the submodule and trust it: if it says
    ``download_complete: true`` they print "already downloaded, aborting", even
    when that folder was emptied or is on another machine; if it names another
    folder they write there. OmniFed's ready-marker is the record that counts
    (this is only called when it is missing), so the FLamby file is stale by
    definition. FLamby rewrites it during the download.
    """
    if not config.is_file():
        return
    import yaml

    try:
        recorded = yaml.safe_load(config.read_text()) or {}
    except Exception:  # unreadable: FLamby starts over anyway
        recorded = {}
    config.unlink()
    print(
        f"[flamby] reset FLamby's {config.name} (dataset_path={recorded.get('dataset_path')}, "
        f"download_complete={recorded.get('download_complete')}) for a fresh download to {target}",
        flush=True,
    )


def flamby_package_dir(package: str = "flamby") -> Path:
    """Folder of the ``flamby`` package (submodule, ``$OMNIFED_FLAMBY_DIR`` or installed)."""
    _ensure_importable(package)
    found = importlib.util.find_spec(package)
    if found is None or not found.origin:
        raise ImportError(
            "FLamby not found: add the submodule third_party/FLamby "
            f"(git submodule update --init third_party/FLamby) or set {FLAMBY_DIR_ENV} to a FLamby checkout"
        )
    return Path(found.origin).resolve().parent


def _agreement_file(name: str) -> Path:
    """FLamby's own record that the licence was accepted (it stops asking on stdin)."""
    flamby_dir = flamby_package_dir(_spec(name).module.split(".")[0])
    return flamby_dir / "datasets" / name / "dataset_creation_scripts" / f"license_agreement_{name}"


def record_flamby_agreement(name: str) -> None:
    """
    Write FLamby's licence-agreement file for ``name`` (needs
    ``$OMNIFED_FLAMBY_ACCEPT_LICENSE=1``). TCGA-BRCA has no download step but
    FLamby asks for its licence every time it is loaded until this file exists.
    """
    spec = _spec(name)
    agreement = _agreement_file(name)
    if agreement.exists():
        return
    if not _license_accepted():
        raise _licence_error(name, spec, f"(or run: {DOWNLOAD_CMD} {name})")
    agreement.parent.mkdir(parents=True, exist_ok=True)
    agreement.touch()


# ════════════════════════════════════════════════════════════════════════════
# Is it there? / Download it
# ════════════════════════════════════════════════════════════════════════════


def _has_files(folder: Path) -> bool:
    return folder.is_dir() and any(p.name != _MARKER for p in folder.iterdir())


def is_flamby_downloaded(name: str, data_dir: Optional[str] = None) -> bool:
    """Whether ``name`` is ready to load (bundled datasets: whether FLamby is present)."""
    spec = _spec(name)
    if spec.bundled or (spec.manual and not spec.takes_data_path):
        try:
            flamby_package_dir(spec.module.split(".")[0])
            return True
        except ImportError:
            return False
    target = flamby_data_root(data_dir) / name
    return _has_files(target) if spec.manual else (target / _MARKER).is_file()


def flamby_dataset_dir(name: str, data_dir: Optional[str] = None) -> Optional[Path]:
    """
    Folder of an already-downloaded dataset, for loading (never downloads).

    Returns ``None`` for datasets that need no folder (bundled with FLamby, or
    located through FLamby's own config). Raises ``FileNotFoundError`` with the
    command to run when the data is missing.
    """
    spec = _spec(name)
    if spec.bundled or (spec.manual and not spec.takes_data_path):
        return None
    target = flamby_data_root(data_dir) / name
    if is_flamby_downloaded(name, data_dir):
        return target
    how = (
        f"prepare it by hand: {spec.manual}{target}" if spec.manual
        else f"download it first ({spec.size}): {DOWNLOAD_CMD} {name}"
    )
    raise FileNotFoundError(f"FLamby dataset {name} is not in {target}; {how}")


def download_flamby_dataset(name: str, data_dir: Optional[str] = None) -> Optional[Path]:
    """
    Download FLamby dataset ``name`` into ``<data_dir>/flamby/<name>`` unless it
    is already there; return that folder (``None`` for datasets without one).

    * Already downloaded: returns at once (no licence needed).
    * Missing: needs ``$OMNIFED_FLAMBY_ACCEPT_LICENSE=1`` (else ``PermissionError``),
      then runs FLamby's download scripts under a file lock and writes the
      ready-marker once they produced files.
    * Bundled (TCGA-BRCA): records FLamby's licence agreement.
    * Manual (Camelyon16, KiTS19): checks the prepared data, else ``RuntimeError``.

    Args:
        name: Key of :data:`FLAMBY_DATASETS`.
        data_dir: Overrides ``$OMNIFED_DATA_DIR``.
    """
    spec = _spec(name)
    if spec.bundled:
        record_flamby_agreement(name)
        return None
    if spec.manual:
        try:
            return flamby_dataset_dir(name, data_dir)
        except FileNotFoundError as e:
            raise RuntimeError(f"{name} cannot be downloaded automatically: {e}") from None

    target = flamby_data_root(data_dir) / name
    marker = target / _MARKER
    if marker.is_file():
        return target
    if not _license_accepted():
        raise _licence_error(name, spec, f"to download it ({spec.size}) to {target}")

    with _file_lock(target.parent / f".{name}.lock"):
        if marker.is_file():  # another process finished while we waited
            return target
        flamby_dir = flamby_package_dir(spec.module.split(".")[0])
        target.mkdir(parents=True, exist_ok=True)
        # the scripts import flamby themselves: make sure they find this copy
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(flamby_dir.parent), env.get("PYTHONPATH")]))
        print(f"[flamby] downloading {name} ({spec.size}) to {target}", flush=True)
        t0 = time.time()
        # our marker is missing, so FLamby's own record is stale: reset it once, before
        # the first step (later steps, e.g. ISIC's resize, read what the first one wrote)
        for folder in dict.fromkeys(_script_path(flamby_dir, script).parent for script, _ in spec.steps):
            _reset_flamby_config(folder / "dataset_location.yaml", target)
        if spec.fetch:
            globals()[spec.fetch](target)
        for script, args in spec.steps:
            path = _script_path(flamby_dir, script)
            script_args = [a.format(out=str(target)) for a in args]
            # through flamby_compat.py: FLamby's 2024 scripts need np.NaN / MONAI's AddChannel
            cmd = [sys.executable, str(_COMPAT_RUNNER), str(path), *script_args]
            print(f"[flamby]   {path.name} {' '.join(script_args)}", flush=True)
            # FLamby asks "have you read and accepted the data terms? (y/n)" on stdin
            # unless it has a saved agreement; acceptance was confirmed above.
            subprocess.run(cmd, cwd=str(path.parent), env=env, input="y\n", text=True, check=True)
        if spec.finish:
            globals()[spec.finish](target)
        if not _has_files(target):
            config = flamby_dir / "datasets" / name / "dataset_creation_scripts" / "dataset_location.yaml"
            raise RuntimeError(
                f"FLamby's download for {name} finished without writing files to {target} "
                f"(see its output above; FLamby's record: {config})."
            )
        marker.write_text(json.dumps({"dataset": name, "license": spec.license, "time": time.time()}))
        print(f"[flamby] {name} ready in {time.time() - t0:.0f}s", flush=True)
    return target


# ════════════════════════════════════════════════════════════════════════════
# Fed-IXI: own download (FLamby's URL is dead)
# ════════════════════════════════════════════════════════════════════════════

#: Mendeley Data record FLamby's IXI-Tiny comes from (version 3, CC BY 4.0):
#: https://data.mendeley.com/datasets/7kd5wj7v7p/3
IXI_FOLDER = "7kd5wj7v7p-3"  # FLamby's FOLDER: the data must be under <target>/<IXI_FOLDER>/IXI_sample
#: TorchIO's IXITiny (566 T1 images + brain masks; ``torchio.datasets.IXITiny``), the
#: data FLamby's IXI-Tiny is derived from, in a flat ``image/`` + ``label/`` layout.
TORCHIO_IXI_TINY_URL = "https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1"
IXI_URLS = (
    "https://data.mendeley.com/public-api/zip/7kd5wj7v7p/download/3",  # Mendeley "Download All"
    TORCHIO_IXI_TINY_URL,
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-3.zip",  # FLamby's (dead)
)
_IXI_MD5 = {TORCHIO_IXI_TINY_URL: "bfb60f4074283d78622760230bfa1f98"}  # from torchio
#: FLamby / Mendeley layout: <subject>/<modality>/<file>.nii.gz, e.g. IXI002-Guys-0828/T1/IXI002-Guys-0828_image.nii.gz
_IXI_MEMBER = re.compile(r"(?:^|/)(IXI\d{3}-[A-Za-z]+-\d+)/([^/]+)/([^/]+\.nii\.gz)$")
#: TorchIO layout: image/<subject>_image.nii.gz and label/<subject>_label.nii.gz
_IXI_FLAT = re.compile(r"(?:^|/)(image|label)/((IXI\d{3}-[A-Za-z]+-\d+)[^/]*\.nii\.gz)$")
_IXI_KEEP = ("T1", "label")  # FLamby's Fed-IXI uses T1 images and brain masks only


def _http_download(url: str, dest: Path) -> None:
    """
    Stream ``url`` to ``dest`` with progress; raises on HTTP errors. An
    interrupted download (``dest.part``) resumes where it stopped when the
    server supports ranges.
    """
    import urllib.request

    tmp = dest.with_name(dest.name + ".part")
    have = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": "omnifed-flamby-download"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        resumed = have and r.status == 206
        if have:
            print(f"[flamby]   {'resuming at' if resumed else 'restarting, server ignored'} {have / 2**20:.0f} MB",
                  flush=True)
        done = have if resumed else 0
        total = int(r.headers.get("Content-Length") or 0) + done
        f = open(tmp, "ab" if resumed else "wb")
        step = 50 * 2**20
        next_report = (done // step + 1) * step
        while True:
            chunk = r.read(2**20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done >= next_report:
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"[flamby]   {done / 2**20:.0f} MB{pct}", flush=True)
                next_report += step
        f.close()
    tmp.replace(dest)


def _download_zip(url: str, dest: Path, max_moves: int = 3) -> None:
    """
    Download a zip archive to ``dest``, following "this file has moved" HTML
    pages (meta refresh / link), which some data hosts serve with HTTP 200.
    Raises ``ValueError`` with what the server sent when it is not a zip.
    """
    import zipfile

    for _ in range(max_moves + 1):
        _http_download(url, dest)
        if zipfile.is_zipfile(dest):
            return
        head = dest.read_bytes()[:4096] if dest.stat().st_size < 2**20 else b""
        dest.unlink()
        text = head.decode("utf-8", "replace")
        moved = re.search(r"""url=([^"'>\s]+)|href=["']([^"']+\.zip[^"']*)["']""", text, re.IGNORECASE)
        if not moved:
            raise ValueError(f"{url} did not return a zip archive; it sent: {' '.join(text.split())[:300]!r}")
        url = moved.group(1) or moved.group(2)
        print(f"[flamby]   moved to {url}", flush=True)
    raise ValueError(f"too many 'moved' pages, last: {url}")


def _collect_ixi(archive: Path, out: Path, depth: int = 0) -> int:
    """Copy T1 + label NIfTIs of every subject in ``archive`` (zip, possibly nested) to ``out/<subject>/``."""
    import shutil
    import tempfile
    import zipfile

    n = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            m = _IXI_MEMBER.search(info.filename)
            flat = None if m else _IXI_FLAT.search(info.filename)
            dest = None
            if m and m.group(2) in _IXI_KEEP:
                dest = out / m.group(1) / m.group(2) / m.group(3)
            elif flat:  # TorchIO: image/ -> T1/, label/ -> label/
                dest = out / flat.group(3) / ("T1" if flat.group(1) == "image" else "label") / flat.group(2)
            if dest is not None:
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                n += 1
            elif info.filename.endswith("IXI.xls") and not (out / "IXI.xls").exists():
                with zf.open(info) as src, open(out / "IXI.xls", "wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif info.filename.lower().endswith(".zip") and depth < 2:  # Mendeley may wrap the files
                with tempfile.TemporaryDirectory(dir=out.parent) as td:
                    inner = Path(td) / "inner.zip"
                    with zf.open(info) as src, open(inner, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n += _collect_ixi(inner, out, depth + 1)
    return n


def _md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flamby_ixi_ids() -> Optional[set]:
    """Patient IDs in FLamby's ``metadata_tiny.csv`` (its train/test split), or None if unavailable."""
    import csv

    try:
        meta = flamby_package_dir() / "datasets" / "fed_ixi" / "metadata" / "metadata_tiny.csv"
    except ImportError:
        return None
    if not meta.is_file():
        return None
    with open(meta, newline="") as f:
        return {int(row["Patient ID"]) for row in csv.DictReader(f)}


def _fetch_ixi(target: Path) -> None:
    """
    Fed-IXI (IXI-Tiny) in the layout FLamby's ``FedIXITiny`` reads:
    ``<target>/7kd5wj7v7p-3/IXI_sample/<subject>/{T1,label}/*.nii.gz`` (it lists
    subjects there) and ``<target>/7kd5wj7v7p-3.zip`` with the same files (it
    reads the images from the zip).

    Source, first that works: a ``.zip`` you put in ``target`` yourself (e.g.
    "Download All" from https://data.mendeley.com/datasets/7kd5wj7v7p/3, or
    TorchIO's ``ixi_tiny.zip``), ``$OMNIFED_FLAMBY_IXI_URL``, then
    :data:`IXI_URLS`. Both layouts (per-subject folders, or TorchIO's flat
    ``image/`` + ``label/``) are accepted, also with extra top folders or nested
    zips. Subjects missing from FLamby's train/test split
    (``metadata_tiny.csv``) are left out, as FLamby could not load them.

    Not a source: the full IXI download (e.g. IXI-T1.tar, ~4.5 GB, from
    brain-development.org) has raw T1 scans but no brain masks, so it cannot
    give FLamby's segmentation task.
    """
    import shutil
    import zipfile

    final_zip = target / f"{IXI_FOLDER}.zip"
    subjects_dir = target / IXI_FOLDER / "IXI_sample"
    sources = [p for p in sorted(target.glob("*.zip")) if p != final_zip]
    downloaded = target / "ixi_download.zip"
    downloaded = downloaded if downloaded in sources else None  # ours, from an interrupted run
    if final_zip.is_file() and not sources:  # e.g. FLamby's own archive placed by hand
        sources = [final_zip.rename(target / "ixi_source.zip")]
    if sources:
        print(f"[flamby]   using {', '.join(p.name for p in sources)} found in {target}", flush=True)
    else:
        urls = [u for u in (os.environ.get(IXI_URL_ENV), *IXI_URLS) if u]
        errors = []
        for url in urls:
            downloaded = target / "ixi_download.zip"
            print(f"[flamby]   fetching {url}", flush=True)
            try:
                _download_zip(url, downloaded)
                if url in _IXI_MD5 and _md5(downloaded) != _IXI_MD5[url]:
                    raise ValueError(f"MD5 mismatch (expected {_IXI_MD5[url]})")
                sources = [downloaded]
                break
            except Exception as e:  # try the next URL
                errors.append(f"{url}: {e}")
                downloaded.unlink(missing_ok=True)
                print(f"[flamby]   failed: {e}", flush=True)
        if not sources:
            raise RuntimeError(
                "could not download Fed-IXI (" + "; ".join(errors) + "). Download the archive by hand "
                f"('Download All' at https://data.mendeley.com/datasets/7kd5wj7v7p/3), put the .zip in "
                f"{target} and rerun, or set {IXI_URL_ENV} to a working URL."
            )

    if subjects_dir.exists():  # leftovers of an interrupted run
        shutil.rmtree(subjects_dir)
    subjects_dir.mkdir(parents=True)
    found = sum(_collect_ixi(src, subjects_dir) for src in sources)
    known = _flamby_ixi_ids()
    if known is not None:
        for p in [p for p in subjects_dir.iterdir() if p.is_dir() and int(p.name[3:6]) not in known]:
            shutil.rmtree(p)
            print(f"[flamby]   left out {p.name}: not in FLamby's metadata_tiny.csv", flush=True)
    subjects = sorted(p for p in subjects_dir.iterdir() if p.is_dir())
    incomplete = [p.name for p in subjects if not (any((p / "T1").glob("*.nii.gz")) and any((p / "label").glob("*.nii.gz")))]
    if not subjects or incomplete:
        raise RuntimeError(
            f"{', '.join(p.name for p in sources)} does not look like IXI-Tiny: found {len(subjects)} subjects "
            f"({found} T1/label files), {len(incomplete)} without both T1 and label "
            f"(e.g. {incomplete[:3]}). Expected <subject>/T1/*.nii.gz and <subject>/label/*.nii.gz."
        )
    print(f"[flamby]   {len(subjects)} subjects; writing {final_zip.name}", flush=True)
    tmp_zip = target / f"{IXI_FOLDER}.zip.part"
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:  # .nii.gz: already compressed
        for f in sorted(subjects_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(target).as_posix())
    tmp_zip.replace(final_zip)
    if downloaded is not None:
        downloaded.unlink(missing_ok=True)  # everything needed is now in the two copies above


# ════════════════════════════════════════════════════════════════════════════
# Fed-ISIC2019: own download (FLamby's needs wget / unzip / rm on PATH)
# ════════════════════════════════════════════════════════════════════════════

#: ISIC 2019 challenge training images (25,331 JPEGs, 9.1 GB); listed at https://challenge.isic-archive.com/data
#: (FLamby's isic-challenge-data.s3.amazonaws.com address now serves a "moved" page pointing here)
ISIC_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip"
_ISIC_RAW = "ISIC_2019_Training_Input"
_ISIC_RESIZED = "ISIC_2019_Training_Input_preprocessed"


def _isic_split_images() -> set:
    """The 23,247 images of FLamby's fixed train/test split (its train_test_split file)."""
    import csv

    split = flamby_package_dir() / "datasets" / "fed_isic2019" / "dataset_creation_scripts" / "train_test_split"
    with open(split, newline="") as f:
        return {row["image"] for row in csv.DictReader(f)}


def _fetch_isic(target: Path) -> None:
    """
    Download the ISIC 2019 training images and keep the ones in FLamby's split,
    in ``<target>/ISIC_2019_Training_Input/`` (what FLamby's download_isic.py
    leaves), then point FLamby's resize_images.py at ``target``.

    Pure Python (no wget / unzip); the download resumes if interrupted; a
    ``ISIC_2019_Training_Input.zip`` already in ``target`` is used as is.
    Source: ``$OMNIFED_FLAMBY_ISIC_URL`` or :data:`ISIC_URL`.
    """
    import shutil
    import zipfile

    import yaml

    wanted = _isic_split_images()
    raw = target / _ISIC_RAW
    have = {p.stem for p in raw.glob("*.jpg")} if raw.is_dir() else set()
    if wanted <= have:
        print(f"[flamby]   {len(wanted)} images already in {raw}", flush=True)
    else:
        archive = target / f"{_ISIC_RAW}.zip"
        if not (archive.is_file() and zipfile.is_zipfile(archive)):
            archive.unlink(missing_ok=True)  # e.g. an HTML page saved by an earlier run
            url = os.environ.get(ISIC_URL_ENV) or ISIC_URL
            print(f"[flamby]   fetching {url} (9.1 GB)", flush=True)
            _download_zip(url, archive)
        print(f"[flamby]   extracting the {len(wanted)} images of FLamby's split from {archive.name}", flush=True)
        raw.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                stem = Path(info.filename).stem
                if info.filename.endswith(".jpg") and stem in wanted and stem not in have:
                    with zf.open(info) as src, open(raw / f"{stem}.jpg", "wb") as dst:
                        shutil.copyfileobj(src, dst)
        missing = wanted - {p.stem for p in raw.glob("*.jpg")}
        if missing:
            raise RuntimeError(
                f"{archive} lacks {len(missing)} images of FLamby's split (e.g. {sorted(missing)[:3]}); "
                "delete it and rerun to download it again."
            )
        archive.unlink()  # 9 GB; everything needed is extracted
    # FLamby's resize_images.py reads the data location from its own config
    scripts = flamby_package_dir() / "datasets" / "fed_isic2019" / "dataset_creation_scripts"
    with open(scripts / "dataset_location.yaml", "w") as f:
        yaml.dump({"dataset_path": str(target), "download_complete": True, "preprocessing_complete": False}, f)


def _finish_isic(target: Path) -> None:
    """Check FLamby's resize produced every image, then drop the full-size copies (~9 GB)."""
    import shutil

    wanted = _isic_split_images()
    resized = {p.stem for p in (target / _ISIC_RESIZED).glob("*.jpg")}
    missing = wanted - resized
    if missing:
        raise RuntimeError(
            f"resize_images.py left {len(missing)} images unresized in {target / _ISIC_RESIZED} "
            f"(e.g. {sorted(missing)[:3]}); rerun the download to finish."
        )
    shutil.rmtree(target / _ISIC_RAW, ignore_errors=True)
    print(f"[flamby]   {len(resized)} resized images in {target / _ISIC_RESIZED}; full-size copies removed", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# Fed-LIDC-IDRI: own download (FLamby's uses TCIA's retired v4 REST API)
# ════════════════════════════════════════════════════════════════════════════

#: Radiologists' XML annotations (8.6 MB), from https://www.cancerimagingarchive.net/collection/lidc-idri/
LIDC_ANNOTATIONS_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip"
#: TCIA's public NBIA search API: one series as a zip of DICOM files
NBIA_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"
_LIDC_NS = "{http://www.nih.gov}"
#: The 9 scans FLamby itself leaves out ("images with missing slices": ``bad_patientID`` in its
#: download_ct_scans.py). metadata.csv lists all 1,018 LIDC CT series, but FLamby never converts
#: these, so Fed-LIDC-IDRI has 1,009 scans; their slice spacing is irregular / duplicated, and
#: FLamby's conversion (dicom-numpy) rejects them anyway. SeriesInstanceUID -> PatientID (via TCIA).
LIDC_EXCLUDED = {
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.103115201714075993579787468219": "LIDC-IDRI-0123",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.110499927630654433643791451680": "LIDC-IDRI-0418",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.156990013635454707781600846659": "LIDC-IDRI-0085",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.220041426531925632952954881401": "LIDC-IDRI-0979",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.245181799370098278918756923992": "LIDC-IDRI-0267",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.329334252028672866365623335798": "LIDC-IDRI-0672",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.331079701650130000691651987949": "LIDC-IDRI-0572",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.410251741986998833890312367579": "LIDC-IDRI-0146",
    "1.3.6.1.4.1.14519.5.2.1.6279.6001.526570782606728516388531252230": "LIDC-IDRI-0514",
}


def _lidc_scripts_dir() -> Path:
    return flamby_package_dir() / "datasets" / "fed_lidc_idri" / "dataset_creation_scripts"


def _lidc_series(limit: int = 0) -> List[str]:
    """
    SeriesInstanceUIDs of FLamby's 1,009 scans: the 1,018 of ``metadata/metadata.csv``
    (center = scanner manufacturer, fixed train/test split) minus the 9 FLamby
    leaves out (:data:`LIDC_EXCLUDED`). With ``limit``, a subset spread evenly over
    the (center, split) groups so every center keeps both.
    """
    import csv

    meta = flamby_package_dir() / "datasets" / "fed_lidc_idri" / "metadata" / "metadata.csv"
    with open(meta, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["SeriesInstanceUID"] not in LIDC_EXCLUDED]
    if not limit or limit >= len(rows):
        return [r["SeriesInstanceUID"] for r in rows]
    groups: Dict[Tuple[str, str], List[str]] = {}
    for r in rows:
        groups.setdefault((r["Manufacturer"], r["Split"]), []).append(r["SeriesInstanceUID"])
    picked: List[str] = []
    queues = [list(v) for _, v in sorted(groups.items())]
    while len(picked) < limit:
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


def _lidc_converted(series_dir: Path) -> bool:
    """FLamby's conversion output, as its dataset class reads it."""
    return all((series_dir / f).is_file() for f in ("patient.nii.gz", "mask_consensus.nii.gz", "mask.nii.gz"))


def _lidc_annotations(target: Path) -> Dict[str, Path]:
    """SeriesInstanceUID -> annotation XML (first file per series, like FLamby)."""
    import xml.etree.ElementTree as ET
    import zipfile

    folder = target / "_annotations"
    if not any(folder.rglob("*.xml")):
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "LIDC-XML-only.zip"
        print(f"[flamby]   fetching radiologists' annotations {LIDC_ANNOTATIONS_URL}", flush=True)
        _download_zip(LIDC_ANNOTATIONS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
        archive.unlink()
        for inner in list(folder.rglob("*.zip")):  # some releases nest one zip per folder
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent / inner.stem)
            inner.unlink()
    mapping: Dict[str, Path] = {}
    for xml in sorted(folder.rglob("*.xml")):
        try:
            uid = next(e.text for e in ET.parse(xml).getroot().iter() if e.tag == f"{_LIDC_NS}SeriesInstanceUid")
        except (StopIteration, ET.ParseError):
            continue
        mapping.setdefault(uid.strip(), xml)
    return mapping


def _has_dicoms(series_dir: Path) -> bool:
    return series_dir.is_dir() and any(series_dir.glob("*.dcm"))


def _lidc_download_idc(target: Path, uids: Sequence[str]) -> None:
    """DICOM series from NCI Imaging Data Commons (public AWS bucket, via idc-index / s5cmd)."""
    from idc_index import IDCClient  # pip install idc-index

    IDCClient().download_from_selection(
        downloadDir=str(target),
        seriesInstanceUID=list(uids),
        dirTemplate="%SeriesInstanceUID",  # <target>/<SeriesInstanceUID>/*.dcm, as FLamby expects
        quiet=True,
        show_progress_bar=True,
        use_s5cmd_sync=True,  # resumes partial series
    )


def _lidc_download_nbia(target: Path, uid: str) -> None:
    """One DICOM series from TCIA's NBIA API (a zip), flattened into ``<target>/<uid>/``."""
    import shutil
    import zipfile

    archive = target / f"{uid}.zip"
    _download_zip(NBIA_IMAGE_URL.format(uid=uid), archive)
    out = target / uid
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".dcm"):
                with zf.open(info) as src, open(out / Path(info.filename).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    archive.unlink()


#: Runs FLamby's process_raw.convert_to_niftis for one scan in a fresh interpreter (after
#: flamby_compat's patches). A plain subprocess, not multiprocessing: works however the
#: caller was started (module, notebook, stdin) and isolates crashes in FLamby's code.
_LIDC_CONVERT = """
import contextlib, io, runpy, sys
from pathlib import Path
compat, scripts, series, xml, keep = sys.argv[1:6]
runpy.run_path(compat, run_name="flamby_compat")["apply"]("flamby.datasets.fed_lidc_idri")
sys.path.insert(0, scripts)
from process_raw import convert_to_niftis
log = io.StringIO()
with contextlib.redirect_stdout(log):
    ok = bool(convert_to_niftis(series, xml, spacing=[1.0, 1.0, 1.0]))
if ok and keep != "1":
    for dcm in Path(series).glob("*.dcm"):
        dcm.unlink()
if not ok:
    print(" ".join(log.getvalue().split())[-300:], file=sys.stderr)
sys.exit(0 if ok else 3)
"""


def _lidc_convert_one(series_dir: Path, xml: Path, keep_dicom: bool) -> Tuple[str, bool, str]:
    """FLamby's DICOM -> NIfTI conversion of one scan (1 mm spacing, as FLamby); (uid, ok, message)."""
    cmd = [sys.executable, "-c", _LIDC_CONVERT, str(_COMPAT_RUNNER), str(_lidc_scripts_dir()),
           str(series_dir), str(xml), "1" if keep_dicom else "0"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    msg = " ".join((out.stderr or out.stdout).split())[-400:]
    return series_dir.name, out.returncode == 0, msg


def _fetch_lidc(target: Path) -> None:
    """
    Fed-LIDC-IDRI in FLamby's layout: ``<target>/<SeriesInstanceUID>/`` with
    ``patient.nii.gz``, ``mask_consensus.nii.gz`` and ``mask.nii.gz``, made by
    FLamby's own conversion (``process_raw.py``) from the CT DICOMs and the
    radiologists' XML annotations.

    * Scans: FLamby's 1,009 (``metadata.csv`` minus :data:`LIDC_EXCLUDED`); ``$OMNIFED_FLAMBY_LIDC_LIMIT=N``
      takes N of them, spread over the 4 centers and both splits (for a quick try).
    * DICOM source (``$OMNIFED_FLAMBY_LIDC_SOURCE``): ``idc`` = NCI Imaging Data
      Commons, the public archive TCIA moves its collections to (needs
      ``pip install idc-index``); ``nbia`` = TCIA's NBIA API; ``auto`` (default) =
      IDC when idc-index is installed, NBIA for anything still missing.
    * Batches of scans are downloaded, converted in parallel
      (``$OMNIFED_FLAMBY_WORKERS``) and their DICOM files deleted (keep them with
      ``$OMNIFED_FLAMBY_LIDC_KEEP_DICOM=1``), so the 133 GB of DICOM never sit on
      disk at once. Rerunning resumes: converted scans are skipped.
    * Scans that fail are listed in ``<target>/lidc_failed.txt`` and the run
      raises, so a rerun retries them.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    limit = int(os.environ.get(LIDC_LIMIT_ENV) or 0)
    source = (os.environ.get(LIDC_SOURCE_ENV) or "auto").strip().lower()
    if source not in ("auto", "idc", "nbia"):
        raise ValueError(f"{LIDC_SOURCE_ENV} must be auto, idc or nbia, not {source!r}")
    have_idc = importlib.util.find_spec("idc_index") is not None
    if source == "idc" and not have_idc:
        raise ImportError(f"{LIDC_SOURCE_ENV}=idc needs: pip install idc-index")
    use_idc = have_idc and source != "nbia"
    keep_dicom = os.environ.get(LIDC_KEEP_DICOM_ENV, "").strip().lower() in ("1", "true", "yes")
    workers = int(os.environ.get(WORKERS_ENV) or max(1, (os.cpu_count() or 2) // 2))

    uids = _lidc_series(limit)
    for uid in LIDC_EXCLUDED:  # DICOMs of the 9 excluded scans, from runs before they were excluded
        if (target / uid).is_dir() and not _lidc_converted(target / uid):
            shutil.rmtree(target / uid)
            print(f"[flamby]   removed {uid} ({LIDC_EXCLUDED[uid]}): FLamby excludes this scan", flush=True)
    annotations = _lidc_annotations(target)
    todo = [u for u in uids if not _lidc_converted(target / u)]
    no_xml = [u for u in todo if u not in annotations]
    todo = [u for u in todo if u in annotations]
    print(
        f"[flamby]   {len(uids)} scans ({len(uids) - len(todo) - len(no_xml)} already converted, {len(todo)} to do); "
        f"DICOM from {'IDC, then NBIA for any missing' if use_idc and source == 'auto' else 'IDC' if use_idc else 'NBIA'}; "
        f"{workers} conversion workers",
        flush=True,
    )
    failed: Dict[str, str] = {u: "no annotation XML for this series" for u in no_xml}
    batch = max(8, 2 * workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:  # each job runs a conversion subprocess
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            missing = [u for u in chunk if not _has_dicoms(target / u)]
            if missing and use_idc:
                try:
                    _lidc_download_idc(target, missing)
                except Exception as e:
                    if source == "idc":
                        raise
                    print(f"[flamby]   IDC download failed ({type(e).__name__}: {e}); trying NBIA", flush=True)
                missing = [u for u in missing if not _has_dicoms(target / u)]
            if missing and source != "idc":
                for u in missing:
                    try:
                        _lidc_download_nbia(target, u)
                    except Exception as e:
                        failed[u] = f"download: {type(e).__name__}: {e}"
            ready = [u for u in chunk if _has_dicoms(target / u) and u not in failed]
            for u in chunk:
                if u not in ready and u not in failed:
                    failed[u] = "no DICOM files downloaded"
            jobs = [pool.submit(_lidc_convert_one, target / u, annotations[u], keep_dicom) for u in ready]
            for job in jobs:
                uid, ok, msg = job.result()
                if not ok:
                    failed[uid] = msg or "conversion returned False"
            done = sum(_lidc_converted(target / u) for u in uids)
            print(f"[flamby]   {done}/{len(uids)} scans converted", flush=True)

    report = target / "lidc_failed.txt"
    if failed:
        report.write_text("".join(f"{u}\t{m}\n" for u, m in sorted(failed.items())))
        raise RuntimeError(
            f"{len(failed)} of {len(uids)} LIDC scans failed (see {report}); rerun to retry them "
            "(converted scans are kept)."
        )
    report.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def _list(data_dir: Optional[str]) -> None:
    try:
        root = flamby_data_root(data_dir)
    except RuntimeError:
        root = None
    print(f"data root: {root if root else f'(set {DATA_DIR_ENV})'}")
    for name, s in FLAMBY_DATASETS.items():
        how = "bundled" if s.bundled else ("manual" if s.manual else "automatic")
        status = ""
        if root is not None or s.bundled:
            status = "ready" if is_flamby_downloaded(name, data_dir) else "missing"
        print(f"{name:18s} {s.num_centers} centers  {how:9s}  {s.size:30s} {status}")
        print(f"{'':18s} {s.task}; licence: {s.license}")
        if s.pip:
            print(f"{'':18s} pip install {s.pip}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Download FLamby datasets into $OMNIFED_DATA_DIR/flamby/<name> if missing "
        f"(needs {FLAMBY_LICENSE_ENV}=1).",
    )
    p.add_argument("names", nargs="*", metavar="NAME",
                   help=f"datasets ({', '.join(FLAMBY_DATASETS)}) or 'all' (every automatic one)")
    p.add_argument("--list", action="store_true", help="show datasets, licences, sizes and what is on disk")
    p.add_argument("--data-dir", default=None, help=f"default ${DATA_DIR_ENV}")
    args = p.parse_args(argv)
    if args.list or not args.names:
        _list(args.data_dir)
        return 0
    if "all" in args.names:
        names = [n for n, s in FLAMBY_DATASETS.items() if not s.manual]
    else:
        unknown = sorted(set(args.names) - set(FLAMBY_DATASETS))
        if unknown:
            p.error(f"unknown datasets {unknown}; choose from {list(FLAMBY_DATASETS)} or 'all'")
        names = args.names
    failed = []
    for name in names:
        try:
            path = download_flamby_dataset(name, args.data_dir)
            print(f"{name}: ready" + (f" at {path}" if path else " (bundled with FLamby)"), flush=True)
        except Exception as e:  # report every dataset, then fail
            failed.append(name)
            print(f"{name}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
