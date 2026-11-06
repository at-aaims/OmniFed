#!/bin/sh

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

# Usage (run from the OmniFed repo root):
#   source ./global_env.sh                         # included in main.sh; exports the variables below into current shell
#   ./global_env.sh --create-conda-env             # optional: create the conda env at $ENV_ROOT if missing, and make sure all of
#                                                  # requirements.txt is installed in it (pip install only if something is missing)
# Without --create-conda-env no environment is created, checked or activated: use whatever env you already have
# and pip install requirements.txt yourself.
# NOTE: variables only reach your shell when this file is *sourced*; running it as ./global_env.sh only does the
# env creation / install (a child process cannot change the environment of the shell that launched it).

# All variables below are DEFAULTS: `export VAR="${VAR:-default}"` keeps a value that is already set (exported in your
# shell, by a script that launches main.sh, ...) and only fills in what is missing; an empty value counts as not set.
# Anything that sets a variable AFTER this file is sourced (a later shell line, os.environ[...] = ... in python) wins too.

# python env paths (always exported); the conda env lives next to the repo at ../$CONDA_ENV_NAME
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-omnifed_env}"
export PARENT_DIR="${PARENT_DIR:-$(cd .. && pwd)}"
echo "[setup] PARENT_DIR=$PARENT_DIR"
export ENV_ROOT="${ENV_ROOT:-${PARENT_DIR%/}/$CONDA_ENV_NAME}"
echo "[setup] ENV_ROOT=$ENV_ROOT"
export PYEXE="${PYEXE:-${ENV_ROOT%/}/bin/python}"
echo "[setup] PYEXE=$PYEXE"

# python module search path: the repo root plus every submodule in third_party/ (FLamby, torchtitan, ...).
# PYTHONPATH is a ':'-separated list; entries it already has are kept and nothing is added twice.
for _p in "$(pwd)" "$(pwd)"/third_party/*/; do
    [ -d "$_p" ] || continue
    _p="${_p%/}"
    case ":${PYTHONPATH:-}:" in
        *":$_p:"*) ;;
        *) PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_p" ;;
    esac
done
export PYTHONPATH
unset _p
echo "[setup] PYTHONPATH=$PYTHONPATH"


# check that every package listed in requirements.txt is installed with an accepted version:
#   _omnifed_check_requirements <python of the env> <requirements file>
# prints what is missing / has the wrong version. Exit status: 0 = all satisfied, 1 = something is missing, 2 = could not check.
# Runs the env's python in isolated mode (-I), so PYTHONPATH (repo, third_party/...) cannot make a package look installed.
_omnifed_check_requirements() {
    "$1" -I - "$2" <<'PY_CHECK'
import re
import sys
import importlib.metadata as md
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # a brand-new env has no 'packaging' yet; pip ships a private copy
    try:
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        print("[setup] could not check requirements.txt: neither 'packaging' nor pip can be imported by this python")
        sys.exit(2)


def source_tree_name(path):
    # distribution name of a source tree, for "-e <path>" lines
    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib
            name = tomllib.loads(pyproject.read_text()).get("project", {}).get("name")
            if name:
                return name
        except Exception:
            pass
        m = re.search(r"^name\s*=\s*[\"']([^\"']+)[\"']", pyproject.read_text(), re.M)
        if m:
            return m.group(1)
    for fname, pattern in (("setup.cfg", r"^\s*name\s*=\s*(\S+)"), ("setup.py", r"name\s*=\s*[\"']([^\"']+)[\"']")):
        f = path / fname
        if f.is_file():
            m = re.search(pattern, f.read_text(), re.M)
            if m:
                return m.group(1)
    return None


def installed_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check(req_file):
    total, bad, skipped = 0, [], []
    for raw in req_file.read_text().splitlines():
        line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()  # drop trailing comments
        if not line or line.startswith("#"):
            continue
        if re.match(r"(-e|--editable)\s+\S", line):  # editable install of a source tree
            target = re.split(r"\s+", line, maxsplit=1)[1].strip()
            name = None
            if "#egg=" in target:
                name = target.split("#egg=", 1)[1]
            elif "://" not in target:
                tree = (req_file.parent / target).resolve()
                if not tree.is_dir():
                    total += 1
                    bad.append(f"{target} (directory not found - git submodule not initialised?)")
                    continue
                name = source_tree_name(tree)
            if not name:
                skipped.append(target)
                continue
            total += 1
            if installed_version(name) is None:
                bad.append(f"{name} (editable install of {target} is missing)")
            continue
        if line.startswith("-"):  # other pip options (-r, --index-url, ...) are not packages
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            skipped.append(line)
            continue
        if req.marker is not None and not req.marker.evaluate():  # not for this platform / python
            continue
        total += 1
        have = installed_version(req.name)
        if have is None:
            bad.append(f"{req.name} (not installed)")
        elif not req.specifier.contains(have, prereleases=True):
            bad.append(f"{req.name} (installed {have}, requirements.txt wants {req.specifier})")
    print(f"[setup] requirements.txt: {total - len(bad)} of {total} packages are installed with an accepted version")
    for item in bad[:20]:
        print(f"[setup]   - {item}")
    if len(bad) > 20:
        print(f"[setup]   ... and {len(bad) - 20} more")
    if skipped:
        print("[setup]   (not checked: " + ", ".join(skipped) + ")")
    return 1 if bad else 0


try:
    sys.exit(check(Path(sys.argv[1])))
except SystemExit:
    raise
except Exception as exc:
    print(f"[setup] could not check requirements.txt: {type(exc).__name__}: {exc}")
    sys.exit(2)
PY_CHECK
}

# create conda env and install python dependencies (only when asked to):
#   ./global_env.sh --create-conda-env
# - $ENV_ROOT does not exist: create the conda env, then pip install -r requirements.txt
# - $ENV_ROOT exists: check that every package in requirements.txt is installed with an accepted version.
#   If all of them are, nothing is installed. If any is missing or has the wrong version, requirements.txt is installed again.
# After an install the check is repeated; if anything is still missing the script stops right there (the variables further
# down are then not exported) instead of carrying on with a broken env.
_create_conda=0
# NOTE: when this file is sourced from another script (main.sh does), "$@" are *that* script's own arguments
# (--config-name ..., hydra overrides), so only our flag is acted on and everything else is ignored.
# A near-miss of our own flag is still an error, to catch typos.
for _arg in "$@"; do
    case "$_arg" in
        --create-conda-env) _create_conda=1 ;;
        --create-conda*)
            echo "[setup] ERROR: unknown option '$_arg' (did you mean --create-conda-env?)" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

if [ "$_create_conda" = 1 ]; then
    # the env's own python (not $PYEXE, which may have been overridden): works when executed as ./global_env.sh too,
    # where 'conda activate' is not available
    _env_python="${ENV_ROOT%/}/bin/python"
    _install=0
    if [ ! -f requirements.txt ]; then
        echo "[setup] ERROR: requirements.txt not found in $(pwd); run this from the OmniFed repo root" >&2
        return 1 2>/dev/null || exit 1
    fi

    if [ -e "$ENV_ROOT" ]; then
        if [ ! -x "$_env_python" ]; then
            echo "[setup] ERROR: $ENV_ROOT exists but $_env_python does not: not a conda env? Remove it, or point ENV_ROOT / CONDA_ENV_NAME elsewhere." >&2
            return 1 2>/dev/null || exit 1
        fi
        echo "[setup] $ENV_ROOT already exists: checking that everything in requirements.txt is installed"
        _rc=0
        _omnifed_check_requirements "$_env_python" requirements.txt || _rc=$?
        if [ "$_rc" = 0 ]; then
            echo "[setup] all requirements are satisfied: nothing to install"
        elif [ "$_rc" = 1 ]; then
            echo "[setup] requirements are missing: installing requirements.txt"
            _install=1
        else
            echo "[setup] ERROR: could not check the requirements in $ENV_ROOT" >&2
            return 1 2>/dev/null || exit 1
        fi
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "[setup] ERROR: conda not found on PATH (on Frontier: module load miniforge3/23.11.0-0)" >&2
            return 1 2>/dev/null || exit 1
        fi
        conda create --prefix "$ENV_ROOT" python=3.12.12 -y || {
            echo "[setup] ERROR: conda create failed (remove $ENV_ROOT if it was left half-created, then re-run)" >&2
            return 1 2>/dev/null || exit 1
        }
        _install=1
    fi

    if [ "$_install" = 1 ]; then
        # -I: pip must only see this env's own packages. PYTHONPATH (repo, third_party/*) contains
        # torchtitan.egg-info, which otherwise makes pip report "Found existing installation: torchtitan".
        "$_env_python" -I -m pip install -r requirements.txt || {
            echo "[setup] ERROR: pip install failed. Fix the error and re-run ./global_env.sh --create-conda-env: it installs again." >&2
            return 1 2>/dev/null || exit 1
        }
        _omnifed_check_requirements "$_env_python" requirements.txt || {
            echo "[setup] ERROR: requirements.txt is still not satisfied in $ENV_ROOT after pip install" >&2
            return 1 2>/dev/null || exit 1
        }
    fi
    echo "[setup] conda env ready: $ENV_ROOT"

    # when sourced, make plain `python` (main.sh) this env's python; only done here, never without the flag
    case ":$PATH:" in
        *":${ENV_ROOT%/}/bin:"*) ;;
        *) export PATH="${ENV_ROOT%/}/bin:$PATH" ;;
    esac
fi
unset _create_conda _arg _env_python _install _rc
unset -f _omnifed_check_requirements


# to compress omnifed repo tar and send to all slurm compute nodes from login node
export ENV_TARBALL="${ENV_TARBALL:-${PARENT_DIR%/}/omnifed.tar.gz}"

# for dataloading, storing and partitioning scheme seed val
export OMNIFED_PARTITION_SEED="${OMNIFED_PARTITION_SEED:-1234}"
# export OMNIFED_DATA_DIR="/Users/ssq/Desktop/datasets/"      # replace with "./datasets" (used in .gitignore)
export OMNIFED_DATA_DIR="${OMNIFED_DATA_DIR:-$(pwd)/datasets}"
export OMNIFED_LOG_DIR="${OMNIFED_LOG_DIR:-$(pwd)/logs}"


# OMNIFED_TOKENIZER_DIR stores where the tokenizer is saved, and OMNIFED_C4_DISK where C4 dataset is stored
### export OMNIFED_TOKENIZER_DIR=""
### export OMNIFED_C4_DISK=""

# HuggingFace cache directory
export HF_HOME="${HF_HOME:-$(pwd)/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"


# To use  FLamby cross-silo datasets
export OMNIFED_FLAMBY_DIR="${OMNIFED_FLAMBY_DIR:-}"      # FLamby repo dir to use if not using third_party/FLamby
export OMNIFED_FLAMBY_ACCEPT_LICENSE="${OMNIFED_FLAMBY_ACCEPT_LICENSE:-1}"
export OMNIFED_FLAMBY_IXI_URL="${OMNIFED_FLAMBY_IXI_URL:-https://data.mendeley.com/datasets/7kd5wj7v7p/3}"
export OMNIFED_FLAMBY_LIDC_LIMIT="${OMNIFED_FLAMBY_LIDC_LIMIT:-16}"     # 16 for testing, then all 1018 scans
export OMNIFED_FLAMBY_LIDC_URL="${OMNIFED_FLAMBY_LIDC_URL:-https://www.cancerimagingarchive.net/wp-content/uploads/LIDC-XML-only.zip}"
