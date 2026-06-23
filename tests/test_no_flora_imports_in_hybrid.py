"""Ensure hybrid package does not import src.flora."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HYBRID_ROOT = REPO / "src" / "omnifed" / "hybrid"


def _flora_imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.flora"):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src.flora"):
                hits.append(node.module)
    return hits


def test_no_flora_imports_under_hybrid():
    offenders: list[str] = []
    for py in HYBRID_ROOT.rglob("*.py"):
        for mod in _flora_imports_in_file(py):
            offenders.append(f"{py.relative_to(REPO)}: {mod}")
    assert offenders == [], "src.flora imports remain:\n" + "\n".join(offenders)
