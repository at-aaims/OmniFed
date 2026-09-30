"""Ensure hierarchical package does not import src.flora."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HIER_ROOT = REPO / "src" / "omnifed" / "hierarchical"


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


class TestNoFloraAndRemovedPackages(unittest.TestCase):
    def test_no_flora_imports_under_hierarchical(self) -> None:
        offenders: list[str] = []
        for py in HIER_ROOT.rglob("*.py"):
            for mod in _flora_imports_in_file(py):
                offenders.append(f"{py.relative_to(REPO)}: {mod}")
        self.assertEqual(offenders, [])

    def test_classic_package_removed(self) -> None:
        self.assertFalse((REPO / "src" / "omnifed" / "classic").exists())

    def test_hybrid_package_removed(self) -> None:
        self.assertFalse((REPO / "src" / "omnifed" / "hybrid").exists())

    def test_layout_group_removed(self) -> None:
        self.assertFalse((REPO / "conf" / "layout").exists())


if __name__ == "__main__":
    unittest.main()
