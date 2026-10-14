"""Validate notebooks: JSON well-formed, every code cell parses as Python.

    python scripts/check_notebooks.py

Shell magics (! and %) are stubbed before parsing.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(path: Path) -> int:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"{path.name}: INVALID JSON at line {e.lineno}: {e.msg}")
        return 1

    cells = nb.get("cells", [])
    print(f"{path.name}: nbformat {nb['nbformat']}.{nb['nbformat_minor']}, {len(cells)} cells")

    failures = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        stubbed = "\n".join(
            "pass" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.split("\n")
        )
        try:
            ast.parse(stubbed)
        except SyntaxError as e:
            print(f"  cell {i}: SYNTAX ERROR line {e.lineno}: {e.msg}")
            failures += 1

        if cell.get("outputs"):
            print(f"  cell {i}: WARNING has saved outputs -- strip before committing")

    if not failures:
        print("  all code cells parse")
    return failures


def main() -> int:
    nbs = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not nbs:
        print("no notebooks found")
        return 0
    return 1 if sum(check(p) for p in nbs) else 0


if __name__ == "__main__":
    sys.exit(main())
