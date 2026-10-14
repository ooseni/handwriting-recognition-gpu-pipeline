"""Execute notebooks end to end, as Colab's "Run all" would, failing on the first
cell error. Outputs are printed, never written back to the notebook.

    .venv-colab\\Scripts\\python scripts\\run_notebook.py notebooks\\01_trocr.ipynb

Off Colab the notebooks switch themselves into a CPU test mode (see their first cell).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    # The kernel is "python3"; make sure it resolves to this interpreter's environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent.resolve())}})
    t0 = time.time()
    try:
        client.execute()
        ok = True
    except CellExecutionError as e:
        ok, failure = False, e

    for i, cell in enumerate(c for c in nb.cells if c.cell_type == "code"):
        text = "".join(o.get("text", "") for o in cell.get("outputs", []) if o.output_type == "stream")
        print(f"--- code cell {i} " + "-" * 50 + ("\n" + text.rstrip() if text.strip() else ""))
    if not ok:
        print(f"\n{path.name}: FAILED after {time.time() - t0:.0f}s\n{failure}")
        return False
    n = sum(c.cell_type == "code" for c in nb.cells)
    print(f"\n{path.name}: all {n} code cells ran clean in {time.time() - t0:.0f}s")
    return True


def main(argv: list[str] | None = None) -> int:
    targets = [Path(p) for p in (sys.argv[1:] if argv is None else argv)]
    if not targets:
        print(__doc__)
        return 2
    if sys.platform == "win32":  # zmq needs a selector event loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Cell output is arbitrary text -- a randomly initialised model emits any token at
    # all -- so never let the console encoding turn printing it into a failure.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return 0 if all([run(p) for p in targets]) else 1


if __name__ == "__main__":
    sys.exit(main())
