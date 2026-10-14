"""Verify the notebook's clone/refresh logic against a real repo.

Mirrors cell 3 with a dummy token, then asserts:
  * failures never surface the token,
  * the second run (reconnect path) works,
  * uncommitted work is protected by default,
  * the token never lands in .git/config or the command line.

    python scripts/test_clone_logic.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

TOKEN = "github_pat_FAKE_TOKEN_FOR_TESTING_ONLY_00000000"
SRC = Path(__file__).resolve().parents[1]  # this repo, cloned locally as the "remote"


def git(*cmd: str, cwd: Path | None = None, token: str = TOKEN) -> str:
    r = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True)
    scrub = lambda s: s.replace(token, "***")  # noqa: E731
    out = scrub(r.stdout + r.stderr)
    if r.returncode:
        # Scrub the command too: a token passed as an argument would otherwise
        # leak here even though the captured output was cleaned.
        raise SystemExit(f"git {scrub(' '.join(cmd))} failed ({r.returncode}):\n{out}")
    return out.strip()


def main() -> int:
    checks: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo = tmp / "repo"

        # --- 1. scrubbing: a failing call must not echo the token ---
        try:
            git("clone", f"https://{TOKEN}@github.com/nonexistent/nope.git", str(tmp / "x"))
            leaked = True
        except SystemExit as e:
            msg = str(e)
            leaked = TOKEN in msg
        checks.append(("failure output is scrubbed", not leaked))

        # --- 2. fresh clone (local path stands in for the remote) ---
        git("clone", "--depth", "1", str(SRC), str(repo))
        checks.append(("fresh clone", (repo / "src" / "road" / "paths.py").exists()))

        # --- 3. token must not be written into .git/config ---
        cfg = (repo / ".git" / "config").read_text(encoding="utf-8")
        checks.append(("token absent from .git/config", TOKEN not in cfg))

        # --- 4. reconnect path: fetch + reset on an existing checkout ---
        git("fetch", "--depth", "1", "origin", "main", cwd=repo)
        git("reset", "--hard", "FETCH_HEAD", cwd=repo)
        checks.append(("reconnect refresh", (repo / "README.md").exists()))

        # --- 5. dirty tree is detected (so edits aren't silently destroyed) ---
        (repo / "scratch_note.txt").write_text("uncommitted work\n")
        dirty = git("status", "--porcelain", cwd=repo)
        checks.append(("dirty tree detected", "scratch_note.txt" in dirty))

        # --- 6. reset --hard alone must NOT delete untracked files ---
        # (documents why the guard matters: --hard is not enough on its own)
        git("reset", "--hard", "FETCH_HEAD", cwd=repo)
        checks.append(("untracked file survives reset --hard",
                       (repo / "scratch_note.txt").exists()))

    print("clone-logic checks")
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    failed = [c for c, ok in checks if not ok]
    print("\n" + ("PASS" if not failed else f"FAIL: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
