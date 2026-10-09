#!/usr/bin/env python3
"""Install a Jupyter kernelspec pinned to this project's virtualenv.

Zed discovers notebook kernels via ``jupyter kernelspec list``. The venv already ships a
``python3`` spec, but its ``argv`` is a bare ``python``, resolved through ``PATH`` — so
which interpreter actually runs depends on how the editor was launched, and a notebook
cell can quietly import a different tickerkit (or none at all) than the one under test.

This writes a spec whose ``argv`` names ``.venv/bin/python`` by absolute path, so the
kernel is correct regardless of ``PATH``.

    uv run python tools/install_kernel.py

Re-run it after ``uv sync`` recreates the venv. The spec lands in the user data
directory rather than inside ``.venv/``, which ``uv`` may delete at any time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KERNEL_NAME = "tickerkit"
DISPLAY_NAME = "tickerkit (.venv)"

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def user_data_dir() -> Path:
    """``~/.local/share/jupyter/kernels`` unless JUPYTER_DATA_DIR says otherwise."""
    override = ROOT / ".jupyter-data"
    if override.is_dir():
        return override / "kernels"
    return Path.home() / ".local" / "share" / "jupyter" / "kernels"


def main() -> int:
    if not VENV_PYTHON.exists():
        print(f"no virtualenv at {VENV_PYTHON}\nrun: uv sync", file=sys.stderr)
        return 1

    target = user_data_dir() / KERNEL_NAME
    target.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [
            str(VENV_PYTHON),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "metadata": {"debugger": True},
    }
    (target / "kernel.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"installed kernelspec {KERNEL_NAME!r} -> {VENV_PYTHON}")
    print(f"  {target / 'kernel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())