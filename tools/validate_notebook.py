#!/usr/bin/env python3
"""Execute a notebook and fail if any cell errors.

A notebook that is committed but never run is a liability: cells rot silently as the API
changes. This runs every code cell in one shared namespace (as a kernel would), with
``await`` permitted at top level, and reports the first failure with its cell index.

    uv run python tools/validate_notebook.py                 # notebook/pipeline.ipynb
    uv run python tools/validate_notebook.py notebook/history.ipynb
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

DEFAULT_NOTEBOOK = Path(__file__).resolve().parent.parent / "notebook" / "pipeline.ipynb"


def check_format(notebook: dict[str, Any]) -> list[str]:
    """Structural problems nbformat would reject, checked without the dependency."""
    problems: list[str] = []
    if notebook.get("nbformat") != 4:
        problems.append(f"expected nbformat 4, got {notebook.get('nbformat')}")
    if "cells" not in notebook:
        problems.append("no cells key")
        return problems
    for index, cell in enumerate(notebook["cells"]):
        kind = cell.get("cell_type")
        if kind not in ("markdown", "code", "raw"):
            problems.append(f"cell {index}: unknown cell_type {kind!r}")
            continue
        source = cell.get("source")
        if not isinstance(source, list) or not all(isinstance(x, str) for x in source):
            problems.append(f"cell {index}: source must be a list of strings")
        if kind == "code":
            if "outputs" not in cell or "execution_count" not in cell:
                problems.append(f"cell {index}: code cell missing outputs/execution_count")
            if not source or not source[0].strip():
                problems.append(f"cell {index}: empty code cell")
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if not kernelspec.get("name"):
        problems.append("missing kernelspec.name")
    return problems


def sources(notebook: dict[str, Any]) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def needs_await(snippet: str) -> bool:
    """Whether the cell uses ``await`` at top level, which plain ``exec`` rejects.

    Detected by parsing rather than compiling: ``ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`` lets
    ``compile`` succeed on a cell with top-level ``await``, but executing that code
    object still raises ``SyntaxError: 'await' outside function``. So the AST is the
    only reliable signal.
    """
    return any(isinstance(node, ast.Await) for node in ast.walk(ast.parse(snippet)))


async def run_cell(index: int, snippet: str, scope: dict[str, Any]) -> None:
    """Run one cell with kernel-like semantics.

    A cell using top-level ``await`` is wrapped in a coroutine so it compiles; the
    coroutine's locals are then merged back into the shared namespace so later cells
    still see the names it defined, exactly as a real kernel would.
    """
    if not needs_await(snippet):
        exec(compile(snippet, f"<cell {index}>", "exec"), scope)
        return

    # The wrapper returns its own locals, because names bound inside a function are
    # invisible to the enclosing scope — without this, `snap = await ...` in cell 2
    # would not exist for cell 4.
    body = textwrap.indent(snippet, "    ")
    wrapper = f"async def __cell_{index}():\n{body}\n    return locals()\n"
    exec(compile(wrapper, f"<cell {index}>", "exec"), scope)
    produced = await scope[f"__cell_{index}"]()  # type: ignore[operator]
    scope.update({k: v for k, v in produced.items() if k != "__cell"})


async def run_all(snippets: list[str]) -> None:
    """Run every cell in one shared namespace, as a kernel would."""
    scope: dict[str, Any] = {"__name__": "__main__"}
    for index, snippet in enumerate(snippets):
        try:
            await run_cell(index, snippet, scope)
        except Exception:
            print(f"\n{'=' * 70}\nFAILED in code cell {index}\n{'=' * 70}")
            print(snippet)
            print("-" * 70)
            traceback.print_exc()
            raise


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    notebook_path = Path(args[0]) if args else DEFAULT_NOTEBOOK
    notebook = json.loads(notebook_path.read_text())
    problems = check_format(notebook)
    if problems:
        print("format problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"format ok: {notebook_path} "
        f"({len(notebook['cells'])} cells, {len(sources(notebook))} code)"
    )

    snippets = sources(notebook)
    try:
        asyncio.run(run_all(snippets))
    except Exception:
        return 1
    print(f"executed {len(snippets)} code cells with no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())