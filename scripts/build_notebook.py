"""Helper for programmatically constructing notebooks cell-by-cell. Not part of
the analysis itself; used to build notebooks/eda.ipynb consistently."""

from __future__ import annotations

import nbformat as nbf


def new_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    made = []
    for kind, source in cells:
        if kind == "md":
            made.append(nbf.v4.new_markdown_cell(source))
        else:
            made.append(nbf.v4.new_code_cell(source))
    nb["cells"] = made
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (credit-risk-venv)", "language": "python", "name": "credit-risk-venv"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def save(nb: nbf.NotebookNode, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
