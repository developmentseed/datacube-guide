# Contributing to the Datacube Guide

Thanks for your interest in improving the guide! This repository is a [`uv`](https://docs.astral.sh/uv/)
workspace holding two things:

- the **MkDocs Material documentation site** (`docs/`), published at
  [developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/), and
- the **`datacube_benchmark` Python package** (`packages/datacube-benchmark/`), a library
  for measuring Zarr read patterns and access costs.

See the [README](README.md) for a higher-level overview.

## Development setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then install the
workspace and its dev dependencies:

```bash
uv sync
```

Run all tooling through `uv run` so it uses the workspace environment.

```bash
uv run -- mkdocs serve            # live-reload docs at http://localhost:8000
uv run -- mkdocs build --strict   # what CI runs; fails on any warning
uv run python main.py             # smoke-test the benchmark library
```

## The strict build is the gate

The **only CI check** (`.github/workflows/docs.yml`) is `mkdocs build --strict`, and a merge
to `main` deploys `site/` to GitHub Pages. `--strict` turns warnings into errors, so broken
links, pages missing from the nav, and autodoc failures all fail the build. Run it locally
before opening a PR:

```bash
uv run -- mkdocs build --strict
```

## Editing the docs

- **`mkdocs.yml` is the source of truth for site structure.** Any new page must be added to
  `nav:`, or the strict build won't surface it.
- **Worst-practices pages are Jupyter notebooks** (`docs/worst-practices/*.ipynb`), rendered
  with `execute=False` — the **committed cell outputs are what gets published**, so re-run a
  notebook locally before committing any change to it. Other pages are plain Markdown.
- API reference pages (`docs/api-reference/`) autodoc the benchmark package via `mkdocstrings`.
  Docstrings are **numpy-style**; broken cross-references fail the strict build.
- `includes/abbreviations.md` is auto-appended to every page for glossary tooltips
  (format: `*[TERM]: definition`).
- Prose uses **American spelling** (e.g. "color", "visualization") for consistency.

### Regenerating diagrams

Some figures are generated from [Graphviz](https://graphviz.org/) sources. The `.dot` files
live next to the rendered `.svg` outputs in `docs/visualization/images/`. After editing a
`.dot` file, regenerate its SVG and commit both. For the figures on the visualization
overview page:

```sh
for f in decision-tree architecture-stack dimensionality-fan scale-funnel; do
  dot -Tsvg "docs/visualization/images/$f.dot" -o "docs/visualization/images/$f.svg"
done
```

Some diagrams (e.g. the static-vs-dynamic comparison and the grid-topologies figure) are
hand-authored SVG with no `.dot` source — edit the `.svg` directly.

## Working on the benchmark package

The public API is re-exported from `packages/datacube-benchmark/src/datacube_benchmark/__init__.py`.
The library is built on `zarr`, `obstore`, `xarray`, `dask`, and `pint`, and routes storage I/O
through `obstore` stores rather than fsspec/paths directly.

```bash
uv run mypy                       # type-check (files configured in pyproject.toml)
uv run ruff check . && uv run ruff format .
```

Note: `uv run mypy` currently reports a handful of pre-existing errors (missing third-party
stubs and string-annotation lookups), so it does not start clean.

## Code quality

[`pre-commit`](https://pre-commit.com/) runs `ruff`, `codespell`, `mypy`, and
`numpydoc-validation`. Install the hooks once, then run them before pushing:

```bash
pre-commit install
pre-commit run --all-files
```

- **`ruff`** handles lint and formatting.
- **`codespell`** has a custom ignore list in `.codespellrc` — add genuine false positives
  there rather than disabling the check.
- **Docstrings are numpy-style**, enforced by `numpydoc-validation` (checks listed in
  `pyproject.toml`); keep signatures within an 80-char line length for autodoc.

## Submitting changes

1. Branch off `main`.
2. Make your change, and ensure `uv run -- mkdocs build --strict` and `pre-commit run --all-files`
   both pass.
3. Open a pull request describing the change. CI runs the strict build; merges to `main`
   publish to GitHub Pages (versioned via [`mike`](https://github.com/jimporter/mike)).

Working notes and TODOs that aren't published content live in `dev-docs/tasks/`.
