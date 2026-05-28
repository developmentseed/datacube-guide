# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two things in one `uv` workspace:

1. **A MkDocs Material documentation site** (`docs/`) — the published guide at
   [developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/),
   covering datacube "worst practices", a visualization tool catalog, and a glossary.
2. **The `datacube_benchmark` Python package** (`packages/datacube-benchmark/`) — a
   library for measuring Zarr read patterns and access costs, used both standalone and
   to generate content in the docs.

The workspace is managed by `uv`; run all tooling through `uv run`.

## Commands

```bash
uv sync                              # install workspace + dev deps
uv run -- mkdocs serve --livereload  # live-reload docs at localhost:8000
uv run -- mkdocs build --strict      # what CI runs; --strict fails on any warning
uv run python main.py                # smoke-test the benchmark library
uv run pytest --cov                  # run tests; enforces the 90% coverage gate
uv run mypy                          # type-check (files set in pyproject.toml)
uv run ruff check . && uv run ruff format .
prek run --all-files           # ruff, codespell, mypy, numpydoc-validation
```

Two CI workflows gate merges:

- **`docs.yml`** runs `mkdocs build --strict` — broken links, missing `nav:` entries,
  and autodoc failures all surface as warnings that fail the build. It deploys `site/`
  to GitHub Pages on push to `main` (`mike` provides versioned docs).
- **`test.yml`** runs `pytest --cov`. The coverage gate (`--cov-fail-under` equivalent)
  lives in `[tool.coverage.report] fail_under = 90` in `pyproject.toml`, enforced by
  `pytest-cov`.

`mypy` and `numpydoc-validation` run via pre-commit (`prek`/`pre-commit`), not in CI —
keep them green locally.

## Docs architecture

- `mkdocs.yml` is the source of truth for site structure: any new page must be added
  to `nav:` or the strict build won't surface it.
- **Worst-practices pages are Jupyter notebooks** (`docs/worst-practices/*.ipynb`),
  rendered by the `mkdocs-jupyter` plugin with `execute=False` — committed cell
  outputs are what gets published, so re-run notebooks locally before committing.
  Other pages are plain Markdown.
- `hooks/mkdocs_jupyter_md_filter.py` is a registered MkDocs hook that stops
  `mkdocs-jupyter` from probing `.md` files (which otherwise breaks `mkdocstrings`
  autodoc on API reference pages). Don't remove it.
- API reference pages (`docs/api-reference/`) use `mkdocstrings` to autodoc the
  benchmark package. Docstrings are **numpy-style**; broken cross-references fail the
  strict build.
- `includes/abbreviations.md` is auto-appended to every page (pymdownx snippets) for
  glossary tooltips.
- `docs/overrides/` holds the Material theme custom dir and CSS.

## Benchmark package

`packages/datacube-benchmark/src/datacube_benchmark/` — public API is re-exported from
`__init__.py` (`Config`, `create_*`, `benchmark_*`). Built on `zarr`, `obstore`,
`xarray`, `dask`, and `pint` (quantities carry units). Storage I/O goes through
`obstore` stores rather than fsspec/paths directly. Note `pint` doesn't know `KB` —
size strings use `MB`/`GB`/`bytes`.

Tests live in `packages/datacube-benchmark/tests/`:

- Pure chunk math (`chunks.py`) uses **Hypothesis** property tests — assert invariants
  (thickness ≥ 1, a "pancake" never chunks the spatial dims, an "over" chunk exceeds
  the target), not exact values — plus a few worked examples.
- Integration tests build real Zarr stores against an `obstore` `LocalStore` under
  `tmp_path` (see `conftest.py` fixtures) with a tiny coarse grid — no cloud. The one
  cloud-only branch (`credential_provider` in `create_or_open_zarr_store`) is
  `# pragma: no cover`; `earthaccess`/`s3fs` are declared deps but unused in source.
- The RNG is seeded (autouse fixture) so the timing/fill sampling is deterministic.

## Conventions

- Numpy-style docstrings, enforced by `numpydoc-validation` (checks listed in
  `pyproject.toml`); 80-char line length for signatures in autodoc.
- `ruff` for lint + format. `codespell` runs in pre-commit with a custom ignore list
  (`fo,ihs,kake,te`) — don't "fix" those.
- `dev-docs/tasks/` holds working notes/TODOs, not published content.
