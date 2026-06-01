# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A MkDocs Material documentation site (`docs/`) — the published guide at
[developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/),
covering datacube "worst practices", a visualization tool catalog, and a glossary.

The worst-practices notebooks depend on the
[`datacube-benchmark`](https://github.com/developmentseed/datacube-benchmark)
library, which is developed in its own repository and pulled in from PyPI like any
other dependency. **Don't edit benchmark library code from this repo** — file
changes against `developmentseed/datacube-benchmark` instead.

The project is managed by `uv`; run all tooling through `uv run`.

## Commands

```bash
uv sync                              # install dev deps
uv run -- mkdocs serve --livereload  # live-reload docs at localhost:8000
uv run -- mkdocs build --strict      # what CI runs; --strict fails on any warning
uv run ruff check . && uv run ruff format .
prek run --all-files                 # ruff, codespell
```

`docs.yml` is the only CI workflow: it runs `mkdocs build --strict` — broken links,
missing `nav:` entries, and autodoc failures all surface as warnings that fail the
build. It deploys `site/` to GitHub Pages on push to `main` (`mike` provides
versioned docs).

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
  `datacube-benchmark` package installed from PyPI. Docstrings are **numpy-style**;
  broken cross-references fail the strict build. To pick up new symbols, bump the
  pinned version in `pyproject.toml` and `uv sync`.
- `includes/abbreviations.md` is auto-appended to every page (pymdownx snippets) for
  glossary tooltips.
- `docs/overrides/` holds the Material theme custom dir and CSS.

## Conventions

- `ruff` for lint + format. `codespell` runs in pre-commit with a custom ignore list
  (`fo,ihs,kake,te`) — don't "fix" those.
- `dev-docs/tasks/` holds working notes/TODOs, not published content.
