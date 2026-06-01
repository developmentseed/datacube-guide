# Contributing to the Datacube Guide

Thanks for your interest in improving the guide! This repository is the
**MkDocs Material documentation site** published at
[developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/),
managed as a [`uv`](https://docs.astral.sh/uv/) project.

The notebooks that generate the worst-practices figures depend on the
[`datacube-benchmark`](https://github.com/developmentseed/datacube-benchmark)
library, which lives in its own repository and is installed here from
PyPI like any other dependency. Library changes go in that repo; this
repo only consumes the released package.

See the [README](README.md) for a higher-level overview.

## Development setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then install the
project and its dev dependencies:

```bash
uv sync
```

Run all tooling through `uv run` so it uses the project environment.

```bash
uv run -- mkdocs serve            # live-reload docs at http://localhost:8000
uv run -- mkdocs build --strict   # what CI runs; fails on any warning
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

## Code quality

[`prek`](https://github.com/j178/prek) — a fast, drop-in replacement for `pre-commit` that
reads the same `.pre-commit-config.yaml` — runs `ruff` and `codespell`. Install it (e.g.
`uv tool install prek`), install the git hooks once, then run the checks before pushing:

```bash
prek install
prek run --all-files
```

- **`ruff`** handles lint and formatting.
- **`codespell`** has a custom ignore list in `.codespellrc` — add genuine false positives
  there rather than disabling the check.

## Submitting changes

1. Branch off `main`.
2. Make your change, and ensure `uv run -- mkdocs build --strict` and `prek run --all-files`
   both pass.
3. Open a pull request describing the change. CI runs the strict build; merges to `main`
   publish to GitHub Pages (versioned via [`mike`](https://github.com/jimporter/mike)).

Working notes and TODOs that aren't published content live in `dev-docs/tasks/`.
