# Datacube Guide

[![Docs](https://img.shields.io/badge/docs-developmentseed.org%2Fdatacube--guide-blue)](https://developmentseed.org/datacube-guide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Practical guidance for producing, using, and visualizing datacubes.

Read the guide at [developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/).

## What's inside

The guide (`docs/`) is published at
[developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/)
and covers:

- **Worst practices** — common pitfalls when producing and using
  multi-dimensional data products: chunking, metadata, datatypes, and
  default library configs (FSSpec, GDAL, Xarray).
- **Visualization** — a catalog and comparison of tools for visualizing
  Zarr-backed datacubes in the browser, covering both server-side
  dynamic tilers (TiTiler, Xpublish-tiles) and client-side rendering
  with deck.gl, MapLibre/Mapbox, and Cesium, plus standalone viewer
  apps.

The notebooks that generate the worst-practices figures use the
[`datacube-benchmark`](https://github.com/developmentseed/datacube-benchmark)
Python package — a library for measuring Zarr read patterns and access
costs — which lives in its own repository and is installable from PyPI:

```bash
pip install datacube-benchmark
```

## Working on the guide locally

```bash
git clone https://github.com/developmentseed/datacube-guide.git
cd datacube-guide
uv sync                              # install dev deps
uv run -- mkdocs serve --livereload  # docs at http://localhost:8000
```

To experiment in a notebook:

```bash
uv run ipython kernel install --user \
  --env VIRTUAL_ENV "$(pwd)/.venv" --name=datacube-guide
uv run --with jupyter jupyter lab
```

For the full development workflow — strict docs build, regenerating
diagrams — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

The Datacube Guide was initiated in partnership with the Microsoft
Planetary Computer team. We recommend checking out the wonderful work
going on as part of the
[Microsoft Planetary Computer Pro service](https://learn.microsoft.com/en-us/azure/planetary-computer/)
as well as the [Open Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/).
We greatly appreciate Microsoft's dedication to supporting open
resources and building impactful geospatial services.

The latest updates to this guide were supported by NASA's Office of
Data Science and Informatics (ODSI) as part of the Data Systems
Evolution team. The Data Systems Evolution team at NASA Marshall Space
Flight Center's Office of Data Science and Informatics enables
scientific exploration and discovery through innovative data
visualization techniques and analysis capabilities that lower the
barrier to entry for cloud-hosted data.

## License

`datacube-guide` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
