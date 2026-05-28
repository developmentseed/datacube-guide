# Datacube Guide

Practical guidance for producing, using, and visualizing datacubes.

Read the guide at [developmentseed.org/datacube-guide](https://developmentseed.org/datacube-guide/).

> Status: First draft, actively iterating.

## What's inside

- **Worst practices** — common pitfalls to avoid when producing and using multi-dimensional data products: chunking, metadata, datatypes, and default library configs (FSSpec, GDAL, Xarray).
- **Visualization** — a catalog and comparison of tools for visualizing Zarr-backed datacubes in the browser, covering both server-side dynamic tilers (TiTiler, Xpublish-tiles) and client-side rendering, with libraries for deck.gl, MapLibre/Mapbox, and Cesium, plus standalone viewer apps. Includes side-by-side comparisons and guidance on choosing an approach.
- **Benchmarking** — a small Python library (`datacube_benchmark`) for measuring read patterns and access costs.

## Installation

```bash
git clone https://github.com/developmentseed/datacube-guide.git
cd datacube-guide
# Serve the documentation
uv run -- mkdocs serve
# Try out the `datacube_benchmark` library in Python
uv run python
# Try out the `datacube_benchmark` library in JupyterLab
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
uv run --with jupyter jupyter lab
```

## Acknowledgements

The Datacube Guide was initiated in partnership with the Microsoft Planetary Computer team. We recommend checking out the wonderful work going on as part of the [Microsoft Planetary Computer Pro service](https://learn.microsoft.com/en-us/azure/planetary-computer/) as well as the [Open Planetary Computer Data Catalog](https://planetarycomputer.microsoft.com/). We greatly appreciate Microsoft's dedication to supporting open resources and building impactful geospatial services.

## License

`datacube-guide` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
