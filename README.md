# Guidance for avoiding common pitfalls when producing and using datacubes.

Status: This is a first draft of the guidance.

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
