# Guidance for avoiding common pitfalls when producing and using datacubes.

STATUS: Work in progress.

## Installation

```bash
git clone https://github.com/pangeo-data/datacube-guide.git
cd datacube-guide
# Serve the documentation
uv run -- mkdocs serve
# Try out the `datacube_benchmark` library in Python
uv run python
# Try out the `datacube_benchmark` library in JupyterLab
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
uv run --with jupyter jupyter lab
```

## License

`datacube-guide` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
