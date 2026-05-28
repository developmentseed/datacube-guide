# datacube-benchmark

[![PyPI](https://img.shields.io/pypi/v/datacube-benchmark.svg)](https://pypi.org/project/datacube-benchmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Utilities for benchmarking [Zarr](https://zarr.dev/) datacubes — generate
synthetic stores with different chunking schemes, compressors, and
dtypes, then measure read performance under realistic access patterns.

Companion package to the [Datacube Guide](https://developmentseed.org/datacube-guide/),
which documents common pitfalls when producing and consuming
multi-dimensional data products.

## Installation

```bash
pip install datacube-benchmark
```

Python 3.12+ is required.

## Quickstart

Create a synthetic Zarr store on local disk and time a few random-access
patterns against it:

```python
from pathlib import Path

import obstore as obs
import zarr

import datacube_benchmark

store = obs.store.LocalStore(Path.cwd() / "data" / "test.zarr")
zarr_store = datacube_benchmark.create_zarr_store(store)

arr = zarr.open_array(zarr_store, zarr_version=3, path="data")
results = datacube_benchmark.benchmark_access_patterns(arr, num_samples=10)
print(results)
```

`create_zarr_store` takes target sizes and chunk shapes as strings or
[`pint`](https://pint.readthedocs.io/) quantities (e.g. `"1 GB"`,
`"10 MB"`), and writes through an [`obstore`](https://developmentseed.org/obstore/)
store — so the same call works against a local directory, S3, GCS, or
Azure by swapping the store.

## What's in the box

- **`create_zarr_store`**, **`create_or_open_zarr_store`**,
  **`create_or_open_zarr_array`**, **`create_empty_dataarray`** — build
  synthetic Zarr datacubes at a target size, resolution, and chunk
  shape.
- **`benchmark_zarr_array`** — time random reads against one access
  pattern (`"point"`, `"time_series"`, `"spatial_slice"`, `"full"`) and
  return summary statistics with units attached.
- **`benchmark_access_patterns`** — run all four access patterns and
  return the combined results as a `pandas.DataFrame`.
- **`benchmark_dataset_open`** — time `xarray.open_dataset` on a Zarr
  store.
- **`Config`** — a dataclass collecting the common knobs (compressor,
  target array size, sample counts, concurrency).

See the [API reference](https://developmentseed.org/datacube-guide/)
for the full signatures and parameter docs.

## License

[MIT](https://github.com/developmentseed/datacube-guide/blob/main/LICENSE.txt)
