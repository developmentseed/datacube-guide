from .create import (
    create_empty_dataarray,
    create_zarr_store,
    create_or_open_zarr_array,
    create_or_open_zarr_store,
)
from .create import Quantity
from .config import Config
from .query import benchmark_zarr_array, benchmark_access_patterns
from .open import benchmark_dataset_open
from .titiler_cmr_benchmark import (
    benchmark_viewport,
    benchmark_tileset,
    benchmark_statistics,
    tiling_benchmark_summary,
    TiTilerCMRBenchmarker,
)

import numpy as np

__all__ = [
    "Config",
    "create_empty_dataarray",
    "create_zarr_store",
    "create_or_open_zarr_array",
    "create_or_open_zarr_store",
    "benchmark_zarr_array",
    "benchmark_access_patterns",
    "benchmark_dataset_open",
    "TiTilerCMRBenchmarker",
    "benchmark_viewport",
    "benchmark_tileset",
    "benchmark_statistics",
    "tiling_benchmark_summary",
]


def main() -> None:
    da = create_empty_dataarray()
    array_size = Quantity(da.nbytes, "bytes")
    chunk_size = Quantity(np.prod(da.data.chunksize) * da.dtype.itemsize, "bytes")
    print(da)
    print(f"Array size: {array_size.to('GB')}")
    print(f"Chunk size: {chunk_size.to('MB')}")
