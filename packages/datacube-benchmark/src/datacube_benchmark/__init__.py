from .create import create_empty_dataarray, create_zarr_store, create_or_open_zarr_array
from .create import Quantity
from .config import Config
from .query import benchmark_zarr_array, benchmark_access_patterns
import numpy as np

__all__ = [
    "Config",
    "create_empty_dataarray",
    "create_zarr_store",
    "create_or_open_zarr_array",
    "benchmark_zarr_array",
    "benchmark_access_patterns",
]


def main() -> None:
    da = create_empty_dataarray()
    array_size = Quantity(da.nbytes, "bytes")
    chunk_size = Quantity(np.prod(da.data.chunksize) * da.dtype.itemsize, "bytes")
    print(da)
    print(f"Array size: {array_size.to('GB')}")
    print(f"Chunk size: {chunk_size.to('MB')}")
