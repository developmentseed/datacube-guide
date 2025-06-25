import zarr
import pandas as pd
import numpy as np
import time
import random
from typing import List, Literal
from .utils import array_storage_size
from pint import Quantity


def _measure_zarr_random_access_performance(
    zarr_array: zarr.Array,
    access_pattern: Literal["point", "time_series", "spatial_slice", "full"] = "point",
    num_samples: int = 100,
    warmup_samples: int = 10,
) -> List[float]:
    """
    Measure time performance for loading random data values from a zarr array.

    Parameters
    ----------
    zarr_array
        The zarr array to test
    access_pattern
        Type of access pattern: "point", "time_series", or "spatial_slice"
    num_samples
        Number of random access operations to perform
    warmup_samples
        Number of warmup operations (not included in timing)

    Returns
    -------
    results
        A list of access times for each random access operation.
    """

    # Get array dimensions
    shape = zarr_array.shape

    if len(shape) < 3 and access_pattern in ["time_series", "spatial_slice"]:
        raise ValueError(
            f"Array must be at least 3D for {access_pattern} access pattern"
        )

    # Generate random indices for all samples (warmup + actual)
    total_samples = warmup_samples + num_samples
    random_indices = []

    idx: tuple[int | slice, ...]
    for _ in range(total_samples):
        if access_pattern == "point":
            # Generate random index for each dimension
            idx = tuple(random.randint(0, dim - 1) for dim in shape)
        elif access_pattern == "time_series":
            # Generate (:, random_y, random_z) for 3D+ arrays
            fixed_indices = [random.randint(0, dim - 1) for dim in shape[1:]]
            idx = (slice(None), *fixed_indices)
        elif access_pattern == "spatial_slice":
            # Generate (random_x, :, :) for 3D+ arrays
            random_first = random.randint(0, shape[0] - 1)
            idx = (random_first, slice(None), slice(None))
            # For arrays with more than 3 dimensions, fix the remaining dimensions
            if len(shape) > 3:
                remaining_indices = [random.randint(0, dim - 1) for dim in shape[3:]]
                idx = idx + tuple(remaining_indices)
        elif access_pattern == "full":
            idx = tuple(slice(None) for _ in shape)

        random_indices.append(idx)

    # Warmup phase - not timed
    for i in range(warmup_samples):
        _ = zarr_array[random_indices[i]]

    # Actual timing phase
    times = []
    for i in range(warmup_samples, total_samples):
        start_time = time.perf_counter()
        value = zarr_array[random_indices[i]]
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        del value

    return times


def benchmark_zarr_array(
    zarr_array: zarr.Array,
    access_pattern: Literal["point", "time_series", "spatial_slice", "full"] = "point",
    num_samples: int = 100,
) -> dict:
    """
    Comprehensive benchmark of zarr array random access performance.

    Returns detailed statistics about the performance.

    Parameters
    ----------
    zarr_array
        The zarr array to benchmark
    access_pattern
        Type of access pattern: "point", "time_series", "spatial_slice", "full"
    num_samples
        Number of random access operations to perform

    Returns
    -------
    dict
        A dictionary containing performance statistics including mean, median, std deviation, min, max access times
        and details about the zarr array such as shape, dtype, and size.
    """

    times = _measure_zarr_random_access_performance(
        zarr_array, access_pattern, num_samples
    )

    stats = {
        "mean_time": Quantity(np.mean(times), "seconds"),
        "median_time": Quantity(np.median(times), "seconds"),
        "std_time": Quantity(np.std(times), "seconds"),
        "min_time": Quantity(np.min(times), "seconds"),
        "max_time": Quantity(np.max(times), "seconds"),
        "total_samples": num_samples,
        "access_pattern": access_pattern,
        "array_shape": zarr_array.shape,
        "chunk_shape": zarr_array.chunks,
        "chunk_size": Quantity(
            np.prod(zarr_array.chunks) * zarr_array.dtype.itemsize, "bytes"
        ).to("MB"),
        "nchunks": zarr_array.nchunks,
        "shard_shape": getattr(
            zarr_array, "shards", None
        ),  # Handle case where shards might not exist
        "array_dtype": zarr_array.dtype,
        "array_size_memory": Quantity(zarr_array.nbytes, "bytes").to("GB"),
        "array_size_store": Quantity(array_storage_size(zarr_array), "bytes").to("GB"),
        "array_compressors": zarr_array.compressors,
    }
    stats["compression_ratio"] = (
        f"{(stats["array_size_memory"] / stats["array_size_store"]).magnitude:.2f}:1"  # type: ignore[operator]
    )
    stats["zarr_concurrency"] = zarr.config.get("async.concurrency")

    return stats


# Convenience function to benchmark all access patterns
def benchmark_access_patterns(
    zarr_array: zarr.Array, num_samples: int = 100
) -> pd.DataFrame:
    """
    Benchmark all three access patterns and return combined results.

    Parameters
    ----------
    zarr_array
        The zarr array to benchmark
    num_samples
        Number of random access operations to perform for each pattern

    Returns
    -------
    pd.DataFrame
        [pandas.DataFrame][] with results for each access pattern
    """

    results = {}
    access_pattern = ["point", "time_series", "spatial_slice", "full"]

    for pattern in access_pattern:
        results[pattern] = benchmark_zarr_array(
            zarr_array,
            access_pattern=pattern,  # type: ignore[arg-type]
            num_samples=num_samples,
        )
    return pd.DataFrame.from_dict(results, orient="index")


__all__ = [
    "benchmark_zarr_array",
    "benchmark_access_patterns",
]
