import xarray as xr
import zarr
import pandas as pd
import numpy as np
import time
from typing import List
from pint import Quantity


def _measure_xarray_open_dataset(
    zarr_store: zarr.abc.store.Store,
    num_samples: int = 10,
    warmup_samples: int = 10,
) -> List[float]:
    """
    Measure time performance for opening the dataset contained in a Zarr store.

    Parameters
    ----------
    zarr_store
        The zarr store to test
    num_samples
        Number of random access operations to perform
    warmup_samples
        Number of warmup operations (not included in timing)

    Returns
    -------
    results
        A list of access times for opening the dataset.
    """
    # Warmup phase - not timed
    for i in range(warmup_samples):
        ds = xr.open_dataset(
            zarr_store,  # type: ignore
            engine="zarr",
            zarr_format=3,
        )
        ds.close()
        del ds

    # Actual timing phase
    times = []
    for i in range(num_samples):
        start_time = time.perf_counter()
        ds = xr.open_dataset(
            zarr_store,  # type: ignore
            engine="zarr",
            zarr_format=3,
        )
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        ds.close
        del ds

    return times


def benchmark_dataset_open(
    zarr_store: zarr.abc.store.Store, num_samples: int = 10, warmup_samples: int = 10
) -> pd.DataFrame:
    """
    Benchmark all three access patterns and return combined results.

    Parameters
    ----------
    zarr_store
        The zarr store to benchmark
    num_samples
        Number of random access operations to perform for each pattern
    warmup_samples
        Number of warmup operations (not included in timing)

    Returns
    -------
    pd.DataFrame
        [pandas.DataFrame][] with results for each access pattern
    """
    times = _measure_xarray_open_dataset(zarr_store, num_samples, warmup_samples)

    stats = {
        "mean_time": Quantity(np.mean(times), "seconds"),
        "median_time": Quantity(np.median(times), "seconds"),
        "std_time": Quantity(np.std(times), "seconds"),
        "min_time": Quantity(np.min(times), "seconds"),
        "max_time": Quantity(np.max(times), "seconds"),
        "total_samples": num_samples,
        "zarr_store": str(zarr_store),
    }
    stats["zarr_concurrency"] = zarr.config.get("async.concurrency")

    return pd.DataFrame.from_dict(stats, orient="index")


__all__ = [
    "benchmark_dataset_open",
]
