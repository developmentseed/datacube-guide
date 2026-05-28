"""Tests for the access-pattern benchmarks in ``datacube_benchmark.query``."""

import pandas as pd
import pytest
import zarr

from datacube_benchmark import benchmark_access_patterns, benchmark_zarr_array
from datacube_benchmark.query import _measure_zarr_random_access_performance

PATTERNS = ["point", "time_series", "spatial_slice", "full"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_benchmark_zarr_array_returns_stats(zarr_array, pattern):
    stats = benchmark_zarr_array(
        zarr_array, access_pattern=pattern, num_samples=2, warmup_samples=1
    )
    assert stats["access_pattern"] == pattern
    assert stats["total_samples"] == 2
    # Timing stats are populated and the derived compression ratio is rendered.
    assert stats["mean_time"].magnitude >= 0
    assert stats["compression_ratio"].endswith(":1")


def test_benchmark_access_patterns_covers_all_patterns(zarr_array):
    df = benchmark_access_patterns(zarr_array, num_samples=2, warmup_samples=1)
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == PATTERNS


@pytest.mark.parametrize("pattern", ["time_series", "spatial_slice"])
def test_low_dimensional_array_rejected(pattern):
    arr_2d = zarr.zeros(shape=(4, 5), dtype="float32")
    with pytest.raises(ValueError, match="at least 3D"):
        benchmark_zarr_array(
            arr_2d, access_pattern=pattern, num_samples=1, warmup_samples=0
        )


def test_spatial_slice_on_4d_array():
    # >3D arrays fix the trailing dimensions of a spatial slice. Exercise the
    # timing helper directly, since benchmark_zarr_array's stats need an
    # obstore-backed array for on-disk size, which the library only makes in 3D.
    arr_4d = zarr.zeros(shape=(3, 4, 5, 6), dtype="float32")
    times = _measure_zarr_random_access_performance(
        arr_4d, access_pattern="spatial_slice", num_samples=2, warmup_samples=0
    )
    assert len(times) == 2
