"""Tests for ``datacube_benchmark.open.benchmark_dataset_open``."""

import pandas as pd

from datacube_benchmark import benchmark_dataset_open


def test_benchmark_dataset_open_returns_dataframe(zarr_store):
    df = benchmark_dataset_open(zarr_store, num_samples=2, warmup_samples=1)
    assert isinstance(df, pd.DataFrame)
    # Stats are returned as a single column indexed by stat name.
    assert "mean_time" in df.index
    assert df.loc["total_samples"].item() == 2
