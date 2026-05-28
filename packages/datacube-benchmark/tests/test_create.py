"""Tests for dataset/array creation in ``datacube_benchmark.create``."""

import numpy as np
import obstore as obs
import pytest
import xarray as xr
import zarr

from datacube_benchmark import (
    Config,
    create_empty_dataarray,
    create_or_open_zarr_array,
    create_or_open_zarr_store,
    create_zarr_store,
)
from datacube_benchmark.create import create_empty_dataset, fill_zarr_array

COARSE = {
    "target_array_size": "1 MB",
    "target_spatial_resolution": "10 degrees",
    "target_chunk_size": "1 MB",
}


# --- create_empty_dataarray / create_empty_dataset ------------------------


def test_empty_dataarray_dims_and_dtype():
    da = create_empty_dataarray(dtype=np.dtype("float64"), **COARSE)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "latitude", "longitude")
    assert da.dtype == np.float64
    # 10-degree grid.
    assert da.sizes["latitude"] == 18
    assert da.sizes["longitude"] == 36
    # Chunked via dask.
    assert da.chunks is not None


def test_empty_dataset_wraps_data_var():
    ds = create_empty_dataset(**COARSE)
    assert isinstance(ds, xr.Dataset)
    assert "data" in ds.data_vars


# --- fill_zarr_array ------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "check"),
    [
        ("zeros", lambda a: np.all(a[:] == 0)),
        ("ones", lambda a: np.all(a[:] == 1)),
        ("arange", lambda a: a[0, 0, 0] == 0 and a[-1, -1, -1] == a.size - 1),
        ("random", lambda a: np.all((a[:] >= 0) & (a[:] < 1))),
    ],
)
def test_fill_zarr_array_methods(method, check):
    arr = zarr.zeros(shape=(2, 3, 4), dtype="float32")
    fill_zarr_array(arr, method=method)
    assert check(arr)


def test_fill_zarr_array_rejects_unknown_method():
    arr = zarr.zeros(shape=(2, 3, 4), dtype="float32")
    with pytest.raises(ValueError, match="Method must be one of"):
        fill_zarr_array(arr, method="sevens")


# --- create_zarr_store ----------------------------------------------------


def test_create_zarr_store_writes_objects(tmp_path):
    store = obs.store.LocalStore(str(tmp_path))
    zarr_store = create_zarr_store(store, fill_method="zeros", **COARSE)
    arr = zarr.open_array(zarr_store, path="data")
    assert arr.shape[1:] == (18, 36)
    assert np.all(arr[:] == 0)


def test_create_zarr_store_with_chunked_coords(tmp_path):
    store = obs.store.LocalStore(str(tmp_path))
    zarr_store = create_zarr_store(
        store, fill_method="zeros", chunked_coords=True, **COARSE
    )
    # Coordinates are written as single-element chunks.
    lon = zarr.open_array(zarr_store, path="longitude")
    assert lon.chunks == (1,)


# --- create_or_open_zarr_store / _array (file:// url, no cloud) ------------


def test_create_then_open_roundtrip(tmp_path):
    url = "file://" + str(tmp_path)
    create_config = Config(create_data=True, target_array_size="1 MB")

    # First call creates the data.
    store = create_or_open_zarr_store(
        url, target_chunk_size="1 MB", config=create_config
    )
    assert isinstance(store, zarr.storage.ObjectStore)

    # Second call with create_data=False opens the existing array read-only.
    open_config = Config(create_data=False, target_array_size="1 MB")
    arr = create_or_open_zarr_array(url, target_chunk_size="1 MB", config=open_config)
    assert isinstance(arr, zarr.Array)
    assert arr.ndim == 3
