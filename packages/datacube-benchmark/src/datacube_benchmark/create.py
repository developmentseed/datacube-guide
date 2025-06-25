from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Literal
from dask import array as da
import numpy as np
import xarray as xr
from pint import UnitRegistry, Quantity
from .defaults import (
    default_longitude_coords,
    default_latitude_coords,
    default_time_coords,
    default_data_attrs,
    default_data_name,
)
from .chunks import calculate_thickness, find_chunk_shape
from .utils import validate_object_store_contains_zarr
import obstore as obs
import zarr

if TYPE_CHECKING:
    from pint import Quantity
    from .types import TARGET_SHAPES
    from numcodecs.abc import Codec
    from zarr.abc.codec import BytesBytesCodec

ureg = UnitRegistry()


def _write_using_obstore(
    ds: xr.Dataset,
    object_store: obs.store.ObjectStore,
    compressor: Codec | BytesBytesCodec | None = None,
) -> zarr.storage.ObjectStore:
    zarr_store = zarr.storage.ObjectStore(
        store=object_store,
        read_only=False,
    )
    ds.to_zarr(
        store=zarr_store,
        mode="w",
        encoding={var: {"compressors": compressor} for var in ds.data_vars},
    )  # type: ignore[call-overload]
    return zarr_store


def create_zarr_store(
    object_store: obs.store.ObjectStore,
    target_array_size: str | Quantity = "1 GB",
    target_spatial_resolution: str | Quantity = ".5 degrees",
    target_chunk_size: str | Quantity = "10 MB",
    target_chunk_shape: TARGET_SHAPES = "dumpling",
    compressor: Codec | BytesBytesCodec | None = None,
    dtype: np.dtype = np.dtype("float32"),
    fill_method: Literal["random", "zeros", "ones", "arange"] = "arange",
) -> zarr.storage.ObjectStore:
    """
    Create a Zarr store in the specified object store with an empty dataset.

    Parameters
    ----------
    object_store
        The object store to write the Zarr dataset to.
    target_array_size
        The size of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_spatial_resolution
        The spatial resolution of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_size
        The size of the chunks in the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_shape
        The shape of the [xarray.DataArray][], default is "dumpling".
    compressor
        The compressor to use for the Zarr store, default is None (no compression).
    dtype
        The data type of the [xarray.DataArray][], default is np.dtype("float32").
    fill_method
        The method to use for filling the Zarr array. Options are:
        - "random": Fill with random values.
        - "zeros": Fill with zeros.
        - "ones": Fill with ones.
        - "arange": Fill with a range of values.

    Returns
    -------
    zarr.storage.ObjectStore
        A Zarr store with the specified parameters.
    """
    warnings.filterwarnings(
        "ignore",
        message="Object at .* is not recognized as a component of a Zarr hierarchy",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Consolidated metadata is currently not part in the Zarr format 3 specification. *",
        category=UserWarning,
    )
    validate_object_store_contains_zarr(object_store=object_store)
    ds = create_empty_dataset(
        target_array_size=target_array_size,
        target_spatial_resolution=target_spatial_resolution,
        target_chunk_size=target_chunk_size,
        target_chunk_shape=target_chunk_shape,
        dtype=dtype,
    )
    zarr_store = _write_using_obstore(
        ds=ds,
        object_store=object_store,
        compressor=compressor,
    )
    arr = zarr.open_array(store=zarr_store, zarr_version=3, path="data")
    fill_zarr_array(arr=arr, method=fill_method)
    return zarr_store


def fill_zarr_array(
    arr=zarr.Array, method=Literal["random", "zeros", "ones", "arange"]
) -> None:
    """
    Fill a Zarr array with specified data.

    Parameters
    ----------
    arr
        The Zarr array to fill.
    method
        The method to use for filling the array. Options are:
        - "random": Fill with random values.
        - "zeros": Fill with zeros.
        - "ones": Fill with ones.
        - "arange": Fill with a range of values.

    Returns
    -------
    None
        The function modifies the Zarr array in place.
    """
    if method == "random":
        arr[:] = np.random.random(arr.shape)
    elif method == "zeros":
        arr[:] = 0
    elif method == "ones":
        arr[:] = 1
    elif method == "arange":
        arr[:] = np.arange(np.prod(arr.shape)).reshape(arr.shape)
    else:
        raise ValueError(
            "Method must be one of 'random', 'zeros', 'ones', or 'arange'."
        )


def create_empty_dataset(
    target_array_size: str | Quantity = "1 GB",
    target_spatial_resolution: str | Quantity = ".5 degrees",
    target_chunk_size: str | Quantity = "10 MB",
    target_chunk_shape: TARGET_SHAPES = "dumpling",
    dtype: np.dtype = np.dtype("float32"),
    name: str = "data",
) -> xr.Dataset:
    """
    Create an empty [xarray.Dataset][] with specified size, shape, and dtype.

    Parameters
    ----------
    target_array_size
        The size of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_spatial_resolution
        The spatial resolution of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_size
        The size of the chunks in the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_shape
        The shape of the [xarray.DataArray][], default is "dumpling".
    dtype
        The data type of the [xarray.DataArray][], default is np.dtype("float32")
    name
        The name of the [xarray.DataArray][] within the [xarray.Dataset], default is "data".

    Returns
    -------
    xr.Dataset
        A [xarray.Dataset][] with the specified parameters.
    """
    da = create_empty_dataarray(
        target_array_size=target_array_size,
        target_spatial_resolution=target_spatial_resolution,
        target_chunk_size=target_chunk_size,
        target_chunk_shape=target_chunk_shape,
        dtype=dtype,
    )
    return da.to_dataset(name="data")


def create_empty_dataarray(
    target_array_size: str | Quantity = "1 GB",
    target_spatial_resolution: str | Quantity = ".5 degrees",
    target_chunk_size: str | Quantity = "10 MB",
    target_chunk_shape: TARGET_SHAPES = "dumpling",
    dtype: np.dtype = np.dtype("float32"),
) -> xr.DataArray:
    """
    Create an empty [xarray.DataArray][] with specified size, shape, and dtype.

    Parameters
    ----------
    target_array_size
        The size of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_spatial_resolution
        The spatial resolution of the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_size
        The size of the chunks in the [xarray.DataArray][], can be a string or a [pint.Quantity][].
        String must be convertible to a [pint.Quantity][].
    target_chunk_shape
        The shape of the [xarray.DataArray][], default is "dumpling".
    dtype
        The data type of the [xarray.DataArray][], default is np.dtype("float32")

    Returns
    -------
    xr.DataArray
        An empty [xarray.DataArray][] with the specified parameters.
    """
    spatial_res: float = (
        (
            Quantity(target_spatial_resolution)
            if isinstance(target_spatial_resolution, str)
            else target_spatial_resolution
        )
        .to("degrees")
        .magnitude
    )
    target_size = (
        Quantity(target_array_size)
        if isinstance(target_array_size, str)
        else target_array_size
    )
    if isinstance(target_chunk_size, str):
        target_chunk_size = Quantity(target_chunk_size)
    longitude_coords = default_longitude_coords(spatial_res)
    latitude_coords = default_latitude_coords(spatial_res)
    item_size = Quantity(dtype.itemsize, "bytes")
    slice_size = (
        item_size * len(longitude_coords["data"]) * len(latitude_coords["data"])
    )
    number_of_timesteps = calculate_thickness(
        slice_size=slice_size,
        target_size=target_size,
        method="over",
    )
    time_coords = default_time_coords(number_of_timesteps)
    chunk_shape = find_chunk_shape(
        array_shape=(
            len(time_coords["data"]),
            len(latitude_coords["data"]),
            len(longitude_coords["data"]),
        ),
        item_size=item_size,
        target_chunk_size=target_chunk_size,
        target_chunk_shape=target_chunk_shape,
    )
    data = da.empty(
        shape=(
            len(time_coords["data"]),
            len(latitude_coords["data"]),
            len(longitude_coords["data"]),
        ),
        dtype=dtype,
        chunks=chunk_shape,
    )
    return xr.DataArray.from_dict(
        {
            "coords": {
                "time": time_coords,
                "longitude": longitude_coords,
                "latitude": latitude_coords,
            },
            "attrs": default_data_attrs,
            "dims": ["time", "latitude", "longitude"],
            "data": data,
            "name": default_data_name,
        }
    )
