from __future__ import annotations

from typing import TYPE_CHECKING
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


if TYPE_CHECKING:
    from pint import Quantity
    from .types import TARGET_SHAPES

ureg = UnitRegistry()


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
