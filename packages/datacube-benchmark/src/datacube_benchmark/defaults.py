import numpy as np

default_resolution = (0.25, 0.25)  # degrees
default_timesteps = 365
default_data_name = "psl"
default_data_attrs = {
    "long_name": "mean sea level pressure",
    "standard_name": "air_pressure_at_sea_level",
    "units": "hPa",
    "grid_mapping": "crs",
}
default_crs = {
    "attrs": {
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }
}


def default_time_coords(
    timesteps: int = default_timesteps,
    start_date: str = "1990-01-01 00:00:00",
) -> dict:
    """
    Create a default time coordinate dictionary.

    Parameters
    ----------
    timesteps : int
        Number of timesteps, default is 365.
    start_date : str
        Start date in the format 'YYYY-MM-DD HH:MM:SS', default is '1990-01-01 00:00:00'.

    Returns
    -------
    dict
        A dictionary representing the time coordinate.
    """
    return {
        "data": np.arange(timesteps, dtype=np.int32),
        "dims": "time",
        "attrs": {"standard_name": "time", "units": f"days since {start_date}"},
    }


def default_longitude_coords(
    resolution: float = default_resolution[1],
) -> dict:
    """
    Create a default longitude coordinate dictionary.

    Parameters
    ----------
    resolution : float
        Longitude resolution in degrees, default is 0.25.

    Returns
    -------
    dict
        A dictionary representing the longitude coordinate.
    """
    return {
        "data": np.arange(-180, 180, resolution, dtype=np.float32),
        "dims": "longitude",
        "attrs": {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
    }


def default_latitude_coords(
    resolution: float = default_resolution[0],
) -> dict:
    """
    Create a default latitude coordinate dictionary.

    Parameters
    ----------
    resolution : float
        Latitude resolution, default is 0.25.

    Returns
    -------
    dict
        A dictionary representing the latitude coordinate.
    """
    return {
        "data": np.arange(-90, 90, resolution, dtype=np.float32),
        "dims": "latitude",
        "attrs": {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
    }
