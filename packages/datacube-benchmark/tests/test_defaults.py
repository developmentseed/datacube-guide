"""Tests for the default coordinate builders in ``datacube_benchmark.defaults``."""

import numpy as np

from datacube_benchmark.defaults import (
    default_latitude_coords,
    default_longitude_coords,
    default_time_coords,
)


def test_time_coords_length_and_metadata():
    coords = default_time_coords(timesteps=10, start_date="2000-01-01 00:00:00")
    assert coords["dims"] == "time"
    assert len(coords["data"]) == 10
    assert coords["data"].dtype == np.int32
    assert "2000-01-01 00:00:00" in coords["attrs"]["units"]
    assert coords["attrs"]["standard_name"] == "time"


def test_time_coords_defaults_to_a_year():
    assert len(default_time_coords()["data"]) == 365


def test_longitude_coords_span_and_units():
    coords = default_longitude_coords(resolution=0.25)
    data = coords["data"]
    assert coords["dims"] == "longitude"
    assert coords["attrs"]["units"] == "degrees_east"
    assert data.dtype == np.float32
    assert data.min() == -180.0
    assert data.max() < 180.0
    assert len(data) == 360 / 0.25


def test_latitude_coords_span_and_units():
    coords = default_latitude_coords(resolution=0.5)
    data = coords["data"]
    assert coords["dims"] == "latitude"
    assert coords["attrs"]["units"] == "degrees_north"
    assert data.min() == -90.0
    assert data.max() < 90.0
    assert len(data) == 180 / 0.5
