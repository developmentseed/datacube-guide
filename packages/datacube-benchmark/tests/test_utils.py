"""Tests for the object-store helpers in ``datacube_benchmark.utils``."""

import obstore as obs
import pytest

from datacube_benchmark.utils import (
    array_storage_size,
    number_of_objects,
    validate_object_store_contains_zarr,
)


def test_validate_passes_for_zarr_store(object_store):
    # The populated store from the fixture is a valid Zarr hierarchy.
    validate_object_store_contains_zarr(object_store)


def test_validate_passes_for_empty_store(tmp_path):
    validate_object_store_contains_zarr(obs.store.LocalStore(str(tmp_path)))


def test_validate_rejects_non_zarr_keys(tmp_path):
    store = obs.store.LocalStore(str(tmp_path))
    store.put("not-a-zarr-key.txt", b"hello")
    with pytest.raises(ValueError, match="Invalid paths found"):
        validate_object_store_contains_zarr(store)


def test_array_storage_size_is_positive(zarr_array):
    assert array_storage_size(zarr_array) > 0


def test_number_of_objects_matches_listing(object_store):
    expected = len(object_store.list().collect())
    assert number_of_objects(object_store) == expected
    assert number_of_objects(object_store) > 0
