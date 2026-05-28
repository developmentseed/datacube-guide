"""Shared fixtures for the integration tests.

The integration tier writes real Zarr stores to a local ``obstore`` store under
a temporary directory -- no cloud required. Datasets are kept intentionally
tiny (a coarse global grid) so the suite stays fast.
"""

import random

import numpy as np
import obstore as obs
import pytest
import zarr

from datacube_benchmark import create_zarr_store

# A coarse 10-degree global grid: 36 lon x 18 lat. Small enough to be fast,
# large enough to span multiple chunks.
SMALL_DATASET = {
    "target_array_size": "1 MB",
    "target_spatial_resolution": "10 degrees",
    "target_chunk_size": "1 MB",
}


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the RNGs the query/fill code samples from, for stable runs."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture(scope="module")
def object_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("zarr_store")
    return obs.store.LocalStore(str(path))


@pytest.fixture(scope="module")
def zarr_store(object_store):
    return create_zarr_store(object_store, fill_method="arange", **SMALL_DATASET)


@pytest.fixture(scope="module")
def zarr_array(zarr_store):
    return zarr.open_array(zarr_store, path="data")
