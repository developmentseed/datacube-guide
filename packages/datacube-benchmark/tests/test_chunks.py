"""Tests for the pure chunk-shape math in ``datacube_benchmark.chunks``.

These functions have clear invariants rather than memorable return values, so
we lean on property-based testing (Hypothesis) for the general guarantees and
keep a few worked examples for the arithmetic itself.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pint import Quantity

from datacube_benchmark.chunks import (
    calculate_thickness,
    find_chunk_shape,
    get_slice_size,
)

# Byte-valued quantities, kept small enough to stay fast but large enough to
# exercise the multi-chunk regime the real library cares about.
byte_sizes = st.integers(min_value=1, max_value=10_000_000)
shapes = st.tuples(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
)


def qty(n: int) -> Quantity:
    return Quantity(n, "bytes")


# --- calculate_thickness ---------------------------------------------------


@given(slice_n=byte_sizes, target_n=byte_sizes)
def test_thickness_is_positive_int(slice_n, target_n):
    for method in ("nearest", "over", "under"):
        thickness = calculate_thickness(qty(slice_n), qty(target_n), method=method)
        assert isinstance(thickness, int)
        assert thickness >= 1


@given(slice_n=byte_sizes, target_n=byte_sizes)
def test_over_rounds_up_past_target(slice_n, target_n):
    """An "over" chunk always holds at least the target's worth of slices."""
    thickness = calculate_thickness(qty(slice_n), qty(target_n), method="over")
    assert thickness * qty(slice_n) > qty(target_n)


@given(slice_n=byte_sizes, target_n=byte_sizes)
def test_over_at_least_under(slice_n, target_n):
    over = calculate_thickness(qty(slice_n), qty(target_n), method="over")
    under = calculate_thickness(qty(slice_n), qty(target_n), method="under")
    assert over >= under


def test_thickness_examples():
    assert calculate_thickness(qty(10), qty(100), method="over") == 11
    assert calculate_thickness(qty(10), qty(100), method="under") == 10
    assert calculate_thickness(qty(10), qty(100), method="nearest") == 10


def test_thickness_clamped_to_one_when_target_smaller_than_slice():
    # target // slice == 0, which would mean an empty chunk; clamp to 1.
    assert calculate_thickness(qty(200), qty(100), method="under") == 1


def test_thickness_rejects_unknown_method():
    with pytest.raises(ValueError, match="Method must be one of"):
        calculate_thickness(qty(10), qty(100), method="sideways")  # type: ignore[arg-type]


# --- get_slice_size --------------------------------------------------------


def test_slice_size_examples():
    item = qty(4)
    shape = (10, 20, 30)
    assert get_slice_size(shape, item, "pancake") == qty(4 * 20 * 30)
    assert get_slice_size(shape, item, "churro") == qty(4 * 10)
    assert get_slice_size(shape, item, "dumpling") == qty(4)


def test_slice_size_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Unrecognized chunk shape"):
        get_slice_size((1, 2, 3), qty(4), "spaghetti")  # type: ignore[arg-type]


# --- find_chunk_shape ------------------------------------------------------


@given(shape=shapes, item_n=byte_sizes, target_n=byte_sizes)
def test_chunk_shape_is_3d_and_positive(shape, item_n, target_n):
    for chunk_kind in ("pancake", "churro", "dumpling"):
        chunk = find_chunk_shape(shape, qty(item_n), qty(target_n), chunk_kind)
        assert len(chunk) == 3
        assert all(dim >= 1 for dim in chunk)


@given(shape=shapes, item_n=byte_sizes, target_n=byte_sizes)
def test_pancake_keeps_spatial_dims_whole(shape, item_n, target_n):
    """A pancake only chunks along the first (time) axis."""
    chunk = find_chunk_shape(shape, qty(item_n), qty(target_n), "pancake")
    assert chunk[1] == shape[1]
    assert chunk[2] == shape[2]
    assert chunk[0] >= 1


def test_find_chunk_shape_rejects_unknown_shape():
    with pytest.raises(ValueError, match="Unrecognized chunk shape"):
        find_chunk_shape((1, 2, 3), qty(4), qty(100), "lasagna")  # type: ignore[arg-type]


def test_pancake_example():
    # 4-byte items, 10x10 spatial grid -> 400 byte slices. A 1000 byte target
    # rounds "over" to floor(1000/400)+1 == 3 time steps per chunk.
    chunk = find_chunk_shape((365, 10, 10), qty(4), qty(1000), "pancake")
    assert chunk == (3, 10, 10)


def test_churro_and_dumpling_widths_are_finite():
    chunk_churro = find_chunk_shape((100, 50, 50), qty(4), qty(10_000), "churro")
    chunk_dumpling = find_chunk_shape((100, 50, 50), qty(4), qty(10_000), "dumpling")
    assert np.isfinite(chunk_churro).all()
    assert np.isfinite(chunk_dumpling).all()
