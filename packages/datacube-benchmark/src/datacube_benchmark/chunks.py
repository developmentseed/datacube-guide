from __future__ import annotations

from typing import Literal, TYPE_CHECKING
import numpy as np
from .types import TARGET_SHAPES

if TYPE_CHECKING:
    from pint import Quantity


def calculate_thickness(
    slice_size: Quantity,
    target_size: Quantity,
    method: Literal["nearest", "over", "under"] = "over",
) -> int:
    """Calculate the thickness of a pancake chunk based on slice size and target chunk size.
    Parameters
    ----------
    slice_size : Quantity
        The size of the slice of the dataset.
    target_chunk_size : Quantity
        The target size of the chunk.
    method : Literal["nearest", "over", "under"]
        The method to use for calculating the thickness. Options are:
        - "nearest": Round to the nearest multiple of slice size.
        - "over": Round up to the next multiple of slice size.
        - "under": Round down to the previous multiple of slice size.
    Returns
    -------
    int
        The number of slices to include in a chunk.
    """
    if method == "nearest":
        chunk_thickness = round(target_size / slice_size)
    elif method == "over":
        chunk_thickness = (target_size // slice_size) + 1
    elif method == "under":
        chunk_thickness = target_size // slice_size
    else:
        raise ValueError("Method must be one of 'nearest', 'over', or 'under'.")
    if chunk_thickness <= 0:
        chunk_thickness = 1  # Ensure at least one slice per chunk
    return int(chunk_thickness)


def get_slice_size(
    array_shape: tuple[int, ...],
    item_size: Quantity,
    target_chunk_shape: TARGET_SHAPES = "dumpling",
) -> Quantity:
    """Calculate the size of a slice based on the shape of the array and the item size.

    Parameters
    ----------
    array_shape : tuple(int, ...)
        The shape of the array.
    item_size : Quantity
        The size of each item in the array.
    target_chunk_shape : target_chunk_shapeS
        The target shape of the chunk.

    Returns
    -------
    Quantity
        The size of a slice in the specified chunk shape.
    """
    if target_chunk_shape == "pancake":
        return item_size * array_shape[1] * array_shape[2]
    elif target_chunk_shape == "churro":
        return item_size * array_shape[0]
    elif target_chunk_shape == "dumpling":
        return item_size
    else:
        raise ValueError(
            f"Unrecognized chunk shape. Got {target_chunk_shape}, expected one of {TARGET_SHAPES}"
        )


def find_chunk_shape(
    array_shape: tuple[int, ...],
    item_size: Quantity,
    target_chunk_size: Quantity,
    target_chunk_shape: TARGET_SHAPES = "dumpling",
) -> tuple[int, ...]:
    """Find a reasonable chunk shape based on the array shape, item size, and target chunk size.
    Parameters
    ----------
    array_shape : tuple(int, ...)
        The shape of the array.
    item_size : Quantity
        The size of each item in the array.
    target_chunk_size : Quantity
        The target size of the chunk.
    target_chunk_shape : TARGET_SHAPES
        The target shape of the chunk. Options are "pancake", "churro", or "dumpling".
    Returns
    -------
    tuple(int, ...)
        A reasonable chunk shape based on the input parameters.
    """
    slice_size = get_slice_size(array_shape, item_size, target_chunk_shape)
    chunk_thickness = calculate_thickness(slice_size, target_chunk_size)

    if target_chunk_shape == "pancake":
        return (chunk_thickness, array_shape[1], array_shape[2])
    elif target_chunk_shape == "churro":
        chunk_width = np.ceil(np.sqrt(chunk_thickness))
        return (array_shape[0], chunk_width, chunk_width)
    elif target_chunk_shape == "dumpling":
        chunk_width = np.ceil(np.cbrt(chunk_thickness))
        return (chunk_width, chunk_width, chunk_width)
    else:
        raise ValueError(
            f"Unrecognized chunk shape. Got {target_chunk_shape}, expected one of {TARGET_SHAPES}"
        )
