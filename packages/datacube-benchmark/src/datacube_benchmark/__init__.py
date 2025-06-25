from .create import create_empty_dataarray
from .create import Quantity
from .chunks import find_chunk_shape
import numpy as np

__all__ = ["create_empty_dataarray", "find_chunk_shape"]


def main() -> None:
    da = create_empty_dataarray()
    array_size = Quantity(da.nbytes, "bytes")
    chunk_size = Quantity(np.prod(da.data.chunksize) * da.dtype.itemsize, "bytes")
    print(da)
    print(f"Array size: {array_size.to('GB')}")
    print(f"Chunk size: {chunk_size.to('MB')}")
