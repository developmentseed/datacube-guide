import zarr
from obstore.store import ObjectStore
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import BytesBytesCodec


def generate_zarr_array(
    store: ObjectStore,
    size: tuple[int, ...] = (100, 100, 100),
    chunk_size: tuple[int, ...] = (10, 10, 10),
    shard_size: None | tuple[int, ...] = None,
    compressors: None | Iterable["BytesBytesCodec"] = None,
    dtype: str = "float32",
    array_name: str = "test_array",
):
    """
    Generate a Zarr array in the given ObjectStore.

    Parameters
    ----------
    store
        The ObjectStore where the Zarr array will be created.
    size, optional
        Array size, by default (100, 100, 100)
    chunk_size, optional
        Array chunk size, by default (10, 10, 10)
    shard_size, optional
        Array shard size, by default None
    compressors, optional
        Compressors to use on the array, by default None
    dtype, optional
        Array dtype, by default "float32"
    array_name, optional
        Name for the array, by default "test_array"

    Returns
    -------
        A Zarr array filled with zeros in the specified ObjectStore.
    """
    zarr_store = zarr.storage.ObjectStore(store=store, read_only=False)
    array = zarr.create_array(zarr_store, shape=size, chunks=chunk_size, dtype=dtype)
    array[:] = 0
    return array
