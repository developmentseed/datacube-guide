import obstore as obs
import pyarrow as pa
import re
from zarr import Array


def validate_object_store_contains_zarr(object_store: obs.store.ObjectStore) -> None:
    """
    Validate that all keys in the object store match the Zarr structure.
    """
    zarr_json_pattern = r"^.*zarr\.json$"
    chunk_pattern = r"^.*/c/\d+(?:/\d+)*$"
    combined_pattern = f"({zarr_json_pattern})|({chunk_pattern})"

    stream = object_store.list(return_arrow=True)
    paths = stream.collect().column("path").to_pylist()
    invalid_paths = []

    for path in paths:
        if not re.match(combined_pattern, path):
            invalid_paths.append(path)
    if invalid_paths:
        raise ValueError(
            f"Invalid paths found under {object_store}'s prefix: {invalid_paths}. "
            "All paths must match the Zarr structure."
        )


def array_storage_size(array: Array) -> int:
    """
    Validate that all keys in the object store match the Zarr structure.
    """
    chunk_pattern = r"^.*/c/\d+(?:/\d+)*$"
    stream = array.store.store.list(return_arrow=True, prefix=array.path)  # type: ignore[attr-defined]
    df = pa.record_batch(stream.collect()).to_pandas()
    df = df[df["path"].str.match(chunk_pattern)]
    return df["size"].sum()


def number_of_objects(
    object_store: obs.store.ObjectStore,
) -> int:
    """
    Count the number of objects in the object store with the given prefix.

    Parameters
    ----------
    object_store
        The object store to count objects in.

    Returns
    -------
    int
        The number of objects with the given prefix.
    """
    return len(object_store.list().collect())


__all__ = ["number_of_objects"]
