from datacube_benchmark import create_empty_dataarray
import obstore as obs
from pathlib import Path
import zarr


def main():
    da = create_empty_dataarray()
    ds = da.to_dataset(name="data")
    prefix = Path.cwd() / "data" / "test_dataset.zarr"
    object_store = obs.store.LocalStore(
        prefix=prefix,
        mkdir=True,
    )
    zarr_store = zarr.storage.ObjectStore(
        store=object_store,
        read_only=False,
    )
    compressor = None  # zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    ds.to_zarr(
        store=zarr_store,
        mode="w",
        encoding={var: {"compressors": compressor} for var in ds.data_vars},
    )
    print(f"Dataset saved to {prefix / 'test_dataset.zarr'}")


if __name__ == "__main__":
    main()
