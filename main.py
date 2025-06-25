import datacube_benchmark
import obstore as obs
from pathlib import Path
import zarr


def main():
    prefix = "data/test_dataset.zarr"
    path = Path.cwd() / prefix
    store = obs.store.LocalStore(path)
    zarr_store = datacube_benchmark.create_zarr_store(store)
    arr = zarr.open_array(zarr_store, zarr_version=3, path="data")
    results = datacube_benchmark.benchmark_access_patterns(arr, num_samples=10)
    results


if __name__ == "__main__":
    main()
