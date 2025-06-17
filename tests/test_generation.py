def test_generate_zarr_array():
    from obstore.store import MemoryStore
    from datacube_practices.generation import generate_zarr_array

    # Create a mock ObjectStore
    store = MemoryStore()

    # Generate a Zarr array
    array = generate_zarr_array(store)

    # Check if the array is created with the correct shape and dtype
    assert array.shape == (100, 100, 100)
    assert array.dtype == "float32"

    # Check if the array is filled with zeros
    assert (array[:] == 0).all()
