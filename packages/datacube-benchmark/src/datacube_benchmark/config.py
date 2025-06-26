from dataclasses import dataclass
import zarr


@dataclass
class Config:
    create_data = True
    compressor = zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
    )
    target_array_size = "25 MB"
    data_var = "data"
    num_samples = 1
    warmup_samples = 0
    credential_provider = None
    zarr_concurrency = 128
