from dataclasses import dataclass, field
from typing import Any
import zarr.codecs


@dataclass
class Config:
    create_data: bool = True
    compressor: Any = field(
        default_factory=lambda: zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        )
    )
    target_array_size: str = "25 MB"
    data_var: str = "data"
    num_samples: int = 1
    warmup_samples: int = 0
    credential_provider: Any = None
    zarr_concurrency: int = 128
