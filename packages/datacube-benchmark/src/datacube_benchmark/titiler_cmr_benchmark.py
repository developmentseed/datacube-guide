"""Benchmarks for titiler-cmr"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Tuple

import httpx
import morecantile
import pytest
import pandas as pd

# -- benchmark a viewport:


def benchmark_titiler_cmr(
    endpoint: str ="https://staging.openveda.cloud/api/titiler-cmr",
    concept_id: str,
    min_zoom: int = 0,
    max_zoom: int = 20,
    tile_scale: int = 3,
    resampling: Resampling = "bilinear",
    start_date: datetime = datetime(2023, 2, 24, 0, 0, 1),
    end_date: datetime = datetime(2023, 2, 25, 0, 0, 1),
    colormap_name: Colormap = "gnbu",
    lng: float = -92.1,
    lat: float = 46.8,
    rescale: None | tuple[int, int] = None,
) -> pd.DataFrame:

    # test concept_id
    concept_id = "C2021957295-LPCLOUD"

    datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"


    # loop over all zoom levels...
    async with httpx.AsyncClient(timeout=30.0) as client:
        for zoom in range(min_zoom, max_zoom + 1):
            print(f"Benchmarking Zoom level: {zoom}")
            center_tile = tms.tile(lng=lng, lat=lat, zoom=zoom)
            tiles_to_fetch = get_surrounding_tiles(center_tile.x, center_tile.y, zoom)

            # Create a list of coroutine tasks
            tasks = [
                fetch_tile(
                    client,
                    endpoint,
                    zoom,
                    x,
                    y,
                    concept_id,
                    datetime_range,
                    assets,
                    resampling,
                    colormap_name,
                    rescale,
                )
                for x, y in tiles_to_fetch
            ]
            responses = await asyncio.gather(*tasks)

            for (x, y), response in zip(tiles_to_fetch, responses):
                results.append(
                    {
                        "zoom": zoom,
                        "x": x,
                        "y": y,
                        "status_code": response.status_code,
                        "response_time_sec": response.elapsed.total_seconds(),
                        "response_size_bytes": len(response.content),
                        "is_error": response.is_error,
                    }
                )

    return pd.DataFrame(results)


# -- titiler -utils
def get_surrounding_tiles(
    x: int, y: int, zoom: int, width: int = TILES_WIDTH, height: int = TILES_HEIGHT
) -> List[Tuple[int, int]]:
    """
    https://github.com/developmentseed/titiler-cmr/blob/develop/tests/test_hls_benchmark.py
    Fetch all tiles for a viewport and return detailed metrics
    """
    tiles = []
    offset_x = width // 2
    offset_y = height // 2

    for y_pos in range(y - offset_y, y + offset_y + 1):
        for x_pos in range(x - offset_x, x + offset_x + 1):
            # Ensure x, y are valid for the zoom level
            max_tile = 2**zoom - 1
            x_valid = max(0, min(x_pos, max_tile))
            y_valid = max(0, min(y_pos, max_tile))
            tiles.append((x_valid, y_valid))

    return tiles


async def fetch_tile(
    client: httpx.AsyncClient,
    endpoint: str,
    z: int,
    x: int,
    y: int,
    concept_id: str,
    datetime_range: str,
    assets: List[str],
    resampling: Resampling,
    colormap_name: Colormap,
    rescale: None | tuple[int, int],
    n_bands: int,
) -> httpx.Response:
    """Fetch a single tile and return the httpx.response object."""
    url = f"{endpoint}/tiles/WebMercatorQuad/{z}/{x}/{y}.png"

    params: Dict[str, Any] = {
        "concept_id": collection_config.concept_id,
        "datetime": datetime_range,
    }

    params.update(get_band_params(collection_config, n_bands))

    start_time = datetime.now()
    try:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        elapsed = (datetime.now() - start_time).total_seconds()

        response.elapsed = timedelta(seconds=elapsed)
        return response
    except Exception:
        # Create a mock response for exceptions
        mock_response = httpx.Response(500, request=httpx.Request("GET", url))
        mock_response.elapsed = datetime.now() - start_time
        return mock_response

