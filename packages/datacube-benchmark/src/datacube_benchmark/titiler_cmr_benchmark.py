"""Benchmarking utility for Titiler-cmr"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Tuple

import httpx
import morecantile
import pandas as pd


TILES_WIDTH = 5
TILES_HEIGHT = 5


# -- benchmark a viewport:
async def benchmark_titiler_cmr(
    concept_id: str,
    endpoint: str = "https://staging.openveda.cloud/api/titiler-cmr",
    *,
    tms: morecantile.TileMatrixSet = morecantile.tms.get("WebMercatorQuad"),
    min_zoom: int = 7,
    max_zoom: int = 10,
    tile_scale: int = 3,
    resampling: str = "nearest",
    colormap_name: Colormap = "gnbu",
    start_date: datetime = datetime(2023, 2, 24, 0, 0, 1),
    end_date: datetime = datetime(2023, 2, 25, 0, 0, 1),
    lng: float = -92.1,
    lat: float = 46.8,
    rescale: None | tuple[int, int] = None,
) -> pd.DataFrame:
    """
    Benchmarks the Titiler-cmr API for a specific viewport across multiple zoom levels.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark results.
    """

    datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"

    # loop over all zoom levels...
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for zoom in range(min_zoom, max_zoom + 1):
            print(f"Benchmarking Zoom level: {zoom}...")
            center_tile = tms.tile(lng=lng, lat=lat, zoom=zoom)
            tiles_to_fetch = get_surrounding_tiles(center_tile.x, center_tile.y, zoom)

            # all tasks to fetch tiles
            tasks = [
                fetch_tile(
                    client=client,
                    endpoint=endpoint,
                    z=zoom,
                    x=x,
                    y=y,
                    concept_id=concept_id,
                    datetime_range=datetime_range,
                    assets=assets or [],
                    resampling=resampling,
                    colormap_name=colormap_name,
                    rescale=rescale,
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
                        "has_data": response.status_code == 200,  # 204 means no data
                    }
                )

    return pd.DataFrame(results)


# -- this could all go to titiler-utils
def get_surrounding_tiles(
    x: int, y: int, zoom: int, width: int = TILES_WIDTH, height: int = TILES_HEIGHT
) -> List[Tuple[int, int]]:
    """
    Get a list of surrounding tile coordinates for a viewport.
    From https://github.com/developmentseed/titiler-cmr/blob/develop/tests/test_hls_benchmark.py
    """

    tiles: List[Tuple[int, int]] = []

    # tiles = []
    offset_x = width // 2
    offset_y = height // 2

    max_tile = 2**zoom - 1

    for y_pos in range(y - offset_y, y + offset_y + 1):
        for x_pos in range(x - offset_x, x + offset_x + 1):
            # Ensure x, y are valid for the zoom level

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
) -> httpx.Response:
    """
    Fetch a single tile and return the httpx.response object."""

    # url = f"{endpoint}/tiles/WebMercatorQuad/{z}/{x}/{y}.png"
    url = f"{endpoint}/tiles/{z}/{x}/{y}.png"

    params: Dict[str, Any] = {
        "concept_id": concept_id,
        "datetime": datetime_range,
        "resampling": resampling,
        "colormap_name": colormap_name,
    }

    # add assets (i.e. bands)
    if assets:
        params["assets"] = assets
    if rescale:
        params["rescale"] = f"{rescale[0]},{rescale[1]}"

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


if __name__ == "__main__":

    async def main():
        print("Starting benchmark with RGB assets...")
        projection = "WebMercatorQuad"
        tms = morecantile.tms.get(projection)

        df_rgb = await benchmark_titiler_cmr(
            concept_id="C2021957295-LPCLOUD",  # HLS L30
            base_date=datetime(2024, 5, 1),
            interval_days=30,
            assets=["B04", "B03", "B02"],  # e.g., RGB
            tms=tms,
            min_zoom=8,
            max_zoom=10,
        )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print("Benchmark finished.")
        print(df_rgb)
        print("\nRGB Benchmark Summary:")
        print(df_rgb.groupby("zoom")["response_time_sec"].describe())

    asyncio.run(main())
