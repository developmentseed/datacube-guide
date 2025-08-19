
"""
Benchmarking utility for Titiler-cmr
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Tuple

import httpx
import morecantile
import pandas as pd

TILES_WIDTH = 5
TILES_HEIGHT = 5

# ------------------------------
# Helpers
# ------------------------------
def _max_tile_index(z: int) -> int:
    """
    Compute the maximum valid XYZ tile index for a given zoom level.

    At zoom level `z`, the map is subdivided into 2**z tiles along each axis
    (x and y). The valid tile indices therefore range from 0 to (2**z - 1).
    This helper returns the maximum valid index for both axes.

    Args:
        z (int): Zoom level (must be >= 0).

    Returns:
        int: The maximum valid tile index at zoom `z`
             (i.e., 2**z - 1).

    Raises:
        ValueError: If `z` is negative.
    """
    if z < 0:
        raise ValueError("zoom must be >= 0")
    return (1 << z) - 1


def get_surrounding_tiles(
    center_x: int,
    center_y: int,
    zoom: int,
    width: int = TILES_WIDTH,
    height: int = TILES_HEIGHT,
) -> list[tuple[int, int]]:
    """
    Get a list of surrounding tile coordinates for a viewport around (center_x, center_y).
    This function builds a `width × height` viewport centered on the given tile at the specified zoom level.
    from https://github.com/developmentseed/titiler-cmr/blob/develop/tests/test_hls_benchmark.py

    Args:
        center_x (int): The x index of the central tile.
        center_y (int): The y index of the central tile.
        zoom (int): The zoom level.
        width (int, optional): The width of the viewport in tiles. Defaults to TILES_WIDTH.
        height (int, optional): The height of the viewport in tiles. Defaults to TILES_HEIGHT.

    Returns:
        list[tuple[int, int]]: A list of (x, y) coordinates for the surrounding tiles.

    Raises:
        ValueError: If `width <= 0` or `height <= 0`.
    
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")

    tiles: list[tuple[int, int]] = []
    offset_x = width // 2
    offset_y = height // 2
    max_tile = _max_tile_index(zoom)

    for y_pos in range(center_y - offset_y, center_y + offset_y + 1):
        for x_pos in range(center_x - offset_x, center_x + offset_x + 1):
            x_valid = max(0, min(x_pos, max_tile))
            y_valid = max(0, min(y_pos, max_tile))
            tiles.append((x_valid, y_valid))
    return tiles


async def fetch_tile(
    client: httpx.AsyncClient,
    endpoint: str,
    format: str = "png",
    *,
    z: int,
    x: int,
    y: int,
    concept_id: str,
    datetime_range: str,
    assets: list[str],
    resampling: str,
    colormap_name: str,
    rescale: tuple[int, int] | None,
) -> httpx.Response:
    """
    Fetch a single tile and return the httpx.response object.

    Parameters
    ----------
        client (httpx.AsyncClient): The HTTP client to use for requests.
        endpoint (str): The API endpoint URL.
        format (str): The image format to request.
        z (int): The zoom level of the tile.
        x (int): The x-coordinate of the tile.
        y (int): The y-coordinate of the tile.
        concept_id (str): The concept ID to use for the request.
        datetime_range (str): The datetime range to use for the request.
        assets (List[str]): The list of asset IDs to include in the request.
        resampling (Resampling): The resampling method to use for the request.
        colormap_name (Colormap): The colormap to use for the request.
        rescale (None | tuple[int, int]): The rescale parameters to use for the request.

    Returns
    -------
        httpx.Response: The HTTP response object.
    """

    url = f"{endpoint}/tiles/{z}/{x}/{y}.{format}"
    params: dict[str, Any] = {
        "concept_id": concept_id,
        "datetime": datetime_range,
        "resampling": resampling,
        "colormap_name": colormap_name,
    }
    if assets:
        params["assets"] = assets
    if rescale:
        params["rescale"] = f"{rescale[0]},{rescale[1]}"

    start_time = time.perf_counter()
    try:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        response.elapsed = time.perf_counter() - start_time
        return response
    except Exception as exc:
        print(f"Error fetching tile {z}/{x}/{y}: {exc}")
        mock_response = httpx.Response(500, request=httpx.Request("GET", url))
        mock_response.elapsed = time.perf_counter() - start_time
        return mock_response

# ---------------------------------------------------------------------
# Benchmark Function
# ---------------------------------------------------------------------
async def benchmark_titiler_cmr(
    concept_id: str,
    endpoint: str = "https://staging.openveda.cloud/api/titiler-cmr",
    *,
    tms: morecantile.TileMatrixSet = morecantile.tms.get("WebMercatorQuad"),
    min_zoom: int = 7,
    max_zoom: int = 10,
    tile_scale: int = 3,
    resampling: str = "nearest",
    colormap_name: str = "gnbu",
    assets: list[str] = None,
    start_date: datetime = datetime(2023, 2, 24, 0, 0, 1),
    end_date: datetime = datetime(2023, 2, 25, 0, 0, 1),
    lng: float = -92.1,
    lat: float = 46.8,
    rescale: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Benchmarks the Titiler-cmr API for a specific viewport across multiple zoom levels.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark results.
    """

    datetime_range = f"{start_date.isoformat()}/{end_date.isoformat()}"
    results = []
    if assets is None:
        assets = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for zoom in range(min_zoom, max_zoom + 1):
            print(f"Benchmarking Zoom level: {zoom}...")
            center_tile = tms.tile(lng=lng, lat=lat, zoom=zoom)
            tiles_to_fetch = get_surrounding_tiles(center_tile.x, center_tile.y, zoom)
            tasks = [
                fetch_tile(
                    client=client,
                    endpoint=endpoint,
                    z=zoom,
                    x=x,
                    y=y,
                    concept_id=concept_id,
                    datetime_range=datetime_range,
                    assets=assets,
                    resampling=resampling,
                    colormap_name=colormap_name,
                    rescale=rescale,
                )
                for x, y in tiles_to_fetch
            ]
            responses = await asyncio.gather(*tasks)
            for (x, y), response in zip(tiles_to_fetch, responses):
                results.append({
                    "zoom": zoom,
                    "x": x,
                    "y": y,
                    "status_code": response.status_code,
                    "response_time_sec": response.elapsed,
                    "response_size_bytes": len(response.content),
                    "is_error": response.is_error,
                    "has_data": response.status_code == 200,
                })
    return pd.DataFrame(results)


if __name__ == "__main__":

    async def main():
        print("Starting benchmark with assets...")
        projection = "WebMercatorQuad"
        tms = morecantile.tms.get(projection)
        concept_id = "C2021957295-LPCLOUD"  # HLS L30

        df_rgb = await benchmark_titiler_cmr(
            concept_id=concept_id,
            start_date=datetime(2024, 5, 1),
            end_date=datetime(2024, 5, 2),
            assets=["B04", "B03", "B02"],  # e.g., bands    
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

    asyncio.run(main())
