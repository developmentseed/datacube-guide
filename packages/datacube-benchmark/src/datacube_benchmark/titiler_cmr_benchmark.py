
"""
Benchmarking utility for Titiler-cmr
"""

import asyncio
from re import I
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

    Parameters
    ----------
    z : int
        Zoom level (must be greater than or equal to 0).

    Returns
    -------
    int
        The maximum valid tile index at zoom ``z``
        (i.e., ``2**z - 1``).

    Raises
    ------
    ValueError
        If `z` is negative.
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

    Parameters
    ----------
        center_x (int): The x index of the central tile.
        center_y (int): The y index of the central tile.
        zoom (int): The zoom level.
        width (int, optional): The width of the viewport in tiles. Defaults to TILES_WIDTH.
        height (int, optional): The height of the viewport in tiles. Defaults to TILES_HEIGHT.

    Returns
    -------
        list[tuple[int, int]]: A list of (x, y) coordinates for the surrounding tiles.

    Raises
    ------
        ValueError
            If `width <= 0` or `height <= 0`.

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
        client : httpx.AsyncClient
            The HTTP client to use for requests.
        endpoint : str
            The API endpoint URL.
        format : str
            The image format to request.
        z : int
            The zoom level of the tile.
        x : int
            The x index of the tile.
        y : int
            The y index of the tile.
        concept_id : str
            The concept ID to use for the request.
        datetime_range : str
            The datetime range to use for the request.
        assets : list[str]
            The list of asset IDs to include in the request.
        resampling : str
            The resampling method to use for the request.
        colormap_name : str
            The colormap to use for the request.
        rescale : tuple[int, int], optional
            The rescale parameters to use for the request.

    Returns
    -------
    httpx.Response
        HTTP response object. If the request fails, a synthetic 500 response
        is returned with an attached `.elapsed` attribute.

    Notes
    -----
    - Exceptions are caught to keep the probe resilient.
    - Caller must inspect ``response.status_code`` to determine success.
    - Response objects are augmented with a synthetic ``elapsed`` attribute
      (seconds as float).
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


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Perform an HTTP GET and parse JSON when available.

    Parameters
    ----------
    client : httpx.AsyncClient
        Active HTTP client.
    url : str
        Absolute URL to request.
    params : dict[str, Any]
        Query string parameters.

    Returns
    -------
    dict[str, Any] or None
        Parsed JSON dictionary if the response is 200 with a JSON
        content-type; `None` otherwise.

    Notes
    -----
    - Exceptions are swallowed to keep the probe resilient; callers treat
      `None` as “no information available.” instead of raising an error.
    """
    try:
        resp = await client.get(url, params=params)
        if (
            resp.status_code == 200
            and resp.headers.get("content-type", "").startswith("application/json")
        ):
            return resp.json()
    except Exception:
        pass
    return None


async def compatibility_test(
    endpoint: str,
    ds: DatasetParams,
    *,
    lng: float,
    lat: float,
    zoom: int = 8,
    formats: Sequence[str] = ("png", "jpeg", "webp"),
    resampling_methods: Sequence[str] = ("nearest", "bilinear"),
    info_paths: Sequence[str] = ("info", "metadata", "collection", "capabilities"),
) -> CompatibilityReport:
    """
    Perform a lightweight compatibility test for a dataset (concept ID).

    The test:
      1) Attempts a single **tile** at a representative (lon, lat, zoom)
         across several formats and resampling modes.
      2) Tries to fetch JSON “info-like” endpoints to extract hints (variables,
         temporal extent, backend) if exposed by the deployment.

    Parameters
    ----------
    endpoint : str
        The API endpoint URL, e.g.:
        ``"https://staging.openveda.cloud/api/titiler-cmr"``.
    ds : DatasetParams
        Dataset parameters (concept ID, datetime, visual/processing hints).
    lng : float
        Longitude used to derive a center tile for the probe.
    lat : float
        Latitude used to derive a center tile for the probe.
    zoom : int, default 8
        Zoom level used for the representative tile request.
    formats : Sequence[str], default ("png", "jpeg", "webp")
        Image formats to try for the tile request.
    resampling_methods : Sequence[str], default ("nearest", "bilinear")
        Resampling methods to try for the tile request.
    info_paths : Sequence[str], optional
        Candidate path segments to query for JSON metadata hints. The first
        responding endpoint with a JSON payload is used.

    Returns
    -------
    CompatibilityReport
        Structured result indicating whether at least one format and one
        resampling method yielded HTTP 200, along with any detected hints.

    Notes
    -----
    - A negative result does not *prove* incompatibility, but strongly suggests
      additional server-side configuration or dataset prep is needed.
    - This function **does not** cache any results; it is intended for
      lightweight probing and should be called once per dataset.
    """
    tms = morecantile.tms.get("WebMercatorQuad")
    center = tms.tile(lng=lng, lat=lat, zoom=zoom)

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=8)
    timeout = httpx.Timeout(30.0)

    params_base: Dict[str, Any] = {
        "concept_id": ds.concept_id,
        "datetime": ds.datetime_range(),
    }
    if ds.assets:
        params_base["assets"] = list(ds.assets)
    if ds.colormap_name:
        params_base["colormap_name"] = ds.colormap_name
    if ds.rescale:
        params_base["rescale"] = f"{ds.rescale[0]},{ds.rescale[1]}"

    tested_formats: Dict[str, bool] = {}
    tested_resampling: Dict[str, bool] = {}
    detected: Dict[str, Any] = {}

    async with httpx.AsyncClient(
        limits=limits, timeout=timeout, http2=True) as client:
        # Try to glean useful hints from typical metadata/info endpoints.
        for path in info_paths:
            url = f"{endpoint}/{path}"
            meta = await _get_json(client, url, params_base)
            if meta:
                if any(k in meta for k in ("variables", "variable", "assets", "bands")):
                    for k in ("variables", "variable", "assets", "bands"):
                        if k in meta and meta[k]:
                            detected["variables"] = meta[k]
                            print(f"Detected variables: {detected['variables']}")
                            break
                for k in ("datetime_range", "time_range", "temporal_extent"):
                    if k in meta and meta[k]:
                        detected["datetime_range"] = meta[k]
                        break
                for k in ("backend", "driver", "reader"):
                    if k in meta and meta[k]:
                        detected["backend"] = meta[k]
                        break
                detected["_raw_info"] = meta
                break

        # Try formats.
        for fmt in formats:
            url = f"{common.endpoint}/tiles/{center.z}/{center.x}/{center.y}.{fmt}"
            resp = await client.get(url, params=params_base | {"resampling": ds.resampling})
            tested_formats[fmt] = (resp.status_code == 200)

        # Try resampling methods (PNG baseline for comparison).
        for method in resampling_methods:
            url = f"{common.endpoint}/tiles/{center.z}/{center.x}/{center.y}.png"
            resp = await client.get(url, params=params_base | {"resampling": method})
            tested_resampling[method] = (resp.status_code == 200)

    ok_any = any(tested_formats.values()) and any(tested_resampling.values())
    return CompatibilityReport(
        concept_id=ds.concept_id,
        ok=ok_any,
        tested_formats=tested_formats,
        tested_resampling=tested_resampling,
        detected=detected,
    )



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
