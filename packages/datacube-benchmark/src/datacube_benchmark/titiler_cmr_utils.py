"""
Benchmarking utility for Titiler-CMR.

This module provides tools to evaluate the performance and compatibility of the 
Titiler-CMR API for geospatial tile rendering.
"""

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, List, Tuple

import httpx
import morecantile
import pandas as pd

TILES_WIDTH = 5
TILES_HEIGHT = 5
SUPPORTED_TILE_FORMATS: set[str] = {
    "png",
    "npy",
    "tiff",
    "jpeg",
    "jpg",
    "jp2",
    "webp",
    "pngraw",
}

# ------------------------------
# Dataclasses
# ------------------------------

@dataclass(frozen=True)
class DatasetParams:
    """
    Dataset-related parameters for requests to TiTiler-CMR.

    Parameters
    ----------
    concept_id : str
        CMR concept ID (e.g., ``"C2021957295-LPCLOUD"``).
    datetime_start : datetime
        Beginning of the ISO 8601 time interval (inclusive).
    datetime_end : datetime
        End of the ISO 8601 time interval.
    assets : Sequence[str] or None
        Optional list of assets/bands/variables to request.
    resampling : str
        Resampling method (e.g., ``"nearest"``, ``"bilinear"``).
    colormap_name : str or None
        Optional colormap for single-band visualization.
    rescale : tuple[float, float] or None
        Optional min/max scaling values for visualization (e.g., ``(0, 3000)``).
    """

    concept_id: str
    datetime_start: datetime
    datetime_end: datetime
    assets: Optional[Sequence[str]] = None
    resampling: str = "nearest"
    colormap_name: Optional[str] = None
    rescale: Optional[Tuple[float, float]] = None

    def datetime_range(self) -> str:
        """Return ISO 8601 datetime interval string ``"<start>/<end>"``."""
        return f"{self.datetime_start.isoformat()}/{self.datetime_end.isoformat()}"


@dataclass(frozen=True)
class TileBenchConfig:
    """
    Configuration for tile benchmarking.
    """

    min_zoom: int = 7
    max_zoom: int = 10
    tiles_width: int = 5
    tiles_height: int = 5
    image_format: str = "png"
    lng: float = -92.1
    lat: float = 46.8

    def validate(self) -> None:
        """Validate configuration values and raise `ValueError` if invalid."""
        if self.max_zoom < self.min_zoom:
            raise ValueError("max_zoom must be >= min_zoom.")
        if self.tiles_width <= 0 or self.tiles_height <= 0:
            raise ValueError("Tile window dimensions must be positive.")
        if self.image_format not in SUPPORTED_TILE_FORMATS:
            raise ValueError(
                f"Format must be one of: {', '.join(SUPPORTED_TILE_FORMATS)}."
            )


@dataclass(frozen=True)
class CompatibilityReport:
    """
    Outcome of a quick, heuristic compatibility probe.
    """

    concept_id: str
    ok: bool
    tested_formats: Dict[str, bool]
    tested_resampling: Dict[str, bool]
    detected: Dict[str, Any]


# ------------------------------
# Helpers
# ------------------------------
def _build_tile_url(
    base_url: str,
    tile_matrix_set_id: str,
    z: int,
    x: int,
    y: int,
    *,
    fmt: Optional[str] = None,
    scale: Optional[int] = None,
) -> str:
    """
    Construct a TiTiler-CMR tile URL using documented patterns:
      /tiles/{tileMatrixSetId}/{z}/{x}/{y}
      /tiles/{tileMatrixSetId}/{z}/{x}/{y}@{scale}x
      /tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}
      /tiles/{tileMatrixSetId}/{z}/{x}/{y}@{scale}x.{format}

    Parameters
    ----------
    base_url : str
        The base URL for the TiTiler-CMR API.
    tile_matrix_set_id : str
        The ID of the tile matrix set.
    z : int
        The zoom level.
    x : int
        The x tile index.
    y : int
        The y tile index.
    fmt : str, optional
        The image format (e.g., "png", "jpeg").
    scale : int, optional
        The scale factor (e.g., 2 for 2x).

    Returns
    -------
    str
        The constructed tile URL.

    Raises
    ------
    ValueError
        If `scale` is provided and less than 1.
    """
    base = base_url.rstrip("/")
    path = f"/tiles/{z}/{x}/{y}"
    if scale is not None:
        if scale < 1:
            raise ValueError("scale must be >= 1 when provided")
        path += f"@{scale}x"
    if fmt:
        path += f".{fmt}"
    return base + path


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
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith(
            "application/json"
        ):
            return resp.json()
    except Exception:
        pass
    return None


async def fetch_tile(
    client: httpx.AsyncClient,
    endpoint: str,
    format: str = "png",
    *,
    tile_matrix_set_id: str = "WebMercatorQuad",
    scale: Optional[int] = None,
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
       tile_matrix_set_id : str
           The tile matrix set ID to use for the request.
        scale: Optional[int]
            The scale factor to use for the request.
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

    url = _build_tile_url(
        endpoint,
        tile_matrix_set_id,
        z,
        x,
        y,
        fmt=format,
        scale=scale,
    )
    print ("url:", url)
    print ("url =", f"{endpoint}/tiles/{z}/{x}/{y}.{format}")
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
# Compatibility Test
# ---------------------------------------------------------------------

async def compatibility_test(
    endpoint: str,
    ds: DatasetParams,
    *,
    lng: float,
    lat: float,
    zoom: int = 8,
    tile_matrix_set_id: str = "WebMercatorQuad",
    scale: Optional[int] = None,
    formats: Sequence[str] = SUPPORTED_TILE_FORMATS,
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
    fmt_list = [f.lower() for f in formats]
    unknown = [f for f in fmt_list if f not in SUPPORTED_TILE_FORMATS]
    if unknown:
        raise ValueError(
            f"Unsupported format(s): {unknown}. "
            f"Supported: {', '.join(SUPPORTED_TILE_FORMATS)}"
        )

    tms = morecantile.tms.get(tile_matrix_set_id)
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

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Try to glean useful hints from typical metadata/info endpoints.
        for path in info_paths:
            print(f"Trying info path: {path}")
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
        for fmt in fmt_list:
            url = f"{endpoint}/tiles/{center.z}/{center.x}/{center.y}.{fmt}"
            resp = await client.get(
                url, params=params_base | {"resampling": ds.resampling}
            )
            tested_formats[fmt] = resp.status_code == 200

        # Try resampling methods (PNG baseline for comparison).
        for method in resampling_methods:
            url = f"{endpoint}/tiles/{center.z}/{center.x}/{center.y}.png"
            resp = await client.get(url, params=params_base | {"resampling": method})
            tested_resampling[method] = resp.status_code == 200

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
    endpoint: str,
    ds: DatasetParams,
    *,
    tms: morecantile.TileMatrixSet = morecantile.tms.get("WebMercatorQuad"),
    min_zoom: int = 7,
    max_zoom: int = 10,
    tile_scale: int = 3,
    lng: float = -92.1,
    lat: float = 46.8,
    timeout_s: float = 30.0,
    format: str = "png",
) -> pd.DataFrame:
    """
    Benchmarks the Titiler-cmr API for a specific viewport across multiple zoom levels.

    Parameters
    ----------
    endpoint : str
        The API endpoint URL.
    ds : DatasetParams
        The dataset parameters.
    tms : morecantile.TileMatrixSet
        The tile matrix set.
    min_zoom : int
        The minimum zoom level.
    max_zoom : int
        The maximum zoom level.
    tile_scale : int
        The tile scale factor.
    lng : float
        The longitude of the center point.
    lat : float
        The latitude of the center point.
    timeout_s : float
        The request timeout in seconds.
    format : str
        The image format (same as the SUPPORTED_TILE_FORMATS)

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark results.
    """

    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for zoom in range(min_zoom, max_zoom + 1):
            print(f"Benchmarking Zoom level: {zoom}...")
            center_tile = tms.tile(lng=lng, lat=lat, zoom=zoom)
            tiles_to_fetch = get_surrounding_tiles(center_tile.x, center_tile.y, zoom)

            tasks = [
                fetch_tile(
                    client=client,
                    endpoint=endpoint,
                    format=format,
                    z=zoom,
                    x=x,
                    y=y,
                    concept_id=ds.concept_id,
                    datetime_range=ds.datetime_range(),
                    assets=list(ds.assets) if ds.assets else [],
                    resampling=ds.resampling,
                    colormap_name=ds.colormap_name,
                    rescale=ds.rescale,
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
                        "response_time_sec": response.elapsed,
                        "response_size_bytes": len(response.content),
                        "is_error": response.is_error,
                        "has_data": response.status_code == 200,
                    }
                )
    return pd.DataFrame(results)


if __name__ == "__main__":

    async def main():
        endpoint = "https://staging.openveda.cloud/api/titiler-cmr"
        projection = "WebMercatorQuad"
        tms = morecantile.tms.get(projection)

        concept_id = "C2021957295-LPCLOUD"  # HLS L30

        ds = DatasetParams(
            concept_id=concept_id,
            datetime_start=datetime(2024, 5, 1),
            datetime_end=datetime(2024, 5, 2),
            assets=["B04", "B03", "B02"],
        )

        compat = await compatibility_test(
            endpoint=endpoint,
            ds=ds,
            lng=-92.1,
            lat=46.8,
            zoom=8,
        )

        print("\n== Compatibility Report ==")
        print(f"concept_id : {compat.concept_id}")
        print(f"compatible : {compat.ok}")
        print(f"formats    : {compat.tested_formats}")
        print(f"resampling : {compat.tested_resampling}")

        if compat.detected:
            # Avoid dumping huge blobs (like full metadata); show highlights.
            detected_view = {
                k: v for k, v in compat.detected.items() if k != "_raw_info"
            }
            print(f"detected   : {detected_view}")

        df_tiles = await benchmark_titiler_cmr(
            endpoint=endpoint,
            ds=ds,
            tms=tms,
            min_zoom=8,
            max_zoom=10,
            format="png",
        )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        print("\n== Tile Benchmark Results ==")
        print(df_tiles)

    asyncio.run(main())
