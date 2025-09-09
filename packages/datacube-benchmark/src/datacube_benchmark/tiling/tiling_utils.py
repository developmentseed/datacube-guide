"""
tiling_utils
============

from datacube_benchmark.tiling.titiler_cmr_params import DatasetParams

Reusable helper functions for working with map tiles, that are independent of
any specific benchmarking or rendering workflow. They are primarily tested
to support TiTiler-CMR benchmarking, but can also be applied in other
contexts where XYZ tile math and asynchronous tile fetching are needed.

Functions
---------
- get_surrounding_tiles:
    Compute the list of (x, y) tile coordinates forming a rectangular
    viewport centered on a given tile index at a specific zoom level.

- get_tileset_tiles:
    Get all tiles for a complete zoom level within geographic bounds.

- fetch_tile:
    Asynchronously fetch one or more tiles from a set of TileJSON
    templates for a given (z, x, y). Returns structured metadata for
    each request, including status code, latency, response size,
    and (optionally) memory usage deltas.

- create_bbox_feature:
    Create a GeoJSON Feature representing a bounding box from
    minx, miny, maxx, maxy coordinates.

- BaseBenchmarker:
    A base class providing shared functionality for TiTiler benchmarking
    classes, including system info, HTTP client setup, and minimal result processing.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import morecantile
import psutil
import pandas as pd
from geojson_pydantic import Feature, Polygon


def get_surrounding_tiles(
    center_x: int,
    center_y: int,
    zoom: int,
    width: int,
    height: int,
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
        width (int): The width of the viewport in tiles.
        height (int): The height of the viewport in tiles.

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

def get_tileset_tiles(
    bounds: List[float], 
    zoom: int, 
    tms: morecantile.TileMatrixSet
) -> List[Tuple[int, int]]:
    """
    Get all tiles for a complete zoom level within bounds.
    
    Parameters
    ----------
    bounds : List[float]
        Bounding box [minx, miny, maxx, maxy] in CRS coordinates
    zoom : int
        Zoom level
    tms : morecantile.TileMatrixSet
        Tile matrix set
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (x, y) tile coordinates
    """
    minx, miny, maxx, maxy = bounds
    
    # Get tile bounds for the bbox
    ul_tile = tms.tile(minx, maxy, zoom)
    lr_tile = tms.tile(maxx, miny, zoom)
    
    tiles = [
        (x, y)
        for x in range(min(ul_tile.x, lr_tile.x), max(ul_tile.x, lr_tile.x) + 1)
        for y in range(min(ul_tile.y, lr_tile.y), max(ul_tile.y, lr_tile.y) + 1)
    ]
    
    return tiles


async def fetch_tile(
    client: httpx.AsyncClient,
    *,
    tiles_endpoints: List[str],
    z: int,
    x: int,
    y: int,
    timeout_s: float = 30.0,
    proc: Optional[psutil.Process] = None,
) -> List[Dict[str, Any]]:
    """
    For a single (z,x,y), iterate over all tiles endpoints, GET the tile, print status,
    and return one record per request.

    Timing semantics:
      - Excludes external queueing/semaphore wait by design (measure that outside).
      - ttfb_sec:      time to first byte (headers) after starting the request
      - transfer_time_sec: body transfer time
      - response_time_sec: total on-wire time = ttfb_sec + transfer_time_sec
      - sched_delay_sec: (optional) time from `started_at` (post-semaphore) to when we begin I/O

    Parameters
    ----------
    client : httpx.AsyncClient
        The HTTP client to use for requests.
    tiles_endpoints : list of str
        URL templates containing {z}, {x}, and {y}.
    z, x, y : int
        Tile coordinates.
    timeout_s : float
        Per-request timeout (seconds).
    proc : psutil.Process, optional
        Process to sample RSS.
    started_at : float, optional
        Timestamp captured *after* you acquire your concurrency gate (e.g., semaphore).

    Returns
    -------
    List[Dict[str, Any]]
        One dictionary per endpoint with fields:
          zoom/z/x/y, timestep_index, url, status_code, ok, no_data, is_error,
          ttfb_sec, transfer_time_sec, response_time_sec,
          response_size_bytes, content_type, error_text, rss_delta,
          sched_delay_sec (if started_at provided)
    """
    rows: List[Dict[str, Any]] = []

    for i, tmpl in enumerate(tiles_endpoints):
        tile_url = tmpl.format(z=z, x=x, y=y)
        t0 = time.perf_counter()
        try:
            resp = await client.get(tile_url, timeout=timeout_s)
            total = time.perf_counter() - t0

            status_code = resp.status_code
            ctype = resp.headers.get("content-type")
            size = len(resp.content)
            is_ok = (status_code == 200)
            is_no_data = (status_code == 204)
            is_error = not (200 <= status_code < 300)

            resp.raise_for_status()

            rows.append(
            {
                "zoom": z,
                "x": x,
                "y": y,
                "status_code": status_code,
                "ok": is_ok,
                "no_data": is_no_data,
                "is_error": is_error,
                "response_time_sec": total,
                "content_type": ctype,
                "response_size_bytes": size,
                "url": tile_url,
                "error_text": None,
                }
            )

        except httpx.HTTPStatusError as ex:
            response = ex.response
            status_code = response.status_code
            error_text = response.text            
            print("~~~~~~~~~~~~~~~~ ERROR FETCHING TILE ~~~~~~~~~~~~~~~~")
            print(f"URL:    {response.request.url}")
            print(f"Error:  {response.status_code} {response.status_reason}")   # <-- status + reason phrase
            print(f":   {response.text}")
            row = (
                {
                "zoom": z,
                "x": x,
                "y": y,
                "status_code": None,
                "ok": False,
                "no_data": False,
                "is_error": True,
                "response_time_sec": float("nan"),
                "response_size_bytes": 0,
                "content_type": None,
                "url": tile_url,
                "error_text": error_text,
                }
            )

    return rows

def create_bbox_feature(minx: float, miny: float, maxx: float, maxy: float) -> Feature:
    """
    Create a GeoJSON Feature from bounding box coordinates.

    Parameters
    ----------
    minx, miny, maxx, maxy : float
        Bounding box coordinates.

    Returns
    -------
    Feature
        GeoJSON Feature representing the bounding box.
    """
    return Feature(
        type="Feature",
        geometry=Polygon.from_bounds(minx, miny, maxx, maxy),
        properties={}
    )

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

# Base benchmarker with shared functionality
class BaseBenchmarker:
    """
    Base class for TiTiler benchmarking infrastructure.

    Provides system info, HTTP client setup, and result processing utilities
    for derived benchmarker classes.
    """
    
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 30.0,
        max_connections: int = 20,
        max_connections_per_host: int = 20
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self._system_info = self._get_system_info()
    
    def _create_http_client(self) -> httpx.AsyncClient:
        """Create configured HTTP client."""
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_connections_per_host,
        )
        return httpx.AsyncClient(limits=limits, timeout=self.timeout_s)
    
    def _get_system_info(self) -> str:
        """Get system information string."""
        return (
            f"{psutil.cpu_count(logical=False)} physical / "
            f"{psutil.cpu_count(logical=True)} logical cores | "
            f"RAM: {self._fmt_bytes(psutil.virtual_memory().total)}"
        )
    
    @staticmethod
    def _fmt_bytes(n: int | float) -> str:
        """
        Convert bytes into a human-readable string (KiB, MiB, GiB...).
        """
        n = float(n)
        
        sign = "-" if n < 0 else ""
        n = abs(n) 
        units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
        i = 0
        while n >= 1024 and i < len(units) - 1:
            n /= 1024.0
            i += 1
        
        return f"{sign}{n:.2f} {units[i]}"
    
    def _process_results(self, results: List[Any]) -> pd.DataFrame:
        """
        Designed for post-processing the output of ``asyncio.gather`` used in
        tile benchmarking. Handles dicts, lists of dicts, and exceptions,
        flattening them into rows and optionally sorting by tiling dimensions.

        Parameters
        ----------
        results : List[Any]
            List of results from ``asyncio.gather``, which may include
            dictionaries, lists of dictionaries, or exceptions.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all successful results, sorted by
            available tiling dimensions (z, y, x, timestep_index, ....)
        """
        all_rows = []
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Task error: {result}")
                continue
            if isinstance(result, list):
                all_rows.extend(result)
            elif isinstance(result, dict):
                all_rows.append(result)
        
        df = pd.DataFrame(all_rows)
        
        if df.empty:
            print("Warning: No successful results")
            return df
        
        sort_cols = [col for col in ["z", "y", "x", "timestep_index"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        return df
    
    def _log_header(self, benchmark_name: str, dataset: DatasetParams) -> None:
        """
        Log standardized benchmark header with system and dataset information.
        
        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark being executed.
        dataset : DatasetParams
            Dataset configuration being benchmarked.
        """
        print(f"=== TiTiler-CMR {benchmark_name} ===")
        print(f"Client: {self._system_info}")
        print(f"Dataset: {dataset.concept_id} ({dataset.backend})")
 