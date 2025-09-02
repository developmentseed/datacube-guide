
"""
Benchmarking utility for TiTiler-CMR.

This module provides tools to evaluate the performance of the
TiTiler-CMR API for geospatial tile rendering across multiple zoom levels.

"""

from __future__ import annotations

import asyncio
import time
import psutil

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tiling import get_surrounding_tiles, fetch_tile, DatasetParams

import httpx
import morecantile
import pandas as pd


TILES_WIDTH = 5
TILES_HEIGHT = 5



async def get_tilejson_info(
    client: httpx.AsyncClient,
    endpoint: str,
    tms_id: str,
    params: list[tuple[str, str]],
    *,
    timeout_s: float = 30.0,
    ) -> dict:
    """
    Query TiTiler-CMR timeseries TileJSON and return parsed entries
    as a list of dictionaries + raw JSON.


    Parameters
    ----------
    client : httpx.AsyncClient
        An active HTTPX async client (connection pool reused).
    endpoint : str
        Base URL of the TiTiler-CMR API.
    tms_id : str
        Tile matrix set identifier (e.g., "WebMercatorQuad").
    params : list of tuple
        Query parameters for the request.
    timeout_s : float, default=30.0
        Timeout for the request.

    Returns
    -------
    dict
        {
          "entries": [list of per-timestep dicts],
          "tilejson": raw JSON response,
          "tiles_endpoints": flattened list of all tile endpoints,
          "bounds": geographic bounds from first entry (if available)
        }
    """
    ts_url = f"{endpoint.rstrip('/')}/timeseries/{tms_id}/tilejson.json"
    resp = await client.get(ts_url, params=params, timeout=timeout_s)
    resp.raise_for_status()

    ts_json = resp.json()
    entries: list[dict] = []
    
    for ts, v in ts_json.items():
        if isinstance(v, dict):
            entries.append({
                "timestamp": ts,
                "tilejson": v.get("tilejson"),
                "version": v.get("version"),
                "scheme": v.get("scheme"),
                "tiles": v.get("tiles", []),
                "minzoom": v.get("minzoom"),
                "maxzoom": v.get("maxzoom"),
                "bounds": v.get("bounds"),
                "center": v.get("center"),
            })

    tiles_endpoints = [tile for entry in entries for tile in entry.get("tiles", [])]
    if not tiles_endpoints:
        raise RuntimeError("No 'tiles' templates found in timeseries TileJSON response.")
    
    # Extract bounds from first entry if available
    bounds = None
    if entries and entries[0].get("bounds"):
        bounds = entries[0]["bounds"]

    return {
        "entries": entries, 
        "tilejson": ts_json, 
        "tiles_endpoints": tiles_endpoints,
        "bounds": bounds
    }


# ------------------------------
# Timeseries-only benchmark (async wrapper)
# Benchmark a viewport
# ------------------------------
async def benchmark_titiler_cmr(
    endpoint: str,
    ds: DatasetParams,
    *,
    tms_id: str = "WebMercatorQuad",
    tile_format: str = "png",
    tile_scale: int = 1,
    min_zoom: int = 7,
    max_zoom: int = 10,

    # -- viewport params
    lng: float,
    lat: float,
    viewport_width: int = TILES_WIDTH,
    viewport_height: int = TILES_HEIGHT,
    # -- 
    timeout_s: float = 30.0,
    max_connections: int = 20,
    max_connections_per_host: int = 20,
    **kwargs: Any, 
    # TODO: review for kwargs......
    ) -> pd.DataFrame:
    """
    Benchmark tile rendering performance for TiTiler-CMR for a viewport.

    Steps:
        1. GET TileJson from /timeseries/{tms_id}/tilejson.json to obtain all valid tile endpoints.
        2. For each zoom & viewport tile, run fetch_tile(...) over all endpoints.
        3. Return a tidy DataFrame with one row per request.

    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR API (e.g., "https://.../api/titiler-cmr").
    ds : DatasetParams
        Dataset and query parameters (concept_id, backend, datetime_range, kwargs).
    tms_id : str, default="WebMercatorQuad"
        Tile matrix set identifier.
    tile_format : str, default="png"
        Tile format to request. Must be one of SUPPORTED_TILE_FORMATS.
    tile_scale : int, default=1
        Tile scale factor (resolution multiplier).
    min_zoom, max_zoom : int, default=7 and 10
        Minimum and maximum zoom levels to benchmark.
    lng, lat : float, default=-92.1, 46.8
        Longitude/latitude of the viewport center.
    timeout_s : float, default=30.0
        Timeout (seconds) for each HTTP request.
    viewport_width, viewport_height : int, default=TILES_WIDTH, TILES_HEIGHT
        Size of the tile viewport around the center tile.
    max_connections : int, default=20
        Maximum concurrent HTTP connections.
    max_connections_per_host : int, default=20
        Maximum concurrent connections per host.
    **kwargs : dict
        Additional parameters passed through to TiTiler-CMR query string.

    Returns
    -------
    pd.DataFrame
        Results for each tile request.
    """
    _print_system_info()


    # Add/override tile_format, minzoom, maxzoom in ds.kwargs
    params: List[Tuple[str, str]] = ds.to_query_params(
        tile_format=tile_format,
        tile_scale=tile_scale,
        **kwargs,
    )
    print("---------- Query Params ----------")
    print(*params, sep="\n")

    tms = morecantile.tms.get(tms_id)

    async with _create_http_client(timeout_s, max_connections, max_connections_per_host) as client:

        # 1) Query Timeseries EndPoint for TileJSON info (entries + endpoint)
        tilejson_info = await get_tilejson_info(client, endpoint, tms_id, params)
        # find all the endpoints for all granules....
        tiles_endpoints = [tile for entry in tilejson_info.get("entries", []) for tile in entry.get("tiles", [])]
        print ('~~~~~~~~~~~~~~~~~~~~~~~~')
        print('\n'.join(tiles_endpoints))
        if not tiles_endpoints:
            raise RuntimeError("No 'tiles' endpoints found in timeseries TileJSON response.")
        n_timesteps = len(tiles_endpoints)

        print(f"TileJSON: timesteps={n_timesteps} | Endpoints={len(tiles_endpoints)}")
        #print(f"Total tile requests: {len(tile_coords)} tiles × {n_timesteps} timesteps = {len(tile_coords) * n_timesteps}")

        # 2) build tasks and fetch concurrently
        proc = psutil.Process()

        tasks: List[asyncio.Task] = []
        for z in range(min_zoom, max_zoom + 1):
            center = tms.tile(lng=lng, lat=lat, zoom=z)
            tiles_xy = get_surrounding_tiles(center.x, center.y, z, viewport_width, viewport_height)
            print(f"\nZoom {z}: {len(tiles_xy)} tiles around ({center.x},{center.y})")

            for x, y in tiles_xy:
                tasks.append(
                    asyncio.create_task(
                        fetch_tile(
                            client,
                            tiles_endpoints=tiles_endpoints,
                            z=z,
                            x=x,
                            y=y,
                            timeout_s=timeout_s,
                            proc=proc,
                        )
                    )
                )

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    # -- process benchmark results into a dataframe
    df = _process_benchmark_results(results_lists)
    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _create_http_client(
    timeout_s: float = 30.0,
    max_connections: int = 20,
    max_connections_per_host: int = 20
) -> httpx.AsyncClient:
    """
    Create configured HTTP client.
    
    Parameters
    ----------
    timeout_s : float
        Request timeout
    max_connections : int
        Maximum total connections
    max_connections_per_host : int
        Maximum connections per host
        
    Returns
    -------
    httpx.AsyncClient
        Configured client
    """
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections_per_host,
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout_s)


def _process_benchmark_results(results_lists: List[Any]) -> pd.DataFrame:
    """
    Process raw benchmark results into a sorted DataFrame.
    
    Parameters
    ----------
    results_lists : List[Any]
        Raw benchmark results from of asyncio.gather(..., return_exceptions=True), where each item is
        either:
          - a List[Dict[str, Any]] of row dicts, or
          - an Exception.
        
    Returns
    -------
    pd.DataFrame
        Processed and sorted DataFrame with benchmark results.
    """
    # -- TODO: save dataframes

    all_rows: List[Dict[str, Any]] = []
    for rows in results_lists:
        if isinstance(rows, Exception):
            print(f"Task error: {rows}")
            continue
        if isinstance(rows, list):
            all_rows.extend(rows)
        else:
            print(f"Unexpected result type: {type(rows)}")


    required_cols = ["z", "y", "x", "timestep_index"]
    df = pd.DataFrame(all_rows)

    if df.empty:
        print("Warning: DataFrame is empty.")
        return df

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"Warning: DataFrame is missing required columns: {missing}")
        # Still try to return something usable
        return df.reset_index(drop=True)

    df = df.sort_values(required_cols).reset_index(drop=True)
    return df







def _fmt_bytes(n: int | float) -> str:
    """Format bytes in a human-friendly way (binary units)."""
    try:
        n = float(n)
    except Exception:
        return "n/a"
    if n < 0:
        sign = "-"
        n = -n
    else:
        sign = ""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{sign}{n:.2f} {units[i]}"

def _print_system_info():
    """Print system information for benchmarking context."""
    print(
        f"Client-side: {psutil.cpu_count(logical=False)} physical / "
        f"{psutil.cpu_count(logical=True)} logical cores | "
        f"RAM: {_fmt_bytes(psutil.virtual_memory().total)}"
    )




async def check_titiler_cmr_compatibility(
    endpoint: str,
    ds: DatasetParams,
    *,
    timeout_s: float = 60.0,
    include_datetimes: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    [] TODO: Min and Max zoom level....
        Simplified compatibility checker for TiTiler-CMR.
    Works with responses like:
        [
            {"concept_id": "C2723754864-GES_DISC", "datetime": "2024-10-01T11:59:59.999500+00:00"},

            ...
        ]

    Returns a clean summary with:
      - Number of granules
      - Time span (start/end)
      - Sorted datetimes list

    Parameters
    ----------
    ds: DatasetParams,
    endpoint: str,
    timeout_s: float = 60.0,
    include_datetimes: bool = False,
    ) -> Dict[str, Any]:
    include_datetimes : bool, optional
        If True, also returns the full sorted list of granule datetimes.
    **kwargs : Any
        Optional extra query parameters to pass to the API.

    Returns
    -------
    Dict[str, Any]
        If `include_datetimes=False` (default):
            {
                "concept_id": <str>,
                "n_granules": <int>,
                "start_time": <str | None>,
                "end_time": <str | None>
            }

        If `include_datetimes=True`:
            {
                "concept_id": <str>,
                "n_granules": <int>,
                "start_time": <str | None>,
                "end_time": <str | None>,
                "granule_datetimes": [<str>, ...]
            }
    """
    url = f"{endpoint.rstrip('/')}/timeseries"
    # Use DatasetParams to build params
    params = dict(ds.to_query_params())

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url, params=params)
        try:
            payload = resp.json()
        except Exception:
            summary = {
                "concept_id": ds.concept_id,
                "n_granules": 0,
                "start_time": None,
                "end_time": None,
            }
            if include_datetimes:
                summary["granule_datetimes"] = []
            return summary

    # Extract available granule datetimes
    if isinstance(payload, list):
        times = [
            item["datetime"]
            for item in payload
            if isinstance(item, dict) and "datetime" in item
        ]
    else:
        times = []

    # Normalize & sort datetimes
    if times:
        ts = pd.to_datetime(times, utc=True, errors="coerce").dropna().sort_values()
        times = [t.isoformat().replace("+00:00", "Z") for t in ts.to_pydatetime()]

    # Build summary
    summary = {
        "concept_id": ds.concept_id,
        "n_granules": len(times),
        "start_time": times[0] if times else None,
        "end_time": times[-1] if times else None,
    }

    if include_datetimes:
        summary["granule_datetimes"] = times

    return summary

# ------------------------------
# Example
# ------------------------------
if __name__ == "__main__":
    async def _run():
        ## ---------------------------------
        # Example 1 : Xarray Backend
        ## ---------------------------------

        endpoint = "https://staging.openveda.cloud/api/titiler-cmr"
        
        min_zoom = 8
        max_zoom = 9
        lng = -92.1
        lat = 46.8
        viewport_width = 5
        viewport_height = 5

        step = "P1W"
        temporal_mode = "interval"

        tile_format = "png"

        backend = "xarray"
        variable = "precipitation"

        concept_id = "C2723754864-GES_DISC"
        #datetime_range = "2024-10-01T00:00:00Z/2024-10-05T23:59:59Z"
        datetime_range = "2024-10-12T00:00:00Z/2024-11-13T00:00:00Z"

        ds_xr = DatasetParams(
            concept_id=concept_id,
            backend=backend,
            datetime_range=datetime_range,
            kwargs={
                "variable": variable,
                "rescale": "0,1",
                "colormap_name": "gnbu",
                "step": step,
                "temporal_mode": temporal_mode,
            }
        )
        result = await check_titiler_cmr_compatibility(
            ds=ds_xr,
            endpoint=endpoint,
        )

        # Access structured results programmatically
        print("Summary object:", result)

        # Example: decide if you should proceed
        if result["n_granules"] == 0:
            print("No granules available in this date range. Skipping benchmark.")
        else:
            print(f"Proceeding: {result['n_granules']} granules found.")

        df = await benchmark_titiler_cmr(
            ds=ds_xr,
            endpoint=endpoint,
            tms_id="WebMercatorQuad",
            tile_format=tile_format,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            lng=lng,
            lat=lat,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        
        print(df.head())

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        zoom_summary = (
            df.assign(
                elapsed_s=pd.to_numeric(df["elapsed_s"], errors="coerce"),
                size_bytes=pd.to_numeric(df["size_bytes"], errors="coerce"),
                rss_delta=pd.to_numeric(df["rss_delta"], errors="coerce"),
            )
            .groupby(["z", "timestep_index"])
            .apply(
                lambda g: pd.Series({
                    "n_tiles": len(g),
                    "ok_pct": 100.0 * g["ok"].sum() / len(g) if len(g) else 0.0,
                    "no_data_pct": 100.0 * g["no_data"].sum() / len(g) if len(g) else 0.0,
                    "error_pct": 100.0 * (
                        len(g) - g["ok"].sum() - g["no_data"].sum()
                    ) / len(g) if len(g) else 0.0,
                    "median_latency_s": g["elapsed_s"].median(),
                    "p95_latency_s": g["elapsed_s"].quantile(0.95),
                    "median_size": _fmt_bytes(g["size_bytes"].median()),
                    "median_rss_delta": _fmt_bytes(g["rss_delta"].median()),
                })
            )
            .reset_index()
            .sort_values(["z", "timestep_index"])
            )

        print("\nSummary by zoom:")
        print (zoom_summary)


        ## ---------------------------------
        # Example 2 : Rasterio Backend
        ## ---------------------------------
        concept_id = "C2036881735-POCLOUD"  # HLS L30
        datetime_range = "2024-10-01T00:00:01Z/2024-10-30T00:00:01Z"

        backend = "rasterio"
        bands = ["B04", "B03", "B02"]
        bands_regex = "B[0-9][0-9]"

        ds_ri = DatasetParams(
            concept_id=concept_id,
            backend=backend,
            datetime_range=datetime_range,
            kwargs={
                "bands": bands,
                "bands_regex": bands_regex,
                "colormap_name": "gnbu",
                "step": step,
                "temporal_mode": "point",
            }
        )
        print("Rasterio example DatasetParams (to_query_params):")
        for k, v in ds_ri.to_query_params():
            print(f"  {k}: {v}")
        df_ri = await benchmark_titiler_cmr(
            ds=ds_ri,
            endpoint=endpoint,
            tms_id="WebMercatorQuad",
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            lng=lng,
            lat=lat,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        print("\nRI head:\n", df_ri.head())

        zoom_summary = (
            df_ri.assign(
                elapsed_s=pd.to_numeric(df_ri["elapsed_s"], errors="coerce"),
                size_bytes=pd.to_numeric(df_ri["size_bytes"], errors="coerce"),
                rss_delta=pd.to_numeric(df_ri["rss_delta"], errors="coerce"),
            )
            .groupby(["z", "timestep_index"])
            .apply(
                lambda g: pd.Series({
                    "n_tiles": len(g),
                    "ok_pct": 100.0 * g["ok"].sum() / len(g) if len(g) else 0.0,
                    "no_data_pct": 100.0 * g["no_data"].sum() / len(g) if len(g) else 0.0,
                    "error_pct": 100.0 * (
                        len(g) - g["ok"].sum() - g["no_data"].sum()
                    ) / len(g) if len(g) else 0.0,
                    "median_latency_s": g["elapsed_s"].median(),
                    "p95_latency_s": g["elapsed_s"].quantile(0.95),
                    "median_size": _fmt_bytes(g["size_bytes"].median()),
                    "median_rss_delta": _fmt_bytes(g["rss_delta"].median()),
                })
            )
            .reset_index()
            .sort_values(["z", "timestep_index"])
            )

        print("\nSummary by zoom:")
        print (zoom_summary)

    asyncio.run(_run())
