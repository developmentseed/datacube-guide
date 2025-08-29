"""
Benchmarking utility for TiTiler-CMR.

This module provides tools to evaluate the performance of the
TiTiler-CMR API for geospatial tile rendering across multiple zoom levels.

TODO: Remove memory profiling (lambda function....)
"""

from __future__ import annotations

import asyncio
import time
import psutil

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tiling import get_surrounding_tiles, fetch_tile

import httpx
import morecantile
import pandas as pd

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

TILES_WIDTH = 5
TILES_HEIGHT = 5
# ------------------------------
# Dataclasses
# ------------------------------

@dataclass
class DatasetParams:
    """
    Encapsulates parameters for requesting tiles from TiTiler-CMR.

    Required parameters:
        concept_id (str): CMR concept ID for the dataset.
        backend (str): Backend type, e.g., "xarray" or "rasterio".
        datetime_range (str): ISO8601 interval, e.g., "2024-10-01T00:00:01Z/2024-10-10T00:00:01Z".

    Optional parameters:
        kwargs (Dict[str, Any]): Additional query parameters, such as:
            - variable (str): For xarray backend, the variable name.
            - bands (Sequence[str]): For rasterio backend, list of bands.
            - bands_regex (str): For rasterio backend, regex for bands selection.
            - rescale (str): Rescale range for visualization.
            - colormap_name (str): Colormap name for visualization.
            - resampling (str): Resampling method.
            - step, temporal_mode, minzoom, maxzoom, tile_format, etc.

    Raises:
        ValueError: If required backend-specific fields are missing or if an unexpected type is encountered in kwargs.
    """
    concept_id: str
    backend: str
    datetime_range: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_query_params(self, **extra_kwargs: Any) -> List[Tuple[str, str]]:
        """
        Build query parameters for TiTiler-CMR tile endpoints.

        Combines required fields and all additional keyword arguments, filtering out None values and converting types as needed.
        Validates that required backend-specific fields are present in kwargs.

        Args:
            extra_kwargs: Additional keyword arguments to include as query params.

        Returns:
            List[Tuple[str, str]]: List of (key, value) pairs for query parameters.

        Raises:
            ValueError: If required backend-specific fields are missing or if an unexpected type is encountered in kwargs.
        """
        params: List[Tuple[str, str]] = [
            ("concept_id", self.concept_id),
            ("backend", self.backend),
            ("datetime", self.datetime_range),
        ]
        all_kwargs = dict(self.kwargs)
        all_kwargs.update(extra_kwargs)

        # Backend-specific validation
        if self.backend == "xarray":
            if not all_kwargs.get("variable"):
                raise ValueError("For backend='xarray', 'variable' must be provided in kwargs.")
        elif self.backend == "rasterio":
            if not (all_kwargs.get("bands") and all_kwargs.get("bands_regex")):
                raise ValueError("For backend='rasterio', 'bands' and 'bands_regex' must be provided in kwargs.")

        for k, v in all_kwargs.items():
            if v is None:
                continue
            if isinstance(v, bool):
                params.append((k, "true" if v else "false"))
            elif isinstance(v, (int, float)):
                params.append((k, str(v)))
            elif isinstance(v, (list, tuple, set)):
                for item in v:
                    if item is not None:
                        params.append((k, str(item)))
            elif isinstance(v, str):
                params.append((k, v))
            else:
                print(f"Unexpected type for param '{k}': {type(v)}. Value: {v}")
        return params

# ------------------------------
# Timeseries-only benchmark (async wrapper)
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
    lng: float = -92.1,
    lat: float = 46.8,
    timeout_s: float = 30.0,
    viewport_width: int = TILES_WIDTH,
    viewport_height: int = TILES_HEIGHT,
    max_connections: int = 20,
    max_connections_per_host: int = 20,
    **kwargs: Any, # review for kwargs......
    ) -> pd.DataFrame:
    """
    1) GET /timeseries/{tms_id}/tilejson.json to obtain tile templates
    2) For each zoom & viewport tile, run fetch_tile(...) over all templates using a shared AsyncClient
    3) Return a tidy DataFrame with one row per request


    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR API (e.g., ``"https://.../api/titiler-cmr"``).
    concept_id : str
        CMR concept ID of the dataset to benchmark.
    datetime_range : str
        ISO8601 interval specifying the temporal subset (e.g., ``"2024-10-01T00:00Z/2024-10-05T23:59Z"``).
    backend : {"xarray", "rasterio"}, default="xarray"
        Backend used by TiTiler-CMR for rendering.
    variable : str, optional
        For the ``xarray`` backend, the variable name to render.
    bands : sequence of str, optional
        For the ``rasterio`` backend, list of bands to request.
    bands_regex : str, optional
        Regex pattern for selecting rasterio bands.
    tms_id : str, default="WebMercatorQuad"
        Tile matrix set identifier (passed to TiTiler-CMR).
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
    rescale : str, optional
        Rescaling range for visualization (e.g., ``"0,1"``).
    colormap_name : str, default="viridis"
        Colormap for rendering tiles.
    resampling : str, optional
        Resampling method used by TiTiler-CMR.
    max_connections : int, default=20
        Maximum concurrent HTTP connections.
    max_connections_per_host : int, default=20
        Maximum concurrent connections per host.
    **kwargs : dict
        Additional parameters passed through to TiTiler-CMR query string.

    """
    print(
        f"System: {psutil.cpu_count(logical=False)} physical / "
        f"{psutil.cpu_count(logical=True)} logical cores | "
        f"RAM: {_fmt_bytes(psutil.virtual_memory().total)}"
    )

    if tile_format not in SUPPORTED_TILE_FORMATS:
        raise ValueError(
            f"Unsupported tile_format '{tile_format}'. "
            f"Supported: {sorted(SUPPORTED_TILE_FORMATS)}"
        )

    # Add/override tile_format, minzoom, maxzoom in ds.kwargs
    params: List[Tuple[str, str]] = ds.to_query_params(
        tile_format=tile_format,
        tile_scale=tile_scale,
        **kwargs,
    )

    print("---------- Query Params ----------")
    print(*params, sep="\n")

    # 1) fetch TileJSON with a short-lived sync call (fine), or use AsyncClient too:
    ts_url = f"{endpoint.rstrip('/')}/timeseries/{tms_id}/tilejson.json"
    tms = morecantile.tms.get(tms_id)

    # Use AsyncClient for consistency
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections_per_host,
    )
    async with httpx.AsyncClient(limits=limits, timeout=timeout_s) as client:
        resp = await client.get(ts_url, params=params)
        #resp.raise_for_status()
        ts_json = resp.json()

        # find all the templates for all granules....
        tiles_endpoints = [tile for v in ts_json.values() for tile in v.get("tiles", [])]
        if not tiles_endpoints:
            raise RuntimeError("No 'tiles' templates found in timeseries TileJSON response.")
        n_timesteps = len(tiles_endpoints)
        
        print(f"TileJSON: timesteps={n_timesteps} | Templates={len(tiles_endpoints)}")

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

    # 3) flatten & frame
    all_rows: List[Dict[str, Any]] = []
    for rows in results_lists:
        if isinstance(rows, Exception):
            print(f"Task error: {rows}")
            continue
        if isinstance(rows, list):
            all_rows.extend(rows)
        else:
            print(f"Unexpected result type: {type(rows)}")

    # -- TODO: save dataframes
    required_cols = ["z", "y", "x", "timestep_index"]
    df = pd.DataFrame(all_rows)
    if not df.empty and all(col in df.columns for col in required_cols):
        df = df.sort_values(required_cols).reset_index(drop=True)
        try:
            rss_after_num = pd.to_numeric(df["rss_after"], errors="coerce")
            rss_peak = int(rss_after_num.max())
            print(f"\nPeak process RSS observed: {_fmt_bytes(rss_peak)}")
        except Exception:
            pass
    else:
        print(f"Warning: DataFrame is empty or missing columns: {required_cols}")
    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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

        concept_id = "C2723754864-GES_DISC"
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
            df.groupby("z").agg(
            n_requests=("ok", "size"),
            ok=("ok", "sum"),
            no_data=("no_data", "sum"),
            median_latency=("elapsed_s", lambda s: pd.to_numeric(s, errors="coerce").median()),
            p95_latency_s=("elapsed_s", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
            median_size=("size_bytes", lambda s: _fmt_bytes(pd.to_numeric(s, errors="coerce").median())),
            median_rss_delta=("rss_delta", lambda s: _fmt_bytes(pd.to_numeric(s, errors="coerce").median())),
            )
            .sort_index()
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

    asyncio.run(_run())
