"""
Benchmarking utility for TiTiler-CMR.

This module provides tools to evaluate the performance of the
TiTiler-CMR API for geospatial tile rendering across multiple zoom levels.
"""

from __future__ import annotations

import asyncio
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
import morecantile
import pandas as pd
import psutil

TILES_WIDTH = 5
TILES_HEIGHT = 5
SUPPORTED_TILE_FORMATS: set[str] = {
    "png", "npy", "tiff", "jpeg", "jpg", "jp2", "webp", "pngraw",
}


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

# ------------------------------
async def fetch_tile(
    client: httpx.AsyncClient,
    *,
    tiles_templates: List[str],
    z: int,
    x: int,
    y: int,
    timeout_s: float = 30.0,
    proc: Optional[psutil.Process] = None,
) -> List[Dict[str, Any]]:
    """
    For a single (z,x,y), iterate over all tiles templates, GET the tile, print status,
    and return one record per request.

     Parameters
    ----------
        tiles_templates : list of str
            A list of URL templates containing ``{z}``, ``{x}``, and ``{y}`` placeholders.
        z : int
            The zoom level of the tile.
        x : int
            The x index of the tile.
        y : int
            The y index of the tile.
        timeout_s : float
            Request timeout in seconds. Default is 30.0 seconds.
        proc : psutil.Process, optional
            The process to monitor for memory usage. If not provided, memory metrics will be unavailable.

    Returns
    -------
    list of dict
        One dictionary per requested template with (at least) the following keys:

        - ``z`` (int): zoom level.
        - ``x`` (int): tile x index.
        - ``y`` (int): tile y index.
        - ``timestep_index`` (int): index of the template within ``tiles_templates``.
        - ``url`` (str): fully formatted request URL.
        - ``status_code`` (int or None): HTTP status code; ``None`` if the request failed.
        - ``elapsed_s`` (float): wall-clock time for the request (seconds).
        - ``size_bytes`` (int): response body size in bytes (``0`` if no content or on failure).
        - ``content_type`` (str or None): value of the ``Content-Type`` header, if present.
        - ``ok`` (bool): ``True`` iff ``status_code == 200``.
        - ``no_data`` (bool): ``True`` iff ``status_code == 204``.
        - ``error_text`` (str or None): short error snippet for non-2xx responses or exceptions.
        - ``rss_before`` (int or float): process RSS (bytes) before the request (``0``/``NaN`` if unavailable).
        - ``rss_after`` (int or float): process RSS (bytes) after the request (``0``/``NaN`` if unavailable).
        - ``rss_delta`` (int or float): ``rss_after - rss_before`` (bytes; ``NaN`` on error).


    """
    rows: List[Dict[str, Any]] = []

    for i, tmpl in enumerate(tiles_templates):
        tile_url = tmpl.format(z=z, x=x, y=y)
        try:
            rss_before = proc.memory_info().rss if proc is not None else 0
            t0 = time.perf_counter()
            response = httpx.get(tile_url, timeout=timeout_s)
            elapsed = time.perf_counter() - t0
            rss_after = proc.memory_info().rss if proc is not None else 0
            rss_delta = rss_after - rss_before

            status = response.status_code
            ctype = response.headers.get("content-type")
            size = len(response.content) if response.content is not None else 0

            # Friendlier console line with readable RSS delta
            # print(
            #     f"[{i}] {status} {ctype or ''} | body={_fmt_bytes(size)} | "
            #     f"rssΔ={_fmt_bytes(rss_delta)} | {tile_url}"
            # )

            if status >= 400:
                try:
                    print(f"[{i}] Error body: {response.text[:400]}")
                except Exception:
                    pass

            rows.append(
                {
                    "z": z,
                    "x": x,
                    "y": y,
                    "timestep_index": i,
                    "url": tile_url,
                    "status_code": status,
                    "elapsed_s": elapsed,
                    "size_bytes": size,
                    "content_type": ctype,
                    "ok": (status == 200),
                    "no_data": (status == 204),
                    "error_text": None if status < 400 else response.text[:400],
                    # memory metrics (in bytes, keep numeric for analysis)
                    "rss_before": rss_before,
                    "rss_after": rss_after,
                    "rss_delta": rss_delta,
                }
            )

        except Exception as ex:
            print(f"[{i}] Request failed: {ex}")
            rows.append(
                {
                    "z": z,
                    "x": x,
                    "y": y,
                    "timestep_index": i,
                    "url": tile_url,
                    "status_code": None,
                    "elapsed_s": float("nan"),  # ensure numeric column
                    "size_bytes": 0,
                    "content_type": None,
                    "ok": False,
                    "no_data": False,
                    "error_text": f"{type(ex).__name__}: {ex}",
                    # memory metrics (NaN on error)
                    "rss_before": float("nan"),
                    "rss_after": float("nan"),
                    "rss_delta": float("nan"),
                }
            )

    return rows


@dataclass
class DatasetParams:
    """
    Parameters needed to request tiles from TiTiler-CMR.

    This class encapsulates the metadata and visualization options needed to build
    query parameters for tile requests against the TiTiler-CMR API. Depending on the
    chosen backend (`xarray` or `rasterio`), different fields are required.

    Attributes
    ----------
    concept_id : str
    datetime_range : str
    backend : str, default="xarray"
    variable : str, optional
    bands : Sequence[str], optional
    bands_regex : str, optional
    rescale : str, optional
    colormap_name : str, optional
    resampling : str, optional

    Methods
    -------
    to_query_params() -> List[Tuple[str, str]]
        Build a list of (key, value) pairs suitable for passing as query parameters
        to TiTiler-CMR tile endpoints. Validates backend-specific requirements.
    """
    concept_id: str
    datetime_range: str  # ISO8601 interval, e.g., "2024-10-01T00:00:01Z/2024-10-10T00:00:01Z"
    backend: str = "xarray"  # "xarray" or "rasterio"

    # xarray
    variable: Optional[str] = None

    # rasterio
    bands: Optional[Sequence[str]] = None
    bands_regex: Optional[str] = None

    # optional visualization/processing
    rescale: Optional[str] = None         # e.g., "0,46"
    colormap_name: Optional[str] = None   # e.g., "viridis"
    resampling: Optional[str] = None      # e.g., "nearest"

    def to_query_params(self) -> List[Tuple[str, str]]:
        """
        Build query params for Tile endpoints depending on backend and options.
        """
        params: List[Tuple[str, str]] = [
            ("concept_id", self.concept_id),
            ("datetime", self.datetime_range),
            ("backend", self.backend),
        ]

        if self.backend == "xarray":
            if not self.variable:
                raise ValueError("For backend='xarray', 'variable' must be provided.")
            params.append(("variable", self.variable))

        elif self.backend == "rasterio":
            # guard for None before .strip()
            if not (self.bands and self.bands_regex and self.bands_regex.strip()):
                raise ValueError("For backend='rasterio', provide both 'bands' and 'bands_regex'.")
            for b in self.bands:
                params.append(("bands", b))
            params.append(("bands_regex", self.bands_regex))

        else:
            raise ValueError("backend must be 'xarray' or 'rasterio'")

        if self.rescale:
            params.append(("rescale", self.rescale))
        if self.colormap_name:
            params.append(("colormap_name", self.colormap_name))
        if self.resampling:
            params.append(("resampling", self.resampling))

        return params


# ------------------------------
# Timeseries-only benchmark (async wrapper)
# ------------------------------
async def benchmark_titiler_cmr(
    endpoint: str,
    concept_id: str,
    datetime_range: str,
    *,
    backend: str = "xarray",
    variable: Optional[str] = None,
    bands: Optional[Sequence[str]] = None,
    bands_regex: Optional[str] = None,
    # ------ titiler-cmr params ------
    tms_id: str = "WebMercatorQuad",
    tile_format: str = "png",
    tile_scale: int = 1,
    min_zoom: int = 7,
    max_zoom: int = 10,
    rescale: Optional[str] = None,
    colormap_name: Optional[str] = "viridis",
    resampling: Optional[str] = None,
    # ---  
    lng: float = -92.1,
    lat: float = 46.8,
    timeout_s: float = 30.0,
    viewport_width: int = 5,
    viewport_height: int = 5,
    # timeseries specifics:
    step: str = "P1W",
    temporal_mode: str = "interval",
    # httpx concurrency limits (tune as needed)
    max_connections: int = 100,
    max_connections_per_host: int = 20,
    **kwargs: Any,

) -> pd.DataFrame:
    """
    1) GET /timeseries/{tms_id}/tilejson.json to obtain tile templates
    2) For each zoom & viewport tile, run fetch_tile(...) over all templates using a shared AsyncClient
    3) Return a tidy DataFrame with one row per request
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

    # Build query params (list of tuples to preserve duplicates like multiple 'bands')
    ds = DatasetParams(
        concept_id=concept_id,
        datetime_range=datetime_range,
        backend=backend,
        variable=variable,
        bands=bands,
        bands_regex=bands_regex,
        rescale=rescale,
        colormap_name=colormap_name,
        resampling=resampling,
    )
    params: List[Tuple[str, str]] = ds.to_query_params() + [
        ("step", step),
        ("temporal_mode", temporal_mode),
        ("minzoom", str(min_zoom)),
        ("maxzoom", str(max_zoom)),
        ("tile_format", tile_format),
    ] + [(k, str(v)) for k, v in kwargs.items()]

    print("----------")
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
        r = await client.get(ts_url, params=params)
        #r.raise_for_status()
        ts_json = r.json()

        tiles_templates = [tile for v in ts_json.values() for tile in v.get("tiles", [])]
        if not tiles_templates:
            raise RuntimeError("No 'tiles' templates found in timeseries TileJSON response.")
        n_timesteps = len(tiles_templates)
        
        print(f"TileJSON: timesteps={n_timesteps} | templates={len(tiles_templates)}")

        # 2) build tasks and fetch concurrently with the SAME client
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
                            tiles_templates=tiles_templates,
                            z=z,
                            x=x,
                            y=y,
                            timeout_s=timeout_s,
                            proc=proc,
                        )
                    )
                )

        results_lists = await asyncio.gather(*tasks)

    # 3) flatten & frame
    all_rows: List[Dict[str, Any]] = [row for rows in results_lists for row in rows]
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["z", "y", "x", "timestep_index"]).reset_index(drop=True)
        try:
            rss_after_num = pd.to_numeric(df["rss_after"], errors="coerce")
            rss_peak = int(rss_after_num.max())
            print(f"\nPeak process RSS observed: {_fmt_bytes(rss_peak)}")
        except Exception:
            pass

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


def _max_tile_index(z: int) -> int:
    """
    Compute the maximum valid XYZ tile index for a given zoom level.

    At zoom level `z`, the map is subdivided into 2**z tiles along each axis
    (x and y). The valid tile indices therefore range from 0 to (2**z - 1).
    """
    if z < 0:
        raise ValueError("zoom must be >= 0")
    return (1 << z) - 1


async def check_titiler_cmr_compatibility(
    endpoint: str,
    concept_id: str,
    datetime_range: str,
    *,
    timeout_s: float = 60.0,
    include_datetimes: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
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
    endpoint : str
        Base URL of the TiTiler-CMR API.
    concept_id : str
        CMR Concept ID of the dataset.
    datetime_range : str
        ISO8601 interval specifying the desired temporal range.
    timeout_s : float, optional
        Request timeout in seconds. Default = 60.0.
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
    params: Dict[str, Any] = {"concept_id": concept_id, "datetime": datetime_range}
    params.update(kwargs or {})

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url, params=params)
        try:
            payload = resp.json()
        except Exception:
            summary = {
                "concept_id": concept_id,
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
        "concept_id": concept_id,
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
        endpoint = "https://staging.openveda.cloud/api/titiler-cmr"
        min_zoom = 8
        max_zoom = 8
        lng = -92.1
        lat = 46.8
        viewport_width = 5
        viewport_height = 5
        step = "P1W"
        temporal_mode = "interval"
        tile_format = "png"

        concept_id = "C2723754864-GES_DISC"
        #datetime_range = "2024-10-12T00:00:00Z/2024-11-13T00:00:00Z"
        backend = "xarray"
        variable = "precipitation"

        # Overwrite for demo range
        concept_id = "C2723754864-GES_DISC"
        datetime_range = "2024-10-01T00:00:00Z/2024-10-05T23:59:59Z"

        # Run compatibility check (concise summary by default)
        result = await check_titiler_cmr_compatibility(
            endpoint=endpoint,
            concept_id=concept_id,
            datetime_range=datetime_range,
        )

        # Access structured results programmatically
        summary = result  # function already returns the flat summary dict
        print("Summary object:", summary)

        # Example: decide if you should proceed
        if summary["n_granules"] == 0:
            print("No granules available in this date range. Skipping benchmark.")
        else:
            print(f"Proceeding: {summary['n_granules']} granules found.")

        df = await benchmark_titiler_cmr(
            endpoint=endpoint,
            concept_id=concept_id,
            datetime_range=datetime_range,
            backend=backend,
            variable=variable,
            rescale="0,1",
            colormap_name="gnbu",
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            lng=lng,
            lat=lat,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            step=step,
            temporal_mode=temporal_mode,
            tile_format=tile_format,
            # max_workers defaults to all logical cores; override if you want
            # max_workers=32,
        )
        # Show head with pretty columns
        print(df.head())

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)

        # Numeric summary first (keeps bytes columns numeric)
        zoom_summary = df.groupby("z").agg(
            n_requests=("ok", "size"),
            ok=("ok", "sum"),
            no_data=("no_data", "sum"),
            median_latency_s=("elapsed_s", "median"),
            p95_latency_s=("elapsed_s", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
            median_size_B=("size_bytes", "median"),
            median_rss_delta_B=("rss_delta", "median"),
            p95_rss_delta_B=("rss_delta", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.95)),
        )
        print("\nSummary by zoom (numeric bytes):")
        print(zoom_summary)

        # Pretty-printed memory summary (string columns for readability)
        pretty_summary = zoom_summary.copy()
        for col in ["median_size_B", "median_rss_delta_B", "p95_rss_delta_B"]:
            pretty_summary[col.replace("_B", "_h")] = pretty_summary[col].map(_fmt_bytes)
        print("\nSummary by zoom (human-readable memory):")
        print(pretty_summary[[
            "n_requests", "ok", "no_data",
            "median_latency_s", "p95_latency_s",
            "median_size_h", "median_rss_delta_h", "p95_rss_delta_h"
        ]])

        # Rasterio example
        concept_id = "C2036881735-POCLOUD"  # HLS L30
        datetime_range = "2024-10-01T00:00:01Z/2024-10-30T00:00:01Z"

        backend = "rasterio"
        bands = ["B04", "B03", "B02"]
        bands_regex = "B[0-9][0-9]"

        df_ri = await benchmark_titiler_cmr(
            endpoint=endpoint,
            concept_id=concept_id,
            datetime_range=datetime_range,
            backend=backend,
            bands=bands,
            bands_regex=bands_regex,
            colormap_name="gnbu",
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            lng=lng,
            lat=lat,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            step=step,
            temporal_mode="point",
            # tile_format="point",  # enable if your API expects this for point mode
            # max_workers=32,
        )
        print("\nRI head:\n", df_ri.head())

    asyncio.run(_run())
