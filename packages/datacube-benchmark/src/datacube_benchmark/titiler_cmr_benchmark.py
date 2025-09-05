"""
Unified benchmarking system for TiTiler-CMR.

This module provides a clean, extensible architecture for benchmarking
TiTiler-CMR performance across different scenarios (viewport, full tileset, statistics).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import httpx
import psutil
import morecantile
import pandas as pd
from geojson_pydantic import Feature

from datacube_benchmark.tiling import (
    get_surrounding_tiles,
    get_tileset_tiles,
    fetch_tile,
    create_bbox_feature,
    BaseBenchmarker,
    DatasetParams,
)


class TiTilerCMRBenchmarker(BaseBenchmarker):
    """
    Main benchmarking utility for TiTiler-CMR.

    Supports benchmarking of tile rendering and statistics endpoints
    across different strategies (viewport, tileset, custom).

    Also supports compatibility checks by querying TileJSON and running
    a preview statistics request.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        tms_id: str = "WebMercatorQuad",
        tile_format: str = "png",
        tile_scale: int = 1,
        min_zoom: int = 7,
        max_zoom: int = 10,
        **base_kwargs: Any,
    ):
        super().__init__(endpoint, **base_kwargs)
        self.tms_id = tms_id
        self.tile_format = tile_format
        self.tile_scale = tile_scale
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    async def benchmark_tiles(
        self,
        dataset: DatasetParams,
        tiling_strategy: Callable,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Benchmark tile rendering performance for TiTiler-CMR.
        It can be adopted for a viewport or whole tileset generation at a zoom level.

        Parameters
        ----------
        dataset : DatasetParams
            Dataset and query parameters (concept_id, backend, datetime_range, kwargs).
        tiling_strategy : Callable
            Function that returns tiles for a given zoom level.
            Signature: (zoom, tms, tilejson_info) -> List[Tuple[int, int]]
        **kwargs : Any
            Additional query parameters for the API.

        Returns
        -------
        pd.DataFrame
            Results for each tile request, including status, latency, and size.
        """
        self._log_header("Tile Benchmark", dataset)
        params = self._dataset_params(dataset, include_tile_opts=True, **kwargs)
        async with self._create_http_client() as client:
            tilejson_info = await self._get_tilejson_info(client, params)
            tiles_endpoints = tilejson_info["tiles_endpoints"]
            print(f"Found {len(tiles_endpoints)} timesteps from TileJson.")
            tasks = await self._build_tile_tasks(client, tiles_endpoints, tiling_strategy, tilejson_info)
            results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._process_results(results)

    async def benchmark_statistics(
        self,
        dataset: DatasetParams,
        geometry: Optional[Union[Feature, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Benchmark statistics endpoint performance with timing and memory metrics.

        Parameters
        ----------
        dataset : DatasetParams
            Dataset configuration.
        geometry : Union[Feature, Dict[str, Any]], optional
            GeoJSON Feature or geometry to analyze. If None, uses bounds from tilejson.
        **kwargs : Any
            Additional query parameters.

        Returns
        -------
        Dict[str, Any]
            Statistics result with timing, memory, and metadata.
        """
        self._log_header("Statistics Benchmark", dataset)
        async with self._create_http_client() as client:
            if geometry is None:
                tile_params = self._dataset_params(dataset, include_tile_opts=False, **kwargs)
                tilejson_info = await self._get_tilejson_info(client, tile_params)
                bounds = tilejson_info.get("bounds")
                if not bounds:
                    raise ValueError("No geometry provided and no bounds available from TileJSON")
                geometry = create_bbox_feature(*bounds)
            return await self._fetch_statistics(client=client, dataset=dataset, geometry=geometry, **kwargs)

    async def check_compatibility(
        self,
        dataset: DatasetParams,
        geometry: Optional[Union[Feature, Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Check TiTiler-CMR compatibility and get dataset overview.

        Queries TileJSON for granule/timestep info, then runs statistics for preview.

        Parameters
        ----------
        dataset : DatasetParams
            Dataset configuration.
        geometry : Union[Feature, Dict[str, Any]], optional
            GeoJSON Feature or geometry. If None, uses bounds from tilejson.
        **kwargs : Any
            Additional query parameters.

        Returns
        -------
        Dict[str, Any]
            Compatibility info including timestep count, granule info, and statistics preview.
        """
        self._log_header("Compatibility Check", dataset)
        async with self._create_http_client() as client:
            tile_params = self._dataset_params(dataset, include_tile_opts=False, **kwargs)
            tilejson_info = await self._get_tilejson_info(client, tile_params)
            tiles_endpoints = tilejson_info["tiles_endpoints"]
            n_timesteps = len(tiles_endpoints)
            print(f"Found {n_timesteps} timesteps/granules from TileJson!")
            if geometry is None:
                bounds = tilejson_info.get("bounds")
                if not bounds:
                    raise ValueError("No geometry provided and no bounds available from TileJSON")
                geometry = create_bbox_feature(*bounds)

        stats_result = await self._fetch_statistics(
            client=client,
            dataset=dataset,
            geometry=geometry,
            **kwargs
        )

        if stats_result["success"] and stats_result["statistics"]:
            print(f"Statistics returned {len(stats_result['statistics'])} timesteps")
        else:
            print("Statistics request failed:", stats_result.get("error"))
            issue_detected = True

        return {
            "concept_id": dataset.concept_id,
            "backend": dataset.backend,
            "n_timesteps": n_timesteps,
            "tilejson_bounds": tilejson_info.get("bounds"),
            "statistics": self._statistics_to_dataframe(stats_result["statistics"]) if stats_result["success"] else pd.DataFrame(),
            "compatibility": "compatible" if n_timesteps > 0 and not issue_detected else "issues_detected",
        }

    async def _fetch_statistics(
        self,
        client: httpx.AsyncClient,
        dataset: DatasetParams,
        geometry: Union[Feature, Dict[str, Any]],
        **extra_params: Any
    ) -> Dict[str, Any]:
        """
        Posts the provided GeoJSON Feature or raw geometry to the TiTiler-CMR
        `/timeseries/statistics` endpoint and returns per-timestep summary
        statistics for pixels intersecting the geometry.
        
        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client for requests.
        dataset : DatasetParams
            Dataset configuration.
        geometry : Union[Feature, Dict[str, Any]]
            GeoJSON Feature or geometry.
        **extra_params : Any
            Additional query parameters.

        Returns
        -------
        Dict[str, Any]
            Statistics result and metadata and timing.
        """
        url = f"{self.endpoint.rstrip('/')}/timeseries/statistics"
        params = dict(dataset.to_query_params(**extra_params))
        if hasattr(geometry, 'model_dump'):
            geojson_body = geometry.model_dump(exclude_none=True)
        elif isinstance(geometry, dict):
            geojson_body = geometry
        else:
            raise ValueError("geometry must be a GeoJSON Feature or dict")
        try:
            data, elapsed, status = await self._request_json(
                client,
                method="POST",
                url=url,
                params=params,
                json_payload=geojson_body,
                timeout_s=self.timeout_s
            )
            stats = data.get("properties", {}).get("statistics", {})
            return {
                "success": True,
                "elapsed_s": elapsed,
                "status_code": status,
                "n_timesteps": len(stats) if isinstance(stats, dict) else 0,
                "url": url,
                "statistics": stats,
                "error": None,
            }
        except Exception as ex:
            return {
                "success": False,
                "elapsed_s": None,
                "status_code": None,
                "n_timesteps": 0,
                "url": url,
                "statistics": {},
                "error": f"{type(ex).__name__}: {ex}",
            }

    async def _build_tile_tasks(
        self,
        client: httpx.AsyncClient,
        tiles_endpoints: List[str],
        tiling_strategy: Callable[[int, morecantile.TileMatrixSet, Dict[str, Any]], List[Tuple[int, int]]],
        tilejson_info: Dict[str, Any]
    ) -> List[asyncio.Task]:
        """
        Build async tasks for all tile requests for a given strategy and tile endpoints.

        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client for requests.
        tiles_endpoints : List[str]
            List of tile endpoint URLs.
        tiling_strategy : Callable
            Tile selection strategy function.
        tilejson_info : Dict[str, Any]
            TileJSON metadata.

        Returns
        -------
        List[asyncio.Task]
            List of asyncio tasks for tile requests.
        """
        tms = morecantile.tms.get(self.tms_id)
        proc = psutil.Process()
        tasks: List[asyncio.Task] = []
        for zoom in range(self.min_zoom, self.max_zoom + 1):
            tiles = tiling_strategy(zoom, tms, tilejson_info)
            nx = len({x for x, _ in tiles})
            ny = len({y for _, y in tiles})
            print(f"Zoom {zoom}: {nx}x{ny} tiles ({len(tiles)} total)")
            for x, y in tiles:
                tasks.append(
                    asyncio.create_task(
                        fetch_tile(
                            client,
                            tiles_endpoints=tiles_endpoints,
                            z=zoom, x=x, y=y,
                            timeout_s=self.timeout_s,
                            proc=proc
                        )
                    )
                )
        return tasks

    async def _get_tilejson_info(
        self, 
        client: httpx.AsyncClient, 
        params: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Query TiTiler-CMR timeseries TileJSON and return parsed entries and metadata.

        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client for requests.
        params : list of tuple
            Query parameters for the request.

        Returns
        -------
        dict
            Dictionary with entries, tilejson, tile endpoints, and bounds.
        """
        url = f"{self.endpoint.rstrip('/')}/timeseries/{self.tms_id}/tilejson.json"
        ts_json, _, _ = await self._request_json(
            client,
            method="GET",
            url=url,
            params=dict(params),
            timeout_s=self.timeout_s,
        )
        entries: List[Dict[str, Any]] = []
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
            raise RuntimeError("No tile endpoints found in TileJSON response")
        bounds = entries[0].get("bounds") if entries else None
        return {
            "entries": entries,
            "tilejson": ts_json,
            "tiles_endpoints": tiles_endpoints,
            "bounds": bounds,
        }

    @staticmethod
    def _statistics_to_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Flatten TiTiler-CMR statistics dict into a DataFrame, assuming
        inner and outer timestamps match. Histogram arrays are dropped.

        Output columns:
          - timestamp (ISO8601 string)
          - scalar metrics (min, max, mean, count, sum, std, median, majority,
            minority, unique, valid_percent, masked_pixels, valid_pixels,
            percentile_2, percentile_98)
        """
        rows: List[Dict[str, Any]] = []
        if not isinstance(stats, dict):
            return pd.DataFrame()
        for _, inner in stats.items():
            if not isinstance(inner, dict) or not inner:
                continue
            inner_ts, metrics = next(iter(inner.items()))
            if not isinstance(metrics, dict):
                continue
            row: Dict[str, Any] = {"timestamp": inner_ts}
            for k, v in metrics.items():
                if k == "histogram":
                    continue
                row[k] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col != "timestamp":
                df[col] = pd.to_numeric(df[col])
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        return df.reset_index(drop=True)

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        *,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = {},
        json_payload: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,  
    ) -> Tuple[Dict[str, Any], float, int]:
        """
        Unified JSON request helper for GET/POST with consistent error handling.

        Returns
        -------
        (payload, elapsed_s, status_code)
        """
        timeout = timeout_s if timeout_s is not None else self.timeout_s

        t0 = time.perf_counter()
        response: Optional[httpx.Response] = None
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = await client.post(url, params=params, json=json_payload, timeout=timeout)
            response.raise_for_status()
            response.elapsed = time.perf_counter() - t0
            data = response.json()
            return data if isinstance(data, dict) else {}, response.elapsed, response.status_code
        except httpx.HTTPStatusError as ex:
            response = ex.response
            response.elapsed = time.perf_counter() - t0

            print("~~~~~~~~~~~~~~~~ ERROR JSON REQUEST ~~~~~~~~~~~~~~~~")
            print(f"URL: {response.request.url}")
            print(f"Error: {response.status_code} {response.reason_phrase}")
            print(f"Body: {response.text}")
            raise

    def _dataset_params(self, dataset: DatasetParams, *, include_tile_opts: bool, **kwargs: Any) -> List[Tuple[str, str]]:
        """
        Build query params for a dataset with optional tile-related arguments added.
        """
        if include_tile_opts:
            params = list(dataset.to_query_params(tile_format=self.tile_format, tile_scale=self.tile_scale, **kwargs))
        else:
            params = list(dataset.to_query_params(**kwargs))
        print(f"Query params: {len(params)} parameters")
        for k, v in params:
            print(f"  {k}: {v}")
        return params


#---------------------------------------
# top level public API
#---------------------------------------

async def benchmark_viewport(
    endpoint: str,
    dataset: DatasetParams,
    lng: float,
    lat: float,
    *,
    viewport_width: int = 5,
    viewport_height: int = 5,
    tms_id: str = "WebMercatorQuad",
    tile_format: str = "png",
    tile_scale: int = 1,
    min_zoom: int = 7,
    max_zoom: int = 10,
    timeout_s: float = 30.0,
    max_connections: int = 20,
    max_connections_per_host: int = 20,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Benchmark tile rendering for a *viewport* centered at (lng, lat).

    This is a high-level convenience wrapper around
    ``TiTilerCMRBenchmarker.benchmark_tiles``. It builds a tiling strategy that
    selects a (viewport_width × viewport_height) neighborhood of tiles around
    the center tile at each zoom in ``[min_zoom, max_zoom]``, then measures
    latency, status, and size for each tile request across all timesteps.

    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR deployment.
    dataset : DatasetParams
        Dataset and query parameters (concept_id, backend, datetime_range, kwargs).
    lng : float
        Center longitude of the viewport.
    lat : float
        Center latitude of the viewport.
    viewport_width : int, optional
        Number of tiles in the X direction (default: 5).
    viewport_height : int, optional
        Number of tiles in the Y direction (default: 5).
    tms_id : str, optional
        Tile matrix set ID (default: "WebMercatorQuad").
    tile_format : str, optional 
        Tile format (default: "png").
    tile_scale : int, optional
        Tile scale factor (default: 1).
    min_zoom : int, optional
        Minimum zoom level (default: 7).
    max_zoom : int, optional
        Maximum zoom level (default: 10).
    timeout_s : float, optional
        Request timeout in seconds (default: 30.0).
    max_connections : int, optional
        Maximum total concurrent connections (default: 20).
    max_connections_per_host : int, optional
        Maximum concurrent connections per host (default: 20).
    **kwargs : Any
        Additional query parameters for the API.

    Returns
    -------
    pd.DataFrame
        Results for each tile request, including status, latency, and size.
    """
    benchmarker = TiTilerCMRBenchmarker(
        endpoint=endpoint,
        tms_id=tms_id,
        tile_format=tile_format,
        tile_scale=tile_scale,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        timeout_s=timeout_s,
        max_connections=max_connections,
        max_connections_per_host=max_connections_per_host
    )

    def viewport_strategy(
        zoom: int,
        tms: morecantile.TileMatrixSet,
        tilejson_info: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        center = tms.tile(lng=lng, lat=lat, zoom=zoom)
        return get_surrounding_tiles(
            center_x=center.x,
            center_y=center.y,
            zoom=zoom,
            width=viewport_width,
            height=viewport_height,
        )

    return await benchmarker.benchmark_tiles(dataset, viewport_strategy, **kwargs)


async def benchmark_tileset(
    endpoint: str,
    dataset: DatasetParams,
    *,
    bounds: Optional[List[float]] = None,
    max_tiles_per_zoom: Optional[int] = 1000,
    tms_id: str = "WebMercatorQuad",
    tile_format: str = "png",
    tile_scale: int = 1,
    min_zoom: int = 7,
    max_zoom: int = 10,
    timeout_s: float = 30.0,
    max_connections: int = 20,
    max_connections_per_host: int = 20,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Benchmark tile rendering for a *full tileset* over given bounds.

    This wrapper enumerates all tiles intersecting the supplied `bounds` (or the
    bounds from TileJSON if omitted) for each zoom level in ``[min_zoom, max_zoom]``.
    Optionally caps the number of tiles per zoom to avoid overly large runs.

    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR deployment.
    dataset : DatasetParams
        Dataset and query parameters (concept_id, backend, datetime_range, kwargs).
    bounds : list of float, optional
        Bounding box [min_lon, min_lat, max_lon, max_lat] to cover.
    max_tiles_per_zoom : int, optional
        If set, limits the number of tiles per zoom level to this count.
    tms_id : str, optional
        Tile matrix set ID (default: "WebMercatorQuad").
    tile_format : str, optional
            Tile image format (e.g., "png", "jpg", "webp"). (default: "png").
    tile_scale : int, optional
            Tile scale factor (default: 1).
    min_zoom : int, optional
        Minimum zoom level (default: 7).
    max_zoom : int, optional
        Maximum zoom level (default: 10).
    timeout_s : float, optional
        Request timeout in seconds (default: 30.0).
    max_connections : int, optional
        Maximum total concurrent connections (default: 20).
    max_connections_per_host : int, optional
        Maximum concurrent connections per host (default: 20).
    **kwargs : Any
        Additional query parameters for the API.    


    Returns
    -------
    pd.DataFrame
        Results for each tile request, including status, latency, and size. 
    
    """
    benchmarker = TiTilerCMRBenchmarker(
        endpoint=endpoint,
        tms_id=tms_id,
        tile_format=tile_format,
        tile_scale=tile_scale,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        timeout_s=timeout_s,
        max_connections=max_connections,
        max_connections_per_host=max_connections_per_host
    )

    def tileset_strategy(
        zoom: int,
        tms: morecantile.TileMatrixSet,
        tilejson_info: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        b = bounds or tilejson_info.get("bounds")
        if not b:
            raise ValueError("No bounds provided and none available in TileJSON.")
        tiles = get_tileset_tiles(bounds=b, zoom=zoom, tms=tms)
        if max_tiles_per_zoom is not None and len(tiles) > max_tiles_per_zoom:
            tiles = tiles[:max_tiles_per_zoom]
        return tiles

    return await benchmarker.benchmark_tiles(dataset, tileset_strategy, **kwargs)


async def benchmark_statistics(
    endpoint: str,
    dataset: DatasetParams,
    geometry: Optional[Union[Feature, Dict[str, Any]]] = None,
    *,
    timeout_s: float = 300.0,
    max_connections: int = 10,
    max_connections_per_host: int = 10,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Benchmark the `/timeseries/statistics` endpoint for a geometry.

    This high-level helper delegates to ``TiTilerCMRBenchmarker.benchmark_statistics``.
    If `geometry` is omitted, the TileJSON bounds for the dataset/time range are
    used to construct a bounding box feature. The result includes timing,
    HTTP status, and the statistics payload keyed by timestep.

    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR deployment.
    dataset : DatasetParams
        Dataset configuration.
    geometry : Union[Feature, Dict[str, Any]], optional 
        GeoJSON Feature or geometry to analyze. If None, uses bounds from tilejson.
    timeout_s : float, optional
        Request timeout in seconds (default: 300.0).    
    max_connections : int, optional
        Maximum total concurrent connections (default: 10).
    max_connections_per_host : int, optional
        Maximum concurrent connections per host (default: 10).
    **kwargs : Any
        Additional query parameters for the API.
    
    Returns
    -------
    Dict[str, Any]
        Statistics result with timing, memory, and metadata.
    """
    benchmarker = TiTilerCMRBenchmarker(
        endpoint=endpoint,
        timeout_s=timeout_s,
        max_connections=max_connections,
        max_connections_per_host=max_connections_per_host
    )
    return await benchmarker.benchmark_statistics(dataset, geometry, **kwargs)


async def check_titiler_cmr_compatibility(
    endpoint: str,
    dataset: DatasetParams,
    geometry: Optional[Union[Feature, Dict[str, Any]]] = None,
    *,
    timeout_s: float = 300.0,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Validate dataset compatibility with a TiTiler-CMR deployment.

    This function queries the timeseries TileJSON for the dataset/time range to
    count available timesteps (granules). It then requests statistics for a
    provided `geometry`—or, if not provided, a geometry derived from the TileJSON
    bounds—to verify end-to-end functionality.

    Parameters
    ----------
    endpoint : str
        Base URL of the TiTiler-CMR deployment.
    dataset : DatasetParams
        Dataset configuration.
    geometry : Union[Feature, Dict[str, Any]], optional
        GeoJSON Feature or geometry. If None, uses bounds from tilejson.
    timeout_s : float, optional
        Request timeout in seconds (default: 300.0).
    **kwargs : Any
        Additional query parameters for the API.
    
    Returns
    -------
    Dict[str, Any]
        Compatibility info including timestep count, granule info, and statistics preview.
    """
    benchmarker = TiTilerCMRBenchmarker(endpoint=endpoint, timeout_s=timeout_s)
    return await benchmarker.check_compatibility(dataset, geometry, **kwargs)


def tiling_benchmark_summary(df: pd.DataFrame, *, silent: bool = False) -> pd.DataFrame:
    """
    Compute and (optionally) print summary statistics for tile benchmark results.

    Groups by zoom level `z` and `timestep_index` and reports:
      - n_tiles
      - ok_pct, no_data_pct, error_pct
      - median_latency_s, p95_latency_s
      - median_size_bytes
      - median_rss_delta

    Parameters
    ----------
    df : pd.DataFrame
        Raw tile benchmark results. Expected columns:
        z, x, timestep_index, ok, no_data, elapsed_s, size_bytes, rss_delta.
    silent : bool, optional
        If False (default), prints a formatted summary table. If True, returns
        the summary without printing.

    Returns
    -------
    pd.DataFrame
        Summary statistics by zoom and timestep. Empty if input is empty or
        required columns are missing.
    """
    required = {"z", "x", "timestep_index", "ok", "no_data", "elapsed_s", "size_bytes", "rss_delta"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    gdf = df.copy()
    for col in ["elapsed_s", "size_bytes", "rss_delta"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    summary = (
        gdf.groupby(["z", "timestep_index"])
           .apply(lambda g: pd.Series({
               "n_tiles": len(g),
               "ok_pct": 100.0 * (g["ok"].sum() / len(g)) if len(g) else 0.0,
               "no_data_pct": 100.0 * (g["no_data"].sum() / len(g)) if len(g) else 0.0,
               "error_pct": 100.0 * ((len(g) - g["ok"].sum() - g["no_data"].sum()) / len(g)) if len(g) else 0.0,
               "median_latency_s": g["elapsed_s"].median(),
               "p95_latency_s": g["elapsed_s"].quantile(0.95),
               "median_size": _fmt_bytes(g["size_bytes"].median()),
               "median_rss_delta": _fmt_bytes(g["rss_delta"].median()),
           }), include_groups=False)
           .reset_index()
           .sort_values(["z", "timestep_index"])
    )
    return summary


__all__ = [
    "benchmark_viewport",
    "benchmark_tileset",
    "benchmark_statistics",
    "check_titiler_cmr_compatibility",
    "tiling_benchmark_summary",
    "TiTilerCMRBenchmarker",
]


if __name__ == "__main__":

    async def main():
        """Example usage of the unified TiTiler-CMR benchmarking system."""
        endpoint = "https://staging.openveda.cloud/api/titiler-cmr"

        ds_xarray = DatasetParams.for_xarray(
            concept_id="C2723754864-GES_DISC",
            datetime_range="2022-03-01T00:00:01Z/2022-03-01T23:59:59Z",
            variable="precipitation",
            step="P1D",
            temporal_mode="point",
        )

        ds_hls = DatasetParams.for_rasterio(
            concept_id="C2036881735-POCLOUD",
            datetime_range="2024-10-01T00:00:01Z/2024-10-10T00:00:01Z",
            bands=["B04", "B03", "B02"],
            bands_regex="B[0-9][0-9]",
            step="P1W",
            temporal_mode="point",
        )

        print("=== Example 1: Compatibility Check ===")
        compat = await check_titiler_cmr_compatibility(
            endpoint=endpoint,
            dataset=ds_xarray,
            timeout_s=1000.0,
        )
        print(f"Compatibility: {compat['compatibility']}")
        print(f"Timesteps: {compat['n_timesteps']}")
        print(f"Bounds: {compat['tilejson_bounds']}")
        if not compat["statistics"].empty:
            print(f"Statistics preview:\n{compat['statistics']}")

        print("\n=== Example 2: Viewport Tile Benchmarking ===")
        if compat["compatibility"] == "compatible":
            df_viewport = await benchmark_viewport(
                endpoint=endpoint,
                dataset=ds_xarray,
                lng=-95.0,
                lat=29.0,
                viewport_width=3,
                viewport_height=3,
                min_zoom=7,
                max_zoom=8,
                timeout_s=60.0,
            )
            print(f"Viewport results: {len(df_viewport)} tile requests")
            print(df_viewport.head())
            tiling_benchmark_summary(df_viewport)

        print("\n=== Example 3: Tileset Tile Benchmarking ===")
        gulf_bounds = [-98.676, 18.857, -81.623, 31.097]
        df_tileset = await benchmark_tileset(
            endpoint=endpoint,
            dataset=ds_hls,
            bounds=gulf_bounds,
            max_tiles_per_zoom=25,
            min_zoom=7,
            max_zoom=7,
            timeout_s=60.0,
        )
        print(f"Tileset results: {len(df_tileset)} tile requests")
        tiling_benchmark_summary(df_tileset)

        print("\n=== Example 4: Statistics Benchmarking ===")
        gulf_geometry = create_bbox_feature(-98.676, 18.857, -81.623, 31.097)
        stats_result = await benchmark_statistics(
            endpoint=endpoint,
            dataset=ds_xarray,
            geometry=gulf_geometry,
            timeout_s=300.0,
        )
        print("Statistics result:")
        print(f"  Success: {stats_result['success']}")
        print(f"  Elapsed: {stats_result['elapsed_s']:.2f}s")
        print(f"  Timesteps: {stats_result['n_timesteps']}")
        print(f"  Statistics: {stats_result['statistics']}")

        print("\n=== Example 5: Using Main Benchmarker Class ===")
        benchmarker = TiTilerCMRBenchmarker(
            endpoint=endpoint,
            min_zoom=7,
            max_zoom=8,
            timeout_s=120.0,
            max_connections=5,
        )

        def custom_viewport_strategy(zoom, tms, tilejson_info):
            center = tms.tile(lng=-92.1, lat=46.8, zoom=zoom)
            return get_surrounding_tiles(
                center_x=center.x, center_y=center.y, zoom=zoom, width=2, height=2
            )

        df_custom = await benchmarker.benchmark_tiles(ds_xarray, custom_viewport_strategy)
        print(f"Custom viewport results: {len(df_custom)} tile requests")

        stats_custom = await benchmarker.benchmark_statistics(dataset=ds_xarray, geometry=None)
        print(f"Custom statistics result: {stats_custom['success']}, {stats_custom['n_timesteps']} timesteps")

        compat_custom = await benchmarker.check_compatibility(dataset=ds_xarray)
        print(f"Custom compatibility: {compat_custom['compatibility']}")

    asyncio.run(main())
