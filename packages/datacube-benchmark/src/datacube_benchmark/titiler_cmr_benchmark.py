"""
Unified benchmarking system for TiTiler-CMR.

This module provides a clean, extensible architecture for benchmarking
TiTiler-CMR performance across different scenarios (viewport, full tileset, statistics).
"""

from __future__ import annotations

import asyncio
import time
import psutil
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, Union

import httpx
import morecantile
import pandas as pd
from geojson_pydantic import Feature, Polygon

from tiling import get_surrounding_tiles, get_tileset_tiles, fetch_tile, create_bbox_feature, BaseBenchmarker, DatasetParams

# Unified benchmarker that handles both tiles timeseries and statistics endpoints
class TiTilerCMRBenchmarker(BaseBenchmarker):
    """
    Main benchmarking orchestrator for TiTiler-CMR.

    Supports benchmarking of tile rendering and statistics endpoints
    across different strategies (viewport, tileset, custom).
    """
    
    def __init__(
        self,
        endpoint: str,
        *,
        # Tile-specific parameters
        tms_id: str = "WebMercatorQuad",
        tile_format: str = "png",
        tile_scale: int = 1,
        min_zoom: int = 7,
        max_zoom: int = 10,
        # Base parameters
        **base_kwargs: Any
    ):
        super().__init__(endpoint, **base_kwargs)
        # Tileset-specific config
        self.tms_id = tms_id
        self.tile_format = tile_format
        self.tile_scale = tile_scale
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

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
        response = None
        payload = None

        # Build query parameters
        params = dict(dataset.to_query_params(**extra_params))

        # Prepare geometry payload
        if hasattr(geometry, 'model_dump'):
            payload = geometry.model_dump(exclude_none=True)
        elif isinstance(geometry, dict):
            payload = geometry
        else:
            raise ValueError("geometry must be a GeoJSON Feature or dict")

        t0 = time.perf_counter()
        try:
            response = await client.post(
                url,
                params=params,
                json=payload,
                timeout=self.timeout_s
            )

            elapsed = time.perf_counter() - t0

            response.raise_for_status()
            result = response.json()

            stats = result.get("properties", {}).get("statistics", {}) if isinstance(result, dict) else {}

            return {
                "success": True,
                "elapsed_s": elapsed,
                "status_code": response.status_code,
                "n_timesteps": len(stats) if isinstance(stats, dict) else 0,
                "url": url,
                "statistics": stats,
                "error": None,
            }

        except Exception as ex:
            elapsed = time.perf_counter() - t0
            return {
                "success": False,
                "elapsed_s": elapsed,
                "status_code": getattr(response, "status_code", None),
                "n_timesteps": 0,
                "url": url,
                "statistics": {},
                "error": f"{type(ex).__name__}: {ex}",
            }

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
        print(f"=== TiTiler-CMR Statistics Benchmark ===")
        print(f"System: {self._system_info}")
        print(f"Dataset: {dataset.concept_id} ({dataset.backend})")
        
        async with self._create_http_client() as client:
            if geometry is None:
                tile_params = list(dataset.to_query_params(**kwargs))
                tilejson_info = await self._get_tilejson_info(client, tile_params)
                bounds = tilejson_info.get("bounds")
                if not bounds:
                    raise ValueError("No geometry provided and no bounds available from TileJSON")
                geometry = create_bbox_feature(*bounds)
                print(f"Using bounds from TileJSON: {bounds}")
            else:
                print("Using provided geometry")

            result = await self._fetch_statistics(
                client=client,
                dataset=dataset,
                geometry=geometry,
                **kwargs
            )
        
        return result
    
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
        print("=== TiTiler-CMR Tile Benchmark ===")
        print(f"System: {self._system_info}")
        print(f"Dataset: {dataset.concept_id} ({dataset.backend})")
        
        # Build query parameters
        params = list(dataset.to_query_params(
            tile_format=self.tile_format,
            tile_scale=self.tile_scale,
            **kwargs
        ))
        
        print(f"Query params: {len(params)} parameters")
        for k, v in params:
            print(f"  {k}: {v}")
        
        async with self._create_http_client() as client:
            tilejson_info = await self._get_tilejson_info(client, params)
            tiles_endpoints = tilejson_info["tiles_endpoints"]
            n_timesteps = len(tiles_endpoints) 

            print(f"Found {n_timesteps} timesteps from TileJson.")
            
            tasks = await self._build_tile_tasks(
                client, tiles_endpoints, tiling_strategy, tilejson_info
            )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        df = self._process_results(results)
        return df
    
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
        print(f"=== TiTiler-CMR Compatibility Check ===")
        print(f"Dataset: {dataset.concept_id} ({dataset.backend})")
        
        async with self._create_http_client() as client:
            # Get TileJSON info for granules/timesteps
            tile_params = list(dataset.to_query_params(**kwargs))
            tilejson_info = await self._get_tilejson_info(client, tile_params)
            tiles_endpoints = tilejson_info["tiles_endpoints"]
            n_timesteps = len(tiles_endpoints)
            
            print(f"Found {n_timesteps} timesteps/granules from TileJson!")
            
            # Get geometry for statistics
            if geometry is None:
                bounds = tilejson_info.get("bounds")
                if not bounds:
                    raise ValueError("No geometry provided and no bounds available from TileJSON")
                geometry = create_bbox_feature(*bounds)
                print(f"Using bounds from TileJSON: {bounds}")
            
            # Run statistics for preview (always with timing)
            stats_result = await self._fetch_statistics(
                client=client,
                dataset=dataset,
                geometry=geometry,
                **kwargs
            )
            
            if stats_result["success"] and stats_result["statistics"]:
                stats = stats_result["statistics"]
                print(f"Statistics returned {len(stats)} timesteps")
                
            else:
                stats_preview = {}
                print("Statistics request failed:", stats_result.get("error"))
            
            return {
                "concept_id": dataset.concept_id,
                "backend": dataset.backend,
                "n_timesteps": n_timesteps,
                "tilejson_bounds": tilejson_info.get("bounds"),
                "statistics": stats_result["statistics"] if stats_result["success"] else {},
                "compatibility": "compatible" if n_timesteps > 0 and stats_result["success"] else "issues_detected"
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
        tasks = []
        
        for zoom in range(self.min_zoom, self.max_zoom + 1):
            tiles = tiling_strategy(zoom, tms, tilejson_info)
            # Calculate unique x and y values for display
            nx = len({x for x, _ in tiles})
            ny = len({y for _, y in tiles})
            print(f"Zoom {zoom}: {nx}x{ny} tiles ({len(tiles)} total)") 
            
            for x, y in tiles:
                task = asyncio.create_task(
                    fetch_tile(
                        client,
                        tiles_endpoints=tiles_endpoints,
                        z=zoom, x=x, y=y,
                        timeout_s=self.timeout_s,
                        proc=proc
                    )
                )
                tasks.append(task)
        
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
        resp = await client.get(url, params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        
        ts_json = resp.json()
        entries = []
        
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
        
        tiles_endpoints = [
            tile for entry in entries 
            for tile in entry.get("tiles", [])
        ]
        
        if not tiles_endpoints:
            raise RuntimeError("No tile endpoints found in TileJSON response")
        
        bounds = entries[0].get("bounds") if entries else None
        
        return {
            "entries": entries,
            "tilejson": ts_json,
            "tiles_endpoints": tiles_endpoints,
            "bounds": bounds,
        }


# Unified analysis utilities
class BenchmarkAnalyzer:
    """
    Utilities for analyzing tile and statistics benchmark results.

    Provides summary statistics and formatted output for tile benchmarks.
    """
    
    @staticmethod
    def create_tile_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics by zoom level and timestep for tile results.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of tile benchmark results.

        Returns
        -------
        pd.DataFrame
            Summary statistics by zoom and timestep.
        """
        if df.empty or "z" not in df.columns:
            return pd.DataFrame()
        
        # Convert columns to numeric
        numeric_cols = ["elapsed_s", "size_bytes", "rss_delta"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        summary = (
            df.groupby(["z", "timestep_index"])
            .apply(lambda g: pd.Series({
                "n_tiles": len(g),
                "ok_pct": 100.0 * g["ok"].sum() / len(g) if len(g) else 0.0,
                "no_data_pct": 100.0 * g["no_data"].sum() / len(g) if len(g) else 0.0,
                "error_pct": 100.0 * (len(g) - g["ok"].sum() - g["no_data"].sum()) / len(g) if len(g) else 0.0,
                "median_latency_s": g["elapsed_s"].median(),
                "p95_latency_s": g["elapsed_s"].quantile(0.95),
                "median_size_bytes": g["size_bytes"].median(),
                "median_rss_delta": g["rss_delta"].median(),
            }), include_groups=False)
            .reset_index()
            .sort_values(["z", "timestep_index"])
        )
        
        return summary
    
    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        """
        Print formatted summary for tile benchmark results.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of benchmark results.
        """
        if df.empty:
            print("No results to summarize")
            return
        
        if "z" in df.columns and "x" in df.columns:
            # Tile results
            summary = BenchmarkAnalyzer.create_tile_summary(df)
            print("\n=== Tile Benchmark Summary ===")
            print(summary.to_string(index=False))
        else:
            print("Unknown result format")

    @staticmethod
    def statistics_to_dataframe(stats: Dict[str, Any]) -> pd.DataFrame:
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
                    continue  # drop histograms entirely
                row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)

        # Convert numeric where possible (leave timestamp as-is)
        non_numeric = {"timestamp"}
        for col in df.columns:
            if col not in non_numeric:
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # Sort by timestamp if we can parse it
        if "timestamp" in df.columns:
            try:
                df = (
                    df.assign(_ts=pd.to_datetime(df["timestamp"], utc=True))
                      .sort_values("_ts")
                      .drop(columns="_ts")
                )
            except Exception:
                df = df.sort_values("timestamp")

        return df.reset_index(drop=True)


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
    """Convenience function for viewport tile benchmarking."""
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
        # compute the center tile from lng/lat, then enumerate a WxH neighborhood
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
    """Convenience function for full tileset benchmarking."""
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
    """Convenience function for statistics benchmarking."""
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
    preview_limit: int = 2,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Check TiTiler-CMR compatibility and get dataset overview.
    
    Parameters
    ----------
    endpoint : str
        TiTiler-CMR endpoint URL
    dataset : DatasetParams
        Dataset configuration
    geometry : Union[Feature, Dict[str, Any]], optional
        GeoJSON Feature or geometry. If None, uses bounds from tilejson
    timeout_s : float, optional
        Request timeout, by default 300.0
    preview_limit : int, optional
        Number of timesteps to show in statistics preview, by default 2
    **kwargs : Any
        Additional query parameters

    Returns
    -------
    Dict[str, Any]
        Compatibility information including timestep count, granule info, and statistics preview
    """
    benchmarker = TiTilerCMRBenchmarker(endpoint=endpoint, timeout_s=timeout_s)
    return await benchmarker.check_compatibility(dataset, geometry,  **kwargs)


# Main example
if __name__ == "__main__":
    
    async def main():
        """Example usage of the unified TiTiler-CMR benchmarking system."""
        
        # Configuration
        endpoint = "https://staging.openveda.cloud/api/titiler-cmr"
        
        # Dataset 1: Xarray Backend (Sea Surface Temperature)
        ds_xarray = DatasetParams.for_xarray(
            concept_id="C2723754864-GES_DISC",
            datetime_range="2022-03-01T00:00:01Z/2022-03-02T23:59:59Z",
            variable="precipitation",
            step="P1D",
            temporal_mode="point"
        )
        
        # Dataset 2: Rasterio Backend (HLS)
        ds_hls = DatasetParams.for_rasterio(
            concept_id="C2036881735-POCLOUD",
            datetime_range="2024-10-01T00:00:01Z/2024-10-10T00:00:01Z",
            bands=["B04", "B03", "B02"],
            bands_regex="B[0-9][0-9]",
            step="P1W",
            temporal_mode="point"
        )
        
        print("=== Example 1: Compatibility Check ===")
        
        # Check compatibility first
        compat = await check_titiler_cmr_compatibility(
            endpoint=endpoint,
            dataset=ds_xarray,
            timeout_s=1000.0,
            preview_limit=2
        )
        
        print(f"Compatibility: {compat['compatibility']}")
        print(f"Timesteps: {compat['n_timesteps']}")
        print(f"Bounds: {compat['tilejson_bounds']}")
        df_stats = BenchmarkAnalyzer.statistics_to_dataframe(compat.get("statistics", {}))
        print(df_stats.head().to_string(index=False))
        
        print("\n=== Example 2: Viewport Tile Benchmarking ===")
        
        if compat['compatibility'] == 'compatible':
            # Run viewport benchmark
            df_viewport = await benchmark_viewport(
                endpoint=endpoint,
                dataset=ds_xarray,
                lng=-95.0,
                lat=29.0,
                viewport_width=3,
                viewport_height=3,
                min_zoom=7,
                max_zoom=8,
                timeout_s=60.0
            )
            
            print(f"Viewport results: {len(df_viewport)} tile requests")
            BenchmarkAnalyzer.print_summary(df_viewport)
        
        print("\n=== Example 3: Tileset Tile Benchmarking ===")
        
        # Run tileset benchmark with bounds
        gulf_bounds = [-98.676, 18.857, -81.623, 31.097]
        df_tileset = await benchmark_tileset(
            endpoint=endpoint,
            dataset=ds_hls,
            bounds=gulf_bounds,
            max_tiles_per_zoom=25,  # Keep small for demo
            min_zoom=7,
            max_zoom=7,  # Single zoom level
            timeout_s=60.0
        )
        
        print(f"Tileset results: {len(df_tileset)} tile requests")
        BenchmarkAnalyzer.print_summary(df_tileset)
        
        print("\n=== Example 4: Statistics Benchmarking ===")
        
        # Create Gulf of Mexico geometry (your example)
        gulf_geometry = create_bbox_feature(-98.676, 18.857, -81.623, 31.097)
        
        # Run statistics benchmark with timing
        stats_result = await benchmark_statistics(
            endpoint=endpoint,
            dataset=ds_xarray,
            geometry=gulf_geometry,
            timeout_s=300.0
        )
        
        print(f"Statistics result:")
        print(f"  Success: {stats_result['success']}")
        print(f"  Elapsed: {stats_result['elapsed_s']:.2f}s")
        print(f"  Timesteps: {stats_result['n_timesteps']}")
        print(f" Statistics: {stats_result['statistics']}")
        

        
        print("\n=== Example 5: Using Main Benchmarker Class ===")
        
        # Create benchmarker instance for more control
        benchmarker = TiTilerCMRBenchmarker(
            endpoint=endpoint,
            min_zoom=7,
            max_zoom=8,
            timeout_s=120.0,
            max_connections=5  # Conservative for demo
        )
        
        # Custom viewport strategy
        def custom_viewport_strategy(zoom, tms, tilejson_info):
            center = tms.tile(lng=-92.1, lat=46.8, zoom=zoom)
            return get_surrounding_tiles(center_x=center.x, center_y=center.y, zoom=zoom, width=2, height=2)
        
        # Run tile benchmark
        df_custom = await benchmarker.benchmark_tiles(ds_xarray, custom_viewport_strategy)
        print(f"Custom viewport results: {len(df_custom)} tile requests")
        
        # Run statistics benchmark with same instance
        stats_custom = await benchmarker.benchmark_statistics(
            dataset=ds_xarray,
            geometry=None  # Will use bounds from tilejson
        )
        print(f"Custom statistics result: {stats_custom['success']}, {stats_custom['n_timesteps']} timesteps")
        
        # Run compatibility check
        compat_custom = await benchmarker.check_compatibility(
            dataset=ds_xarray,
        )
        print(f"Custom compatibility: {compat_custom['compatibility']}")
        
    
    # Run the examples
    asyncio.run(main())