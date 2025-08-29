"""
tiling_utils
============

Reusable helper functions for working with map tiles, that are independent of
any specific benchmarking or rendering workflow. They are primarily tested
to support TiTiler-CMR benchmarking, but can also be applied in other
contexts where XYZ tile math and asynchronous tile fetching are needed.

Functions
---------
- get_surrounding_tiles:
    Compute the list of (x, y) tile coordinates forming a rectangular
    viewport centered on a given tile index at a specific zoom level.

- fetch_tile:
    Asynchronously fetch one or more tiles from a set of TileJSON
    templates for a given (z, x, y). Returns structured metadata for
    each request, including status code, latency, response size,
    and (optionally) memory usage deltas.

"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


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

     Parameters
    ----------
        client : httpx.AsyncClient
            The HTTP client to use for requests.
        tiles_endpoints : list of str
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
        - ``timestep_index`` (int): index of the template within ``tiles_endpoints``.
        - ``url`` (str): fully formatted request URL.
        - ``status_code`` (int or None): HTTP status code; ``None`` if the request failed.
        - ``elapsed_s`` (float): wall-clock time for the request (seconds).
        - ``size_bytes`` (int): response body size in bytes (``0`` if no content or on failure).
        - ``content_type`` (str or None): value of the ``Content-Type`` header, if present.
        - ``ok`` (bool): ``True`` if ``status_code == 200``.
        - ``no_data`` (bool): ``True`` iff ``status_code == 204``.
        - ``error_text`` (str or None): short error snippet for non-2xx responses or exceptions.
        - ``rss_delta`` (int or float): ``rss_after - rss_before`` (bytes; ``NaN`` on error).


    """
    rows: List[Dict[str, Any]] = []

    for i, tmpl in enumerate(tiles_endpoints):
        tile_url = tmpl.format(z=z, x=x, y=y)
        try:
            rss_before = proc.memory_info().rss if proc is not None else 0
            t0 = time.perf_counter()
            response = await client.get(tile_url, timeout=timeout_s)
            response.elapsed = time.perf_counter() - t0
            rss_after = proc.memory_info().rss if proc is not None else 0
            rss_delta = rss_after - rss_before

            status = response.status_code
            ctype = response.headers.get("content-type")
            size = len(response.content) if response.content is not None else 0

            if status >= 400:
                try:
                    print(f"[{i}] Error body: {response.text[:400]}")
                except Exception:
                    pass

            if proc is not None:
                print(
                    f"[{i}] Tile {z}/{x}/{y} | RSS before: {rss_before:,} "
                    f"after: {rss_after:,} delta: {rss_delta:+,} bytes"
                )

            rows.append(
                {
                    "z": z,
                    "x": x,
                    "y": y,
                    "timestep_index": i,
                    "url": tile_url,
                    "response": response, 
                    "status_code": status,
                    "elapsed_s": response.elapsed,
                    "size_bytes": size,
                    "content_type": ctype,
                    "ok": (status == 200),
                    "no_data": (status == 204),
                    "error_text": None if status < 400 else response.text[:400],
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
                    "response": response,
                    "status_code": None,
                    "elapsed_s": float("nan"),
                    "size_bytes": 0,
                    "content_type": None,
                    "ok": False,
                    "no_data": False,
                    "error_text": f"{type(ex).__name__}: {ex}",
                    "rss_delta": float("nan"),
                }
            )

    return rows

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