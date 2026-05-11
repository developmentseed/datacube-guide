# xpublish-tiles

A FastAPI-based xpublish plugin that turns xarray datasets into raster and vector tiles, with a focus on the unstructured and curvilinear grids used in operational ocean and atmospheric models. Maintained by Earth Mover under the Apache 2.0 license.

## At a glance

- **Repo** — [earth-mover/xpublish-tiles](https://github.com/earth-mover/xpublish-tiles)
- **Shape** — Server-side library, a pair of xpublish plugins (`TilesPlugin`, `WMSPlugin`)
- **Web framework** — FastAPI, mounted via xpublish
- **Render engine** — [Datashader](https://datashader.org/) with Numba JIT compilation, plus a custom pyproj-based reprojection pipeline (no GDAL or rasterio)
- **Zarr versions** — v2 and v3 (`zarr>=3.1.2`); also reads anything xarray can open (NetCDF, Icechunk)
- **Conventions** — CF-xarray for axis detection, CF time decoding, multiple `grid_mapping` types

## What it does

xpublish-tiles takes an xarray dataset (regular, curvilinear, triangular, or HEALPix) and serves OGC Tiles, OGC WMS, and (newer) vector tiles directly from it, without a pre-rendering step. Its design centre is the operational geoscience use case: ocean models on ROMS, FVCOM, and SELFE grids; HEALPix and cubed-sphere from climate models; HRRR and RTOFS on 2D non-dimensional grids. These are precisely the grid topologies that GDAL-based tile servers either don't handle or handle awkwardly.

The plugin lives inside the xpublish ecosystem; see the [Xpublish ecosystem overview](overview.md) for the broader picture, and the [Dynamic tiling ecosystem comparison](../ecosystem-comparison.md) for how it stacks up against TiTiler.

## Architecture

Two plugins ship in the same repository and are loaded independently.

- **`TilesPlugin`** — OGC API Tiles 1.0 endpoints. Returns raster (PNG, JPEG) and, more recently, vector tiles (MVT, GeoJSON).
- **`WMSPlugin`** — OGC WMS 1.1.1 / 1.3.0 endpoints. Marked as still in active development; not yet at parity with the Tiles plugin or with the older [xpublish-wms](https://github.com/xpublish-community/xpublish-wms).

The rendering pipeline is:

1. **Coordinate transform.** A custom pyproj-based implementation, with a separable specialization for EPSG:4326→3857 that preserves the input grid structure (no per-pixel warp) and a thread-pool blocked transform for general cases. Tunable via `TRANSFORM_CHUNK_SIZE` and the thread-pool size.
2. **Resampling and aggregation.** Datashader chooses an engine based on grid type: a fast rectilinear path (3–10× faster than the alternatives), `quadmesh` for curvilinear grids, and `trimesh` with Delaunay triangulation for unstructured / UGRID. `numbagg` handles nearest-neighbour aggregation for discrete data.
3. **Styling.** Continuous data uses `colorscalerange` plus a colormap LUT; categorical data uses `flag_values` / `flag_meanings` / `flag_colors`; `abovemaxcolor` and `belowmincolor` parameters colour out-of-range values explicitly rather than clamping.
4. **Encoding.** Pillow for raster output; protocol buffers for MVT vector tiles.

Async loading uses `xarray.load_async()`, configurable via the `async_load` setting. Caching keys are built from a required `_xpublish_id` dataset attribute plus the requested dimension and variable.

## Supported grid systems

xpublish-tiles' grid-system abstraction is its differentiator. The currently supported systems:

- **Regular lat/lon** — RasterIndex
- **Curvilinear** — ROMS-family models including CBOFS, DBOFS, TBOFS
- **FVCOM triangular** — LOOFS, LSOFS, LMHOFS, NGOFS2 and similar
- **SELFE unstructured** — CREOFS
- **2D non-dimensional rectilinear** — RTOFS, HRRR-CONUS
- **Polar** — radar and satellite products in polar grids
- **Cubed-sphere** — faceted grid system (`FacetedGridSystem`) for cubed-sphere climate-model output
- **HEALPix** — cell-index coordinates with nested indexing (currently restricted to the `polygons` style)

The grid system is detected from CF metadata where possible; custom grid systems can be registered.

## Tile API surface

Selected request parameters; see the in-repo OpenAPI docs for the full surface.

| Category | Parameter | Notes |
|---|---|---|
| Variable | `variables` | List, required |
| Dimension selection | `dimension={method}::{value}` | DSL with `nearest`, `ffill`/`pad`, `bfill`/`backfill`, `exact` |
| Style | `style={type}/{colormap}` | Continuous, categorical, vector subtypes |
| Color range | `colorscalerange` | Tuple `(min, max)` |
| Out-of-range | `abovemaxcolor`, `belowmincolor` | Explicit colours for clipped values |
| Custom colormap | JSON | `0–255` → hex mapping |
| Image dimensions | `width`, `height` | Multiples of 256 |
| Image format | `f` | `image/png`, `image/jpeg`, MVT, GeoJSON |
| Error handling | `render_errors` | Boolean; render errors as image tiles vs HTTP 5xx |

Notable endpoints beyond the standard OGC Tiles set:

- **Legend endpoint** (`/tiles/legend`) — returns a rendered legend image or JSON colour-stop description for the requested style.
- **Vector tiles** — `vector/cells`, `vector/points`, `vector/contours` styles emit MVT or GeoJSON with cell geometries, suitable for use as the data layer of a MapLibre/Mapbox vector style.

## Examples

The same endpoint shape covers raster output, vector output, and the legend; the `style` query parameter (and `f` for explicit format negotiation) selects which.

=== "Raster tile (PNG)"

    ```http
    GET /tiles/WebMercatorQuad/4/8/5
        ?variables=temperature
        &style=raster/viridis
        &colorscalerange=270,310
        &width=512
        &height=512
    ```

    Returns `image/png`. The `raster/{colormap}` style is the standard continuous-data path; swap the colormap suffix for any of Datashader's named maps.

=== "Vector tile (MVT)"

    ```http
    GET /tiles/WebMercatorQuad/4/8/5
        ?variables=temperature
        &style=vector/cells
        &f=application/vnd.mapbox-vector-tile
    ```

    Returns Mapbox Vector Tiles with cell-polygon geometries and the variable as a property. Suitable as the data source for a MapLibre or Mapbox vector style. `vector/points` and `vector/contours` are the other vector styles.

=== "Legend"

    ```http
    GET /tiles/legend
        ?variables=temperature
        &style=raster/viridis
        &colorscalerange=270,310
    ```

    Returns a rendered legend image matching the style above. Add `f=application/json` to get the colour stops as structured data instead, useful for rendering a custom legend in a frontend rather than using the prebaked image.

A few request shapes worth knowing:

- `dimension={method}::{value}` is the dimension-selection DSL (e.g. `time=nearest::2024-06-15T12:00:00Z`).
- `abovemaxcolor` and `belowmincolor` colour out-of-range values explicitly rather than clamping them, useful for highlighting outliers.
- For categorical data, use `style=raster/categorical` and let the dataset's `flag_values` / `flag_colors` drive the rendering.

## Where it fits

Choose xpublish-tiles when the data lives on a non-rectilinear grid that the climate or operational-oceanography community uses (ROMS curvilinear, FVCOM triangular, SELFE, HEALPix, cubed-sphere) and you need a server that respects that topology rather than regridding upstream. It is also a strong fit when the wider stack is xpublish-based (xpublish-edr for queries, xpublish-opendap, etc.) because everything mounts on the same FastAPI app over the same dataset accessor.

The trade-off versus TiTiler is breadth versus depth: TiTiler has the broader format ecosystem (COG, STAC, NASA CMR), wider deployment story (Lambda, ECS), Redis caching, and band-math expressions; xpublish-tiles has the unstructured-grid renderers, CF integration, and async chunked loading that operational geoscience needs. The two ecosystems are increasingly complementary rather than overlapping; see the [comparison](../ecosystem-comparison.md) for details.

## Configuration knobs

Set via environment variables or a [donfig](https://github.com/pytroll/donfig) config file:

- `NUMBA_NUM_THREADS` — Datashader/Numba thread count for JIT rendering.
- `TRANSFORM_CHUNK_SIZE` — chunk size (NxN) for blocked coordinate transforms.
- Thread-pool size — for the chunked transform pipeline.
- `async_load` — toggles `xarray.load_async()`.

Worth knowing: first request after startup is slow due to Numba JIT compilation. Warm up the renderers in production deployments rather than serving the first user request cold.

## Links

- Source: [earth-mover/xpublish-tiles](https://github.com/earth-mover/xpublish-tiles)
- Parent project: [xpublish](https://github.com/xpublish-community/xpublish)
- Dependencies: [Datashader](https://datashader.org/), [pyproj](https://pyproj4.github.io/pyproj/), [cf-xarray](https://github.com/xarray-contrib/cf-xarray), [morecantile](https://github.com/developmentseed/morecantile), [numbagg](https://github.com/numbagg/numbagg)
- See also: [xpublish ecosystem overview](overview.md), [TiTiler vs Xpublish comparison](../ecosystem-comparison.md)
