# Dynamic tiling ecosystem comparison

This page compares the two main FastAPI-based ecosystems for serving Zarr-backed datacubes as tiles: **TiTiler** (Development Seed) and **Xpublish** with its plugins (UCAR, Earth Mover, and the wider xarray community). For per-ecosystem detail see the [Titiler ecosystem overview](titiler/overview.md) and the [Xpublish ecosystem overview](xpublish/overview.md). For the client-side equivalent (visualization libraries that bypass a tile server entirely), see the [client-side rendering comparison](client-side-comparison.md).

## Summary

| | TiTiler | Xpublish |
|---|---|---|
| Maintainer | Development Seed (Vincent Sarago, Aimee Barciauskas) | xarray community (Joe Hamman, Alex Kerney) + UCAR governance; `xpublish-tiles` by Earth Mover |
| Shape | Layered Python stack: `rio-tiler` → `titiler.core` → `titiler.xarray` → applications (`titiler.application`, `titiler-cmr`, `titiler-multidim`, `titiler-eopf`) | Small `xpublish` core with independent plugins (`xpublish-tiles`, `xpublish-wms`, `xpublish-edr`, `opendap-protocol`) |
| Rendering engine | `rio-tiler` + GDAL/rasterio | Datashader (Numba-JIT) for raster, trimesh for unstructured |
| Primary inputs | COG, STAC; Zarr/NetCDF/Icechunk via `titiler.xarray` | Xarray-native: Zarr (primary), NetCDF, Icechunk |
| Grid topologies | Regular and projected lat/lon are first-class; curvilinear limited; unstructured out of scope | Regular, curvilinear, FVCOM triangular, SELFE, 2D non-dimensional, HEALPix, cubed-sphere, polar |
| Reprojection | GDAL warp (full kernel set) | Custom pyproj: separable 4326→3857 fast path, blocked thread-pool transform for general CRS |
| Tile endpoints | XYZ, WMTS, TileJSON, POST `/statistics`; no vector tiles | XYZ, WMTS, OGC Tiles 1.0, full WMS, MVT/GeoJSON vector tiles, OGC EDR, OPeNDAP |
| Conventions | CF via rioxarray (basic), STAC native | CF via `cf-xarray` (full), `flag_values`/`flag_meanings`/`flag_colors` for categorical, no STAC |
| Caching | Redis (dataset-level) | Plugin-configurable; `xpublish-tiles` keys on `_xpublish_id` + dim + variable |
| First request | Fast | Slow (Numba JIT; warm-up required) |
| Deployment | Official Docker images, AWS Lambda/ECS CDK examples | No official images; deployment is hands-on |
| License | MIT | Apache 2.0 |

## Project framing

**TiTiler** is a layered library stack with `rio-tiler` at the foundation, a generic FastAPI factory in `titiler.core`, and a small set of opinionated applications: `titiler.application` (reference), `titiler-cmr` (NASA CMR), `titiler-multidim` (VEDA), `titiler-eopf` (ESA Copernicus). Xarray-backed Zarr/NetCDF support lives in `titiler.xarray` and has been exposed by the reference application under `/zarr/*` since 2.0. The history is COG-first — `rio-tiler` and `titiler.core` predate the Zarr work — and the resulting strengths are GDAL-native raster handling, STAC integration, and a deep set of cloud-deploy recipes.

**Xpublish** is a small core (plugin extension points and an `xpublish` xarray accessor) with most user-facing capabilities in independent plugins. The design centre is xarray-native scientific data, with serious investment in irregular grid topologies (curvilinear ROMS, triangular FVCOM, SELFE, HEALPix, cubed-sphere). `xpublish-wms` predates `xpublish-tiles` and remains the more mature WMS path. Adoption is concentrated in operational geoscience — NOAA forecast systems (CBOFS/LOOFS/CREOFS) on the ROMS/FVCOM/SELFE side, climate models on the HEALPix/ICON side.

## Inputs and grid topologies

**TiTiler**'s grid support reflects GDAL's bias toward regular rectangular rasters. Regular lat/lon and any PROJ-defined projected CRS are first-class; curvilinear data works with caveats; unstructured meshes (FVCOM triangles, SELFE) are out of scope. Conversely, COG and STAC are native: `rio-tiler` was built to serve COGs, and STAC items work as a first-class asset-discovery layer — `titiler-cmr` and `titiler-eopf` lean heavily on this.

**Xpublish-tiles** treats unstructured and irregular grids as first-class. The regular case uses datashader's raster pipeline; curvilinear goes through datashader's quadmesh with an optional approximate-rectilinear fast path detected via a Numba-optimized 1-pixel threshold check; triangular meshes use datashader's trimesh with Delaunay triangulation. Additional systems exist for HEALPix (cell-index coordinates with nested indexing), cubed-sphere (`FacetedGridSystem`), and polar grids (for radar data). COG and STAC are not in scope; the assumption is that data is xarray-readable (Zarr, NetCDF, or another xarray backend).

## Rendering pipeline

**TiTiler** renders through `rio-tiler`'s GDAL stack: read a window, optionally warp, rescale, colormap, encode. The full GDAL resampling kernel set is exposed for both raster resampling and coordinate warping (nearest, bilinear, cubic, cubic spline, Lanczos, average, mode, max, min, med, q1, q3, sum, rms). Multi-band composites, band math via the `expression` parameter (e.g. `b1/b2` for NDVI), and algorithm extensions (hillshade, NDVI as a named algorithm) are supported. First-request latency is low; there is no JIT step.

**Xpublish-tiles** renders through datashader (Numba-JIT) for raster and trimesh paths, with a custom pyproj-based reprojection that has a separable optimization for the 4326→3857 hot path and a configurable thread-pool blocked transform (`TRANSFORM_CHUNK_SIZE`) for the general case. Categorical rendering is a first-class concern: `flag_values`/`flag_meanings`/`flag_colors` in CF metadata map directly to discrete colormaps, and out-of-range colours can be set explicitly via `abovemaxcolor`/`belowmincolor`. First request is slow because Numba JIT-compiles on first invocation; production deployments must warm up the rendering paths. The trade-off versus TiTiler is worse performance on the first cold tile, much better performance on curvilinear and triangular data — datashader raster is reportedly 3–10x faster than quadmesh for the rectilinear case, and trimesh is the only practical option for FVCOM-class meshes.

## Endpoint surface

**TiTiler** exposes XYZ raster tiles (`/tiles/{z}/{x}/{y}`), WMTS, TileJSON, `/info` and `/info.geojson`, and a POST `/statistics` endpoint that accepts a GeoJSON geometry. Vector tiles are not native. STAC asset discovery is integrated via `titiler-cmr` and `titiler-eopf`. Configuration grain is per-application: each of the four applications has its own router set and dependency wiring.

**Xpublish** spreads endpoints across plugins. `xpublish-tiles` provides XYZ tiles, OGC Tiles 1.0, a dedicated `/tiles/legend` endpoint (rendered image or JSON colour stops), and three vector-tile styles (`vector/cells`, `vector/points`, `vector/contours`) producing MVT or GeoJSON. `xpublish-wms` provides full WMS 1.1.1/1.3.0 — more mature than the tiles plugin's WMS path. `xpublish-edr` provides OGC EDR position, area, and cube queries with multiple output formats (NetCDF, Parquet, CSV, GeoJSON, GeoTIFF). `opendap-protocol` implements DAP 2.0. The mix-and-match story is the point: pick the plugins your clients need.

## Metadata and conventions

**TiTiler** handles CF through `rioxarray`: lat/lon/x/y auto-detection, basic grid-mapping support, `decode_times` as a parameter. The CF styling primitives (`flag_values`/`flag_meanings`/`flag_colors`) are not used directly. STAC item and asset metadata are native and are the integration story for `titiler-cmr` and `titiler-eopf`. CRS handling is rioxarray + pyproj; tile-matrix sets come from `morecantile`.

**Xpublish-tiles** uses `cf-xarray` for full CF support: axis detection (X/Y/Z/T), multiple grid_mapping handling, CF time parsing, vertical-coordinate detection, and bounds-aware cell representation. Categorical rendering reads `flag_values`/`flag_meanings`/`flag_colors` directly. STAC is not used. CRS handling is pyproj-direct via CF `grid_mapping_name`; tile-matrix sets come from `morecantile` through the plugin.

## Deployment and performance

**TiTiler** ships official Docker images and provides AWS CDK examples for Lambda and ECS. Deployment is stateless given an external cache; Redis is the documented dataset-cache option. Horizontal scaling is well-trodden. The critical tuning settings are Redis configuration and worker count. Image size carries GDAL overhead but is moderate.

**Xpublish-tiles** has no official Docker images and no Lambda/ECS recipes; deployment is hands-on. Dataset loading happens at startup (with `async_load` an option for large catalogs), JIT warm-up is required for the datashader pipelines (first request to each path is slow and unavoidable), and the critical tuning settings are `NUMBA_NUM_THREADS`, the coordinate-transform thread-pool size, and `TRANSFORM_CHUNK_SIZE`. Image size is larger (scientific Python stack) and CPU usage is higher (Numba JIT plus thread pools).

## Picking the right tool

- **COG and STAC** are the design centre, or you need a mix of raster formats from one stack: **TiTiler**.
- **Operational scientific data on irregular grids** (FVCOM, SELFE, ROMS curvilinear, HEALPix, ICON cubed-sphere): **Xpublish** with `xpublish-tiles` and/or `xpublish-wms`.
- **OGC EDR queries** (position/area/cube extraction, time-series, profiles): **Xpublish** with `xpublish-edr`.
- **OPeNDAP** clients: **Xpublish** with `opendap-protocol`.
- **Categorical raster styling** from CF `flag_values`/`flag_meanings`/`flag_colors`, vector tiles, or a legend endpoint: **Xpublish-tiles**.
- **NASA CMR**, **NASA VEDA**, or **ESA EOPF** data: **TiTiler** via `titiler-cmr`, `titiler-multidim`, or `titiler-eopf` respectively.

The two stacks compose. A common hybrid is TiTiler for public-facing slippy-map tiles (where its Redis cache and Lambda/ECS deploy story shine) and Xpublish-tiles/EDR for research-oriented access to the same datasets on their native grids.

## Related

- [Client-side rendering comparison](client-side-comparison.md): browser-side libraries (deck.gl-raster, `@carbonplan/maps`, zarr-layer, zarr-cesium) and viewer apps (Browzarr, GridLook) that read Zarr directly with no tile server.
- [Titiler ecosystem overview](titiler/overview.md), [Xpublish ecosystem overview](xpublish/overview.md): per-ecosystem detail.
- [Xpublish-tiles detail page](xpublish/xpublish-tiles.md).
