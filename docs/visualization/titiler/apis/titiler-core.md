# titiler.core API reference

`titiler.core` is the foundation of the TiTiler ecosystem: a FastAPI framework with factory patterns for building dynamic tile servers, plus the reference application that combines it with `titiler.mosaic` and (since 2.0) `/zarr/*` endpoints from `titiler.xarray`. Currently at v2.0.x, requires `rio-tiler>=9,<10` and Python 3.11–3.14.

## Key features

- **COG support** — Cloud Optimized GeoTIFF processing via `rio-tiler`.
- **STAC integration** — full SpatioTemporal Asset Catalog support, including remote item URLs.
- **Zarr endpoints** — `/zarr/*` is now part of the default reference application via `titiler.xarray`.
- **OGC compliance** — XYZ, WMTS, and partial OGC Maps API support.
- **Extensible architecture** — factory pattern that the other applications (`titiler-cmr`, `titiler-multidim`, `titiler-eopf`) all build on.

## Notable changes in 2.0

- `256x256` is no longer the implicit default tile size; `/tilejson.json` defaults to `tilesize=512` and `/map.html` to `tilesize=256`, and `tilesize` is an optional query parameter on tile and tilejson endpoints.
- `MultiBandTilerFactory` removed. Multi-band selection is handled through the standard factory.
- The `@{scale}x` suffix on tile endpoints removed; use `tilesize` instead.
- Deprecated parsers `AssetsBidxParams`, `BandsParams`, and related dependency helpers removed.
- WMTS endpoints expose tile dependencies in OpenAPI; performance improvements landed in 2.0.1.

## Installation

The bare `titiler` PyPI package was dropped in late 2025. Install subpackages directly:

```bash
pip install titiler.core titiler.mosaic titiler.xarray
# or, for the bundled reference application:
pip install titiler.application
```

## Interactive API documentation

The complete, interactive API documentation from the public demo deployment is below. Please be kind with this API.

<iframe src="https://titiler.xyz/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick links

- [Open API docs in new tab](https://titiler.xyz/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://titiler.xyz/api){:target="_blank"}
- [TiTiler demo landing page](https://titiler.xyz/){:target="_blank"}
- [Source on GitHub](https://github.com/developmentseed/titiler){:target="_blank"}

## Main endpoint categories

- `/cog/*` — Cloud Optimized GeoTIFF processing.
- `/stac/*` — SpatioTemporal Asset Catalog integration.
- `/mosaicjson/*` — Multi-source mosaicking via `titiler.mosaic`.
- `/zarr/*` — Zarr / xarray reading via `titiler.xarray` (new standard endpoint as of 2.0).
- `/algorithms` — Available processing algorithms.
- `/colorMaps` — Available colormaps.
- `/tileMatrixSets` — Supported tiling schemes.

!!! info "Foundation layer"
    `titiler.core` is the foundation that the other TiTiler applications build on. Anything documented for the reference application here is generally available, possibly under a different route prefix, in the application-specific pages.
