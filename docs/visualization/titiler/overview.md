# Titiler ecosystem overview

The TiTiler ecosystem is a layered Python stack for building dynamic tile servers from geospatial datasets. Each component fills a specific role: a foundation library, several focused extensions, and a set of opinionated applications targeting concrete data ecosystems (NASA CMR, NASA VEDA, ESA EOPF, generic COG/STAC). Components are released independently and pinned to compatible major-version ranges.

For a side-by-side comparison with the Xpublish ecosystem, see the [Dynamic tiling ecosystem comparison](../ecosystem-comparison.md).

## Foundation

- **[rio-tiler](https://github.com/cogeotiff/rio-tiler)** — core tile generation engine. TiTiler 2.x requires `rio-tiler>=9,<10`.
- **[titiler.core](https://github.com/developmentseed/titiler/tree/main/src/titiler/core)** — the base FastAPI framework, factory patterns, and dependency primitives used by every TiTiler application.

## Extensions

- **[titiler.xarray](https://github.com/developmentseed/titiler/tree/main/src/titiler/xarray)** — multidimensional support that extends `titiler.core` with xarray-based readers for NetCDF, Zarr, and similar formats. As of TiTiler 2.0 the application also exposes `/zarr/*` endpoints by default.
- **[titiler.extensions](https://github.com/developmentseed/titiler/tree/main/src/titiler/extensions)** — plugin system for custom factory behaviour (viewers, custom endpoints, dataset metadata, etc.).
- **[titiler.mosaic](https://github.com/developmentseed/titiler/tree/main/src/titiler/mosaic)** — multi-source mosaic tiling on top of MosaicJSON.

## Applications

- **[titiler.application](https://github.com/developmentseed/titiler/tree/main/src/titiler/application)** — reference application bundling `titiler.core`, `titiler.mosaic`, and (since 2.0) the `/zarr/*` endpoints from `titiler.xarray`. Public demo at [titiler.xyz](https://titiler.xyz/api.html).
- **[titiler-cmr](apis/titiler-cmr.md)** — NASA Common Metadata Repository application. Now built on both `titiler.core` and `titiler.xarray` with dual `/xarray/` and `/rasterio/` backend prefixes (formerly `/collections/*`, which still redirect).
- **[titiler-multidim](apis/titiler-multidim.md)** — VEDA-deployed multidimensional application built on `titiler.xarray`. Adds Redis caching, OpenTelemetry tracing, and (since v0.7) icechunk support. No longer labelled a prototype.
- **[titiler-eopf](apis/titiler-eopf.md)** — ESA Copernicus / Earth Observation Processing Framework application. Built on `titiler.xarray` plus `titiler.stacapi`, with a custom GeoZarr reader for hierarchical Zarr DataTrees. Can deploy as either a TiTiler REST API or an OpenEO backend from the same image.

## Installation note

The bare `titiler` metapackage on PyPI was dropped in late 2025. Install the specific subpackages you need: `pip install titiler.core titiler.xarray titiler.mosaic`, or one of the application packages directly.

## Layering at a glance

```
rio-tiler
   ↑
titiler.core ─── titiler.extensions ─── titiler.mosaic
   ↑                                         ↑
   └────────── titiler.xarray ───────────────┤
                       ↑                     │
                       ├── titiler.application
                       ├── titiler-cmr (also uses titiler.core + titiler.mosaic directly)
                       ├── titiler-multidim
                       └── titiler-eopf (also uses titiler.stacapi)
```

Python support across the stack is currently 3.11 through 3.14.
