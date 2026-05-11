# Xpublish ecosystem overview

[Xpublish](https://xpublish.readthedocs.io/) is a FastAPI-based framework for serving xarray datasets via HTTP APIs, enabling web-based access to scientific datasets. The ecosystem follows a plugin-oriented architecture with an intentionally small core library, and most user-facing capabilities are added by separate plugins.

## Foundation

- **[xpublish](https://github.com/xpublish-community/xpublish)** — core library that defines plugin extension points and an `xpublish` accessor for serving xarray datasets over HTTP APIs.

## Plugins

- **[xpublish-tiles](xpublish-tiles.md)** ([source](https://github.com/earth-mover/xpublish-tiles)) — transforms xarray datasets into raster and vector tiles. Provides OGC Tiles and OGC WMS endpoints as separate plugins (TilesPlugin and WMSPlugin). The detail page covers its grid-system support (regular, curvilinear, triangular, SELFE, polar, cubed-sphere, HEALPix), tile API surface, and rendering pipeline.
- **[xpublish-wms](https://github.com/xpublish-community/xpublish-wms)** — a dedicated Web Map Service implementation. Requires CF-compliant datasets. Predates xpublish-tiles' WMS plugin and is the more mature WMS path today.
- **[xpublish-edr](https://github.com/xpublish-community/xpublish-edr)** — routers for the [OGC EDR API](https://ogcapi.ogc.org/edr/), supporting position, area, and cube queries; useful for time-series and profile extraction rather than map tiles.
- **[opendap-protocol](https://github.com/xpublish-community/opendap-protocol)** — bare-minimum implementation of the DAP 2.0 protocol, for clients that expect OPeNDAP semantics.

For a side-by-side comparison with the TiTiler ecosystem covering input formats, grid support, endpoints, tile-request parameters, resampling, and deployment, see the [Dynamic tiling ecosystem comparison](../ecosystem-comparison.md).
