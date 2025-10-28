# Xpublish Ecosystem Overview

[Xpublish](https://xpublish.readthedocs.io/) is a framework built on FastAPI for serving xarray datasets via HTTP APIs , enabling web-based access to scientific datasets. The ecosystem includes several plugins that extend functionality for different use cases, particularly in the geospatial and datacube domains.

## Architecture and Relationships

The XPublish ecosystem follows a plugin-oriented architecture with an intentionally small core library.

### Foundation Layer

- **[xpublish](https://github.com/xpublish-community/xpublish)**: Core library that defines plugin extension points and an accessor for serving xarray datasets over HTTP APIS

### Plugins

- **[xpublish-tiles](https://github.com/earth-mover/xpublish-tiles)**: Transforms xarray datasets to raster, vector, and other tiles. Contains both OGC Tiles and OGC WMS endpoints as separate plugins
- **[xpublish-wms](https://github.com/xpublish-community/xpublish-wms)**: Dedicated Web Map Service implementation, requires CF compliant datasets
- **[opendap-protocol](https://github.com/xpublish-community/opendap-protocol)**: Bare minimum implementation of the DAP 2.0 protocol.
- **[xpublish-edr](https://github.com/xpublish-community/xpublish-edr)**: Routers for the [OGC EDR API](https://ogcapi.ogc.org/edr/), supporting position query, area query, cube query
