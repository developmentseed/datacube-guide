# Components of the TiTiler Ecosystem

The TiTiler ecosystem is a comprehensive suite of Python tools for creating dynamic tile servers from geospatial datasets. The components are organized by their primary function:

## Core TiTiler Framework

**titiler.core** - Foundation libraries for creating FastAPI applications that serve dynamic tiles from Cloud Optimized GeoTIFFs (COGs) and SpatioTemporal Asset Catalog (STAC) items.

**titiler.xarray** - Specialized libraries for creating dynamic tile servers from multi-dimensional datasets stored in Zarr or NetCDF formats.

**titiler.extensions** - Plugin system providing additional functionality for TiTiler factories, such as custom algorithms, authentication, and data processing extensions.

**titiler.mosaic** - Libraries for creating dynamic tile servers that can serve mosaicked imagery from multiple sources using the MosaicJSON specification.

**titiler.application** - Complete reference implementation demonstrating a FastAPI application with full support for COGs, STAC items, and MosaicJSON mosaics.

## Specialized Applications

**titiler-cmr** - NASA-focused application that accepts Concept IDs and uses the Common Metadata Repository (CMR) to discover and serve associated granules as tiles.

**titiler-multidim** - Application specifically designed for multi-dimensional datasets, built on titiler.xarray for handling complex scientific data formats.

## Core Libraries

**rio-tiler** - The foundational library that handles the core tile generation logic, dynamically creating map tiles from raster data sources including COGs.

**rio-cogeo** - Command-line tool and library for creating and validating Cloud Optimized GeoTIFFs, ensuring optimal performance for tile serving.

**rio-viz** - Lightweight visualization tool for locally exploring and debugging raster datasets during development.

## Infrastructure Components

**cogeo-mosaic** - Serverless infrastructure toolkit (designed for AWS Lambda) that implements the MosaicJSON specification for creating and serving mosaicked tile sets.

## Development Tools

**tilebench** - Performance analysis tool that measures how many HTTP requests are required to generate tiles from different data sources, helping optimize tile server configurations.

## Extensions and Plugins

**rio-tiler-mvt** - Plugin that extends rio-tiler to generate Mapbox Vector Tiles (MVT) from raster data sources, enabling vector-based visualizations.

## Standards and Specifications

**MosaicJSON** - Open specification for creating spatial indexes that efficiently link multiple COGs to XYZ tile coordinates, enabling seamless mosaicking of large datasets.

## Legacy Components

**riotiler_mosaic** - **(Deprecated)** - Former rio-tiler plugin for creating tiles from multiple observations. This functionality has been integrated directly into rio-tiler and cogeo-mosaic.

---

*The TiTiler ecosystem provides a complete stack for building scalable, cloud-native tile servers that can handle everything from simple COG serving to complex multi-dimensional scientific datasets.*
