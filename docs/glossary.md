# Glossary

Definitions for terms used throughout this guide. Distinctive terms also appear
as hover tooltips wherever they occur in the documentation.

## Interface architecture

These describe the layers of a dynamic datacube visualization, from the user's
input down to the pixels on screen (see the [visualization
overview](visualization/overview.md)).

User interaction
:   The layer where the user interacts with the page to indicate what data and
    display options they want, through controls like time sliders, layer
    toggles, and zoom controls.

Data orchestration
:   Based on the user's selections, the logic that determines *how* to fetch
    data: API integration (such as STAC discovery) and coordination with backend
    services.

Framework
:   A visualization library that wraps a rendering engine with higher-level
    abstractions such as layer management, data binding, and interaction
    patterns. Examples include deck.gl and MapLibre.

Rendering engine
:   The engine that generates an image from input data, via a graphics context
    (WebGL, SVG, or DOM elements), drawing primitives, and coordinate systems.

## Visualization

Static visualization
:   A visualization whose contents do not change after creation, like a map
    printed to paper.

Dynamic visualization
:   A visualization that responds to user input, such as zooming, panning, or
    changing the color scheme.

Dynamic tiler
:   A backend service that renders map tiles from source data on demand rather
    than from pre-rendered files. Examples include TiTiler and xpublish-tiles.

Tile
:   A small image (commonly 256×256 pixels) covering one region at one zoom
    level; the unit a web map fetches and draws.

Zoom level
:   An integer describing the resolution of a tiled map. Level 0 covers the
    whole world in a single tile, and each level adds detail by splitting every
    tile into four, doubling the resolution.

Multiscales
:   Precomputed downsampled copies of a dataset at multiple resolutions (a
    "pyramid"), so a viewer reads only the resolution it needs.

Colormap
:   A mapping from data values to colors, used to render numeric data as an
    image.

CRS (coordinate reference system)
:   A definition of how coordinates map to locations on Earth. Visualization
    often transforms between data, projected, and display coordinate reference
    systems.

Web Mercator
:   The map projection (EPSG:3857) used by most web map tiles.

Range request
:   An HTTP request for a byte range of a file, letting a client read part of a
    large file without downloading the whole thing.

## Data and formats

Datacube
:   A multi-dimensional array of data, for example time × level × latitude ×
    longitude × band, typically spanning 3 to 5 dimensions.

Chunk
:   A contiguous block of a chunked array, read and written as a unit. Chunk
    size strongly affects performance.

Zarr
:   A cloud-optimized format for chunked, compressed N-dimensional arrays.

GeoZarr
:   A geospatial convention layered on top of Zarr, covering multiscales and CRS
    encoding (EPSG, WKT2, or PROJJSON).

COG (Cloud-Optimized GeoTIFF)
:   A GeoTIFF structured for efficient range-request access on object storage.

NetCDF
:   A self-describing array format common in earth-science data.

GRIB
:   A binary format for gridded meteorological data.

STAC (SpatioTemporal Asset Catalog)
:   A JSON specification for describing and discovering geospatial datasets.

OPeNDAP
:   A protocol for remote access to subsets of scientific datasets over HTTP.
