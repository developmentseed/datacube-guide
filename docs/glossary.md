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
    longitude × band, typically spanning 3 to 5 dimensions. See
    [Data structures](concepts/data-structures.md#datacube).

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

## Data structures

Foundational concepts covered on the
[Data structures](concepts/data-structures.md) page.

Index space
:   The integer-indexed structure of an array. A value is addressed by its
    position `(i, j[, k])`; has no inherent units. See
    [Data structures](concepts/data-structures.md#two-spaces).

World space
:   Where each value actually sits in a CRS — also "physical space" or
    "geographic space". Has units defined by the CRS (meters, degrees, …). See
    [Data structures](concepts/data-structures.md#two-spaces).

Gridded
:   Data tied to the cells (or nodes) of a grid; a value's location is implied
    by its index in the array. See
    [Data structures](concepts/data-structures.md#gridded-vs-ungridded).

Ungridded
:   Scattered or point observations that are not arranged on a grid; each
    value carries its own explicit coordinates. See
    [Data structures](concepts/data-structures.md#gridded-vs-ungridded).

Structured grid
:   A grid whose cells form a regular logical array addressable by integer
    indices, with connectivity implicit. Includes regular (rectilinear) and
    curvilinear grids. See
    [Data structures](concepts/data-structures.md#structured-vs-unstructured).

Unstructured grid
:   A mesh whose cells are joined by an explicit connectivity list, with
    variable numbers of neighbors per node. See
    [Data structures](concepts/data-structures.md#structured-vs-unstructured).

DGGS (Discrete Global Grid System)
:   A global tessellation of the sphere by a single cell family (often
    equal-area), with hierarchical refinement and a specialized cell-ID
    indexing scheme — connectivity is implicit in the ID arithmetic rather
    than in `(i, j)` array shape or an explicit connectivity list. Examples:
    HEALPix, H3, S2, cubed-sphere. See
    [Data structures](concepts/data-structures.md#structured-vs-unstructured).

Regridding (resampling)
:   The operation that moves data from one spatial sampling to another, for
    example from ungridded points onto a regular grid. Method matters: nearest,
    bilinear, and conservative each suit different quantities. See
    [Data structures](concepts/data-structures.md#regridding-resampling).

## Satellite data products

Pointers to canonical external references for satellite Earth-observation
product taxonomy. These terms are out of scope for an in-depth treatment here.

Swath
:   The strip of Earth's surface observed by a sensor as the platform moves
    along its orbit; the sensor's native acquisition geometry, indexed by
    along-track × across-track with 2-D geolocation arrays. Typical of
    Level-1/Level-2 products, distinct from a Level-3 product resampled onto a
    regular map grid. See ESA Sentinel Online —
    [Sentinel-1 product types and processing levels][s1-products].

Analysis-ready data (ARD)
:   Satellite data processed to a minimum set of requirements and organized so
    it can be analyzed immediately, with minimal further user effort, and
    interoperably across products. The formal reference is **CEOS-ARD**
    (formerly CARD4L). See CEOS — [Analysis Ready Data][ceos-ard].

Data processing levels
:   A maturity ladder describing how far a product has been processed, from raw
    instrument data (Level 0) to model output (Level 4). The numbers are *not*
    portable across agencies — NASA, ESA/Copernicus, and USGS each define their
    own scheme. See NASA Earthdata —
    [Data Processing Levels][nasa-levels].

Timeliness (NRT/STC/NTC)
:   ESA's latency axis for a product: Near Real Time (hours), Short Time
    Critical, and Non Time Critical (best calibration accuracy). Independent of
    processing level. See Copernicus SentiWiki — [Sentinel-3 products][s3-products].

[s1-products]: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels
[ceos-ard]: https://ceos.org/ard/
[nasa-levels]: https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/data-processing-levels
[s3-products]: https://sentiwiki.copernicus.eu/web/s3-products
