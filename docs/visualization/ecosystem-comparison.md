# Dynamic tiling ecosystem comparison

## Overview

Both titiler and Xpublish provide FastAPI-based web services for publishing geospatial and scientific datasets, but with different architectural philosophies and target use cases:

- **TiTiler**: Specialized for dynamic tile generation with a layered architecture built on rio-tiler
- **Xpublish**: Plugin-based data publishing platform with protocol compliance focus

## Detailed Comparison

| Factor | TiTiler Ecosystem | Xpublish Ecosystem |
|--------|------------------|-------------------|
| **License** | MIT License (Development Seed) | Apache License 2.0 (UCAR) |
| **Organizational Maintainer** | Development Seed | UCAR/Xarray community |
| **Individual Maintainers** | Vincent Sarago, Aimee Barciauskas | Joe Hamman, Alex Kerney, distributed community |

### Maintenance, Governance, and Development Models

| Aspect | TiTiler | Xpublish |
|--------|---------|----------|
| **Governance Model** | Open source, built by Development Seed | Open source, community-driven (scientific community) |
| **Development Focus** | Tile server optimization and performance | Protocol compliance and scientific data standards |
| **Release Cadence** | Regular releases with coordinated ecosystem updates | Community-driven releases, plugin-independent versioning |
| **Commercial Support** | Available through Development Seed | Community support through ESIP/scientific networks |
| **Contributor Base** | Concentrated around geospatial tile serving | Distributed across oceanographic and climate science communities |

### Tested Input Formats

| Format | TiTiler Support | Xpublish Support |
|--------|----------------|------------------|
| **Native Zarr** | ✅ Full support via titiler.xarray | ✅ Primary format with optimal performance |
| **NetCDF** | ✅ Full support via h5netcdf engine | ✅ Full support via Xarray integration |
| **Virtual Zarr** | ✅ Supported through zarr v3+ interfaces | ✅ Reference-based access to remote datasets |
| **Cloud Optimized GeoTIFF (COG)** | ✅ Primary format for titiler.core | ❌ Not directly supported |
| **STAC Items** | ✅ Native support for asset discovery | ❌ Not directly supported |
| **HDF5/NetCDF via references** | ✅ Via virtual references | ✅ Via Zarr reference spec |
| **Multi-file datasets** | ✅ Via CMR integration (NASA datasets) | ✅ Via catalog systems (Intake) |
| **Icechunk stores** | ⚠️ Limited experimental support | ✅ Full support for versioned Zarr |

### IO Methods and Data Access

| Method | TiTiler | Xpublish |
|--------|---------|----------|
| **Core Libraries** | | |
| • Xarray integration | ✅ Via titiler.xarray package | ✅ Native, primary data structure |
| • Rasterio/GDAL | ✅ Via rio-tiler (titiler.core) | ❌ Not used |
| • Zarr direct access | ✅ Via xarray.open_zarr | ✅ Native zarr.open_consolidated |
| • fsspec support | ✅ For remote datasets (S3, HTTP, etc.) | ✅ For all remote access |
| • h5netcdf engine | ✅ For NetCDF files | ✅ Via xarray backends |
| **Data Loading** | | |
| • Lazy loading | ✅ Via Dask arrays | ✅ Via Dask arrays |
| • Async loading | ⚠️ Limited support | ✅ xarray.load_async() support |
| • Caching | ✅ Redis-based dataset caching | ✅ Plugin-configurable |
| • Chunk-aware | ✅ Leverages Zarr/NetCDF chunking | ✅ Optimized for chunked access |
| **Coordinate Handling** | | |
| • CRS transformations | ✅ Via rioxarray + pyproj | ✅ Via pyproj with thread pool |
| • Coordinate renaming | ✅ Auto-detect lat/lon/x/y | ✅ CF-xarray for detection |
| • Antimeridian handling | ✅ Longitude normalization | ✅ LongitudeCellIndex for wrapping |
| **Indexing Methods** | | |
| • Label-based selection | ✅ Xarray sel() with method parameter | ✅ Direct dimension indexing |
| • Spatial indexing | ✅ Bounding box queries | ✅ Custom spatial indexes (CellTreeIndex) |
| • Temporal indexing | ✅ Time dimension selection | ✅ CF time coordinate support |

### Tested Grid Structures

| Grid Type | TiTiler Support | Xpublish Support |
|-----------|----------------|------------------|
| **Regular lat/lon grids** | ✅ Optimized for rectangular grids | ✅ Full support via CF conventions |
| **Projected coordinate systems** | ✅ Via morecantile TileMatrixSets | ✅ Via CF-compliant grid mappings |
| **Curvilinear grids** | ⚠️ Basic support with limitations | ✅ Full support (ROMS: CBOFS, DBOFS, TBOFS, etc.) |
| **FVCOM triangular grids** | ❌ No native support | ✅ Full support (LOOFS, LSOFS, LMHOFS, NGOFS2) |
| **SELFE grids** | ❌ No native support | ✅ Full support (CREOFS) |
| **Irregular/unstructured grids** | ❌ Limited support | ✅ Extensible grid system for custom types |
| **2D non-dimensional grids** | ⚠️ Basic support | ✅ Full support (RTOFS, HRRR-Conus) |

### Supported Endpoints

| Endpoint Category | TiTiler | Xpublish |
|------------------|---------|----------|
| **Tile Generation** | | |
| • XYZ tiles | ✅ `/tiles/{z}/{x}/{y}` with multiple TMS | ✅ Via xpublish-wms plugin |
| • WMTS | ✅ OGC WMTS compliance | ✅ Via xpublish-wms (GetMap, GetCapabilities) |
| • TileJSON | ✅ Tile layer metadata | ✅ Via WMS plugin |
| **Data Access** | | |
| • Zarr API | ✅ Via xarray reader | ✅ Native `.zmetadata`, chunk access |
| • OpenDAP | ❌ No direct support | ✅ Via xpublish-opendap plugin |
| • OGC EDR | ❌ No direct support | ✅ Via xpublish-edr (position, area, cube queries) |
| **Metadata** | | |
| • Dataset info | ✅ `/info` and `/info.geojson` | ✅ Built-in dataset information endpoints |
| • Variable listing | ✅ `/variables` (deprecated) | ✅ Automatic metadata exposure |
| **Analysis** | | |
| • Statistics/Histograms | ✅ `/statistics` (POST) with geometry support | ✅ Via EDR plugin with multiple output formats |
| • Timeseries extraction | ✅ Temporal indexing and selection | ✅ Via EDR temporal querying |
| • Spatial querying | ✅ Bbox and feature queries | ✅ EDR position, area, cube queries |

### Tile Request Parameters

| Parameter Category | TiTiler | Xpublish-Tiles |
|-------------------|---------|----------------|
| **Dataset Selection** | | |
| • Variable/Asset | `variable` (xarray), `assets` (STAC) | `variables` (list, required) |
| • Band indexes | `bidx` for multi-band selection | Single variable focus |
| • Group/hierarchy | `group` for Zarr/HDF5 groups | Dataset-level only |
| **Dimension Selection** | | |
| • Dimension indexing | `sel={dim}={value}` (list of strings) | Any dimension as query param |
| • Selection method | `sel_method` (nearest, pad, ffill, etc.) | Automatic dtype casting |
| • Time selection | Via sel parameter | Direct time parameter with ISO8601 |
| • Decode times | `decode_times` (boolean) | N/A (always decoded) |
| **Styling & Rendering** | | |
| • Colormap | `colormap_name` or `colormap` (JSON) | `style={type}/{colormap}` format |
| • Color range | `rescale` (min,max) | `colorscalerange` (tuple) |
| • Custom colormap | JSON with int→hex mapping | JSON with 0-255→hex mapping |
| • Band math | `expression` (e.g., "b1/b2") | Not supported |
| **Image Parameters** | | |
| • Output format | `f` (jpeg, png, webp, tiff, etc.) | `f` (image/png, image/jpeg) |
| • Tile size | `tile_scale` (1=256x256, 2=512x512) | `width`, `height` (multiples of 256) |
| • Quality | Format-specific params | Not exposed |
| **Reprojection** | | |
| • Resampling | `resampling` (RIOResampling enum) | Automatic via datashader |
| • Warp resampling | `reproject` (WarpResampling enum) | Coordinate transform in pipeline |
| • Nodata | `nodata` value override | Automatic from metadata |
| **Processing** | | |
| • Histogram equalization | Via rescale algorithms | Not supported |
| • Hillshade | Via extensions | Not supported |
| • Statistics | POST to `/statistics` endpoint | Via EDR plugin |
| **Error Handling** | | |
| • Render errors | Standard HTTP errors | `render_errors` (boolean) for image tiles |
| • Missing data | Configurable nodata handling | Transparent or nodata fill |

### Supported Tiling Features

| Feature Category | TiTiler | Xpublish |
|-----------------|---------|----------|
| **Output Formats** | | |
| • Raster formats | JPEG, PNG, WebP, TIFF, JP2, NumpyTile | PNG, JPEG via WMS plugin |
| • Vector formats | ❌ | GeoJSON, CSV via EDR plugin |
| • Scientific formats | Limited | NetCDF, Parquet, GeoTIFF via EDR plugin |
| **Rendering** | | |
| • Rescaling | Linear, histogram-based, custom functions | Basic rescaling via WMS |
| • Colormaps | Built-in + custom JSON colormaps | WMS styling capabilities |
| • Band combinations | Multi-band composites and band math | Single variable visualization focus |
| • Algorithms | NDVI, hillshade via extensions | Limited processing algorithms |
| **Performance** | | |
| • Caching | Redis-based response caching | Plugin-configurable caching |
| • Concurrent access | Multi-threaded tile generation | FastAPI async processing |
| • Chunk optimization | Leverages Zarr/NetCDF chunking | Optimized for chunked scientific data |

### Resampling Implementations

| Resampling Aspect | TiTiler | Xpublish-Tiles |
|-------------------|---------|----------------|
| **Raster Resampling (RIOResampling)** | | |
| • Available methods | nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, gauss, max, min, med, q1, q3 | nearest (via datashader), automatic for continuous data |
| • Default method | nearest | Adaptive based on data type |
| • Usage context | Tile generation, preview, bbox queries | Internal rendering pipeline |
| **Warp Resampling (coordinate reprojection)** | | |
| • Available methods | nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, max, min, med, q1, q3, sum, rms | Custom implementation via pyproj |
| • Default method | nearest | Optimized for 4326→3857 (separable transform) |
| • Thread pool | Not used | ✅ Configurable chunk-based parallel transform |
| **Downsampling/Coarsening** | | |
| • Method | Via GDAL overviews or on-the-fly | DataArray.coarsen().mean() with even factors |
| • Automatic trigger | Based on zoom level | When data exceeds max_renderable_size |
| • Boundary handling | GDAL standard | Configurable padding for edge effects |
| **Grid-Specific Optimizations** | | |
| • Rectilinear grids | Standard rasterio resampling | Datashader raster (3-10x faster than quadmesh) |
| • Curvilinear grids | Limited support | Datashader quadmesh with optional rectilinear approximation |
| • Triangular/UGRID | Not supported | Datashader trimesh with Delaunay triangulation |
| • Categorical data | Mode resampling | Numbagg for nearest-neighbor on discrete data |
| **Coordinate Transform Optimizations** | | |
| • 4326→3857 | Via rioxarray/GDAL | Custom separable implementation (preserves grid structure) |
| • General transforms | GDAL warp with selected kernel | Blocked transformation with thread pool |
| • Chunking strategy | N/A | Configurable TRANSFORM_CHUNK_SIZE (NxN chunks) |
| • Approximate rectilinear | Not implemented | Numba-optimized detection (1-pixel threshold) |
| **Rendering Performance** | | |
| • Rendering engine | rio-tiler (GDAL-based) | Datashader (Numba JIT-compiled) |
| • JIT compilation | Not used | First-invocation blocking (unavoidable) |
| • Numba threads | N/A | Configurable NUMBA_NUM_THREADS |

### Metadata Conventions

| Convention Category | TiTiler | Xpublish-Tiles |
|---------------------|---------|----------------|
| **CF Conventions** | | |
| • Standard names | ⚠️ Basic support via rioxarray | ✅ Full support via cf-xarray |
| • Coordinate detection | lat/lon/x/y auto-detection | CF axis detection (X, Y, Z, T) |
| • Grid mappings | Via rioxarray | Multiple grid_mapping support |
| • Vertical coordinates | Limited | Z axis detection and handling |
| • Time coordinates | decode_times parameter | CF-compliant time parsing |
| • Bounds/cells | Not used | CF bounds for accurate cell representation |
| **Zarr Metadata** | | |
| • Format version | Zarr v2/v3 support | Zarr v2 primary, v3 compatible |
| • Consolidated metadata | ✅ .zmetadata support | ✅ zarr.open_consolidated() |
| • Chunk encoding | Standard zarr chunks | Standard + compressor/filters |
| • Dimension names | _ARRAY_DIMENSIONS | _ARRAY_DIMENSIONS |
| • Fill values | _FillValue handling | _FillValue + automatic detection |
| **Custom Attributes** | | |
| • valid_min/valid_max | ✅ Used for rescaling | ✅ Used for continuous data colorscale |
| • valid_range | ✅ Converted to valid_min/max | Standard processing |
| • flag_values | Not directly used | ✅ For categorical/discrete rendering |
| • flag_meanings | Not directly used | ✅ Category labels |
| • flag_colors | Not directly used | ✅ Custom categorical colormaps |
| • long_name | Available in metadata | ✅ Used in tile metadata |
| **OGC Standards** | | |
| • OGC WMTS | ✅ Full compliance | Via WMS plugin |
| • OGC Tiles API | ⚠️ Partial | ✅ Full compliance (1.0) |
| • TileJSON | ✅ 3.0.0 support | ✅ 3.0.0 support |
| • WMS | Limited | ✅ 1.1.1/1.3.0 compliance |
| **STAC Metadata** | | |
| • STAC Items | ✅ Native support | ❌ Not used |
| • Asset metadata | ✅ Full integration | N/A |
| • Temporal extent | Via STAC properties | Via CF time coordinates |
| **Coordinate Reference Systems** | | |
| • CRS detection | rioxarray CRS | CF grid_mapping + PyProj |
| • EPSG codes | ✅ Standard support | ✅ Full PyProj CRS support |
| • Custom projections | Via PROJ strings | Via CF grid_mapping_name |
| • Default CRS | Explicit or epsg:4326 | epsg:4326 fallback |
| **Dataset-Level Metadata** | | |
| • Title/description | Via STAC or dataset attrs | Dataset.attrs with fallbacks |
| • Keywords | Via STAC | attrs["keywords"] |
| • License | Via STAC | attrs["license"] |
| • Attribution | Via STAC | attrs["attribution"] |
| • Contact | Via STAC | attrs["contact"] |
| • Version | Not standard | attrs["version"] |
| **Internal Identifiers** | | |
| • Dataset ID | URL-based | attrs["_xpublish_id"] (required for caching) |
| • Cache keys | URL + parameters | Dataset ID + dimension + variable |

### Plugin/Extension Points

| Extension Type | TiTiler | Xpublish |
|---------------|---------|----------|
| **Architecture** | Factory-based endpoint creation | Plugin-based router system |
| **Custom I/O** | Pluggable readers (Xarray, Rasterio) | Dataset provider plugins |
| **Protocol Support** | Limited to tile/analysis endpoints | Full protocol plugins (WMS, EDR, OpenDAP) |
| **Authentication** | Extensions for custom auth | Pluggable auth systems |
| **Data Processing** | Algorithm extensions and middleware | Data transformation plugins |
| **Deployment** | Application factory pattern | Configurable server distributions |

### Architecture and Core Dependencies

| Component | TiTiler | Xpublish-Tiles |
|-----------|---------|----------------|
| **Web Framework** | | |
| • Framework | FastAPI | FastAPI (via xpublish) |
| • Async support | ✅ ASGI-based | ✅ ASGI-based |
| • OpenAPI docs | ✅ Automatic | ✅ Automatic |
| **Core Data Libraries** | | |
| • Primary reader | rio-tiler (rasterio/GDAL) | Xarray with custom grid systems |
| • Xarray support | Via titiler.xarray addon | Native and required |
| • Rasterio | ✅ Core dependency | ❌ Not used |
| • GDAL | ✅ Via rasterio | ❌ Not used |
| **Rendering Engines** | | |
| • Tile rendering | rio-tiler (GDAL-based) | Datashader (Numba JIT) |
| • Image encoding | rio-tiler + Pillow | Pillow |
| • Resampling | GDAL/rasterio | Datashader + pyproj |
| **Geospatial Libraries** | | |
| • CRS handling | rioxarray + pyproj | pyproj |
| • Coordinate transforms | GDAL warp | Custom pyproj implementation |
| • Tile matrix sets | morecantile | morecantile (via plugin) |
| **Scientific Computing** | | |
| • Xarray | Optional (titiler.xarray) | Required, core |
| • Dask | Via xarray | Via xarray |
| • NumPy | Via dependencies | Direct use |
| • Numba | Not used | ✅ For JIT compilation |
| • numbagg | Not used | ✅ For aggregations |
| • cf-xarray | Not used | ✅ For CF conventions |
| **Grid/Mesh Support** | | |
| • Regular grids | ✅ Native via GDAL | ✅ Via RasterIndex |
| • Curvilinear grids | ⚠️ Limited | ✅ Full support |
| • Unstructured meshes | ❌ No support | ✅ Via scipy.spatial.Delaunay |
| • UGRID | ❌ No support | ✅ Via CellTreeIndex |
| **Caching** | | |
| • Implementation | Redis for datasets | Plugin-configurable |
| • Cache strategy | Dataset-level | Dataset + grid system |
| • Cache invalidation | TTL-based | Internal via _xpublish_id |
| **Configuration** | | |
| • Environment vars | Via Pydantic settings | Via Pydantic settings |
| • Config files | Application-specific | Config system with env vars |
| • Runtime tuning | Limited | Thread pool, async load, chunk size |

### Deployment Considerations

| Aspect | TiTiler | Xpublish-Tiles |
|--------|---------|----------------|
| **Container Support** | | |
| • Official images | ✅ Docker Hub + GitHub registry | No official images |
| • Base requirements | Python + GDAL | Python + scientific stack |
| • Image size | Moderate (GDAL overhead) | Larger (scientific libraries) |
| **Cloud Deployment** | | |
| • AWS Lambda | ✅ CDK examples provided | Possible but not documented |
| • AWS ECS | ✅ CDK examples provided | Manual setup |
| • Kubernetes | Community deployments | Manual setup |
| **Scaling Considerations** | | |
| • Stateless | ✅ With external cache | ✅ Dataset loading needed |
| • Horizontal scaling | ✅ Well-suited | ✅ With shared cache |
| • Resource requirements | Moderate CPU + memory | High CPU for JIT, configurable threads |
| **Performance Tuning** | | |
| • Critical settings | Redis config, worker count | NUMBA_NUM_THREADS, thread pool size, async_load |
| • First request | Fast (no JIT) | Slow (JIT compilation) |
| • Warm-up needed | Only for cache | ✅ Required for datashader JIT |
| • Memory footprint | Dataset + tile buffer | Dataset + coordinate cache + grid cache |

## Use Case Recommendations

### Choose TiTiler When:
- **Primary need**: High-performance tile serving and web map visualization
- **Data types**: COGs, STAC catalogs, regular gridded scientific data
- **Requirements**: Fast tile generation, web mapping standards compliance, cloud-native architecture
- **Users**: Web mapping applications, tile-based visualizations, GIS workflows

### Choose Xpublish When:
- **Primary need**: Multi-protocol data access and scientific data sharing
- **Data types**: Complex gridded scientific datasets (ocean models, climate data)
- **Requirements**: Protocol compliance (OpenDAP, EDR), flexible data access patterns, research workflows
- **Users**: Scientific communities, oceanographic/climate data centers, research institutions

## Hybrid Approaches

For comprehensive data serving solutions, consider:

1. **Complementary deployment**: Use TiTiler for tile-based visualization and Xpublish for data access protocols
2. **Protocol bridging**: Leverage both ecosystems' strengths for different access patterns
3. **Staged architecture**: TiTiler for public-facing maps, Xpublish for research data access

---

*Both ecosystems continue to evolve, with increasing interoperability and shared standards adoption across the scientific data serving landscape.*