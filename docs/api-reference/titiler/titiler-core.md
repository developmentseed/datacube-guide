# TiTiler Core API Reference

TiTiler Core provides the foundational API patterns used across all TiTiler applications. It handles Cloud Optimized GeoTIFFs (COGs) and SpatioTemporal Asset Catalog (STAC) items.

## Key Features

- **COG Support**: Optimized Cloud Optimized GeoTIFF processing
- **STAC Integration**: Full SpatioTemporal Asset Catalog support
- **OGC Compliance**: Standards-compliant tile serving
- **Extensible Architecture**: Foundation for specialized applications
- **High Performance**: Optimized for cloud-native workflows

## Interactive API Documentation

The complete, interactive API documentation from the Development demo deployment is below. Please be kind with this API.

<iframe src="https://titiler.xyz/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick Links

- [Open API docs in new tab](https://titiler.xyz/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://titiler.xyz/api){:target="_blank"}
- [TiTiler Demo Landing Page](https://titiler.xyz/){:target="_blank"}

## Main Endpoint Categories

- **COG Endpoints**: `/cog/*` - Cloud Optimized GeoTIFF processing
- **STAC Endpoints**: `/stac/*` - SpatioTemporal Asset Catalog integration
- **Mosaic Endpoints**: `/mosaicjson/*` - Multi-source mosaicking
- **Algorithms**: `/algorithms` - Available processing algorithms
- **Color Maps**: `/colorMaps` - Available visualization color schemes
- **TMS**: `/tileMatrixSets` - Supported tiling schemes

!!! info "Foundation Layer"
    TiTiler Core serves as the foundation that all other TiTiler applications build upon, providing consistent API patterns and core functionality.
