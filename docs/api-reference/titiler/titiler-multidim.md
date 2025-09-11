# TiTiler Multidim API Reference

TiTiler Multidim is a comprehensive application built on `titiler.xarray` specifically designed for multi-dimensional datasets like NetCDF and Zarr files.

## Key Features

- **Multi-dimensional Support**: Native handling of 3D, 4D, and 5D datasets
- **Temporal Processing**: Advanced time-series analysis and animation support
- **Performance Optimizations**: Redis caching and optimized chunking strategies
- **Scientific Data Formats**: NetCDF, Zarr, HDF, and other research data formats
- **VEDA Integration**: Optimized for NASA's VEDA platform infrastructure

## Interactive API Documentation

The complete, interactive API documentation from the OpenVEDA Cloud deployment is below. Please be kind with this API.

<iframe src="https://staging.openveda.cloud/api/titiler-multidim/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick Links

- [Open API docs in new tab](https://staging.openveda.cloud/api/titiler-multidim/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://staging.openveda.cloud/api/titiler-multidim/api){:target="_blank"}

## Main Endpoint Categories

- **Dataset Info**: `/info` - Dataset metadata and structure
- **Statistics**: `/statistics` - Statistical analysis across dimensions
- **Tiles**: `/tiles/{z}/{x}/{y}` - Map tile generation
- **Temporal Selection**: Time-based data slicing and selection
- **Dimensional Analysis**: Multi-dimensional data exploration
- **Rendering**: Advanced visualization and color mapping

!!! note "Prototype Application"
    TiTiler Multidim serves as a prototype application demonstrating advanced multidimensional data processing capabilities with various optimizations for production use.
