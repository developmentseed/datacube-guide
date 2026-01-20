# TiTiler API Reference

This section provides interactive API documentation for the TiTiler ecosystem components. Each application has its own specialized API while sharing common patterns from the core framework.

## Available Applications

<div class="grid cards" markdown>

-   **TiTiler Core**

    ---

    Foundation API for COGs and STAC items. Base functionality that all other applications extend.

    [:octicons-arrow-right-24: Core API Reference](titiler-core.md)

-   **TiTiler CMR**

    ---

    NASA CMR-focused application for satellite data collections with time series support.

    [:octicons-arrow-right-24: CMR API Reference](titiler-cmr.md)

-   **TiTiler Multidim**

    ---

    Multi-dimensional dataset processing for NetCDF, Zarr, and scientific data formats.

    [:octicons-arrow-right-24: Multidim API Reference](titiler-multidim.md)

</div>

## Common API Patterns

All TiTiler applications follow consistent patterns:

### Authentication
- **API Keys**: Some endpoints require authentication via API keys
- **CORS**: Cross-Origin Resource Sharing is configured for web applications
- **Rate Limiting**: Default rate limits may apply

### Response Formats
- **JSON**: Metadata, statistics, and configuration responses
- **Images**: PNG, JPEG, WebP tiles and previews
- **GeoJSON**: Spatial data responses
- **HTML**: Interactive viewers and documentation

### Error Handling
- **HTTP Status Codes**: Standard codes (200, 400, 404, 500, etc.)
- **Error Messages**: Detailed error descriptions in JSON format
- **Validation**: Parameter validation with helpful error messages

### Performance
- **Caching**: Response caching for improved performance
- **Compression**: Automatic response compression
- **Streaming**: Efficient data streaming for large responses

!!! tip "Testing APIs"
    Use the embedded interactive documentation to test endpoints directly in your browser. Each API reference page includes a full interactive interface for exploring available endpoints and parameters.
