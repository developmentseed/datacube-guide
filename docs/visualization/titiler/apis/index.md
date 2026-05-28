# TiTiler API reference

This section provides interactive API documentation for the four TiTiler applications most relevant to datacube serving. Each application has its own specialized API while sharing common patterns from `titiler.core`.

## Available applications

<div class="grid cards" markdown>

-   **titiler.core (reference application)**

    ---

    Foundation API for COGs, STAC, MosaicJSON, and (since 2.0) Zarr. The base everything else builds on.

    [:octicons-arrow-right-24: titiler.core API reference](titiler-core.md)

-   **titiler-cmr**

    ---

    NASA Common Metadata Repository application with dual `/xarray/` and `/rasterio/` backends, granule filtering, and per-DAAC EDL credentials.

    [:octicons-arrow-right-24: titiler-cmr API reference](titiler-cmr.md)

-   **titiler-multidim**

    ---

    NASA VEDA multidimensional application built on `titiler.xarray`. Adds Redis caching, OpenTelemetry tracing, and Icechunk support.

    [:octicons-arrow-right-24: titiler-multidim API reference](titiler-multidim.md)

-   **titiler-eopf**

    ---

    ESA Copernicus / EOPF application with a GeoZarr DataTree reader. Deploys as either a TiTiler REST API or an OpenEO backend.

    [:octicons-arrow-right-24: titiler-eopf API reference](titiler-eopf.md)

</div>

## Common API patterns

All TiTiler applications share most of these patterns:

### Authentication

- **API keys** — some endpoints require authentication via API keys.
- **Earthdata Login** — `titiler-cmr` uses EDL bearer tokens, scoped per DAAC.
- **CORS** — Cross-Origin Resource Sharing is configured for web applications.
- **Rate limiting** — default rate limits may apply on shared deployments.

### Response formats

- **JSON** — metadata, statistics, and configuration responses.
- **Images** — PNG, JPEG, WebP tiles and previews.
- **GeoJSON** — spatial responses, including granule footprints in `titiler-cmr`.
- **HTML** — interactive viewers and OpenAPI documentation.

### Error handling

- **HTTP status codes** — standard codes (200, 400, 404, 500, etc.).
- **Error messages** — detailed error descriptions in JSON.
- **Validation** — parameter validation with helpful error messages.

### Performance

- **Caching** — response caching via Redis (in `titiler-multidim`) and via standard CDN front-ends.
- **Compression** — automatic response compression.
- **Streaming** — efficient streaming for large responses.

!!! tip "Testing APIs"
    Use the embedded interactive documentation on each per-application page to test endpoints directly in your browser. Each reference page includes the deployed Swagger UI for exploring endpoints and parameters.
