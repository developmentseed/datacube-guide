# TiTiler CMR API Reference

TiTiler CMR is a NASA-focused application that accepts Concept IDs and uses the Common Metadata Repository (CMR) to discover and serve associated granules as tiles.

## Key Features

- **CMR Integration**: Direct integration with NASA's Common Metadata Repository
- **Earth Science Data**: Specialized for NASA Earth science data collections
- **Time Series Support**: Built-in temporal analysis capabilities
- **Granule Discovery**: Automatic discovery and aggregation of data granules

## Interactive API Documentation

The complete, interactive API documentation from the OpenVEDA Cloud deployment is below. Please be kind with this API.

<iframe src="https://staging.openveda.cloud/api/titiler-cmr/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick Links

- [Open API docs in new tab](https://staging.openveda.cloud/api/titiler-cmr/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://staging.openveda.cloud/api/titiler-cmr/api){:target="_blank"}

## Main Endpoint Categories

- **Collections**: `/collections/{collection_id}` - Work with CMR collections
- **Statistics**: `/collections/{collection_id}/statistics` - Extract statistical data
- **Time Series**: `/collections/{collection_id}/timeseries` - Temporal analysis
- **Tiles**: `/collections/{collection_id}/tiles` - Generate map tiles
- **Items**: Individual granule access and processing

!!! tip "Authentication"
    Some endpoints may require authentication depending on the data collection's access restrictions.
