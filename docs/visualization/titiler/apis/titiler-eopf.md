# titiler-eopf API reference

`titiler-eopf` is a TiTiler application purpose-built for Earth Observation Processing Framework (EOPF) datasets in the ESA Copernicus ecosystem. It serves Sentinel and other EOPF-shaped data from cloud-hosted Zarr stores via either a standard TiTiler REST API or an OpenEO-compatible backend, from the same image. Apache 2.0-licensed; v0.8.x; maintained by the [EOPF-Explorer](https://github.com/EOPF-Explorer) organization with development from Development Seed.

## Key features

- **GeoZarr-native reader** — a custom `GeoZarrReader` understands hierarchical Zarr DataTrees (xarray-style groups) and detects the GeoZarr v1 conventions, lifting spatial metadata out of Zarr attributes rather than expecting flat 2D arrays.
- **EOPF data conventions** — endpoints follow `/collections/{collection_id}/items/{item_id}/...` aligned with Copernicus naming; the backing store is configured via `TITILER_EOPF_STORE_URL` (S3, Azure, GCS, ...) using `obstore`.
- **Dual REST + OpenEO deployment** — the same image can run as a TiTiler REST API (`titiler.eopf.main:app`) or as an OpenEO backend (`titiler.eopf.openeo.main:app`) via the upstream `titiler-openeo` package and Helm chart.
- **EOPF-specific factory extensions** — `DatasetMetadataExtension` (DataTree introspection), `EOPFChunkVizExtension` (Zarr chunk visualization), `EOPFViewerExtension` (map viewer), `EOPFwmtsExtension` (WMTS service).
- **STAC API integration** — collection-level mosaic tiling when a STAC API is configured, via `titiler.stacapi`.

## Backend dependency

```toml
"titiler-core>=2.0,<3.0"
"titiler-xarray>=2.0,<3.0"
"titiler-extensions>=2.0,<3.0"
"titiler-stacapi>=2.0,<3.0"
"zarr>=3.1.3"
```

Optional: `titiler-openeo` for OpenEO-mode deployment.

## Interactive API documentation

A live deployment is operated by ESA Copernicus Dataspace; consult the project README for the current endpoint URL. The OpenAPI surface is also available locally when running the app.

## Quick links

- [Source on GitHub](https://github.com/EOPF-Explorer/titiler-eopf){:target="_blank"}
- [Copernicus EOPF Explorer](https://api.explorer.eopf.copernicus.eu/stac/){:target="_blank"} — the live STAC API used by this application.
- [OpenEO API spec](https://openeo.org/){:target="_blank"} — for the alternate deployment mode.

## Main endpoint categories

Item-level routes:

- `/collections/{collection_id}/items/{item_id}/info`
- `/collections/{collection_id}/items/{item_id}/tiles/{z}/{x}/{y}`
- `/collections/{collection_id}/items/{item_id}/tilejson.json`
- `/collections/{collection_id}/items/{item_id}/preview`
- `/collections/{collection_id}/items/{item_id}/point`
- `/collections/{collection_id}/items/{item_id}/part` — bounding-box crops.
- `/collections/{collection_id}/items/{item_id}/dataset/`, `/dataset/dict`, `/dataset/groups` — DataTree introspection.

Collection-level mosaic (when a STAC API is configured):

- `/collections/{collection_id}/tiles/{z}/{x}/{y}`

Standard OGC endpoints: `/conformance`, `/tileMatrixSets`, `/colorMaps`, `/algorithms`.

Operational: `/_mgmt/ping`, `/_mgmt/health`, `/_mgmt/cache`.

## Where it fits

Choose `titiler-eopf` when working with ESA Copernicus / EOPF data, GeoZarr-shaped hierarchical Zarr stores, or when an OpenEO-compatible deployment is part of the requirement. For NASA CMR data, see [titiler-cmr](titiler-cmr.md). For VEDA-shaped multidimensional workflows, see [titiler-multidim](titiler-multidim.md). For generic COG and STAC, the [reference titiler.application](titiler-core.md) is a better fit.
