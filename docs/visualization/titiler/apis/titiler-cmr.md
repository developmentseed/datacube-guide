# titiler-cmr API reference

`titiler-cmr` is a NASA-focused application that accepts CMR Concept IDs and uses the Common Metadata Repository to discover and serve associated granules as tiles. Currently at v1.0.x, MIT-licensed, maintained by Development Seed.

## Key features

- **Dual-backend architecture** — `/xarray/*` routes for NetCDF/HDF5 via `titiler.xarray`, `/rasterio/*` routes for COG-style data via `titiler.core` + `titiler.mosaic`. Selected at the URL prefix; users no longer have to pick a backend out of band.
- **CMR integration** — direct integration with NASA's Common Metadata Repository for granule discovery.
- **Granule filtering** — request-time filters including `orbit_number`, `cloud_cover`, `attribute`, `sort_key`, plus rio-tiler-style `skipcovered`, `coverage_tolerance`, and `exitwhenfull`.
- **Cloud-native I/O** — the xarray backend now uses `obstore` and `obspec-utils` for cloud-native data loading, replacing the older fsspec path.
- **Per-DAAC EDL credentials** — Earthdata Login bearer-token credentials cached per DAAC (5-minute refresh), replacing the older AWS IAM role flow.
- **Expression translation** — named-asset and named-variable expressions (e.g. `B04/B03`) are automatically rewritten to `b1/b2` form for `rio-tiler>=9` compatibility.
- **GeoJSON granule output** — `/granules` endpoints accept `f=geojson` to return FeatureCollections of granule footprints.
- **Multi-variable xarray** — `variables` parameter accepts repeated values for multi-band requests; superseded the singular `variable` parameter.

## Backend dependency

`titiler-cmr` v1.0+ depends on **both** `titiler.core` and `titiler.xarray` (plus `titiler.mosaic`):

```toml
"titiler-core>=2.0,<3.0"
"titiler-mosaic>=2.0,<3.0"
"titiler-xarray>=2.0,<3.0"
```

This resolves the older [issue #35](https://github.com/developmentseed/titiler-cmr/issues/35) by adopting a dual-backend pattern rather than picking one or the other.

## Examples

The same endpoint families are mirrored across both backends; the path prefix selects which one runs.

### Tile request

=== "xarray backend"

    ```http
    GET /xarray/tiles/WebMercatorQuad/8/40/95
        ?collection_concept_id=C2036881712-POCLOUD
        &variables=analysed_sst
        &temporal=2024-06-15T12:00:00Z
        &colormap_name=thermal
        &rescale=270,305
    ```

    Backed by `titiler.xarray`. Use for NetCDF, HDF5, and Zarr granules. Pass repeated `variables=` for multi-variable requests.

=== "rasterio backend"

    ```http
    GET /rasterio/tiles/WebMercatorQuad/8/40/95
        ?collection_concept_id=C2021957295-LPCLOUD
        &assets=B04,B03,B02
        &temporal=2024-06-15
        &rescale=0,3000
    ```

    Backed by `titiler.core` and `titiler.mosaic`. Use for COG-style granules. Multi-band selection via `assets=`; band-math via `expression=`.

### Statistics

=== "xarray backend"

    ```http
    GET /xarray/statistics
        ?collection_concept_id=C2036881712-POCLOUD
        &variables=analysed_sst
        &temporal=2024-06-15T12:00:00Z
        &bbox=-130,30,-60,50
    ```

=== "rasterio backend"

    ```http
    GET /rasterio/statistics
        ?collection_concept_id=C2021957295-LPCLOUD
        &assets=B04
        &temporal=2024-06-15
        &bbox=-130,30,-60,50
    ```

The granule-search routes (`/bbox/.../granules`, `/point/.../granules`, `/tiles/.../granules`) and the `/timeseries/*` family are backend-independent and live at the application root rather than under the `/xarray/` or `/rasterio/` prefixes.

## Interactive API documentation

The complete, interactive API documentation from the OpenVEDA Cloud staging deployment is below. Please be kind with this API.

<iframe src="https://staging.openveda.cloud/api/titiler-cmr/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick links

- [Open API docs in new tab](https://staging.openveda.cloud/api/titiler-cmr/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://staging.openveda.cloud/api/titiler-cmr/api){:target="_blank"}
- [Source on GitHub](https://github.com/developmentseed/titiler-cmr){:target="_blank"}

## Main endpoint categories

- `/xarray/{endpoint}` — xarray-backed routes (`/tiles`, `/statistics`, `/part`, `/preview`, etc.) for NetCDF and HDF5 granules.
- `/rasterio/{endpoint}` — rasterio/rio-tiler-backed routes for COG-style granules.
- `/granules/*` and granule-search routes:
  - `/bbox/{minx},{miny},{maxx},{maxy}/granules`
  - `/point/{lon},{lat}/granules`
  - `/tiles/{tileMatrixSetId}/{z}/{x}/{y}/granules`
- `/timeseries/*` — dedicated time-series endpoints for temporal analysis across granules.
- Standard helper routes: `/algorithms`, `/colorMaps`, `/tileMatrixSets`, `/conformance`.

!!! tip "Legacy URL compatibility"
    Older `/collections/{collection_id}/...` paths still resolve via 301/308 redirects to the new `/xarray/` or `/rasterio/` prefixes, with parameter renames applied automatically (`concept_id` → `collection_concept_id`, `datetime` → `temporal`, `bands` → `assets`, `bands_regex` → `assets_regex`). Existing integrations should keep working but new code should target the canonical routes.

!!! tip "Authentication"
    Some endpoints may require Earthdata Login authentication depending on the data collection's access restrictions. Bearer tokens are scoped per DAAC and cached server-side.
