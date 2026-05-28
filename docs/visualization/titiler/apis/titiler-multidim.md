# titiler-multidim API reference

`titiler-multidim` is a TiTiler application built on `titiler.xarray`, designed for multi-dimensional datasets like NetCDF, Zarr, and (since v0.7) Icechunk repositories. Originally framed as a prototype demonstrating advanced multidimensional capabilities, it is now operationally deployed via NASA's VEDA platform and tracks `titiler.xarray` v0.25.x.

## Key features

- **Multi-dimensional support** — native handling of 3D, 4D, and 5D datasets via `titiler.xarray`.
- **Icechunk support** — full `opener_icechunk()` reads Icechunk repositories locally or from S3 (added in v0.7.0). Includes an authorization config for virtual chunk access (`authorized_chunk_access`).
- **Zarr v3** — works against Zarr v3 stores via `zarr>=3.1`.
- **Redis caching** — optional response caching via `TITILER_MULTIDIM_ENABLE_CACHE`. Uses `fakeredis` in tests.
- **OpenTelemetry tracing** — X-Ray and OTel tracing enabled for Lambda and other tracing-aware deployments (added in v0.6.0).
- **VEDA integration** — production deployments orchestrated via [NASA-IMPACT/veda-deploy](https://github.com/NASA-IMPACT/veda-deploy); PRs labeled `deploy-dev` trigger ephemeral environments.

## Backend dependency

```toml
"titiler-core>=0.25.0,<0.26"
"titiler-xarray>=0.25.0,<0.26"
```

Note the version pinning here is to the older `titiler.xarray` 0.25 line rather than the 2.x line used by `titiler.application` and `titiler-cmr`. Bumping to `titiler.xarray` 2.x is tracked in the repository.

## Interactive API documentation

The complete, interactive API documentation from the OpenVEDA Cloud staging deployment is below. Please be kind with this API.

<iframe src="https://staging.openveda.cloud/api/titiler-multidim/api.html"
        width="100%"
        height="800px"
        frameborder="0"
        style="border: 1px solid #ddd; border-radius: 4px;">
</iframe>

## Quick links

- [Open API docs in new tab](https://staging.openveda.cloud/api/titiler-multidim/api.html){:target="_blank"}
- [OpenAPI Schema JSON](https://staging.openveda.cloud/api/titiler-multidim/api){:target="_blank"}
- [Source on GitHub](https://github.com/developmentseed/titiler-multidim){:target="_blank"}

## Main endpoint categories

Inherited from `titiler.xarray`'s `BaseTilerFactory`:

- `/info` — dataset metadata and structure.
- `/statistics` — statistical analysis across selected dimensions.
- `/tiles/{z}/{x}/{y}` and `/tilejson.json` — map tile generation and tile metadata.

Custom multidim endpoints (in `factory.py`):

- `/variables` — list of variables in the dataset.
- `/histogram` — value histogram for a selected variable.
- `/{tileMatrixSetId}/map` — interactive map viewer for the dataset.

Operational:

- `/healthz` — health check.

Format support: NetCDF (via `h5netcdf`), Zarr v3 (via `zarr>=3.1`), Icechunk (via the bundled opener), and virtual references (via `obstore` and `virtualizarr` in dev dependencies).

!!! note "Status"
    `titiler-multidim` was historically described as a prototype. It is now deployed in production via VEDA and is the recommended path for VEDA-shaped multidimensional workflows. The "prototype" framing has been retired.
