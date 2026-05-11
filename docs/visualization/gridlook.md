# GridLook

A WebGL viewer for Earth system model output, with first-class support for the unstructured and reduced grids that climate models actually produce. Maintained by DKRZ and MPI-M under the MIT license.

## At a glance

- **Repo** — [dkrz-icon/gridlook](https://github.com/dkrz-icon/gridlook)
- **Hosted** — [gridlook.pages.dev](https://gridlook.pages.dev)
- **Shape** — Vue 3 single-page app, also embeddable as a Vue component
- **Render API** — three.js with custom GLSL shaders
- **Zarr versions** — v2 and v3 via zarrita
- **Conventions** — CF (`grid_mapping`, `_FillValue`, `missing_value`, lat/lon coordinate variables)

## What it does

GridLook's defining feature is grid-topology awareness. Where the other viewers in this section assume a regular lat-lon raster (or accept that as input after preprocessing), GridLook detects and renders eight grid types directly from the data:

1. Regular lat-lon
2. Rotated regular (CORDEX-style, detected via `grid_mapping_name="rotated_latitude_longitude"`)
3. HEALPix (hierarchical, levels 0 to 11+)
4. Triangular meshes (ICON, detected via `vertex_of_cell`)
5. Gaussian reduced (ERA5-style, longitude tapers toward the poles)
6. Curvilinear (2D lat/lon coordinate arrays)
7. Irregular (CMIP6 unstructured)
8. Irregular Delaunay (on-the-fly Delaunay fallback)

The README is explicit that GridLook is "not a competitor to high-end visualization suites" but a tool for intuitive, interactive exploration of cloud-hosted ESM output. Operational use includes the WCRP Global Hackathon 2025 HEALPix datasets and DKRZ-hosted ICON, IFS DYAMOND3, and EERIE collections.

## How it renders

The primary view is a 3D nearside-perspective globe with orbit controls. Eight 2D cartographic projections are also available: Mercator, Robinson, Mollweide, Cylindrical Equal Area, Equirectangular, Azimuthal Equidistant, and a custom Azimuthal Hybrid for seamless polar plus global views. Projection math is computed once on the CPU (via d3-geo and d3-geo-projection) and the per-pixel reprojection runs in GLSL, so panning between projections stays smooth.

Coastlines and graticules use Natural Earth GeoJSON at 50m resolution and are rendered via WebGL after a recent refactor; both are toggleable. Land/sea masking is generated CPU-side from GeoJSON onto a canvas, then applied as a GPU texture.

The data pipeline is load chunk, normalize via offset and scale-factor, optionally posterize, look up against one of 58 built-in colormaps (viridis, inferno, coolwarm, RdBu, BrBG, diff, curl, phase, ...), then render. Histogram-based auto-scaling and per-variable default colormaps and ranges are baked into the dataset catalog.

## Zarr handling

Reads use zarrita 0.7.x with bitround codec support. Chunk loading targets 1 to 5 MB requests, with byte-range coalescing and a 512-entry LRU byte cache. CF metadata (`grid_mapping`, `_FillValue`, `missing_value`, NaN) drives both grid detection and nodata masking. Catalog files (JSON) list datasets for institutional deployments and are how DKRZ exposes its hosted collections.

## Where it fits

Choose GridLook when the data lives on a non-rectilinear grid that the climate community actually uses (HEALPix, ICON triangular, Gaussian-reduced, rotated lat-lon) and you want it visualized on that native grid rather than after a regridding step. The map-hosted libraries above either require regular gridded input or assume web-map tiles, so they are not substitutes for the unstructured-grid case. The trade-off is that GridLook is its own scene, not a layer on top of a basemap; if you need a vector basemap underneath, look elsewhere.

## Links

- Source: [dkrz-icon/gridlook](https://github.com/dkrz-icon/gridlook)
- Hosted: [gridlook.pages.dev](https://gridlook.pages.dev)
- Authors: Andrej Fast (DKRZ), Tobi Kölling (MPI-M), Fabian Wachsmann (DKRZ), Lukas Kluft (MPI-M)
- Related: [zarrita](https://github.com/manzt/zarrita.js), d3-geo, d3-geo-projection
