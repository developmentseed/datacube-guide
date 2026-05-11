# Browzarr

A browser-native viewer for multi-dimensional Zarr and NetCDF datasets, with first-class 3D volumetric rendering. Maintained by the Max Planck Institute for Biogeochemistry (EarthyScience) under the Apache 2.0 license.

## At a glance

- **Repo** — [EarthyScience/Browzarr](https://github.com/EarthyScience/Browzarr)
- **Live demo** — [browzarr.io](https://browzarr.io)
- **Shape** — Viewer application (Next.js, static export). Not currently published as an embeddable npm package.
- **Render API** — three.js + react-three-fiber, GLSL3 shaders, with WebGPU support enabled via `webgpu-utils`
- **Zarr versions** — v2 and v3 via zarrita; also reads NetCDF (via `netcdf4-wasm`) and Icechunk stores (via `icechunk-js`)
- **Conventions** — CF `_ARRAY_DIMENSIONS` for axis naming

## What it does

Browzarr is the most "data-centric" of the client-side options listed here. The user points the hosted app at a remote Zarr URL or uploads a local file, and the dataset is rendered directly in the browser without a backing tile server. The distinguishing capability is genuine 3D volumetric rendering of Zarr cubes, not just slices of them.

## How it renders

Three rendering modes share the same data pipeline.

- **Data Cube** — full 3D volume rendering via fragment-shader raycasting, with both orthographic and perspective cameras. Implemented in `volFragment.glsl`.
- **FlatMap** — 2D orthographic slicing with pan and zoom. Implemented in `flatFrag.glsl`.
- **Sphere** — globe rendering with displacement mapping for elevation, useful for global datasets. Implemented in `sphereVertex.glsl`.

Additional point cloud and block modes handle sparse data. Animation is driven by GSAP for temporal transitions, and post-processing (shadows, bloom) is provided by react-three-postprocessing.

The data pipeline is normalize-to-uint8 then sample. Raw float values are scaled to [0, 1] and uploaded as 8-bit textures; the fragment shader looks them up against one of 70+ colormaps from `js-colormaps-es` (magma, viridis, turbo, Spectral, ...). Adjustable `cOffset` and `cScale` uniforms allow live remapping of the value range without re-uploading texture data, and a separate NaN colour and alpha control nodata appearance.

## Zarr handling

Reads use zarrita 0.6.x. An LRU cache (`ZarrLoaderLRU.ts`) holds recently accessed chunks. Geographic datasets get latitude and longitude bounds passed as shader uniforms, and the `useCoordBounds()` hook computes UV mapping for the sphere render mode. There is no GeoZarr support, no tile pyramid, no projection beyond geographic lat/lon, and no basemap concept; the Earth, when present, is just a textured sphere.

## Where it fits

Choose Browzarr when the goal is exploration of a multi-dimensional cube as a cube, with volumetric rendering and free 3D camera movement, rather than as a 2D web map. It is also the right starting point for users who want a hosted viewer to point at a Zarr URL with no integration work. It is not a substitute for a basemap-anchored 2D map: if the dataset belongs on top of a vector basemap with picking and panning, deck.gl-raster or zarr-layer are the fits.

## Links

- Source: [EarthyScience/Browzarr](https://github.com/EarthyScience/Browzarr)
- Hosted: [browzarr.io](https://browzarr.io)
- Authors: Jeran Poehls and Lazaro Alonso (MPI-BGC Jena)
- Funding: EU Horizon Europe (AI4PEX) and ESA SeasFire
- Related: [zarrita](https://github.com/manzt/zarrita.js), [icechunk-js](https://github.com/earth-mover/icechunk), [@earthyscience/netcdf4-wasm](https://github.com/EarthyScience/netcdf4-wasm)
