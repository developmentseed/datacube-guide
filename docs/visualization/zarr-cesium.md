# zarr-cesium

CesiumJS provides for visualizing Zarr datasets directly on the Cesium 3D globe, with separate paths for 2D imagery, 3D volumetric slices, and animated vector fields. Maintained by the National Oceanography Centre (NOC) UK under the MIT license.

## At a glance

- **Repo** — [NOC-OI/zarr-cesium](https://github.com/NOC-OI/zarr-cesium)
- **Shape** — Library (TypeScript)
- **Map host** — CesiumJS (not Resium, not deck.gl-cesium)
- **Render API** — WebGL2 with custom GLSL fragment shaders, integrated as Cesium imagery and primitive APIs
- **Zarr versions** — v2 and v3, autodetected via zarrita
- **Conventions** — CF time decoding (days/hours since reference), CF dimension-name aliasing (`time`/`t`/`Time`/`time_counter`); supports `ndpyramid` multiscales

## What it does

zarr-cesium is the only Zarr-native option in this section that targets CesiumJS. It bridges Cesium's existing geometry and imagery APIs by exposing three provider classes, each fitting a different Cesium extension point:

- **`ZarrLayerProvider`** produces a Cesium `ImageryLayer` for 2D rasters draped on the globe.
- **`ZarrCubeProvider`** produces Cesium `Primitive` geometry for 3D volumetric data, rendered as draped horizontal and vertical slices through the cube.
- **`ZarrCubeVelocityProvider`** feeds U/V components into `cesium-wind-layer` for animated particle visualization of ocean currents and winds.

Demo datasets focus on oceanography and atmospheric reanalysis (NEMO model output for salinity, temperature, depth profiles; Hurricane Florence ERA5 wind and pressure), which reflects the NOC use case driving development.

## How it renders

All three providers share the same color pipeline. A fragment shader (`src/shaders.ts`) samples a single-channel data texture, applies `scale_factor` and `add_offset`, masks NaN and nodata, normalizes against per-layer min and max, and looks up a 1D colormap texture built from `jsColormaps` (matplotlib-compatible: viridis, plasma, inferno, jet, ...). All color mapping happens GPU-side; there is no CPU canvas stage.

Where the providers diverge is in geometry:

- The 2D `ZarrLayerProvider` plugs into Cesium's `TilingScheme` (Geographic EPSG:4326 or Web Mercator EPSG:3857), automatically selects a multiscale level for the current zoom, and lets Cesium handle tile lifecycle.
- The 3D `ZarrCubeProvider` loads the requested cube into memory as an `ndarray` and renders user-positioned slices as `RectangleGeometry` primitives with textured materials. This is a slice-rendering approach to volumetric data, distinct from Browzarr's full raycasting.
- The vector provider hands U/V slices off to `cesium-wind-layer`, which runs the particle simulation.

CRS is autodetected (coordinate-range analysis plus metadata inspection); only EPSG:4326 and EPSG:3857 are supported. Curvilinear or rotated-pole grids must be reprojected before storage.

## Zarr handling

Reads use zarrita and try v3 first, falling back to v2. Multiscale pyramids in the `ndpyramid` convention are supported, with automatic level selection in the 2D provider and an explicit `multiscaleLevel` argument in the 3D provider. CF time decoding (`src/decodeCFTime.ts`) handles common reference-date encodings, and time and depth dimensions are sliced dynamically via `updateSelectors()`. Sharding and GeoZarr are not explicitly handled.

## Where it fits

Choose zarr-cesium when the host is Cesium and the data is environmental, oceanographic, or atmospheric, especially when the value of the visualization comes from sitting on a 3D globe rather than a 2D web map. The 3D slice rendering and the wind-layer integration are unique among the libraries in this section. The trade-off is that you commit to the Cesium runtime, and the 3D rendering loads full cubes into memory rather than streaming them, so dataset size for the cube provider is bounded by browser memory.

## Links

- Source: [NOC-OI/zarr-cesium](https://github.com/NOC-OI/zarr-cesium)
- Docs: in-repo under `docs/`
- Author: Tobias Ferreira (NOC), as part of the Atlantis project
- Related: [CesiumJS](https://cesium.com/platform/cesiumjs/), [cesium-wind-layer](https://github.com/hongfaqiu/cesium-wind-layer), [zarrita](https://github.com/manzt/zarrita.js), [ndpyramid](https://github.com/carbonplan/ndpyramid)
