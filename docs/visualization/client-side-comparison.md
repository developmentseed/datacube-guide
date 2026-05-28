# Client-side rendering comparison

Several browser-side libraries and applications now read Zarr directly (via [zarrita](https://github.com/manzt/zarrita.js) or [zarr-js](https://github.com/freeman-lab/zarr-js)) and render on the GPU, no tile server required. They differ in what they sit on top of, what they do with the data, and what they assume about its shape. This page is the cross-cutting view; per-library detail lives in the sibling pages under "Rendering layers" and "Applications".

A parallel comparison for server-side tooling (TiTiler vs Xpublish) is at [Dynamic tiling ecosystem comparison](ecosystem-comparison.md).

The four libraries below are ordered roughly by host: deck.gl, MapLibre/Mapbox (with `@carbonplan/maps` and zarr-layer as React-component-vs-custom-layer alternatives on the same host), and Cesium. The two applications are scene-first viewers with no basemap.

## Rendering layers

| | [deck.gl-raster](deck.gl-raster.md) | [@carbonplan/maps](carbonplan-maps.md) | [zarr-layer](zarr-layer.md) | [zarr-cesium](zarr-cesium.md) |
|---|---|---|---|---|
| Primary author | Development Seed | CarbonPlan | CarbonPlan | NOC UK (Atlantis) |
| Shape | Library (custom deck.gl layer) | React component library | Library (custom map layer) | Library (3 Cesium provider classes) |
| Map host | deck.gl | Mapbox GL v1 (bundled) or MapLibre/Mapbox via `/core` | MapLibre GL / Mapbox GL v3 | CesiumJS (3D globe) |
| Render API | WebGL2 via luma.gl shader modules | regl, hand-written GLSL | WebGL2, hand-written GLSL | WebGL2, custom GLSL inside Cesium primitives |
| Geographic context | Pairs with an external basemap (MapLibre/Mapbox, Google Maps, ...); not used in isolation | Mapbox/MapLibre basemaps | MapLibre/Mapbox basemaps, optional adaptive-mesh reprojection | Cesium globe with imagery layers underneath |
| Native projection support | Web Mercator, plus arbitrary CRS support | Web Mercator only (validated, throws otherwise) | EPSG:3857/4326 tiled, arbitrary CRS via proj4 in untiled mode | EPSG:4326 and EPSG:3857 only, autodetected |
| Zarr versions | v3 | v2 (primary) and v3 via `version` prop, read with `zarr-js` | v2 and v3, autodetected, via zarrita | v2 and v3, autodetected, via zarrita |
| Conventions | GeoZarr (spatial, multiscales, proj) | Strict `ndpyramid` Web Mercator pyramids, `multiscales` in `.zattrs` | Tiled XYZ, experimental GeoZarr `multiscales` support, user-supplied `spatialDimensions` + proj4 string | `ndpyramid` multiscales, CF time decoding, CF dim-name aliasing |
| Dimensionality | N-D, user passes `selection` per non-spatial dim | N-D via selector arrays (e.g. `{ month: [1,2,3] }` → uniforms `month_1`, `month_2`, `month_3`) | N-D via `selector` (multi-band selectors load multiple textures) | 2D imagery, 3D volumetric via draped slices, animated U/V vector fields |
| Custom shader injection | Composable luma.gl shader modules + custom modules | `frag` prop on `<Raster>` | User-injectable fragment-shader strings, multi-band selectors expose named GLSL variables | Single fixed shader; geometry varies by provider |
| License | MIT | MIT | MIT | MIT |
| Maturity | Released regularly; Zarr support added early 2026 | Mature; powers CarbonPlan's published visualizations | Pre-1.0, "active experiment" | Pre-1.0, early development |

## Applications

| | [Browzarr](browzarr.md) | [GridLook](gridlook.md) |
|---|---|---|
| Primary author | Jeran Poehls (MPI-BGC) | DKRZ + MPI-M |
| Shape | Hosted viewer app (Next.js) | Vue 3 SPA, also embeddable as a Vue component |
| Scene host | three.js + react-three-fiber | three.js |
| Render API | WebGL / WebGPU, GLSL3 shaders | WebGL, custom GLSL |
| Geographic context | None, just a sphere or flat plane with lat/lon UVs | Coastlines and graticules from Natural Earth, no basemap tiles |
| Native projection support | Globe + flat ortho only | 8 projections plus 3D nearside-perspective globe |
| Zarr versions | v2 and v3, plus NetCDF (wasm) and Icechunk | v2 and v3 |
| Conventions | CF `_ARRAY_DIMENSIONS`, no GeoZarr | CF (`grid_mapping`, `_FillValue`, lat/lon coords) |
| Dimensionality | N-D, with 3D volume raycasting as a first-class mode | 2D fields with time slider; 8 grid topologies (regular, rotated, HEALPix, ICON triangular, Gaussian-reduced, curvilinear, irregular Delaunay) |
| License | Apache 2.0 | MIT |
| Maturity | Pre-1.0 | 1.0 released April 2026 |

## Project framing

**deck.gl-raster** is the most architecturally ambitious: a monorepo of small packages layered so that the raster pipeline is generic over its tile source. Zarr is one backend, COG is the other, and they share the same renderer. It plugs into deck.gl, so users get the rest of the deck.gl ecosystem for free. (Supersedes the earlier `numeric-data-layer`.)

**`@carbonplan/maps`** is the original "dynamic client" library. A React component library built on `regl` and Mapbox GL v1, it is the rendering backbone of CarbonPlan's published visualizations (CMIP6 downscaling, seaweed farming, forest carbon). It assumes data is in a Web Mercator `ndpyramid` and validates that strictly; the trade-off for that constraint is a small, opinionated `<Map>` + `<Raster>` API and a polished `frag` prop for custom-shader injection.

**zarr-layer** is the newer CarbonPlan project, designed to relax `@carbonplan/maps`'s assumptions. It implements MapLibre/Mapbox's `CustomLayerInterface` directly (not React-shaped), supports both tiled and untiled modes, handles arbitrary CRS via proj4 with adaptive Delaunay reprojection, and works with both v2 and v3 via zarrita. The trade-off is that it is less mature (an "active experiment") than the production-tested `@carbonplan/maps`. (Also supersedes the earlier `zarr-gl`.)

**zarr-cesium** is the same library shape (plug into an existing map host) but the host is Cesium. It exposes three Cesium provider classes for three different visualization shapes: a `ZarrLayerProvider` for 2D imagery on the globe, a `ZarrCubeProvider` for 3D volumetric data rendered as draped slices, and a `ZarrCubeVelocityProvider` that hands U/V components to `cesium-wind-layer` for animated particle visualization. Built by NOC for oceanographic and atmospheric use cases.

**Browzarr** is a hosted viewer rather than a library, built on three.js and react-three-fiber. Its distinguishing feature is genuine 3D volumetric rendering via raycasting. There is no embeddable npm package and no basemap; the Earth is rendered as a textured sphere when geographic, otherwise the data is a free-floating cube.

**GridLook** is a Vue 3 viewer with a clearly geoscience-shaped scope: visualizing climate-model output on the native grid topology the model used. Its raison d'être is the eight grid-type detectors (HEALPix, ICON triangular, Gaussian-reduced, rotated lat-lon, ...), not raster pyramids.

## Where the GPU work happens

All of these normalize values, apply a colormap, and handle nodata on the GPU; CPU only does I/O and bookkeeping. Differences are in composition.

- **deck.gl-raster** exposes the pipeline as composable luma.gl shader modules. A user passes `renderPipeline: [LinearRescale, Colormap, FilterNoDataVal, ...]` and the framework wires uniforms and texture bindings. `CompositeBands` allows up to four band textures with per-band UV transforms, which is how the published NDVI and AlphaEarth-embeddings examples work.
- **`@carbonplan/maps`** uses hand-written GLSL via regl. Multi-band is expressed via selector arrays that lift to GLSL uniforms (`{ month: [1,2,3] }` becomes `month_1`, `month_2`, `month_3` in the shader), and the `frag` prop on `<Raster>` lets users inject custom fragment-shader logic for averaging, differencing, and conditional masking.
- **zarr-layer** uses fixed shader templates with user-injectable fragment-shader strings. Multi-band is supported via `selector: { band: ['B08', 'B04'] }`, which loads each band into its own texture and exposes them as named GLSL variables, so NDVI is `(B08 - B04) / (B08 + B04)` in a custom fragment.
- **zarr-cesium** runs a single fragment shader that handles `scale_factor`/`add_offset`, NaN/nodata masking, normalization, and colormap LUT lookup. The interesting variation is in geometry rather than shading: the 2D provider tiles into Cesium's standard imagery scheme, the 3D provider loads a cube into memory as an `ndarray` and renders user-positioned horizontal/vertical slices as Cesium primitives, and the velocity provider feeds U/V slices to the third-party `cesium-wind-layer` particle simulator.
- **Browzarr** has fixed shaders per render mode (volumetric raycasting, sphere displacement, flat orthographic). The pipeline is normalize-to-uint8, sample colormap LUT, apply transfer function with NaN color and alpha. Live remap via `cOffset`/`cScale` uniforms avoids re-uploading texture data when the user drags the color-range slider.
- **GridLook** also uses purpose-built shaders, with the twist that the *projection* runs on the GPU. d3-geo computes geometry on the CPU once, then GLSL handles per-pixel reprojection so panning a Mollweide globe stays smooth. Colormap is a 1D LUT after normalization (offset + scale-factor) and optional posterization.

## Geospatial integration

The host environment is the cleanest dividing line.

**deck.gl-raster**, **`@carbonplan/maps`**, and **zarr-layer** are 2D-map-first. They sit inside an existing web-map ecosystem (deck.gl and Mapbox/MapLibre respectively), inherit basemap, attribution, picking, and projection handling, and primarily produce slippy-tile views. `@carbonplan/maps` enforces Web Mercator strictly; deck.gl-raster and zarr-layer have escape hatches for non-Mercator data, but the default mental model is "raster on a web map."

deck.gl is not itself a basemap provider, and most deck.gl users [pair it with a basemap](https://deck.gl/docs/get-started/using-with-map) — most commonly MapLibre/Mapbox, but also Google Maps, OpenLayers, or ArcGIS. When interleaved with MapLibre/Mapbox, deck.gl renders through the *same* `CustomLayerInterface` that zarr-layer implements directly, so the Zarr layer draws inside the MapLibre layer stack (the [ECMWF example](https://developmentseed.org/deck.gl-raster/examples/dynamical-zarr-ecmwf/) renders the Zarr data underneath the basemap's text labels this way). The difference from zarr-layer is that deck.gl is an abstraction that is *not* tied to MapLibre/Mapbox: you can use it with them, but you are not forced to.

**zarr-cesium** is globe-first. The host is CesiumJS, with its imagery layers underneath the data, so the integration story matches the 2D-map libraries (plug into the host's standard extension points), but the canvas is a 3D globe rather than a 2D web map.

**Browzarr** and **GridLook** are scene-first with no basemap. Browzarr renders into a free three.js scene where the geographic case is just one of several render modes; volumetric raycasting has no map analogue. GridLook draws the Earth as a 3D sphere with coastlines and graticules, plus eight 2D projections, but at no point is there a tile basemap underneath.

## Zarr depth and conventions

**deck.gl-raster** has the most opinionated metadata stance: a separate `@developmentseed/geozarr` package parses the GeoZarr conventions (spatial, multiscales, proj). For datasets without GeoZarr metadata, you can pass in GeoZarr metadata by hand. It produces a generic `TilesetDescriptor`, which `@developmentseed/deck.gl-zarr`'s `ZarrLayer` feeds into the same renderer that COG uses. Zarr v2, OME-NGFF, and CF are listed as future work.

**`@carbonplan/maps`** is strict in a different way: it requires the data already be an `ndpyramid` Web Mercator pyramid with `multiscales` metadata in `.zattrs`. There is no in-browser CRS reprojection and no untiled mode; the assumption is all preprocessing happens upstream.

**zarr-layer** is convention-light by comparison: tiled XYZ where the user supplies the spatial-dim mapping, plus experimental support for the GeoZarr `multiscales` convention. It does not parse the rest of GeoZarr (CRS, geo-projection metadata) — projection is user-supplied as a proj4 string.

**zarr-cesium** uses `ndpyramid` for multiscale, decodes CF time, and aliases common CF dimension names. No GeoZarr; only EPSG:4326 and EPSG:3857 are supported, so curvilinear or rotated grids must be reprojected upstream.

**Browzarr** reads `_ARRAY_DIMENSIONS` (CF) for axis names but isn't trying to be a geospatial tool, so projections, CRS, and tiling conventions don't really apply.

**GridLook** leans hard on CF (`grid_mapping`, `_FillValue`/`missing_value`, lat/lon variable detection) and on grid-topology detection rather than on GeoZarr. The HEALPix and ICON-triangular paths in particular are well outside what the others handle.

## Picking the right tool

- For web mapping in deck.gl (optionally [integrating with](https://deck.gl/docs/get-started/using-with-map) basemaps such as MapLibre/Mapbox, Google Maps, OpenLayers, ArcGIS): **deck.gl-raster** with its new Zarr layer.
- For Zarr on MapLibre/Mapbox without a deck.gl dependency: **zarr-layer**.
- For environmental, oceanographic, or atmospheric data on a Cesium 3D globe, especially with 3D slices or animated vector fields: **zarr-cesium**.
- For exploratory science visualization, especially anything volumetric or where you want a hosted viewer for a Zarr URL: **Browzarr**.
- For climate-model output on its native grid (HEALPix, ICON, rotated lat-lon, Gaussian-reduced) with multiple cartographic projections: **GridLook**.

`@carbonplan/maps` is intentionally omitted from the recommendation list. It is the rendering backbone of CarbonPlan's published visualizations, but for new work the requirement that data be pre-baked into `ndpyramid` Web Mercator pyramids carries the same trade-offs as any pre-rendering pipeline: frozen projection, frozen pyramid structure, and a regen step on every data update. zarr-layer is the more flexible CarbonPlan path for new MapLibre/Mapbox integrations.

Roughly: deck.gl-raster, zarr-layer, and zarr-cesium are libraries you embed in a host map or globe; Browzarr and GridLook are scenes/apps you point at data. Within the libraries, the host (deck.gl, MapLibre/Mapbox, Cesium) is the choice. Within the apps, the choice is exploratory 3D versus ESM-grid-aware 2D/3D.

## Related

- [Dynamic tiling ecosystem comparison](ecosystem-comparison.md): server-side tooling (TiTiler vs Xpublish).
- [NASA-IMPACT Zarr Visualization Report](https://nasa-impact.github.io/zarr-visualization-report/): empirical benchmarks for Zarr v3 sharding, pyramid choices, AWS region effects, and pixels-per-tile, from the 2024 study.
