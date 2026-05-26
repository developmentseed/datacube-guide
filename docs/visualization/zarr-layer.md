# zarr-layer

A custom map layer that renders Zarr directly inside MapLibre GL or Mapbox GL JS, without going through deck.gl. Maintained by CarbonPlan under the MIT license. Supersedes [zarr-gl](zarr-gl.md).

## At a glance

- **Repo** — [carbonplan/zarr-layer](https://github.com/carbonplan/zarr-layer)
- **Shape** — Library, single npm package
- **Map host** — MapLibre GL or Mapbox GL JS v3+, via the native `CustomLayerInterface`
- **Render API** — WebGL2, hand-written GLSL
- **Zarr versions** — v2 and v3, autodetected
- **Conventions** — Tiled XYZ; experimental support for GeoZarr `multiscales` in untiled mode (other GeoZarr metadata — CRS, geo-projection — is not parsed)

## What it does

zarr-layer is a focused alternative to deck.gl-raster for users who want Zarr on a web map but do not want to introduce deck.gl. It plugs into MapLibre or Mapbox v3 as a native custom layer, so it composes with vector basemaps, Mapbox-style projections, and existing layer-management code without the deck.gl runtime. The README acknowledges deck.gl-raster as inspiration and reuses `@developmentseed/raster-reproject` for mesh generation, but the surface area and dependency footprint are deliberately smaller.

## How it renders

The layer compiles WebGL2 shaders that normalize values via `(value - clim.min) / (clim.max - clim.min)`, clamp, then look up a 1D RGB16F colormap texture. Multi-band selectors load multiple variables into separate textures and expose them as named GLSL variables (sanitized for shader identifiers), so a NDVI fragment can be written as `(B08 - B04) / (B08 + B04)` directly. Custom fragment shader strings are a supported extension point.

Two rendering modes coexist. **Tiled mode** assumes the data is already in Web Mercator or geographic XYZ tiles and renders against the standard MapLibre/Mapbox tile grid. **Untiled mode**, marked experimental, loads chunks intersecting the viewport at a zoom-appropriate level and supports arbitrary CRS via a proj4 string plus an adaptive Delaunay reprojection mesh. Polar coverage works on MapLibre out of the box; on Mapbox it requires an experimental `renderPoles: true` flag that uses Mapbox internals.

## Zarr handling

Reads use zarrita and detect v3 first, falling back to v2. Codecs supported through zarrita include bytes, zlib, gzip, blosc, lz4, zstd, transpose, crc32c, and bitround, with custom codecs registrable. Spatial dimensions are configured via the `spatialDimensions` option; non-spatial dimensions are sliced via a `selector` API that defaults to index 0 for unspecified dims and accepts either positional indices or coordinate-value lookups. CF conventions like `scale_factor`, `add_offset`, and `_FillValue` are honoured during normalization. Only the `multiscales` portion of GeoZarr is implemented (experimental, untiled mode); CRS and geo-projection metadata are not parsed and must be supplied by the caller.

## Where it fits

Choose zarr-layer when the host map is MapLibre or Mapbox and you want a small, native custom layer rather than the deck.gl ecosystem. It is the most direct path from "I have a Zarr in S3" to "it shows up on my MapLibre map." The trade-off versus deck.gl-raster is less metadata machinery (only the `multiscales` portion of GeoZarr is parsed; CRS and geo-projection metadata are caller-supplied) and no shared rendering stack with COG; the trade-off versus the viewer apps below is that zarr-layer is a layer, not a UI, so time sliders, colorbar widgets, and dataset pickers are the integrator's job.

## Links

- Source: [carbonplan/zarr-layer](https://github.com/carbonplan/zarr-layer)
- Sibling project: [@carbonplan/maps](carbonplan-maps.md), CarbonPlan's earlier regl-based React component library; still actively maintained, with a stricter `ndpyramid` Web Mercator assumption and a React-component API surface
- Related: [zarrita](https://github.com/manzt/zarrita.js), proj4js, `@developmentseed/raster-reproject`, delaunator
