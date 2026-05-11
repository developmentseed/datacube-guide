# carbonplan/maps

A React component library for rendering Zarr-backed raster data on Mapbox or MapLibre maps, built around `regl`. Maintained by CarbonPlan under the MIT license. The original "dynamic client" library that informed CarbonPlan's later [zarr-layer](zarr-layer.md), and still actively maintained.

## At a glance

- **Repo** — [carbonplan/maps](https://github.com/carbonplan/maps)
- **Shape** — React component library, single npm package (`@carbonplan/maps`)
- **Map host** — Mapbox GL JS v1.13.1 bundled in the default export, or bring-your-own Mapbox/MapLibre via the `/core` export with `<MapProvider>`
- **Render API** — `regl` (WebGL abstraction) with hand-written GLSL
- **Zarr versions** — v2 (primary) and v3 (via `version` prop), read with `zarr-js` (not zarrita)
- **Conventions** — Strict `ndpyramid` Web Mercator pyramids; reads `.zmetadata` with `multiscales` in `.zattrs`

## What it does

`@carbonplan/maps` exposes a small set of React components, `<Map>`, `<Raster>`, `<Fill>`, `<Line>`, `<RegionPicker>`, plus hooks like `useMap()`, `useRegl()`, `useControls()`, `useRegion()`, that let an application compose a Zarr-backed map view declaratively. It is the rendering backbone of CarbonPlan's published visualizations, including the CMIP6 downscaling, seaweed farming, and forest carbon explainers, so its design choices reflect a production track record rather than a research prototype.

## How it renders

The vertex shader handles Web Mercator projection math and tile geometry. The fragment shader looks up data values from a per-tile texture, normalizes against the user-supplied `clim` range, and samples a 1D RGB colormap texture (typically supplied via the sibling `@carbonplan/colormaps` package). Custom fragment-shader logic can be injected via the `frag` prop on `<Raster>`, which is how multi-band math (averaging, differencing, conditional masking) is expressed.

Multi-band selectors lift naturally into shader uniforms: passing `selector={{ month: [1, 2, 3] }}` loads three arrays simultaneously and exposes them as `month_1`, `month_2`, `month_3` GLSL uniforms.

## Zarr handling

Reads use `zarr-js`. Data is required to be in a Web Mercator pyramid built with `ndpyramid`, with consistent `pixels_per_tile` across zoom levels, and the library validates this at load time and throws if the projection is anything else. The `version` prop selects between Zarr v2 and v3 store layouts. There is no GeoZarr support, no untiled mode, and no in-browser CRS reprojection; the assumption is that all preprocessing happens upstream.

## Where it fits

`@carbonplan/maps` is documented here for completeness and is appropriate if you are already using it (e.g., extending one of CarbonPlan's existing visualizations) or if you are committed to a CarbonPlan-style workflow that pre-bakes data into `ndpyramid` Web Mercator pyramids.

For new MapLibre/Mapbox integrations, this guide does not recommend `@carbonplan/maps` as the starting point. The strict `ndpyramid` Web Mercator pyramid requirement carries the same trade-offs as any pre-rendering pipeline: a frozen projection, a frozen pyramid structure, and a regen step on every data update. CarbonPlan's own newer project, [zarr-layer](zarr-layer.md), removes those constraints (untiled mode, arbitrary CRS via proj4, vanilla-JS API) at the cost of being less mature. For the deck.gl ecosystem, [deck.gl-raster](deck.gl-raster.md) is the better fit since it shares a rendering stack with COG and parses GeoZarr metadata directly.

## Links

- Source: [carbonplan/maps](https://github.com/carbonplan/maps)
- Companion: [@carbonplan/colormaps](https://github.com/carbonplan/colormaps)
- Pyramid generator: [ndpyramid](https://github.com/carbonplan/ndpyramid)
- Newer sibling: [zarr-layer](zarr-layer.md), CarbonPlan's vanilla-JS MapLibre/Mapbox custom-layer approach
- Production users: CarbonPlan visualizations (CMIP6 downscaling, seaweed farming, forest carbon)
