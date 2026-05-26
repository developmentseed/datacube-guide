# deck.gl-raster

GPU-accelerated raster visualization for deck.gl, with backends for both Cloud-Optimized GeoTIFF (COG) and Zarr. Maintained by Development Seed under the MIT license. Supersedes [numeric-data-layer](numeric-data-layer.md).

## At a glance

- **Repo** — [developmentseed/deck.gl-raster](https://github.com/developmentseed/deck.gl-raster)
- **Shape** — Library, distributed as 12 npm packages under the `@developmentseed` namespace
- **Map host** — deck.gl (typically with MapLibre GL, Mapbox GL, or Google Maps as the basemap)
- **Render API** — WebGL2 via luma.gl shader modules
- **Zarr versions** — v3 (in the current `@developmentseed/deck.gl-zarr` package)
- **Conventions** — GeoZarr (multiscales, proj, CRS as EPSG, WKT2, or PROJJSON)

## What it does

deck.gl-raster provides a generic raster pipeline that is independent of the tile source. The same `RasterTileLayer` and `RasterLayer` render either COG tiles (via `@developmentseed/cogeotiff`) or GeoZarr tiles (via `@developmentseed/deck.gl-zarr`), because the architecture introduces a common `TilesetDescriptor` interface that both backends satisfy. The result is that a project standardising on deck.gl can mix Zarr and COG layers in the same scene without bringing in a second rendering stack.

## How it renders

The pipeline is composed from luma.gl shader modules. A user constructs a render pipeline as a list, for example `renderPipeline: [LinearRescale, Colormap, FilterNoDataVal]`, and the framework wires uniforms and texture bindings on the GPU. Built-in modules cover linear rescaling to [0, 1], colormap LUT sampling against a 256-entry RGBA texture, nodata masking, and band compositing of up to four input bands with per-band UV transforms. Users can write custom fragment shaders to express arbitrary band math; the published examples include NDVI from NIR and Red textures and an AlphaEarth Foundation embeddings mosaic.

Reprojection from the source CRS to Web Mercator is performed once on the CPU as a triangle mesh (using `@developmentseed/raster-reproject`), then rasterized on the GPU. This keeps non-Mercator data renderable inside an otherwise Mercator deck.gl scene.

## Zarr handling

Zarr support is split across two new packages. `@developmentseed/geozarr` parses the GeoZarr conventions, including multiscales, axis order, and CRS, and produces a `TilesetDescriptor`. `@developmentseed/deck.gl-zarr` wraps that descriptor in a `ZarrLayer` that delegates to the generic `RasterTileLayer`. Non-spatial dimensions are addressed via an explicit `selection: Record<string, SliceInput>` map keyed by dimension name, so time, band, depth, and similar axes are first-class.

Reads use zarrita. Zarr v2, OME-NGFF, and CF conventions are listed as future work.

## Where it fits

Choose deck.gl-raster when the wider scene is already deck.gl, when you want a single rendering stack across COG and Zarr, or when you need GeoZarr metadata handling out of the box. The cost is the deck.gl dependency and the assumption of a tile-pyramid view of the data; this is not the right tool for in-browser volumetric exploration of a Zarr cube.

## Links

- Source: [developmentseed/deck.gl-raster](https://github.com/developmentseed/deck.gl-raster)
- Key packages: `@developmentseed/raster-layer`, `@developmentseed/deck.gl-zarr`, `@developmentseed/geozarr`, `@developmentseed/raster-reproject`
- Related: deck.gl, luma.gl, [zarrita](https://github.com/manzt/zarrita.js)
