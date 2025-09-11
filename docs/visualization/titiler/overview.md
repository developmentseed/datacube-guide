# Titiler Ecosystem Overview

The TiTiler ecosystem is a comprehensive suite of Python tools for creating dynamic tile servers from geospatial datasets. The components are organized by their primary function and built upon a layered architecture that provides flexibility and specialization for different use cases.

## Architecture and Relationships

The TiTiler ecosystem follows a layered architecture that promotes code reuse and specialization:

### Foundation Layer
- **[rio-tiler](https://github.com/cogeotiff/rio-tiler)**: Core tile generation engine
- **[titiler.core](https://github.com/developmentseed/titiler/tree/main/src/titiler/core)**: Base FastAPI framework and patterns

### Extension Layer
- **[titiler.xarray](https://github.com/developmentseed/titiler/tree/main/src/titiler/xarray)**: Multidimensional data support extending titiler.core
- **[titiler.extensions](https://github.com/developmentseed/titiler/tree/main/src/titiler/extensions)**: Plugin system for custom functionality
- **[titiler.mosaic](https://github.com/developmentseed/titiler/tree/main/src/titiler/mosaic)**: Multi-source tiling capabilities

### Application Layer
- **[titiler.application](https://github.com/developmentseed/titiler/tree/main/src/titiler/application)**: Reference implementation using titiler.core
- **[titiler-multidim](https://github.com/developmentseed/titiler-multidim)**: Prototype application using titiler.xarray + optimizations
- **[titiler-cmr](https://github.com/developmentseed/titiler-cmr)**: NASA-specific application using titiler.core + CMR integration
- **[titiler-eopf](https://github.com/EOPF-Explorer/titiler-eopf)**: ESA-specific application using titiler.xarray + EOPF integration

### Key Relationships

#### **titiler.core → titiler.xarray**

`titiler.xarray` extends the core framework with xarray-based readers and multidimensional dataset support, inheriting all core functionality while adding temporal and dimensional processing capabilities.

#### **titiler.xarray → titiler-multidim / titiler-eopf applications**

Both `titiler-multidim` and `titiler.eopf` are built on `titiler.xarray`, leveraging its multidimensional capabilities while adding their own optimizations:

- `titiler-multidim` adds Redis caching, VEDA platform integration, and prototypes of optimizations
- `titiler-eopf` adds EOPF-specific data structures, collection/item routing, and ESA workflow integrations

#### **titiler.core → titiler-cmr application**

`titiler-cmr` is built directly on `titiler.core` rather than `titiler.xarray` due to the development timeline of the two projects. In the future, we anticipate
`titiler-cmr` will depend on `titiler.xarray`, with progress tracked by [titiler-cmr issue #35](https://github.com/developmentseed/titiler-cmr/issues/35).

---

*The TiTiler ecosystem provides a complete stack for building scalable, cloud-native tile servers that can handle everything from simple COG serving to complex multi-dimensional scientific datasets.*
