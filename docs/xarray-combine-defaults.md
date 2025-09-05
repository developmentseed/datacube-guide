# Default Xarray combine parameters

Xarray's default combine operations  are overly permissive and can lead to unintuitive behavior and poor performance. Guidance for better arguments can be found in the [xarray documentation](https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets) and [issue proposing stricter defaults](https://github.com/pydata/xarray/issues/8778).

With Xarray `v2025.08.0` or later, you can opt into more consistent behavior that does not load data by default using `xarray.set_options(use_new_combine_kwarg_defaults=True)`.
