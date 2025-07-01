# Default Xarray combine parameters

Xarray's default combine operations  are overly permissive and can lead to unintuitive behavior and poor performance. Guidance for better arguments can be found in the [xarray documentation](https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets) and [issue proposing stricter defaults](https://github.com/pydata/xarray/issues/8778), which is being addressed in an [open pull request](https://github.com/pydata/xarray/pull/10062).

