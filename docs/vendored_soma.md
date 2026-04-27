# Vendored SOMA Runtime

`dataset_converter` vendors the SOMA Python runtime under:

```text
dataset_converter/src/soma/
```

This makes `import soma` resolve from the installed `dataset_converter` package instead of relying on an external source-tree path being injected into `sys.path`.

## Source

The vendored runtime was copied from the local SOMA source tree:

```text
/home/hpx/HPX_LOCO_2/SOMA-X/soma
```

The original files include SPDX headers from NVIDIA. Keep those headers intact when updating the vendored code.

## What Is Not Vendored

Large SOMA assets are not copied into the Python package. They remain an explicit runtime input:

```bash
export SOMA_ASSETS_ROOT=/path/to/soma/assets
export SMPL_MODEL_PATH=/path/to/soma/assets/SMPL/SMPL_NEUTRAL.npz
```

This keeps the source package small while removing the dependency on the external SOMA source tree.

## Future Environment Work

The package is now structured so a future `uv` environment can declare the SOMA runtime dependencies explicitly. That environment is not created yet.
