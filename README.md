# dataset_converter

`dataset_converter` is the new unified package for converting motion datasets in this repository.

It currently owns the CPU/IO-friendly annotation and SMPL export stages, plus the shared SOMA BVH writing/orchestration code used by CUDA export:

- `dataset_converter.hdf5`: Xperience/HDF5 `annotation.hdf5` data.
- `dataset_converter.nymeria`: Nymeria MVNX body motion data.

The old `hdf5_parse/` and `nymeria_parse/` folders are still kept for compatibility. New code and deployment should use this package.

## Install

From this package directory:

```bash
cd dataset_converter
pip install -e .
```

This exposes Python imports such as:

```python
from dataset_converter.hdf5.batch import discover_hdf5_episode_tasks
from dataset_converter.nymeria.batch import discover_nymeria_sequence_tasks
```

It also installs command line entry points:

```bash
dataset-converter-hdf5-batch --help
dataset-converter-nymeria-batch --help
```

You can also run the same CLIs without installing console scripts:

```bash
python -m dataset_converter.hdf5.cli.batch_export --help
python -m dataset_converter.nymeria.cli.batch_export --help
```

## External Assets

The package does not hard-code machine-specific paths. For SOMA/SMPL export, provide paths with CLI arguments or environment variables:

```bash
export SOMA_ASSETS_ROOT=/path/to/soma/assets
export SMPL_MODEL_PATH=/path/to/soma/assets/SMPL/SMPL_NEUTRAL.npz
```

Equivalent CLI arguments:

```bash
--soma-assets-root /path/to/soma/assets
--smpl-model-path /path/to/SMPL_NEUTRAL.npz
```

`annotation` and `smpl` export stages do not require SOMA assets and no longer import implementation code from `hdf5_parse` or `nymeria_parse`. `soma-bvh` requires the SOMA Python runtime to be importable in the active environment.

## Environment Direction

The package is structured so it can later own an independent `uv` environment. That work has not started yet. For now, use the existing `mimic_baseline` environment and install this package editable with `pip install -e dataset_converter`.

The SOMA Python runtime is vendored in `dataset_converter/src/soma`, so `import soma` comes from this package after installation. Large SOMA assets are still explicit runtime inputs through `SOMA_ASSETS_ROOT`.

## Data Layout

HDF5/Xperience:

```text
hdf5_parse/test_data/
└── <subset_id>/
    └── <episode_id>/
        └── annotation.hdf5
```

Nymeria:

```text
nymeria_parse/test_data/
└── <sequence_id>/
    └── body_xdata_mvnx
```

Default outputs are written under:

```text
hdf5_parse/out/batch/
nymeria_parse/out/batch/
```

Use `--output-root` to choose another location.

## Quick Commands

Batch HDF5 annotation and SMPL export:

```bash
dataset-converter-hdf5-batch \
  --test-data-root hdf5_parse/test_data \
  --output-root hdf5_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing
```

Batch Nymeria annotation and SMPL export:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing
```

Batch SOMA BVH export is intentionally sequential because it uses CUDA:

```bash
dataset-converter-hdf5-batch \
  --exports soma-bvh \
  --soma-assets-root "$SOMA_ASSETS_ROOT" \
  --smpl-model-path "$SMPL_MODEL_PATH" \
  --batch-size 128 \
  --skip-existing
```

## More Docs

- [Quickstart](docs/quickstart.md)
- [Deployment](docs/deployment.md)
- [Architecture](docs/architecture.md)
- [Vendored SOMA Runtime](docs/vendored_soma.md)
