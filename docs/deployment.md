# Deployment

This page is for setting up the converter on a new machine.

## Install Editable Package

Recommended Python version: 3.11.

Standalone `uv` environment:

```bash
cd /path/to/mimic_baseline/dataset_converter
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Existing conda environment:

```bash
cd /path/to/mimic_baseline/dataset_converter
pip install -e .
```

The editable install makes `dataset_converter` importable and installs the package CLI entry points. Run commands from the `mimic_baseline` workspace root, or pass explicit `--test-data-root` and `--output-root` paths.

## Dependency Profiles

Base CPU/IO dependencies are listed in `requirements.txt` and installed by the package:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

SOMA/GPU export dependencies are optional:

```bash
uv pip install -e ".[soma]"
```

The `soma` extra includes `torch`, `smplx`, `trimesh`, and `warp-lang`. Because CUDA-enabled PyTorch wheels are platform-specific, install the matching `torch` build first if your host needs a custom PyTorch index.

## Configure External Assets

Do not edit source files to change local paths. Use environment variables or CLI arguments.

Recommended environment variables:

```bash
export SOMA_ASSETS_ROOT=/path/to/soma/assets
export SMPL_MODEL_PATH=/path/to/soma/assets/SMPL/SMPL_NEUTRAL.npz
export SMPLH_MODEL_PATH=/path/to/SMPLH_NEUTRAL.pkl
```

Only `SOMA_ASSETS_ROOT` and `SMPL_MODEL_PATH` are needed for `soma-bvh` export. Basic annotation and SMPL extraction do not need SOMA assets.

The package can now be installed with `uv`; keep using the current `mimic_baseline` environment when you need to reuse already-installed CUDA/PyTorch wheels.

The SOMA Python runtime is included in this package under `dataset_converter/src/soma`. After editable install, this should work without adding an external source tree to `PYTHONPATH`:

```bash
python -c "import soma; print(soma.__file__)"
```

## Validate Installation

```bash
python -m dataset_converter.hdf5.cli.batch_export --help
python -m dataset_converter.nymeria.cli.batch_export --help
```

Then run a tiny frame range:

```bash
dataset-converter-hdf5-batch \
  --exports annotation smpl \
  --end-frame 2 \
  --output-root /tmp/hdf5_batch_smoke
```

```bash
dataset-converter-nymeria-batch \
  --exports annotation smpl \
  --end-frame 2 \
  --output-root /tmp/nymeria_batch_smoke
```

## Common Problems

`SOMA assets root was not found`

Set `SOMA_ASSETS_ROOT` or pass `--soma-assets-root`.

`The SOMA runtime package is not importable`

Install or expose the `soma` Python package in the active environment. The converter no longer modifies `sys.path` to discover an external source tree.

`SMPL model was not found`

Set `SMPL_MODEL_PATH` or pass `--smpl-model-path`.

CUDA out of memory during SOMA export

Lower `--batch-size`. SOMA export is sequential by design, but each sequence is still processed in GPU chunks.

No tasks discovered

Check `--test-data-root` and the expected data layouts:

```text
hdf5_parse/test_data/<subset_id>/<episode_id>/annotation.hdf5
nymeria_parse/test_data/<sequence_id>/body_xdata_mvnx
```
