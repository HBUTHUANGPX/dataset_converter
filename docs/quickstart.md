# Quickstart

This guide assumes `dataset_converter/` lives inside the larger `mimic_baseline` workspace.

## 1. Install The Package

Recommended Python version: 3.11.

Using `uv`:

```bash
cd dataset_converter
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cd ..
```

If you are staying in the existing `mimic_baseline` conda environment, `pip install -e dataset_converter` is also fine.

After installation, these commands should work:

```bash
dataset-converter-hdf5-batch --help
dataset-converter-nymeria-batch --help
```

If you do not want to rely on console scripts, use:

```bash
python -m dataset_converter.hdf5.cli.batch_export --help
python -m dataset_converter.nymeria.cli.batch_export --help
```

## 2. Prepare Data

HDF5/Xperience data should look like:

```text
hdf5_parse/test_data/<subset_id>/<episode_id>/annotation.hdf5
```

Nymeria data should look like:

```text
nymeria_parse/test_data/<sequence_id>/body_xdata_mvnx
```

You can put data elsewhere. If so, pass that location with `--test-data-root`.

## 3. Export Annotation And SMPL

These stages are CPU/IO friendly and can use multiple processes.

HDF5:

```bash
dataset-converter-hdf5-batch \
  --test-data-root hdf5_parse/test_data \
  --output-root hdf5_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing \
  --summary-path hdf5_parse/out/batch/summary.jsonl
```

Nymeria:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing \
  --summary-path nymeria_parse/out/batch/summary.jsonl
```

## 4. Export SOMA BVH

SOMA BVH export uses CUDA and requires the SOMA Python runtime to be importable in the active environment. It is intentionally sequential even in batch mode.

Install SOMA/GPU dependencies when needed:

```bash
cd dataset_converter
uv pip install -e ".[soma]"
cd ..
```

If your PyTorch CUDA wheel needs a custom index, install the matching `torch` build first, then install `.[soma]`.

Set paths once:

```bash
export SOMA_ASSETS_ROOT=/path/to/soma/assets
export SMPL_MODEL_PATH=/path/to/soma/assets/SMPL/SMPL_NEUTRAL.npz
```

Then run HDF5 SOMA BVH export:

```bash
dataset-converter-hdf5-batch \
  --exports soma-bvh \
  --soma-assets-root "$SOMA_ASSETS_ROOT" \
  --batch-size 128 \
  --skip-existing
```

Or Nymeria SOMA BVH export:

```bash
dataset-converter-nymeria-batch \
  --exports soma-bvh \
  --soma-assets-root "$SOMA_ASSETS_ROOT" \
  --batch-size 128 \
  --skip-existing
```

If your GPU runs out of memory, lower `--batch-size`, for example `64` or `32`.

## 5. Check Output

Each summary line is JSON:

```json
{"stage": "smpl", "task_id": "example/ep1", "ok": true, "outputs": ["..."], "error": ""}
```

Failed tasks have `"ok": false` and include the Python exception string in `"error"`.
