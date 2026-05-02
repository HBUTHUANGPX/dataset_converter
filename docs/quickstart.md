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

The exported SOMA BVH follows the SOMA retargeter convention: `Root` stays as a zero virtual root, `Hips` carries the root motion in SOMA's Y-up BVH frame, and all BVH position channels are written in centimeters. `Hips` position channels are written as the SOMA reference offset plus motion relative to the first valid frame, so absolute source heights from MVNX/HDF5 are not double-counted. HDF5 raw SMPL root motion is converted to that Y-up frame before SOMA inversion. If you are regenerating files after changing exporter code, remove old `soma_bvh/*.bvh` files or omit `--skip-existing`; otherwise existing BVHs are intentionally left untouched.

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

## 5. Export Nymeria Head Video

Nymeria head-mounted video lives in Project Aria VRS files. Install the optional video dependencies:

```bash
cd dataset_converter
uv pip install -e ".[video]"
cd ..
```

The exporter automatically uses system `ffmpeg` when available, and falls back to OpenCV `mp4v` when it is not.

Then export the two SLAM camera streams:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports head-video \
  --workers 2 \
  --skip-existing
```

Each sequence writes:

```text
<output-root>/<sequence_id>/head_video/
├── slam_left.mp4
├── slam_right.mp4
└── timestamps.npz
```

The timestamp sidecar stores original VRS capture timestamps and frame indices for each stream. Use it for precise alignment with MVNX/body frames.

The SLAM left/right streams are stereo grayscale cameras. If you need color video, export the RGB stream:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports head-video \
  --video-streams rgb \
  --skip-existing
```

Useful options:

```bash
--video-streams slam-left slam-right rgb
--video-fps 30
--video-max-frames 300
--video-rotate-degrees 90
--stride 2
```

## 6. Check Output

Each summary line is JSON:

```json
{"stage": "smpl", "task_id": "example/ep1", "ok": true, "outputs": ["..."], "error": ""}
```

Failed tasks have `"ok": false` and include the Python exception string in `"error"`.
