# CLI Reference

This page documents the installed command line tools in `dataset_converter`.

Both commands are batch-oriented. They discover tasks under `--test-data-root`, write outputs under `--output-root`, and optionally write a JSONL summary with `--summary-path`.

## Install

Base annotation and SMPL export:

```bash
cd dataset_converter
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

SOMA BVH export:

```bash
uv pip install -e ".[soma]"
```

Nymeria VRS video export:

```bash
uv pip install -e ".[video]"
```

If `ffmpeg` is available on `PATH`, Nymeria MP4 export uses it automatically. Otherwise it falls back to OpenCV `mp4v`.

Converted Nymeria visualization:

```bash
uv pip install -e ".[viewer]"
```

## Shared Batch Behavior

`--exports` selects one or more stages. Requested stages run in this order when the command supports them:

```text
annotation -> smpl -> soma-bvh -> head-video
```

`--workers` is used only by CPU/IO stages. SOMA BVH is always sequential because it uses CUDA and large model state.

`--start-frame`, `--end-frame`, and `--stride` slice the input timeline before export. `--end-frame -1` means "until the end".

`--skip-existing` skips a stage when its expected output already exists.

`--summary-path` writes one JSON object per task and stage:

```json
{"stage": "smpl", "task_id": "subset/ep1", "ok": true, "outputs": ["..."], "error": ""}
```

`--fail-fast` stops after the first failed stage. Without it, later requested stages still run.

## `dataset-converter-hdf5-batch`

Converts Xperience/HDF5 episodes.

Required input layout:

```text
<test-data-root>/
└── <subset_id>/
    └── <episode_id>/
        └── annotation.hdf5
```

Task id format:

```text
<subset_id>/<episode_id>
```

Default output layout:

```text
<output-root>/<subset_id>/<episode_id>/
├── annotation_soma.npz
├── smpl/
│   └── annotation_<start_ts>_<end_ts>.npz
└── soma_bvh/
    └── annotation_<start_ts>_<end_ts>.bvh
```

### HDF5 Stages

`annotation`

Exports the text/timeline payload to `annotation_soma.npz`. It requires `caption`, `full_body_mocap/frame_nums`, and video timestamp fields. Missing `caption` is treated as a failed task.

`smpl`

Exports segmented SMPL `.npz` files. It requires:

```text
full_body_mocap/frame_nums
full_body_mocap/Ts_world_root
full_body_mocap/body_quats
full_body_mocap/betas
video/frame_number
video/device_timestamp
```

`soma-bvh`

Runs SMPL-to-SOMA conversion and writes segmented BVH files. It requires the same body fields as `smpl`, plus SOMA assets and a SMPL model.

### HDF5 Common Options

`--test-data-root PATH`

Root containing `<subset>/<episode>/annotation.hdf5`.

`--output-root PATH`

Root where converted files are written.

`--exports {annotation,smpl,soma-bvh} [...]`

Stages to run. Default: `annotation smpl`.

`--workers N`

Worker count for `annotation` and `smpl`.

`--start-frame N`, `--end-frame N`, `--stride N`

Slice `full_body_mocap` rows by index before export.

`--skip-existing`

Skip already exported stage outputs.

`--summary-path PATH`

Write JSONL status for all tasks/stages.

`--fail-fast`

Stop after the first failing stage.

### HDF5 SMPL Options

`--filename-prefix TEXT`

Prefix for segmented `smpl` and `soma-bvh` filenames. Default: `annotation`.

`--smpl-frame {soma_y_up,raw}`

Coordinate frame for exported SMPL `.npz`. Default: `soma_y_up`.

### HDF5 SOMA BVH Options

These options are used only when `--exports` includes `soma-bvh`.

`--device TEXT`

Torch device for SOMA inversion. Default: `cuda`.

`--batch-size N`

GPU batch size. Smaller values reduce memory usage.

`--soma-assets-root PATH`

SOMA assets directory. Falls back to `SOMA_ASSETS_ROOT`.

`--smpl-model-path PATH`

SMPL neutral model path. Falls back to `SMPL_MODEL_PATH` or the SOMA assets directory.

### HDF5 Examples

Annotation and SMPL:

```bash
dataset-converter-hdf5-batch \
  --test-data-root ../xperience-10m \
  --output-root test_out/xp_batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing \
  --summary-path test_out/xp_batch/summary.jsonl
```

SOMA BVH:

```bash
dataset-converter-hdf5-batch \
  --test-data-root ../xperience-10m \
  --output-root test_out/xp_batch \
  --exports soma-bvh \
  --soma-assets-root "$SOMA_ASSETS_ROOT" \
  --smpl-model-path "$SMPL_MODEL_PATH" \
  --batch-size 64 \
  --skip-existing
```

## `dataset-converter-nymeria-batch`

Converts Nymeria sequences.

Required input layout:

```text
<test-data-root>/
└── <sequence_id>/
    ├── body_xdata_mvnx
    ├── narration/
    │   ├── activity_summarization.csv
    │   └── atomic_action.csv
    └── recording_head/
        └── data/
            └── data.vrs
```

`recording_head/data/data.vrs` is required only for `head-video`.
`recording_head/data/motion.vrs` is not enough for video export: it contains motion sensor streams, while RGB/SLAM image streams live in `data.vrs`.

Task id format:

```text
<sequence_id>
```

Default output layout:

```text
<output-root>/<sequence_id>/
├── annotation.npz
├── smpl/
│   └── nymeria_smpl.npz
├── soma_bvh/
│   └── nymeria_soma.bvh
└── head_video/
    ├── slam_left.mp4
    ├── slam_right.mp4
    └── timestamps.npz
```

### Nymeria Stages

`annotation`

Exports text/timeline metadata from MVNX timestamps and narration CSV files. `activity_summarization.csv` is saved as sub task text. `atomic_action.csv` is saved as current action text. Main task and interaction default to `UNKNOWN`.

The output `annotation.npz` includes a corrected MVNX timeline (`frame_timestamps`, `frame_timestamps_ns`) rebuilt from MVNX frame indices and `frameRate`, plus `raw_frame_timestamps` for the original MVNX `ms` attributes. This keeps the first MVNX clock sample but fixes Nymeria files whose MVNX frame deltas are 10x too large. If this stage is exported together with `head-video --video-streams rgb`, it also includes `time_zero_ns`, `time_zero_source`, and `relative_frame_timestamps_ns`, where zero is the first RGB frame VRS `TIME_CODE` timestamp. Text segment start/end timestamps are also saved in absolute nanoseconds, with relative versions when `time_zero_ns` is available. Narration rows are anchored to the first exported MVNX/body frame while preserving their original segment durations and intervals.

`smpl`

Converts MVNX body motion to standard SMPL `.npz` fields.

The output `smpl/nymeria_smpl.npz` includes SMPL pose fields plus corrected `timestamps_ns`, original `raw_timestamps_ns`, and `frame_indices`. If RGB time zero is available from the same batch command, it also includes `time_zero_ns`, `time_zero_source`, and `relative_timestamps_ns`.

`soma-bvh`

Runs SMPL-to-SOMA conversion and writes a BVH file. Requires SOMA assets and a SMPL model.

`head-video`

Reads `recording_head/data/data.vrs` and writes MP4 files plus `timestamps.npz`. Default streams are `slam-left` and `slam-right`; these are stereo grayscale streams. Use `--video-streams rgb` for the color camera.

The timestamp sidecar stores each exported stream's VRS `TIME_CODE` timestamps in `*_timestamps_ns`, original device/capture timestamps in `*_capture_timestamps_ns` for diagnostics, frame indices, estimated FPS, `time_domain=time_code`, and `capture_time_domain=device_time`. When RGB is exported, `rgb_timestamps_ns[0]` becomes `time_zero_ns` and each stream receives `*_relative_timestamps_ns`.

### Nymeria Common Options

`--test-data-root PATH`

Root containing `<sequence>/body_xdata_mvnx`.

`--output-root PATH`

Root where converted files are written.

`--exports {annotation,smpl,soma-bvh,head-video} [...]`

Stages to run. Default: `annotation smpl`.

`--workers N`

Worker count for `annotation`, `smpl`, and `head-video`.

`--start-frame N`, `--end-frame N`, `--stride N`

Slice MVNX rows and VRS stream frame indices before export.

`--skip-existing`, `--summary-path PATH`, `--fail-fast`

Same behavior as the HDF5 command.

### Nymeria SOMA BVH Options

These options are used only when `--exports` includes `soma-bvh`.

`--device TEXT`

Torch device for SOMA inversion. Default: `cuda`.

`--batch-size N`

GPU batch size. Default is tuned for Nymeria but can be lowered for out-of-memory errors.

`--soma-assets-root PATH`

SOMA assets directory. Falls back to `SOMA_ASSETS_ROOT`.

`--smpl-model-path PATH`

SMPL neutral model path. Falls back to `SMPL_MODEL_PATH` or the SOMA assets directory.

### Nymeria Head-Video Options

These options are used only when `--exports` includes `head-video`.

`--video-streams {slam-left,slam-right,rgb} [...]`

VRS camera streams to export. Default: `slam-left slam-right`.

`--video-fps FLOAT`

Override MP4 FPS. If omitted, FPS is estimated from VRS timestamps.

`--video-max-frames N`

Limit frames per stream for previews or quick tests.

`--video-rotate-degrees {0,90,180,270}`

Rotate exported frames clockwise.

### Nymeria Examples

Annotation and SMPL:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports annotation smpl \
  --workers 4 \
  --skip-existing \
  --summary-path nymeria_parse/out/batch/summary.jsonl
```

Color head RGB video:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports head-video \
  --video-streams rgb \
  --workers 2 \
  --skip-existing
```

Annotation, SMPL, and RGB video on a shared RGB-zero timeline:

```bash
dataset-converter-nymeria-batch \
  --test-data-root dataset_converter/test_data/nymeria_test_data \
  --output-root dataset_converter/test_out/nymeria_batch \
  --exports annotation smpl head-video \
  --video-streams rgb \
  --workers 4 \
  --skip-existing \
  --summary-path dataset_converter/test_out/nymeria_batch/summary.jsonl
```

This is the recommended command when you want to inspect skeleton, text, and video together. The command resolves the first RGB VRS `TIME_CODE` timestamp first, then writes the same `time_zero_ns` into `annotation.npz`, `smpl/nymeria_smpl.npz`, and `head_video/timestamps.npz`.

Short video preview:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root /tmp/nymeria_preview \
  --exports head-video \
  --video-streams rgb \
  --video-max-frames 300 \
  --stride 2
```

SOMA BVH:

```bash
dataset-converter-nymeria-batch \
  --test-data-root nymeria_parse/test_data \
  --output-root nymeria_parse/out/batch \
  --exports soma-bvh \
  --soma-assets-root "$SOMA_ASSETS_ROOT" \
  --smpl-model-path "$SMPL_MODEL_PATH" \
  --batch-size 64 \
  --skip-existing
```

## Failure Handling

The commands do not invent missing data.

Missing HDF5 `full_body_mocap` fields fail and are skipped because SMPL cannot be recovered without body motion.

Missing HDF5 `caption` fails the `annotation` stage by design.

Missing Nymeria narration rows leave affected frames as `UNKNOWN`; missing required MVNX/VRS files fail the relevant stage.

For Nymeria `head-video`, a missing `recording_head/data/data.vrs` fails only the video stage. `annotation` and `smpl` can still be exported from `body_xdata_mvnx`, but they will not receive RGB-relative timestamps unless the RGB `data.vrs` is available in the same batch run.

## `dataset-converter-nymeria-viewer`

Visualizes one converted Nymeria sequence with Rerun. This tool is intentionally independent from the official Nymeria viewer source code. It reads only `dataset_converter` outputs:

```text
<sequence-output-dir>/
├── annotation.npz
├── smpl/
│   └── nymeria_smpl.npz
└── head_video/
    ├── rgb.mp4
    └── timestamps.npz
```

The Rerun timeline prefers the relative nanosecond timestamps written by the batch command: `annotation.npz/relative_frame_timestamps_ns` for SMPL/text and `head_video/timestamps.npz/rgb_relative_timestamps_ns` for video. If those fields are missing, it falls back to the older absolute timestamp fields. At each motion frame it logs SMPL joints/bones and the four text categories, so the Rerun time panel can scrub skeleton, text, and RGB video together around the RGB first-frame zero point.

The viewer also logs a static `world/text/time_axis` document. It shows whether motion/text and RGB are using relative or absolute timestamps, the `time_domain`, and the `time_zero_source`.

### Viewer Options

`--sequence-dir PATH`

Converted sequence directory containing `annotation.npz` and `smpl/nymeria_smpl.npz`.

`--smpl-model-path PATH`

Path to `SMPL_NEUTRAL.npz`. The viewer uses this model locally to compute SMPL joint positions from `global_orient`, `body_pose`, `transl`, and `betas`.

`--output-rrd PATH`

Write a Rerun `.rrd` recording to this path instead of spawning the interactive viewer.

`--save-rrd`

Shortcut for writing `<sequence-dir>/converted_nymeria.rrd`.

`--stride N`

Skeleton/text frame stride. Default: `1`.

`--video-stride N`

RGB video frame stride. Default: `1`.

### Viewer Examples

Interactive viewer:

```bash
dataset-converter-nymeria-viewer \
  --sequence-dir nymeria_parse/out/batch/<sequence_id> \
  --smpl-model-path "$SMPL_MODEL_PATH"
```

Save a smaller preview recording:

```bash
dataset-converter-nymeria-viewer \
  --sequence-dir nymeria_parse/out/batch/<sequence_id> \
  --smpl-model-path "$SMPL_MODEL_PATH" \
  --output-rrd /tmp/nymeria_preview.rrd \
  --stride 4 \
  --video-stride 4
```
