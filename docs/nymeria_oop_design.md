# Nymeria OOP Design

This document summarizes the object-oriented structure used by the Nymeria conversion path.

## Scope

Only `dataset_converter.nymeria` is covered here. HDF5 conversion remains unchanged.

## MVNX Motion

`MvnxFrameSliceConfig` stores frame slicing options and validates stride.

`MvnxTimestampNormalizer` detects MVNX files whose `ms` values are 10x larger than the shared Nymeria timeline, then rebuilds timestamps from frame indices and `frameRate`.

`MvnxXmlMotionReader` streams `body_xdata_mvnx` and delegates timestamp correction to an injected normalizer.

`MvnxMotion` is the immutable parsed motion data object used by annotation, SMPL, and SOMA export.

## Annotation

`NarrationCsvReader` loads `activity_summarization.csv` and `atomic_action.csv`.

`NarrationTimelineAligner` anchors the first narration row to the first exported MVNX/body frame while preserving original text durations and intervals.

`FrameTextIndexer` maps narration rows onto body frames and builds text pools plus per-frame indices.

`AnnotationPayloadBuilder` coordinates MVNX loading, narration loading, frame indexing, and final `annotation.npz` payload assembly.

## SMPL

`SmplMotionConverter` converts parsed Xsens segment motion to SMPL `global_orient`, `body_pose`, and `transl`.

`SmplPayloadBuilder` adds SMPL arrays and timestamp metadata to create `smpl/nymeria_smpl.npz`.

## Head Video

`HeadVideoStreamRegistry` validates and resolves stream metadata for `rgb`, `slam-left`, and `slam-right`.

`HeadVideoTimestampMapper` converts VRS device/capture timestamps to VRS `TIME_CODE` timestamps.

`HeadVideoExportConfig` groups video export options.

`FfmpegVideoWriter` and the OpenCV fallback both satisfy the `VideoWriter` protocol.

## Batch

`NymeriaSequenceTask` carries one sequence source/output mapping.

`ConvertedOutputGuard` checks required NPZ keys and rejects stale `nymeria_time_alignment_version` values when `--skip-existing` is used.

## Compatibility

Existing public functions such as `load_mvnx_motion`, `build_annotation_payload`, `build_smpl_motion_payload`, and `export_head_videos` remain available. They now delegate to the OOP classes above, so CLI behavior is unchanged.
