from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_converter.common.rotations import quat_wxyz_to_rotvec
from dataset_converter.common.smpl import SMPLBodyMotion, convert_smpl_motion_to_soma_y_up_frame, slice_smpl_body_motion
from dataset_converter.hdf5.io import BodyFrameSelection, load_body_frame_selection, split_pose7_qwxyz_xyz


SMPL_EXPORT_FRAMES = ("soma_y_up", "raw")


def selection_to_smpl_body_motion(selection: BodyFrameSelection) -> SMPLBodyMotion:
    root_quat_wxyz, transl = split_pose7_qwxyz_xyz(selection.root_pose7)
    global_orient = quat_wxyz_to_rotvec(root_quat_wxyz)
    body_pose = quat_wxyz_to_rotvec(selection.body_quats).reshape(selection.num_frames, -1)
    if body_pose.shape[-1] == 63:
        body_pose = np.concatenate([body_pose, np.zeros((selection.num_frames, 6), dtype=np.float32)], axis=-1)
    if body_pose.shape[-1] != 69:
        raise ValueError(f"Expected SMPL body pose with 69 dims after conversion, got {body_pose.shape}.")
    return SMPLBodyMotion(
        global_orient=np.asarray(global_orient, dtype=np.float32),
        body_pose=np.asarray(body_pose, dtype=np.float32),
        transl=np.asarray(transl, dtype=np.float32),
        betas=np.asarray(selection.betas, dtype=np.float32),
        frame_nums=np.asarray(selection.frame_nums, dtype=np.int32),
        frame_timestamps=np.asarray(selection.frame_timestamps, dtype=np.int64),
        fps=float(selection.fps),
    )


def split_contiguous_frame_ranges(frame_nums: np.ndarray, *, expected_step: int = 1) -> list[tuple[int, int]]:
    frame_nums = np.asarray(frame_nums, dtype=np.int64).reshape(-1)
    if frame_nums.size == 0:
        return []
    if expected_step <= 0:
        raise ValueError(f"expected_step must be positive, got {expected_step}.")
    split_points = np.flatnonzero(np.diff(frame_nums) != int(expected_step)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), split_points])
    ends = np.concatenate([split_points, np.array([frame_nums.shape[0]], dtype=np.int64)])
    return [(int(start), int(end)) for start, end in zip(starts.tolist(), ends.tolist())]


def build_segment_file_stem(frame_timestamps: np.ndarray, *, prefix: str = "annotation") -> str:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if frame_timestamps.size == 0:
        raise ValueError("frame_timestamps must contain at least one element.")
    return f"{prefix}_{int(frame_timestamps[0])}_{int(frame_timestamps[-1])}"


def prepare_smpl_motion_for_export(motion: SMPLBodyMotion, *, smpl_frame: str = "soma_y_up") -> SMPLBodyMotion:
    if smpl_frame == "raw":
        return motion
    if smpl_frame == "soma_y_up":
        return convert_smpl_motion_to_soma_y_up_frame(motion)
    raise ValueError(f"Unsupported smpl_frame {smpl_frame!r}. Expected one of {SMPL_EXPORT_FRAMES}.")


def save_smpl_motion_npz(
    motion: SMPLBodyMotion,
    output_path: str | Path,
    *,
    smpl_frame: str = "soma_y_up",
) -> Path:
    export_motion = prepare_smpl_motion_for_export(motion, smpl_frame=smpl_frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.asarray(int(round(float(export_motion.fps))), dtype=np.int32),
        num_frames=np.asarray(export_motion.num_frames, dtype=np.int32),
        frame_nums=np.asarray(export_motion.frame_nums, dtype=np.int32),
        frame_timestamps=np.asarray(export_motion.frame_timestamps, dtype=np.int64),
        smpl_frame=np.asarray(str(smpl_frame)),
        smpl_global_orient=np.asarray(export_motion.global_orient, dtype=np.float32),
        smpl_body_pose=np.asarray(export_motion.body_pose, dtype=np.float32),
        smpl_transl=np.asarray(export_motion.transl, dtype=np.float32),
        smpl_betas=np.asarray(export_motion.betas, dtype=np.float32),
    )
    return output_path


def export_segmented_smpl_npz(
    hdf5_path: str | Path,
    *,
    smpl_output_dir: str | Path,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    filename_prefix: str = "annotation",
    smpl_frame: str = "soma_y_up",
) -> list[Path]:
    selection = load_body_frame_selection(hdf5_path, start_frame=start_frame, end_frame=end_frame, stride=stride)
    ranges = split_contiguous_frame_ranges(selection.frame_nums, expected_step=max(1, int(stride)))
    if not ranges:
        return []

    motion = selection_to_smpl_body_motion(selection)
    smpl_output_dir = Path(smpl_output_dir)
    smpl_output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for start_idx, end_idx in ranges:
        segment_motion = slice_smpl_body_motion(motion, start_idx, end_idx)
        stem = build_segment_file_stem(segment_motion.frame_timestamps, prefix=filename_prefix)
        outputs.append(save_smpl_motion_npz(segment_motion, smpl_output_dir / f"{stem}.npz", smpl_frame=smpl_frame))
    return outputs
