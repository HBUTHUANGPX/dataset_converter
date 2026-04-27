from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class BodyFrameSelection:
    root_pose7: np.ndarray
    body_quats: np.ndarray
    betas: np.ndarray
    frame_nums: np.ndarray
    frame_timestamps: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.frame_nums.shape[0])


def compute_fps(num_frames: int, video_length_sec: float | None) -> float:
    if video_length_sec is None or video_length_sec <= 0:
        return 20.0
    return float(num_frames) / float(video_length_sec)


def split_pose7_qwxyz_xyz(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape[-1] != 7:
        raise ValueError(f"Expected pose7 trailing dim 7, got {pose7.shape}.")
    return pose7[..., :4], pose7[..., 4:]


def _array_is_finite(array: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(array), axis=tuple(range(1, array.ndim)))


def build_body_valid_mask(*, root_pose7: np.ndarray, body_quats: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce(
        [
            _array_is_finite(np.asarray(root_pose7)),
            _array_is_finite(np.asarray(body_quats)),
            _array_is_finite(np.asarray(betas)),
        ]
    )


def build_frame_timestamp_lookup(
    *, video_frame_numbers: np.ndarray, video_timestamps: np.ndarray
) -> dict[int, int]:
    frame_numbers = np.asarray(video_frame_numbers, dtype=np.int32).reshape(-1)
    timestamps = np.asarray(video_timestamps, dtype=np.int64).reshape(-1)
    if frame_numbers.shape[0] != timestamps.shape[0]:
        raise ValueError("video_frame_numbers and video_timestamps must have the same length.")
    return {int(frame_num): int(timestamp) for frame_num, timestamp in zip(frame_numbers, timestamps)}


def load_body_frame_selection(
    hdf5_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> BodyFrameSelection:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5_file:
        total_frames = int(h5_file["full_body_mocap/frame_nums"].shape[0])
        stop = total_frames if end_frame in (-1, None) else min(int(end_frame), total_frames)
        frame_slice = slice(int(start_frame), stop, int(stride))

        root_pose7 = np.asarray(h5_file["full_body_mocap/Ts_world_root"][frame_slice], dtype=np.float32)
        body_quats = np.asarray(h5_file["full_body_mocap/body_quats"][frame_slice], dtype=np.float32)
        betas = np.asarray(h5_file["full_body_mocap/betas"][frame_slice], dtype=np.float32)
        frame_nums = np.asarray(h5_file["full_body_mocap/frame_nums"][frame_slice], dtype=np.int32)
        frame_timestamp_lookup = build_frame_timestamp_lookup(
            video_frame_numbers=np.asarray(h5_file["video/frame_number"][:], dtype=np.int32),
            video_timestamps=np.asarray(h5_file["video/device_timestamp"][:], dtype=np.int64),
        )

        video_length_sec = None
        if "video/length_sec" in h5_file:
            video_length_sec = float(np.asarray(h5_file["video/length_sec"][()]).flat[0])
        fps = compute_fps(total_frames, video_length_sec)

    frame_timestamps = np.asarray([frame_timestamp_lookup[int(frame_num)] for frame_num in frame_nums], dtype=np.int64)
    valid_mask = build_body_valid_mask(root_pose7=root_pose7, body_quats=body_quats, betas=betas)
    return BodyFrameSelection(
        root_pose7=root_pose7[valid_mask],
        body_quats=body_quats[valid_mask],
        betas=betas[valid_mask],
        frame_nums=frame_nums[valid_mask],
        frame_timestamps=frame_timestamps[valid_mask],
        fps=float(fps),
    )
