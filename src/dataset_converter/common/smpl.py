from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_converter.common.rotations import convert_root_to_soma_y_up


@dataclass(frozen=True)
class SMPLBodyMotion:
    global_orient: np.ndarray
    body_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    frame_nums: np.ndarray
    frame_timestamps: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.frame_nums.shape[0])


def convert_smpl_motion_to_soma_y_up_frame(motion: SMPLBodyMotion) -> SMPLBodyMotion:
    global_orient, transl = convert_root_to_soma_y_up(motion.global_orient, motion.transl)
    return SMPLBodyMotion(
        global_orient=np.asarray(global_orient, dtype=np.float32),
        body_pose=np.asarray(motion.body_pose, dtype=np.float32),
        transl=np.asarray(transl, dtype=np.float32),
        betas=np.asarray(motion.betas, dtype=np.float32),
        frame_nums=np.asarray(motion.frame_nums, dtype=np.int32),
        frame_timestamps=np.asarray(motion.frame_timestamps, dtype=np.int64),
        fps=float(motion.fps),
    )


def slice_smpl_body_motion(motion: SMPLBodyMotion, start_idx: int, end_idx: int) -> SMPLBodyMotion:
    return SMPLBodyMotion(
        global_orient=np.asarray(motion.global_orient[start_idx:end_idx], dtype=np.float32),
        body_pose=np.asarray(motion.body_pose[start_idx:end_idx], dtype=np.float32),
        transl=np.asarray(motion.transl[start_idx:end_idx], dtype=np.float32),
        betas=np.asarray(motion.betas[start_idx:end_idx], dtype=np.float32),
        frame_nums=np.asarray(motion.frame_nums[start_idx:end_idx], dtype=np.int32),
        frame_timestamps=np.asarray(motion.frame_timestamps[start_idx:end_idx], dtype=np.int64),
        fps=float(motion.fps),
    )
