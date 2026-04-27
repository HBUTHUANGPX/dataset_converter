from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_converter.nymeria.mvnx import load_mvnx_motion
from dataset_converter.nymeria.xsens_smpl import (
    convert_xsens_root_pos_to_smpl_transl,
    global_to_local_rotations,
    map_xsens_global_rotations_to_smpl,
    matrices_to_rotvec,
)


def build_smpl_motion_payload(
    sequence_dir: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    sequence_dir = Path(sequence_dir)
    motion = load_mvnx_motion(sequence_dir / "body_xdata_mvnx", start_frame=start_frame, end_frame=end_frame, stride=stride)
    smpl_global = map_xsens_global_rotations_to_smpl(motion.segment_quat_wxyz)
    smpl_local = global_to_local_rotations(smpl_global)
    smpl_rotvec = matrices_to_rotvec(smpl_local)
    transl = convert_xsens_root_pos_to_smpl_transl(motion.segment_pos_xyz)
    frame_count = motion.num_frames
    return {
        "global_orient": np.asarray(smpl_rotvec[:, 0], dtype=np.float32),
        "body_pose": np.asarray(smpl_rotvec[:, 1:].reshape(frame_count, 69), dtype=np.float32),
        "transl": np.asarray(transl, dtype=np.float32),
        "betas": np.zeros((frame_count, 10), dtype=np.float32),
    }


def save_smpl_motion_npz(payload: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
