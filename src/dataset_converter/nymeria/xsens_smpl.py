from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


XSENS_TO_SMPL = [
    0,
    19,
    15,
    1,
    20,
    16,
    3,
    21,
    17,
    4,
    22,
    18,
    5,
    11,
    7,
    6,
    12,
    8,
    13,
    9,
    13,
    9,
    13,
    9,
]

SMPL_PARENT_INDICES = np.asarray(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ],
    dtype=np.int32,
)


def convert_xsens_quat_to_smpl_quat_wxyz(segment_quat_wxyz: np.ndarray) -> np.ndarray:
    segment_quat_wxyz = np.asarray(segment_quat_wxyz, dtype=np.float32)
    converted = segment_quat_wxyz.copy()
    converted[..., 1] = segment_quat_wxyz[..., 2]
    converted[..., 2] = segment_quat_wxyz[..., 3]
    converted[..., 3] = segment_quat_wxyz[..., 1]
    return converted


def convert_xsens_root_pos_to_smpl_transl(segment_pos_xyz: np.ndarray) -> np.ndarray:
    root_pos = np.asarray(segment_pos_xyz, dtype=np.float32)[:, 0]
    return root_pos[:, [1, 2, 0]].astype(np.float32)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(quat_wxyz.shape[:-1] + (3, 3)).astype(np.float32)


def map_xsens_global_rotations_to_smpl(segment_quat_wxyz: np.ndarray) -> np.ndarray:
    smpl_quat_wxyz = convert_xsens_quat_to_smpl_quat_wxyz(segment_quat_wxyz)
    xsens_global_mats = quat_wxyz_to_matrix(smpl_quat_wxyz)
    frame_count = xsens_global_mats.shape[0]
    smpl_global = np.broadcast_to(np.eye(3, dtype=np.float32), (frame_count, 24, 3, 3)).copy()
    for smpl_idx, xsens_idx in enumerate(XSENS_TO_SMPL):
        smpl_global[:, smpl_idx] = xsens_global_mats[:, xsens_idx]
    return smpl_global


def global_to_local_rotations(global_rotations: np.ndarray, parent_indices: np.ndarray = SMPL_PARENT_INDICES) -> np.ndarray:
    global_rotations = np.asarray(global_rotations, dtype=np.float32)
    local = np.empty_like(global_rotations)
    for joint_idx, parent_idx in enumerate(np.asarray(parent_indices, dtype=np.int32).tolist()):
        if parent_idx < 0:
            local[:, joint_idx] = global_rotations[:, joint_idx]
        else:
            local[:, joint_idx] = np.einsum(
                "fij,fjk->fik",
                np.swapaxes(global_rotations[:, parent_idx], -1, -2),
                global_rotations[:, joint_idx],
            )
    return local.astype(np.float32)


def matrices_to_rotvec(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    return Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_rotvec().reshape(rotations.shape[:-2] + (3,)).astype(np.float32)
