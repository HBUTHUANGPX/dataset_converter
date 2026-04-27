from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_converter.common.rotations import (
    VISUALIZATION_FRAME_QUAT_XYZW,
    quat_conjugate_xyzw,
    quat_mul_xyzw,
    quat_rotate_xyzw,
)


def normalize_root_parent_index(parent_indices: np.ndarray) -> np.ndarray:
    normalized = np.asarray(parent_indices, dtype=np.int32).copy()
    if normalized.size > 0 and normalized[0] == 0:
        normalized[0] = -1
    return normalized


def compute_global_joint_transforms_xyzw(
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_transforms = np.asarray(local_transforms, dtype=np.float32)
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
    num_frames, num_joints = local_transforms.shape[:2]
    global_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_positions = local_transforms[..., :3]
    local_rotations = local_transforms[..., 3:7]

    for joint_idx in range(num_joints):
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx < 0:
            global_positions[:, joint_idx] = local_positions[:, joint_idx]
            global_rotations[:, joint_idx] = local_rotations[:, joint_idx]
        else:
            parent_rot = global_rotations[:, parent_idx]
            parent_pos = global_positions[:, parent_idx]
            global_positions[:, joint_idx] = parent_pos + quat_rotate_xyzw(parent_rot, local_positions[:, joint_idx])
            global_rotations[:, joint_idx] = quat_mul_xyzw(parent_rot, local_rotations[:, joint_idx])
    return global_positions, global_rotations


def apply_visualization_frame_xyzw(
    positions: np.ndarray,
    rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=np.float32)
    expanded = np.broadcast_to(VISUALIZATION_FRAME_QUAT_XYZW, rotations.shape)
    corrected_positions = quat_rotate_xyzw(expanded, positions)
    corrected_rotations = quat_mul_xyzw(quat_mul_xyzw(expanded, rotations), quat_conjugate_xyzw(expanded))
    return corrected_positions.astype(np.float32, copy=False), corrected_rotations.astype(np.float32, copy=False)


def convert_root_to_pre_visualization_frame_xyzw(local_transforms: np.ndarray) -> np.ndarray:
    local_transforms = np.asarray(local_transforms, dtype=np.float32).copy()
    inverse_frame = quat_conjugate_xyzw(VISUALIZATION_FRAME_QUAT_XYZW)
    local_transforms[:, 0, :3] = quat_rotate_xyzw(inverse_frame, local_transforms[:, 0, :3])
    local_transforms[:, 0, 3:7] = quat_mul_xyzw(
        np.broadcast_to(inverse_frame, local_transforms[:, 0, 3:7].shape),
        local_transforms[:, 0, 3:7],
    )
    return local_transforms


def ensure_local_transforms_pre_visualization_frame(
    *,
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    local_transforms = np.asarray(local_transforms, dtype=np.float32)
    if "Hips" not in joint_names or "Head" not in joint_names:
        return local_transforms
    hips_idx = joint_names.index("Hips")
    head_idx = joint_names.index("Head")
    global_positions, _ = compute_global_joint_transforms_xyzw(local_transforms, parent_indices)
    spine_vector = np.asarray(global_positions[:, head_idx] - global_positions[:, hips_idx], dtype=np.float32)
    mean_abs = np.mean(np.abs(spine_vector), axis=0)
    if mean_abs[2] > mean_abs[1]:
        return convert_root_to_pre_visualization_frame_xyzw(local_transforms)
    return local_transforms


def project_rotation_matrices(rot_mats: np.ndarray) -> np.ndarray:
    original = np.asarray(rot_mats, dtype=np.float64)
    flat = original.reshape(-1, 3, 3)
    u, _, vh = np.linalg.svd(flat)
    projected = u @ vh
    det = np.linalg.det(projected)
    bad_handedness = det < 0.0
    if np.any(bad_handedness):
        u[bad_handedness, :, -1] *= -1.0
        projected[bad_handedness] = u[bad_handedness] @ vh[bad_handedness]
    return projected.reshape(original.shape)


def rotation_matrices_to_quat_xyzw(rot_mats: np.ndarray) -> np.ndarray:
    projected = project_rotation_matrices(rot_mats)
    rotations = Rotation.from_matrix(projected.reshape(-1, 3, 3))
    return rotations.as_quat().reshape(np.asarray(rot_mats).shape[:-2] + (4,)).astype(np.float32)


def pose7_from_transforms(transforms: np.ndarray) -> np.ndarray:
    positions = np.asarray(transforms[..., :3, 3], dtype=np.float32)
    quats = rotation_matrices_to_quat_xyzw(transforms[..., :3, :3])
    return np.concatenate([positions, quats], axis=-1).astype(np.float32)
