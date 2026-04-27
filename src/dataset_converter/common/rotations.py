from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


VISUALIZATION_FRAME_QUAT_XYZW = Rotation.from_euler("x", 90.0, degrees=True).as_quat().astype(np.float32)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternion trailing dim 4, got {quat_wxyz.shape}.")
    return quat_wxyz[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected quaternion trailing dim 4, got {quat_xyzw.shape}.")
    return quat_xyzw[..., [3, 0, 1, 2]]


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
    rotations = Rotation.from_quat(quat_xyzw.reshape(-1, 4))
    return rotations.as_rotvec().reshape(quat_xyzw.shape[:-1] + (3,)).astype(np.float32)


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32)
    rotations = Rotation.from_rotvec(rotvec.reshape(-1, 3))
    return rotations.as_quat().reshape(rotvec.shape[:-1] + (4,)).astype(np.float32)


def quat_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    rotations = Rotation.from_quat(quat_xyzw.reshape(-1, 4))
    return rotations.as_rotvec().reshape(quat_xyzw.shape[:-1] + (3,)).astype(np.float32)


def quat_conjugate_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    out = quat_xyzw.copy()
    out[..., :3] *= -1.0
    return out


def quat_mul_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    lx, ly, lz, lw = np.moveaxis(lhs, -1, 0)
    rx, ry, rz, rw = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_rotate_xyzw(quat_xyzw: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    vector_quat = np.concatenate([vectors, np.zeros(vectors.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    return quat_mul_xyzw(
        quat_mul_xyzw(np.broadcast_to(quat_xyzw, vector_quat.shape), vector_quat),
        np.broadcast_to(quat_conjugate_xyzw(quat_xyzw), vector_quat.shape),
    )[..., :3].astype(np.float32)


def convert_root_to_soma_y_up(global_orient: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw SMPL root motion into the Y-up frame used by SOMA BVH exports."""
    root_quat_xyzw = rotvec_to_quat_xyzw(global_orient)
    inverse_frame = quat_conjugate_xyzw(VISUALIZATION_FRAME_QUAT_XYZW)
    converted_quat = quat_mul_xyzw(np.broadcast_to(inverse_frame, root_quat_xyzw.shape), root_quat_xyzw)
    converted_transl = quat_rotate_xyzw(inverse_frame, np.asarray(transl, dtype=np.float32))
    return quat_xyzw_to_rotvec(converted_quat), converted_transl
