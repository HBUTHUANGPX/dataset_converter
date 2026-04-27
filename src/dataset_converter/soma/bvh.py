from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from dataset_converter.common.rotations import quat_mul_xyzw, quat_rotate_xyzw


DEFAULT_POSITION_CHANNEL_JOINTS = ("Root", "Hips")
BVH_ROTATION_ORDER = "ZYX"


def canonicalize_motion_local_transforms_for_bvh(
    *,
    local_transforms: np.ndarray,
    parent_indices: np.ndarray,
) -> np.ndarray:
    local_transforms = np.asarray(local_transforms, dtype=np.float32).copy()
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
    root_indices = np.flatnonzero(parent_indices < 0)
    if root_indices.size != 1:
        raise ValueError(f"Expected exactly one root joint, got indices {root_indices.tolist()}.")
    root_idx = int(root_indices[0])
    children = np.flatnonzero(parent_indices == root_idx)
    if children.size == 0:
        return local_transforms

    root_pos = local_transforms[:, root_idx, :3].copy()
    root_quat = local_transforms[:, root_idx, 3:7].copy()
    for child_idx in children.tolist():
        child_pos = local_transforms[:, child_idx, :3].copy()
        child_quat = local_transforms[:, child_idx, 3:7].copy()
        local_transforms[:, child_idx, :3] = root_pos + quat_rotate_xyzw(root_quat, child_pos)
        local_transforms[:, child_idx, 3:7] = quat_mul_xyzw(root_quat, child_quat)

    local_transforms[:, root_idx, :3] = 0.0
    local_transforms[:, root_idx, 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return local_transforms


def write_soma_bvh(
    *,
    output_path: str | Path,
    joint_names: list[str],
    parent_indices: np.ndarray,
    reference_local_transforms: np.ndarray,
    local_transforms: np.ndarray,
    fps: float,
    position_channel_joints: tuple[str, ...] = DEFAULT_POSITION_CHANNEL_JOINTS,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joint_names = [str(name) for name in joint_names]
    parent_indices = np.asarray(parent_indices, dtype=np.int32)
    reference_local_transforms = np.asarray(reference_local_transforms, dtype=np.float32)
    local_transforms = np.asarray(local_transforms, dtype=np.float32)
    if local_transforms.ndim != 3 or local_transforms.shape[-1] != 7:
        raise ValueError(f"Expected local_transforms with shape (F, J, 7), got {local_transforms.shape}.")
    if reference_local_transforms.shape != (len(joint_names), 7):
        raise ValueError(
            f"Expected reference_local_transforms with shape ({len(joint_names)}, 7), got {reference_local_transforms.shape}."
        )
    if parent_indices.shape != (len(joint_names),):
        raise ValueError(f"Expected parent_indices with shape ({len(joint_names)},), got {parent_indices.shape}.")
    if local_transforms.shape[1] != len(joint_names):
        raise ValueError(f"Expected local_transforms second dimension {len(joint_names)}, got {local_transforms.shape[1]}.")

    position_channel_set = set(position_channel_joints)
    root_indices = np.flatnonzero(parent_indices < 0)
    if root_indices.size != 1:
        raise ValueError(f"Expected exactly one root joint, got indices {root_indices.tolist()}.")
    root_idx = int(root_indices[0])

    children = _build_children(parent_indices)
    traversal = _depth_first_order(root_idx, children)
    hierarchy_lines = ["HIERARCHY"]
    hierarchy_lines.extend(
        _emit_joint_hierarchy(
            joint_idx=root_idx,
            joint_names=joint_names,
            children=children,
            reference_local_transforms=reference_local_transforms,
            position_channel_set=position_channel_set,
            indent=0,
            is_root=True,
        )
    )
    motion_lines = [
        "MOTION",
        f"Frames: {local_transforms.shape[0]}",
        f"Frame Time: {1.0 / float(fps):.6f}",
    ]
    motion_lines.extend(
        _build_motion_lines(
            local_transforms=local_transforms,
            traversal=traversal,
            joint_names=joint_names,
            position_channel_set=position_channel_set,
        )
    )
    output_path.write_text("\n".join([*hierarchy_lines, *motion_lines, ""]), encoding="utf-8")
    return output_path


def _build_children(parent_indices: np.ndarray) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {idx: [] for idx in range(int(parent_indices.shape[0]))}
    for joint_idx, parent_idx in enumerate(np.asarray(parent_indices, dtype=np.int32).tolist()):
        if parent_idx >= 0:
            children[int(parent_idx)].append(int(joint_idx))
    return children


def _depth_first_order(root_idx: int, children: dict[int, list[int]]) -> list[int]:
    order = [root_idx]
    for child_idx in children[root_idx]:
        order.extend(_depth_first_order(child_idx, children))
    return order


def _emit_joint_hierarchy(
    *,
    joint_idx: int,
    joint_names: list[str],
    children: dict[int, list[int]],
    reference_local_transforms: np.ndarray,
    position_channel_set: set[str],
    indent: int,
    is_root: bool,
) -> list[str]:
    pad = " " * indent
    joint_name = joint_names[joint_idx]
    joint_type = "ROOT" if is_root else "JOINT"
    offset_bvh_units = reference_local_transforms[joint_idx, :3]
    has_position = joint_name in position_channel_set
    if is_root or has_position:
        channel_line = "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
    else:
        channel_line = "CHANNELS 3 Zrotation Yrotation Xrotation"

    lines = [
        f"{pad}{joint_type} {joint_name}",
        f"{pad}{{",
        f"{pad}  OFFSET {offset_bvh_units[0]:.6f} {offset_bvh_units[1]:.6f} {offset_bvh_units[2]:.6f}",
        f"{pad}  {channel_line}",
    ]
    for child_idx in children[joint_idx]:
        lines.extend(
            _emit_joint_hierarchy(
                joint_idx=child_idx,
                joint_names=joint_names,
                children=children,
                reference_local_transforms=reference_local_transforms,
                position_channel_set=position_channel_set,
                indent=indent + 1,
                is_root=False,
            )
        )
    lines.append(f"{pad}}}")
    return lines


def _build_motion_lines(
    *,
    local_transforms: np.ndarray,
    traversal: list[int],
    joint_names: list[str],
    position_channel_set: set[str],
) -> list[str]:
    frame_lines: list[str] = []
    frame_iter = tqdm(
        range(local_transforms.shape[0]),
        desc="Writing BVH frames",
        unit="frame",
        dynamic_ncols=True,
        disable=local_transforms.shape[0] <= 1,
    )
    for frame_idx in frame_iter:
        frame_values: list[str] = []
        for joint_idx in traversal:
            joint_name = joint_names[joint_idx]
            joint_local = local_transforms[frame_idx, joint_idx]
            if joint_name in position_channel_set:
                frame_values.extend(f"{value:.6f}" for value in joint_local[:3].tolist())
            rotation_deg = _quat_xyzw_to_bvh_zyx_deg(joint_local[3:7])
            frame_values.extend(f"{value:.6f}" for value in rotation_deg.tolist())
        frame_lines.append(" ".join(frame_values))
    return frame_lines


def _quat_xyzw_to_bvh_zyx_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if quat_xyzw.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat_xyzw.shape}.")
    return Rotation.from_quat(quat_xyzw).as_euler(BVH_ROTATION_ORDER, degrees=True).astype(np.float32)
