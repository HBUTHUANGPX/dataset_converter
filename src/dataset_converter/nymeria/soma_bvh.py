from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_converter.common.smpl import SMPLBodyMotion
from dataset_converter.nymeria.mvnx import load_mvnx_motion
from dataset_converter.nymeria.smpl import build_smpl_motion_payload
from dataset_converter.soma.bvh import canonicalize_motion_local_transforms_for_bvh, write_soma_bvh
from dataset_converter.soma.inversion import run_soma_inversion
from dataset_converter.soma.transforms import ensure_local_transforms_pre_visualization_frame, normalize_root_parent_index


def smpl_payload_to_motion(
    payload: dict[str, np.ndarray],
    *,
    frame_nums: np.ndarray,
    frame_timestamps: np.ndarray,
    fps: float,
) -> SMPLBodyMotion:
    return SMPLBodyMotion(
        global_orient=np.asarray(payload["global_orient"], dtype=np.float32),
        body_pose=np.asarray(payload["body_pose"], dtype=np.float32),
        transl=np.asarray(payload["transl"], dtype=np.float32),
        betas=np.asarray(payload["betas"], dtype=np.float32),
        frame_nums=np.asarray(frame_nums, dtype=np.int32),
        frame_timestamps=np.asarray(frame_timestamps, dtype=np.int64),
        fps=float(fps),
    )


def export_nymeria_to_soma_bvh(
    sequence_dir: str | Path,
    *,
    output_path: str | Path,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None = None,
) -> Path:
    smpl_payload = build_smpl_motion_payload(sequence_dir, start_frame=start_frame, end_frame=end_frame, stride=stride)
    mvnx_motion = load_mvnx_motion(
        Path(sequence_dir) / "body_xdata_mvnx",
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    motion = smpl_payload_to_motion(
        smpl_payload,
        frame_nums=mvnx_motion.frame_indices,
        frame_timestamps=mvnx_motion.frame_timestamps,
        fps=mvnx_motion.fps,
    )
    soma_output = run_soma_inversion(
        motion,
        device=device,
        batch_size=batch_size,
        soma_assets_root=soma_assets_root,
        smpl_model_path=smpl_model_path,
    )
    joint_names = list(soma_output["joint_names"])
    parent_indices = normalize_root_parent_index(soma_output["parent_indices"])
    reference_local_transforms = np.asarray(soma_output["reference_local_transforms"], dtype=np.float32)
    local_transforms = canonicalize_motion_local_transforms_for_bvh(
        local_transforms=ensure_local_transforms_pre_visualization_frame(
            local_transforms=np.asarray(soma_output["local_transforms"], dtype=np.float32),
            parent_indices=parent_indices,
            joint_names=joint_names,
        ),
        parent_indices=parent_indices,
    )
    return write_soma_bvh(
        output_path=output_path,
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        local_transforms=local_transforms,
        fps=float(motion.fps),
    )
