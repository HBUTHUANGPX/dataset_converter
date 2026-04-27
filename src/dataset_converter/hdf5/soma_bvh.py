from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_converter.common.smpl import slice_smpl_body_motion
from dataset_converter.hdf5.io import load_body_frame_selection
from dataset_converter.hdf5.smpl import build_segment_file_stem, selection_to_smpl_body_motion, split_contiguous_frame_ranges
from dataset_converter.soma.bvh import canonicalize_motion_local_transforms_for_bvh, write_soma_bvh
from dataset_converter.soma.inversion import run_soma_inversion
from dataset_converter.soma.transforms import ensure_local_transforms_pre_visualization_frame, normalize_root_parent_index


def export_segmented_soma_bvh(
    hdf5_path: str | Path,
    *,
    soma_bvh_output_dir: str | Path,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = "annotation",
) -> list[Path]:
    selection = load_body_frame_selection(hdf5_path, start_frame=start_frame, end_frame=end_frame, stride=stride)
    ranges = split_contiguous_frame_ranges(selection.frame_nums, expected_step=max(1, int(stride)))
    if not ranges:
        return []

    motion = selection_to_smpl_body_motion(selection)
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
    human_local_transforms = canonicalize_motion_local_transforms_for_bvh(
        local_transforms=ensure_local_transforms_pre_visualization_frame(
            local_transforms=np.asarray(soma_output["local_transforms"], dtype=np.float32),
            parent_indices=parent_indices,
            joint_names=joint_names,
        ),
        parent_indices=parent_indices,
    )

    output_dir = Path(soma_bvh_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for start_idx, end_idx in ranges:
        segment_motion = slice_smpl_body_motion(motion, start_idx, end_idx)
        stem = build_segment_file_stem(segment_motion.frame_timestamps, prefix=filename_prefix)
        outputs.append(
            write_soma_bvh(
                output_path=output_dir / f"{stem}.bvh",
                joint_names=joint_names,
                parent_indices=parent_indices,
                reference_local_transforms=reference_local_transforms,
                local_transforms=human_local_transforms[start_idx:end_idx],
                fps=float(motion.fps),
            )
        )
    return outputs
