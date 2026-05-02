from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def test_prepare_soma_bvh_motion_transforms_writes_position_channels_in_centimeters_relative_to_first_frame() -> None:
    from dataset_converter.soma.bvh import prepare_soma_bvh_motion_transforms

    joint_names = ["Root", "Hips", "Head"]
    parent_indices = np.asarray([-1, 0, 1], dtype=np.int32)
    reference_local_transforms = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 101.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    root_quat = Rotation.from_euler("z", 90.0, degrees=True).as_quat().astype(np.float32)
    local_transforms = np.asarray(
        [
            [
                [1.0, 2.0, 3.0, *root_quat.tolist()],
                [0.5, 0.70, 2.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.25, 2.5, 3.75, *root_quat.tolist()],
                [0.75, 0.80, 2.25, 0.0, 0.0, 0.0, 1.0],
                [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    prepared = prepare_soma_bvh_motion_transforms(
        joint_names=joint_names,
        parent_indices=parent_indices,
        reference_local_transforms=reference_local_transforms,
        local_transforms=local_transforms,
    )

    collapsed_hips = local_transforms[:, 0, :3] + local_transforms[:, 1, :3]
    expected_hips_position = reference_local_transforms[1, :3] + (collapsed_hips - collapsed_hips[:1]) * 100.0
    np.testing.assert_allclose(prepared[:, 0, :3], np.zeros((2, 3), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        prepared[:, 0, 3:7],
        np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 4)),
        atol=1e-6,
    )
    np.testing.assert_allclose(prepared[:, 1, :3], expected_hips_position, atol=1e-5)


def test_hdf5_soma_bvh_exports_y_up_motion_without_post_inversion_frame_rotation(monkeypatch, tmp_path: Path) -> None:
    import dataset_converter.hdf5.soma_bvh as export_module
    from dataset_converter.common.smpl import convert_smpl_motion_to_soma_y_up_frame
    from dataset_converter.hdf5.io import BodyFrameSelection
    from dataset_converter.hdf5.smpl import selection_to_smpl_body_motion

    selection = BodyFrameSelection(
        root_pose7=np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        body_quats=np.broadcast_to(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 23, 4)).copy(),
        betas=np.zeros((1, 10), dtype=np.float32),
        frame_nums=np.asarray([7], dtype=np.int32),
        frame_timestamps=np.asarray([1234], dtype=np.int64),
        fps=20.0,
    )
    expected_motion = convert_smpl_motion_to_soma_y_up_frame(selection_to_smpl_body_motion(selection))
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(export_module, "load_body_frame_selection", lambda *args, **kwargs: selection)

    def fake_run_soma_inversion(motion, **kwargs):
        captured["motion_transl"] = motion.transl.copy()
        return _fake_soma_output_with_z_spine()

    monkeypatch.setattr(export_module, "run_soma_inversion", fake_run_soma_inversion)
    monkeypatch.setattr(export_module, "write_soma_bvh", _capture_write_soma_bvh(captured))

    export_module.export_segmented_soma_bvh(
        "fake.hdf5",
        soma_bvh_output_dir=tmp_path,
        soma_assets_root=tmp_path,
        smpl_model_path=tmp_path / "SMPL_NEUTRAL.npz",
    )

    np.testing.assert_allclose(captured["motion_transl"], expected_motion.transl, atol=1e-6)
    np.testing.assert_allclose(captured["written_local_transforms"][0, 1, :3], [0.0, 101.0, 0.0], atol=1e-6)


def test_nymeria_soma_bvh_does_not_rotate_post_inversion_root_translation(monkeypatch, tmp_path: Path) -> None:
    import dataset_converter.nymeria.soma_bvh as export_module

    captured: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        export_module,
        "build_smpl_motion_payload",
        lambda *args, **kwargs: {
            "global_orient": np.zeros((1, 3), dtype=np.float32),
            "body_pose": np.zeros((1, 69), dtype=np.float32),
            "transl": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "betas": np.zeros((1, 10), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        export_module,
        "load_mvnx_motion",
        lambda *args, **kwargs: SimpleNamespace(
            frame_indices=np.asarray([0], dtype=np.int32),
            frame_timestamps=np.asarray([1000], dtype=np.int64),
            fps=240.0,
        ),
    )

    def fake_run_soma_inversion(motion, **kwargs):
        captured["motion_transl"] = motion.transl.copy()
        return _fake_soma_output_with_z_spine()

    monkeypatch.setattr(export_module, "run_soma_inversion", fake_run_soma_inversion)
    monkeypatch.setattr(export_module, "write_soma_bvh", _capture_write_soma_bvh(captured))

    export_module.export_nymeria_to_soma_bvh(
        tmp_path,
        output_path=tmp_path / "nymeria_soma.bvh",
        soma_assets_root=tmp_path,
        smpl_model_path=tmp_path / "SMPL_NEUTRAL.npz",
    )

    np.testing.assert_allclose(captured["motion_transl"], [[1.0, 2.0, 3.0]], atol=1e-6)
    np.testing.assert_allclose(captured["written_local_transforms"][0, 1, :3], [0.0, 101.0, 0.0], atol=1e-6)


def _fake_soma_output_with_z_spine() -> dict[str, np.ndarray | list[str]]:
    local_transforms = np.asarray(
        [
            [
                [0.10, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    return {
        "joint_names": ["Root", "Hips", "Head"],
        "parent_indices": np.asarray([-1, 0, 1], dtype=np.int32),
        "reference_local_transforms": np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 101.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "local_transforms": local_transforms,
    }


def _capture_write_soma_bvh(captured: dict[str, np.ndarray]):
    def fake_write_soma_bvh(*, output_path, local_transforms, **kwargs):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        captured["written_local_transforms"] = np.asarray(local_transforms, dtype=np.float32).copy()
        return output_path

    return fake_write_soma_bvh
