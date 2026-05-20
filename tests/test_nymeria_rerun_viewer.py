from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_annotation(path: Path) -> Path:
    np.savez(
        path,
        frame_timestamps=np.asarray([1000, 2000, 3000], dtype=np.int64),
        frame_timestamps_ns=np.asarray([1_000_000_000, 2_000_000_000, 3_000_000_000], dtype=np.int64),
        relative_frame_timestamps_ns=np.asarray([-100, 200, 500], dtype=np.int64),
        time_zero_ns=np.asarray(1_000_000_100, dtype=np.int64),
        time_zero_source=np.asarray("recording_head/rgb/frame_0"),
        time_domain=np.asarray("time_code"),
        timeline_frame_indices=np.asarray([10, 20, 30], dtype=np.int32),
        main_task_texts=np.asarray(["UNKNOWN"], dtype=object),
        main_task_text_indices=np.asarray([0, 0, 0], dtype=np.int32),
        sub_task_texts=np.asarray(["UNKNOWN", "walk"], dtype=object),
        sub_task_text_indices=np.asarray([0, 1, 1], dtype=np.int32),
        current_action_texts=np.asarray(["UNKNOWN", "step", "turn"], dtype=object),
        current_action_text_indices=np.asarray([0, 1, 2], dtype=np.int32),
        interaction_texts=np.asarray(["UNKNOWN"], dtype=object),
        interaction_text_indices=np.asarray([0, 0, 0], dtype=np.int32),
    )
    return path


def test_sequence_loader_decodes_text_for_each_timestamp(tmp_path: Path) -> None:
    from dataset_converter.visualization.nymeria_rerun import ConvertedNymeriaSequence

    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    _write_annotation(sequence_dir / "annotation.npz")
    smpl_dir = sequence_dir / "smpl"
    smpl_dir.mkdir()
    np.savez(
        smpl_dir / "nymeria_smpl.npz",
        global_orient=np.zeros((3, 3), dtype=np.float32),
        body_pose=np.zeros((3, 69), dtype=np.float32),
        transl=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros((3, 10), dtype=np.float32),
    )

    sequence = ConvertedNymeriaSequence.load(sequence_dir)

    assert sequence.timestamps_ns.tolist() == [-100, 200, 500]
    assert sequence.text_at(0)["sub_task"] == "UNKNOWN"
    assert sequence.text_at(1)["sub_task"] == "walk"
    assert sequence.text_at(2)["current_action"] == "turn"


def test_sequence_loader_missing_annotation_error_shows_expected_layout(tmp_path: Path) -> None:
    from dataset_converter.visualization.nymeria_rerun import ConvertedNymeriaSequence

    sequence_dir = tmp_path / "missing"

    try:
        ConvertedNymeriaSequence.load(sequence_dir)
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected missing annotation to raise FileNotFoundError.")

    assert "Expected converted sequence layout" in message
    assert "annotation.npz" in message
    assert "smpl/nymeria_smpl.npz" in message
    assert "Current working directory" in message


def test_time_axis_markdown_describes_relative_rgb_zero(tmp_path: Path) -> None:
    from dataset_converter.visualization.nymeria_rerun import ConvertedNymeriaSequence, _format_time_axis_markdown

    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    _write_annotation(sequence_dir / "annotation.npz")
    smpl_dir = sequence_dir / "smpl"
    smpl_dir.mkdir()
    np.savez(
        smpl_dir / "nymeria_smpl.npz",
        global_orient=np.zeros((3, 3), dtype=np.float32),
        body_pose=np.zeros((3, 69), dtype=np.float32),
        transl=np.zeros((3, 3), dtype=np.float32),
        betas=np.zeros((3, 10), dtype=np.float32),
    )

    markdown = _format_time_axis_markdown(ConvertedNymeriaSequence.load(sequence_dir))

    assert "relative" in markdown
    assert "recording_head/rgb/frame_0" in markdown
    assert "time_code" in markdown


def test_nearest_index_uses_sorted_timestamps() -> None:
    from dataset_converter.visualization.nymeria_rerun import nearest_index

    timestamps = np.asarray([1000, 2000, 3000], dtype=np.int64)

    assert nearest_index(timestamps, 100) == 0
    assert nearest_index(timestamps, 1600) == 1
    assert nearest_index(timestamps, 2600) == 2
    assert nearest_index(timestamps, 9999) == 2


def test_load_rgb_video_timestamps_reads_export_payload(tmp_path: Path) -> None:
    from dataset_converter.visualization.nymeria_rerun import load_rgb_video_metadata

    video_dir = tmp_path / "head_video"
    video_dir.mkdir()
    (video_dir / "rgb.mp4").write_bytes(b"not a real mp4")
    np.savez(
        video_dir / "timestamps.npz",
        rgb_timestamps_ns=np.asarray([11, 22], dtype=np.int64),
        rgb_relative_timestamps_ns=np.asarray([-9, 2], dtype=np.int64),
        rgb_frame_indices=np.asarray([0, 1], dtype=np.int32),
        rgb_stream_fps=np.asarray(30.0, dtype=np.float32),
    )

    metadata = load_rgb_video_metadata(video_dir)

    assert metadata is not None
    assert metadata.video_path == video_dir / "rgb.mp4"
    assert metadata.timestamps_ns.tolist() == [-9, 2]
    assert metadata.timestamp_kind == "relative"
    assert float(metadata.fps) == 30.0


def test_smpl_skeleton_uses_model_rest_joints_and_translation(tmp_path: Path) -> None:
    from dataset_converter.visualization.nymeria_rerun import SMPL_PARENT_INDICES, load_smpl_skeleton

    model_path = tmp_path / "SMPL_NEUTRAL.npz"
    rest_vertices = np.zeros((24, 3), dtype=np.float32)
    rest_vertices[:, 2] = np.arange(24, dtype=np.float32)
    np.savez(
        model_path,
        v_template=rest_vertices,
        shapedirs=np.zeros((24, 3, 10), dtype=np.float32),
        J_regressor=np.eye(24, dtype=np.float32),
        kintree_table=np.vstack([np.maximum(SMPL_PARENT_INDICES, 0), np.arange(24)]).astype(np.int64),
    )
    smpl_payload = {
        "global_orient": np.zeros((1, 3), dtype=np.float32),
        "body_pose": np.zeros((1, 69), dtype=np.float32),
        "transl": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "betas": np.zeros((1, 10), dtype=np.float32),
    }

    skeleton = load_smpl_skeleton(smpl_payload, smpl_model_path=model_path)

    assert skeleton.joint_positions.shape == (1, 24, 3)
    np.testing.assert_allclose(skeleton.joint_positions[0, 0], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(skeleton.joint_positions[0, 15], rest_vertices[15] + [1.0, 2.0, 3.0], atol=1e-6)
