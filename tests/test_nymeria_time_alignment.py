from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_mvnx(path: Path) -> Path:
    quat_frame = " ".join(["1 0 0 0"] * 23)
    pos_frame0 = " ".join(["0 0 0"] * 23)
    pos_frame1 = " ".join(["1 0 0"] * 23)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<mvnx>
  <subject frameRate="240" segmentCount="23">
    <frames>
      <frame type="normal" index="0" ms="1000">
        <orientation>{quat_frame}</orientation>
        <position>{pos_frame0}</position>
      </frame>
      <frame type="normal" index="1" ms="1042">
        <orientation>{quat_frame}</orientation>
        <position>{pos_frame1}</position>
      </frame>
    </frames>
  </subject>
</mvnx>
""",
        encoding="utf-8",
    )
    return path


def test_annotation_payload_writes_absolute_and_video_relative_time_fields(tmp_path: Path) -> None:
    from dataset_converter.nymeria.annotation import build_annotation_payload

    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    _write_mvnx(sequence_dir / "body_xdata_mvnx")

    payload, _ = build_annotation_payload(sequence_dir, time_zero_ns=900_000_000)

    np.testing.assert_array_equal(payload["frame_timestamps"], [100, 104])
    np.testing.assert_array_equal(payload["raw_frame_timestamps"], [1000, 1042])
    np.testing.assert_array_equal(payload["frame_timestamps_ns"], [100_000_000, 104_000_000])
    np.testing.assert_array_equal(payload["relative_frame_timestamps_ns"], [-800_000_000, -796_000_000])
    assert int(payload["time_zero_ns"]) == 900_000_000
    assert str(payload["time_zero_time_domain"]) == "time_code"
    assert str(payload["time_domain"]) == "time_code"
    assert str(payload["mvnx_timestamp_source"]) == "mvnx_frame_index_fps_scaled_0.1"
    assert int(payload["nymeria_time_alignment_version"]) == 4


def test_narration_time_anchor_matches_motion_start_and_preserves_segment_duration(tmp_path: Path) -> None:
    from dataset_converter.nymeria.annotation import load_narration_rows

    csv_path = tmp_path / "atomic_action.csv"
    csv_path.write_text(
        "start_time,end_time,Describe my atomic actions\n"
        "5831.0,5836.0,first action\n"
        "5836.0,5841.0,second action\n",
        encoding="utf-8",
    )

    rows = load_narration_rows(csv_path, target_anchor_ms=100)

    assert [(row.start_time_ms, row.end_time_ms, row.text) for row in rows] == [
        (100, 5_100, "first action"),
        (5_100, 10_100, "second action"),
    ]


def test_smpl_payload_writes_time_fields_aligned_to_video_zero(tmp_path: Path) -> None:
    from dataset_converter.nymeria.smpl import build_smpl_motion_payload

    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    _write_mvnx(sequence_dir / "body_xdata_mvnx")

    payload = build_smpl_motion_payload(sequence_dir, time_zero_ns=900_000_000)

    np.testing.assert_array_equal(payload["timestamps_ns"], [100_000_000, 104_000_000])
    np.testing.assert_array_equal(payload["raw_timestamps_ns"], [1_000_000_000, 1_042_000_000])
    np.testing.assert_array_equal(payload["relative_timestamps_ns"], [-800_000_000, -796_000_000])
    np.testing.assert_array_equal(payload["frame_indices"], [0, 1])
    assert str(payload["mvnx_timestamp_source"]) == "mvnx_frame_index_fps_scaled_0.1"
    assert str(payload["time_zero_time_domain"]) == "time_code"
    assert str(payload["time_zero_source"]) == "recording_head/rgb/frame_0"
    assert int(payload["nymeria_time_alignment_version"]) == 4


class _FakeImage:
    def is_valid(self) -> bool:
        return True

    def to_numpy_array(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeProvider:
    def get_stream_id_from_label(self, _label: str):
        return "rgb"

    def get_num_data(self, _stream_id) -> int:
        return 2

    def get_first_time_ns(self, _stream_id, _time_domain) -> int:
        return 900_000_000

    def get_last_time_ns(self, _stream_id, _time_domain) -> int:
        return 966_666_667

    def get_image_data_by_index(self, _stream_id, index: int):
        return _FakeImage(), SimpleNamespace(capture_timestamp_ns=900_000_000 + index * 66_666_667)

    def convert_from_device_time_to_timecode_ns(self, device_time_ns: int) -> int:
        return int(device_time_ns) + 123_000_000


class _FakeWriter:
    def __init__(self) -> None:
        self.frames = 0

    def write(self, _frame: np.ndarray) -> None:
        self.frames += 1

    def release(self) -> None:
        return None


def test_head_video_export_writes_relative_timestamps_and_time_zero(tmp_path: Path) -> None:
    from dataset_converter.nymeria.video import export_head_videos

    sequence_dir = tmp_path / "seq"
    (sequence_dir / "recording_head" / "data").mkdir(parents=True)
    (sequence_dir / "recording_head" / "data" / "data.vrs").write_bytes(b"fake")

    export = export_head_videos(
        sequence_dir,
        output_dir=tmp_path / "out",
        streams=("rgb",),
        provider_factory=lambda _path: _FakeProvider(),
        writer_factory=lambda *_args: _FakeWriter(),
    )

    with np.load(export.timestamps_path, allow_pickle=False) as data:
        assert int(data["time_zero_ns"]) == 1_023_000_000
        assert str(data["time_zero_source"]) == "recording_head/rgb/frame_0"
        assert str(data["time_domain"]) == "time_code"
        assert str(data["capture_time_domain"]) == "device_time"
        assert int(data["nymeria_time_alignment_version"]) == 4
        np.testing.assert_array_equal(data["rgb_timestamps_ns"], [1_023_000_000, 1_089_666_667])
        np.testing.assert_array_equal(data["rgb_capture_timestamps_ns"], [900_000_000, 966_666_667])
        np.testing.assert_array_equal(data["rgb_relative_timestamps_ns"], [0, 66_666_667])


def test_batch_time_zero_resolution_skips_bad_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataset_converter.nymeria.batch import NymeriaSequenceTask, resolve_batch_rgb_time_zero

    good = NymeriaSequenceTask(
        sequence_id="good",
        sequence_dir=tmp_path / "good",
        output_dir=tmp_path / "out" / "good",
    )
    bad = NymeriaSequenceTask(
        sequence_id="bad",
        sequence_dir=tmp_path / "bad",
        output_dir=tmp_path / "out" / "bad",
    )

    def _fake_resolve(sequence_dir: Path, *, stream_name: str, start_frame: int) -> int:
        if sequence_dir.name == "bad":
            raise FileNotFoundError("missing VRS")
        return 123

    monkeypatch.setattr("dataset_converter.nymeria.batch.resolve_head_video_time_zero_ns", _fake_resolve)

    assert resolve_batch_rgb_time_zero([good, bad]) == {"good": 123}


def test_skip_existing_requires_current_time_alignment_version(tmp_path: Path) -> None:
    from dataset_converter.nymeria.batch import _npz_has_keys

    path = tmp_path / "old_annotation.npz"
    np.savez(path, nymeria_time_alignment_version=np.asarray(3, dtype=np.int32), mvnx_timestamp_source=np.asarray("old"))

    assert not _npz_has_keys(
        path,
        ("nymeria_time_alignment_version", "mvnx_timestamp_source"),
        time_alignment_version=4,
    )

    path = tmp_path / "new_annotation.npz"
    np.savez(path, nymeria_time_alignment_version=np.asarray(4, dtype=np.int32), mvnx_timestamp_source=np.asarray("new"))

    assert _npz_has_keys(
        path,
        ("nymeria_time_alignment_version", "mvnx_timestamp_source"),
        time_alignment_version=4,
    )


def test_rgb_time_zero_requires_head_data_vrs(tmp_path: Path) -> None:
    from dataset_converter.nymeria.video import export_head_videos, resolve_head_video_time_zero_ns

    sequence_dir = tmp_path / "seq"
    data_dir = sequence_dir / "recording_head" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "motion.vrs").write_bytes(b"motion only")

    with pytest.raises(FileNotFoundError, match="data.vrs"):
        resolve_head_video_time_zero_ns(
            sequence_dir,
            stream_name="rgb",
            provider_factory=lambda _path: pytest.fail("motion.vrs should not be opened for RGB video"),
        )
    with pytest.raises(FileNotFoundError, match="data.vrs"):
        export_head_videos(
            sequence_dir,
            output_dir=tmp_path / "out",
            streams=("rgb",),
            provider_factory=lambda _path: pytest.fail("motion.vrs should not be opened for RGB video"),
        )
