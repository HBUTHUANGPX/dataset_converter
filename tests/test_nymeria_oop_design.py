from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_mvnx(path: Path) -> Path:
    quat_frame = " ".join(["1 0 0 0"] * 23)
    pos_frame = " ".join(["0 0 0"] * 23)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<mvnx>
  <subject frameRate="240" segmentCount="23">
    <frames>
      <frame type="normal" index="0" ms="1000">
        <orientation>{quat_frame}</orientation>
        <position>{pos_frame}</position>
      </frame>
      <frame type="normal" index="1" ms="1042">
        <orientation>{quat_frame}</orientation>
        <position>{pos_frame}</position>
      </frame>
    </frames>
  </subject>
</mvnx>
""",
        encoding="utf-8",
    )
    return path


def test_mvnx_reader_uses_injected_timestamp_normalizer(tmp_path: Path) -> None:
    from dataset_converter.nymeria.mvnx import MvnxFrameSliceConfig, MvnxTimestampNormalizer, MvnxXmlMotionReader

    mvnx_path = _write_mvnx(tmp_path / "body_xdata_mvnx")
    reader = MvnxXmlMotionReader(timestamp_normalizer=MvnxTimestampNormalizer())

    motion = reader.read(mvnx_path, frame_slice=MvnxFrameSliceConfig())

    assert motion.timestamp_source == "mvnx_frame_index_fps_scaled_0.1"
    np.testing.assert_array_equal(motion.frame_timestamps, [100, 104])
    np.testing.assert_array_equal(motion.raw_frame_timestamps, [1000, 1042])


def test_annotation_payload_builder_anchors_text_to_motion_start(tmp_path: Path) -> None:
    from dataset_converter.nymeria.annotation import AnnotationPayloadBuilder

    sequence_dir = tmp_path / "seq"
    (sequence_dir / "narration").mkdir(parents=True)
    _write_mvnx(sequence_dir / "body_xdata_mvnx")
    (sequence_dir / "narration" / "atomic_action.csv").write_text(
        "start_time,end_time,Describe my atomic actions\n"
        "5831.0,5836.0,first action\n",
        encoding="utf-8",
    )

    payload, _summary = AnnotationPayloadBuilder().build(sequence_dir, time_zero_ns=0)

    assert int(payload["text_anchor_ms"]) == 100
    np.testing.assert_array_equal(payload["current_action_segment_start_timestamps_ns"], [100_000_000])
    np.testing.assert_array_equal(payload["current_action_segment_end_timestamps_ns"], [5_100_000_000])


def test_converted_output_guard_rejects_stale_alignment_version(tmp_path: Path) -> None:
    from dataset_converter.nymeria.batch import ConvertedOutputGuard

    path = tmp_path / "annotation.npz"
    np.savez(path, nymeria_time_alignment_version=np.asarray(3, dtype=np.int32), mvnx_timestamp_source=np.asarray("old"))

    guard = ConvertedOutputGuard(required_time_alignment_version=4)

    assert not guard.has_required_npz_fields(path, ("nymeria_time_alignment_version", "mvnx_timestamp_source"))


def test_head_video_timestamp_mapper_uses_vrs_time_code() -> None:
    from dataset_converter.nymeria.video import HeadVideoTimestampMapper

    class FakeProvider:
        def convert_from_device_time_to_timecode_ns(self, device_time_ns: int) -> int:
            return int(device_time_ns) + 123

    mapper = HeadVideoTimestampMapper()

    assert mapper.device_to_timecode_ns(FakeProvider(), 1000) == 1123
