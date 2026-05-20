from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_minimal_mvnx(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<mvnx>
  <subject frameRate="240" segmentCount="2">
    <frames>
      <frame type="normal" index="0" ms="1000">
        <orientation>1 0 0 0 1 0 0 0</orientation>
        <position>0 0 0 1 1 1</position>
      </frame>
      <frame type="normal" index="1" ms="1004">
        <orientation>1 0 0 0 1 0 0 0</orientation>
        <position>0 0 0 2 2 2</position>
      </frame>
    </frames>
  </subject>
</mvnx>
""",
        encoding="utf-8",
    )
    return path


def test_load_mvnx_motion_uses_one_streaming_parse(tmp_path: Path, monkeypatch) -> None:
    from dataset_converter.nymeria import mvnx

    mvnx_path = _write_minimal_mvnx(tmp_path / "body_xdata_mvnx")
    original_iterparse = mvnx.ET.iterparse
    calls = []

    def recording_iterparse(*args, **kwargs):
        calls.append(args[0])
        return original_iterparse(*args, **kwargs)

    monkeypatch.setattr(mvnx.ET, "iterparse", recording_iterparse)

    motion = mvnx.load_mvnx_motion(mvnx_path)

    assert len(calls) == 1
    assert motion.fps == 240.0
    assert motion.segment_count == 2
    np.testing.assert_array_equal(motion.frame_indices, [0, 1])
    np.testing.assert_array_equal(motion.raw_frame_timestamps, [1000, 1004])
    np.testing.assert_array_equal(motion.frame_timestamps, [1000, 1004])


def test_load_mvnx_motion_rebuilds_timestamps_from_frame_rate(tmp_path: Path) -> None:
    from dataset_converter.nymeria import mvnx

    path = tmp_path / "body_xdata_mvnx"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<mvnx>
  <subject frameRate="240" segmentCount="2">
    <frames>
      <frame type="normal" index="100" ms="5674564">
        <orientation>1 0 0 0 1 0 0 0</orientation>
        <position>0 0 0 1 1 1</position>
      </frame>
      <frame type="normal" index="101" ms="5674606">
        <orientation>1 0 0 0 1 0 0 0</orientation>
        <position>0 0 0 2 2 2</position>
      </frame>
      <frame type="normal" index="340" ms="5684644">
        <orientation>1 0 0 0 1 0 0 0</orientation>
        <position>0 0 0 3 3 3</position>
      </frame>
    </frames>
  </subject>
</mvnx>
""",
        encoding="utf-8",
    )

    motion = mvnx.load_mvnx_motion(path)

    np.testing.assert_array_equal(motion.raw_frame_timestamps, [5674564, 5674606, 5684644])
    np.testing.assert_array_equal(motion.frame_timestamps, [567456, 567461, 568456])
    assert motion.timestamp_source == "mvnx_frame_index_fps_scaled_0.1"


def test_load_mvnx_motion_wraps_oserror_with_path_context(tmp_path: Path, monkeypatch) -> None:
    from dataset_converter.nymeria import mvnx

    mvnx_path = _write_minimal_mvnx(tmp_path / "body_xdata_mvnx")

    def failing_iterparse(*args, **kwargs):
        raise OSError(22, "invalid argument")

    monkeypatch.setattr(mvnx.ET, "iterparse", failing_iterparse)

    with pytest.raises(RuntimeError, match=r"body_xdata_mvnx.*OSError.*invalid argument"):
        mvnx.load_mvnx_motion(mvnx_path)
