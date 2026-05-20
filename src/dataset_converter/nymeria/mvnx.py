from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MvnxMotion:
    segment_quat_wxyz: np.ndarray
    segment_pos_xyz: np.ndarray
    frame_indices: np.ndarray
    frame_timestamps: np.ndarray
    raw_frame_timestamps: np.ndarray
    timestamp_source: str
    fps: float
    segment_count: int

    @property
    def num_frames(self) -> int:
        return int(self.frame_indices.shape[0])


def _tag_name(element: ET.Element) -> str:
    return element.tag.split("}", 1)[-1]


def _parse_float_array(text: str | None, *, width: int) -> np.ndarray:
    if not text:
        return np.empty((0, width), dtype=np.float32)
    values = np.fromstring(text, sep=" ", dtype=np.float32)
    if values.size % width != 0:
        raise ValueError(f"Expected value count divisible by {width}, got {values.size}.")
    return values.reshape(-1, width)


def _infer_mvnx_time_scale(raw_timestamps_ms: np.ndarray, frame_indices: np.ndarray, fps: float) -> float:
    if raw_timestamps_ms.size < 2 or fps <= 0:
        return 1.0
    frame_deltas = np.diff(frame_indices.astype(np.float64))
    time_deltas = np.diff(raw_timestamps_ms.astype(np.float64))
    valid = (frame_deltas > 0) & (time_deltas > 0)
    deltas = time_deltas[valid] / frame_deltas[valid]
    if deltas.size == 0:
        return 1.0
    nominal_ms = 1000.0 / float(fps)
    ratio = float(np.median(deltas) / nominal_ms)
    return 0.1 if 8.0 <= ratio <= 12.0 else 1.0


def load_mvnx_motion(
    mvnx_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> MvnxMotion:
    mvnx_path = Path(mvnx_path)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if not mvnx_path.is_file():
        raise FileNotFoundError(f"MVNX file not found: {mvnx_path}")

    stop = None if end_frame in (-1, None) else int(end_frame)
    start_frame = int(start_frame)
    stride = int(stride)
    fps: float | None = None
    segment_count: int | None = None

    quats: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps_ms: list[int] = []

    try:
        with mvnx_path.open("rb") as handle:
            for event, elem in ET.iterparse(handle, events=("start", "end")):
                elem_name = _tag_name(elem)
                if event == "start" and elem_name == "subject":
                    fps = float(elem.attrib.get("frameRate", 240))
                    segment_count = int(elem.attrib.get("segmentCount", 23))
                    continue
                if event != "end" or elem_name != "frame" or elem.attrib.get("type") != "normal":
                    continue
                if segment_count is None:
                    raise ValueError(f"No subject metadata found before frame data in {mvnx_path}.")

                frame_idx = int(elem.attrib["index"])
                if frame_idx < start_frame or (stop is not None and frame_idx >= stop) or (frame_idx - start_frame) % stride != 0:
                    elem.clear()
                    continue

                orientation_text = None
                position_text = None
                for child in elem:
                    child_name = _tag_name(child)
                    if child_name == "orientation":
                        orientation_text = child.text
                    elif child_name == "position":
                        position_text = child.text
                quat = _parse_float_array(orientation_text, width=4)
                pos = _parse_float_array(position_text, width=3)
                if quat.shape != (segment_count, 4):
                    raise ValueError(f"Frame {frame_idx} orientation shape {quat.shape}, expected ({segment_count}, 4).")
                if pos.shape != (segment_count, 3):
                    raise ValueError(f"Frame {frame_idx} position shape {pos.shape}, expected ({segment_count}, 3).")

                quats.append(quat)
                positions.append(pos)
                frame_indices.append(frame_idx)
                timestamps_ms.append(int(elem.attrib["ms"]))
                elem.clear()
    except OSError as exc:
        raise RuntimeError(f"Failed to read MVNX file {mvnx_path}: {type(exc).__name__}: {exc}") from exc

    if fps is None or segment_count is None:
        raise ValueError(f"No subject metadata found in {mvnx_path}.")

    raw_timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64)
    frame_indices_array = np.asarray(frame_indices, dtype=np.int32)
    if raw_timestamps_ms.size:
        # Some Nymeria MVNX frame "ms" values are 10x larger than the shared
        # timeline used by narration and VRS time code. Normalize that scale
        # before rebuilding frame deltas from frame indices and frameRate.
        time_scale = _infer_mvnx_time_scale(raw_timestamps_ms, frame_indices_array, float(fps))
        corrected_timestamps_ms = np.rint(
            float(raw_timestamps_ms[0]) * time_scale
            + (frame_indices_array.astype(np.float64) - float(frame_indices_array[0])) * 1000.0 / float(fps)
        ).astype(np.int64)
        timestamp_source = "mvnx_frame_index_fps_scaled_0.1" if time_scale == 0.1 else "mvnx_frame_index_fps"
    else:
        corrected_timestamps_ms = raw_timestamps_ms
        timestamp_source = "mvnx_frame_index_fps"

    return MvnxMotion(
        segment_quat_wxyz=np.asarray(quats, dtype=np.float32),
        segment_pos_xyz=np.asarray(positions, dtype=np.float32),
        frame_indices=frame_indices_array,
        frame_timestamps=corrected_timestamps_ms,
        raw_frame_timestamps=raw_timestamps_ms,
        timestamp_source=timestamp_source,
        fps=float(fps),
        segment_count=int(segment_count),
    )
