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


def _read_subject_metadata(mvnx_path: Path) -> tuple[float, int]:
    for _, elem in ET.iterparse(mvnx_path, events=("start",)):
        if _tag_name(elem) == "subject":
            fps = float(elem.attrib.get("frameRate", 240))
            segment_count = int(elem.attrib.get("segmentCount", 23))
            return fps, segment_count
    raise ValueError(f"No subject metadata found in {mvnx_path}.")


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

    fps, segment_count = _read_subject_metadata(mvnx_path)
    stop = None if end_frame in (-1, None) else int(end_frame)
    start_frame = int(start_frame)
    stride = int(stride)

    quats: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps_ms: list[int] = []

    for _, elem in ET.iterparse(mvnx_path, events=("end",)):
        if _tag_name(elem) != "frame" or elem.attrib.get("type") != "normal":
            continue

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

    return MvnxMotion(
        segment_quat_wxyz=np.asarray(quats, dtype=np.float32),
        segment_pos_xyz=np.asarray(positions, dtype=np.float32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        frame_timestamps=np.asarray(timestamps_ms, dtype=np.int64),
        fps=float(fps),
        segment_count=int(segment_count),
    )
