from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MvnxMotion:
    """Parsed MVNX body motion.

    Responsibilities:
        Store segment rotations, segment positions, frame indices, corrected
        timestamps, raw timestamps, and source metadata.
    Preconditions:
        All arrays describe the same frame count.
    Postconditions:
        Consumers can use ``frame_timestamps`` as the normalized motion
        timeline and ``raw_frame_timestamps`` for diagnostics only.
    """

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
        """Return the number of parsed MVNX frames.

        Preconditions:
            ``frame_indices`` is a one-dimensional numpy array.
        Postconditions:
            Returns a non-negative integer frame count.
        """

        return int(self.frame_indices.shape[0])


@dataclass(frozen=True)
class MvnxFrameSliceConfig:
    """Frame slicing configuration for MVNX parsing.

    Responsibilities:
        Encapsulate start, stop, and stride choices so readers do not need
        loose primitive parameters.
    Preconditions:
        ``stride`` must be positive.
    Postconditions:
        ``includes`` returns whether a frame index should be parsed.
    """

    start_frame: int = 0
    end_frame: int = -1
    stride: int = 1

    @property
    def stop_frame(self) -> int | None:
        """Return the exclusive stop frame, or ``None`` for no stop.

        Preconditions:
            ``end_frame`` follows the CLI convention where ``-1`` means all
            remaining frames.
        Postconditions:
            Returns ``None`` or an integer suitable for ``frame_idx >= stop``.
        """

        return None if self.end_frame in (-1, None) else int(self.end_frame)

    def validate(self) -> None:
        """Validate the slicing configuration.

        Preconditions:
            The instance has been constructed.
        Postconditions:
            Raises ``ValueError`` if ``stride`` is not positive; otherwise
            returns ``None``.
        """

        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

    def includes(self, frame_index: int) -> bool:
        """Return whether ``frame_index`` is selected by this slice.

        Preconditions:
            ``validate`` has succeeded.
        Postconditions:
            Returns ``True`` only for frames inside the requested range and on
            the requested stride.
        """

        stop = self.stop_frame
        return (
            frame_index >= int(self.start_frame)
            and (stop is None or frame_index < stop)
            and (frame_index - int(self.start_frame)) % int(self.stride) == 0
        )


@dataclass(frozen=True)
class NormalizedTimestampResult:
    """Result of MVNX timestamp normalization.

    Responsibilities:
        Carry corrected millisecond timestamps and a human-readable source
        label.
    Preconditions:
        ``timestamps_ms`` is one-dimensional and aligned to the input frames.
    Postconditions:
        Consumers can store ``source`` in output NPZ files for diagnostics.
    """

    timestamps_ms: np.ndarray
    source: str


class TimestampNormalizer(Protocol):
    """Protocol for objects that normalize MVNX timestamps.

    Responsibilities:
        Provide a substitution point for timestamp normalization strategies.
    Preconditions:
        Inputs are one-dimensional frame index and raw timestamp arrays.
    Postconditions:
        Implementations return corrected timestamps without mutating inputs.
    """

    def normalize(self, raw_timestamps_ms: np.ndarray, frame_indices: np.ndarray, fps: float) -> NormalizedTimestampResult:
        """Normalize raw MVNX timestamps.

        Preconditions:
            ``raw_timestamps_ms`` and ``frame_indices`` have the same length.
        Postconditions:
            Returns normalized millisecond timestamps and a source label.
        """


class MvnxTimestampNormalizer:
    """Default timestamp normalization strategy for Nymeria MVNX files.

    Responsibilities:
        Detect MVNX files whose ``ms`` attributes are 10x larger than the
        shared time axis, then rebuild timestamps from frame indices and fps.
    Preconditions:
        Frame indices and timestamps are sorted in parse order.
    Postconditions:
        Returns a stable timeline suitable for SMPL, annotation, and video
        alignment.
    """

    _SCALED_SOURCE = "mvnx_frame_index_fps_scaled_0.1"
    _DEFAULT_SOURCE = "mvnx_frame_index_fps"

    def normalize(self, raw_timestamps_ms: np.ndarray, frame_indices: np.ndarray, fps: float) -> NormalizedTimestampResult:
        """Normalize MVNX timestamps.

        Preconditions:
            ``fps`` is positive for meaningful frame reconstruction.
        Postconditions:
            Returns raw timestamps unchanged when there are no frames;
            otherwise returns timestamps rebuilt from frame index deltas.
        """

        raw_timestamps_ms = np.asarray(raw_timestamps_ms, dtype=np.int64)
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        if raw_timestamps_ms.size == 0:
            return NormalizedTimestampResult(raw_timestamps_ms, self._DEFAULT_SOURCE)

        time_scale = self._infer_time_scale(raw_timestamps_ms, frame_indices, float(fps))
        corrected_timestamps_ms = np.rint(
            float(raw_timestamps_ms[0]) * time_scale
            + (frame_indices.astype(np.float64) - float(frame_indices[0])) * 1000.0 / float(fps)
        ).astype(np.int64)
        source = self._SCALED_SOURCE if time_scale == 0.1 else self._DEFAULT_SOURCE
        return NormalizedTimestampResult(corrected_timestamps_ms, source)

    def _infer_time_scale(self, raw_timestamps_ms: np.ndarray, frame_indices: np.ndarray, fps: float) -> float:
        """Infer whether MVNX timestamps use a 10x scale.

        Preconditions:
            Inputs describe at least zero frames; ``fps`` may be invalid.
        Postconditions:
            Returns ``0.1`` only when the observed per-frame clock delta is
            roughly 10x the nominal frame delta, otherwise ``1.0``.
        """

        return _infer_mvnx_time_scale(raw_timestamps_ms, frame_indices, fps)


class MvnxXmlMotionReader:
    """Streaming XML reader for Nymeria ``body_xdata_mvnx`` files.

    Responsibilities:
        Parse MVNX frames from disk and delegate timestamp correction to an
        injected ``TimestampNormalizer``.
    Preconditions:
        The input path exists and points to a valid MVNX XML file.
    Postconditions:
        Returns ``MvnxMotion`` without retaining XML tree state in memory.
    """

    def __init__(self, timestamp_normalizer: TimestampNormalizer | None = None) -> None:
        """Create an MVNX reader.

        Preconditions:
            ``timestamp_normalizer`` implements ``TimestampNormalizer`` when
            provided.
        Postconditions:
            The reader is ready to parse MVNX files.
        """

        self._timestamp_normalizer = timestamp_normalizer or MvnxTimestampNormalizer()

    @property
    def timestamp_normalizer(self) -> TimestampNormalizer:
        """Return the configured timestamp normalizer.

        Preconditions:
            The reader has been constructed.
        Postconditions:
            Returns the normalizer object used during ``read``.
        """

        return self._timestamp_normalizer

    def read(self, mvnx_path: str | Path, *, frame_slice: MvnxFrameSliceConfig | None = None) -> MvnxMotion:
        """Parse a Nymeria MVNX motion file.

        Preconditions:
            ``mvnx_path`` exists; ``frame_slice`` is valid or ``None``.
        Postconditions:
            Returns parsed motion with normalized timestamps.
        """

        mvnx_path = Path(mvnx_path)
        frame_slice = frame_slice or MvnxFrameSliceConfig()
        frame_slice.validate()
        if not mvnx_path.is_file():
            raise FileNotFoundError(f"MVNX file not found: {mvnx_path}")

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
                    if not frame_slice.includes(frame_idx):
                        elem.clear()
                        continue

                    quat, pos = self._parse_frame_arrays(elem, segment_count=segment_count, frame_index=frame_idx)
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
        normalized = self.timestamp_normalizer.normalize(raw_timestamps_ms, frame_indices_array, float(fps))
        return MvnxMotion(
            segment_quat_wxyz=np.asarray(quats, dtype=np.float32),
            segment_pos_xyz=np.asarray(positions, dtype=np.float32),
            frame_indices=frame_indices_array,
            frame_timestamps=normalized.timestamps_ms,
            raw_frame_timestamps=raw_timestamps_ms,
            timestamp_source=normalized.source,
            fps=float(fps),
            segment_count=int(segment_count),
        )

    def _parse_frame_arrays(self, element: ET.Element, *, segment_count: int, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Parse orientation and position arrays from one MVNX frame.

        Preconditions:
            ``element`` is a normal MVNX frame element.
        Postconditions:
            Returns quaternion and position arrays with ``segment_count`` rows.
        """

        orientation_text = None
        position_text = None
        for child in element:
            child_name = _tag_name(child)
            if child_name == "orientation":
                orientation_text = child.text
            elif child_name == "position":
                position_text = child.text
        quat = _parse_float_array(orientation_text, width=4)
        pos = _parse_float_array(position_text, width=3)
        if quat.shape != (segment_count, 4):
            raise ValueError(f"Frame {frame_index} orientation shape {quat.shape}, expected ({segment_count}, 4).")
        if pos.shape != (segment_count, 3):
            raise ValueError(f"Frame {frame_index} position shape {pos.shape}, expected ({segment_count}, 3).")
        return quat, pos


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
    """Load a Nymeria MVNX motion file.

    Preconditions:
        ``mvnx_path`` exists and ``stride`` is positive.
    Postconditions:
        Returns parsed motion with normalized timestamps. This function is a
        backward-compatible wrapper around ``MvnxXmlMotionReader``.
    """

    return MvnxXmlMotionReader().read(
        mvnx_path,
        frame_slice=MvnxFrameSliceConfig(start_frame=int(start_frame), end_frame=int(end_frame), stride=int(stride)),
    )
