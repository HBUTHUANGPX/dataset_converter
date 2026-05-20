from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from dataset_converter.common.text import UNKNOWN_TEXT, build_index_sequence, normalize_text_value
from dataset_converter.nymeria.constants import NYMERIA_TIME_ALIGNMENT_VERSION


@dataclass(frozen=True)
class NarrationRow:
    """One normalized narration segment.

    Responsibilities:
        Store a text label and its active time range in milliseconds.
    Preconditions:
        ``start_time_ms`` and ``end_time_ms`` use the same timeline.
    Postconditions:
        Consumers can compare motion frame timestamps directly against the row.
    """

    start_time_ms: int
    end_time_ms: int
    text: str


class NarrationReader(Protocol):
    """Protocol for narration row readers.

    Responsibilities:
        Provide an interface for loading text rows from a source.
    Preconditions:
        The source exists or the reader can intentionally return no rows.
    Postconditions:
        Returns normalized ``NarrationRow`` instances.
    """

    def read(self, csv_path: str | Path, *, target_anchor_ms: int | None = None) -> list[NarrationRow]:
        """Read narration rows.

        Preconditions:
            ``target_anchor_ms`` is either ``None`` or a millisecond timestamp.
        Postconditions:
            Returned rows preserve source durations and use target anchoring.
        """


class NarrationTimelineAligner:
    """Align narration timestamps to an exported motion timeline.

    Responsibilities:
        Anchor the first narration timestamp to the first exported body frame
        while preserving each row's original duration and interval.
    Preconditions:
        ``source_anchor_seconds`` comes from the first row of one narration CSV.
    Postconditions:
        Converted timestamps are in milliseconds on the motion timeline.
    """

    def normalize_seconds_to_ms(self, value: float, *, source_anchor_seconds: float, target_anchor_ms: int) -> int:
        """Normalize a narration timestamp.

        Preconditions:
            ``target_anchor_ms`` is in the desired motion timeline.
        Postconditions:
            Returns an integer millisecond timestamp anchored to motion start.
        """

        return _normalize_seconds_to_ms(value, source_anchor_seconds=source_anchor_seconds, target_anchor_ms=target_anchor_ms)


class NarrationCsvReader:
    """CSV reader for Nymeria narration files.

    Responsibilities:
        Parse Nymeria narration CSV files and normalize timing through a
        ``NarrationTimelineAligner``.
    Preconditions:
        CSV files use ``start_time``, ``end_time``, and one ``Describe my...``
        text column when present.
    Postconditions:
        Missing or incompatible files yield an empty row list.
    """

    def __init__(self, timeline_aligner: NarrationTimelineAligner | None = None) -> None:
        """Create a narration CSV reader.

        Preconditions:
            ``timeline_aligner`` implements the required normalization method
            when provided.
        Postconditions:
            The reader is ready to parse narration CSV files.
        """

        self._timeline_aligner = timeline_aligner or NarrationTimelineAligner()

    @property
    def timeline_aligner(self) -> NarrationTimelineAligner:
        """Return the timeline aligner used by this reader.

        Preconditions:
            The reader has been constructed.
        Postconditions:
            Returns a non-null aligner instance.
        """

        return self._timeline_aligner

    def read(self, csv_path: str | Path, *, target_anchor_ms: int | None = None) -> list[NarrationRow]:
        """Read normalized narration rows from a CSV file.

        Preconditions:
            ``csv_path`` may be missing; missing files are allowed.
        Postconditions:
            Returns normalized rows, or an empty list when there is no usable
            narration content.
        """

        csv_path = Path(csv_path)
        if not csv_path.is_file():
            return []
        rows: list[NarrationRow] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            text_columns = [name for name in (reader.fieldnames or []) if name.startswith("Describe my")]
            if not text_columns:
                return []
            text_column = text_columns[0]
            raw_rows = list(reader)
            if not raw_rows:
                return []
            source_anchor_seconds = float(raw_rows[0]["start_time"])
            target_anchor_ms = int(source_anchor_seconds * 1000.0) if target_anchor_ms is None else int(target_anchor_ms)
            for row in raw_rows:
                rows.append(
                    NarrationRow(
                        start_time_ms=self.timeline_aligner.normalize_seconds_to_ms(
                            float(row["start_time"]),
                            source_anchor_seconds=source_anchor_seconds,
                            target_anchor_ms=target_anchor_ms,
                        ),
                        end_time_ms=self.timeline_aligner.normalize_seconds_to_ms(
                            float(row["end_time"]),
                            source_anchor_seconds=source_anchor_seconds,
                            target_anchor_ms=target_anchor_ms,
                        ),
                        text=normalize_text_value(row[text_column]),
                    )
                )
        return rows


class FrameTextIndexer:
    """Build per-frame text indices from normalized narration rows.

    Responsibilities:
        Map text segments to motion frame timestamps and build compact text
        pools plus index arrays.
    Preconditions:
        Frame timestamps are one-dimensional millisecond timestamps.
    Postconditions:
        Unknown/uncovered frames point to ``UNKNOWN`` through index arrays.
    """

    def align_rows_to_frames(self, rows: list[NarrationRow], frame_timestamps: np.ndarray) -> np.ndarray:
        """Return per-frame text values for ``rows``.

        Preconditions:
            ``rows`` are normalized to the same time axis as ``frame_timestamps``.
        Postconditions:
            Returns an object array with one value per frame.
        """

        return _align_rows_to_frames(rows, frame_timestamps)

    def build_pool_and_indices(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build a de-duplicated text pool and per-frame indices.

        Preconditions:
            ``values`` is a one-dimensional object array of text values.
        Postconditions:
            Returns ``(texts, indices)`` suitable for NPZ storage.
        """

        return build_index_sequence(values)


class AnnotationPayloadBuilder:
    """Build ``annotation.npz`` payloads for Nymeria sequences.

    Responsibilities:
        Coordinate MVNX loading, narration reading, text indexing, and final
        NPZ payload assembly.
    Preconditions:
        ``sequence_dir`` contains ``body_xdata_mvnx`` and may contain narration
        CSV files.
    Postconditions:
        Returns a payload dictionary and summary metadata without writing files.
    """

    def __init__(
        self,
        *,
        motion_reader: object | None = None,
        narration_reader: NarrationReader | None = None,
        frame_text_indexer: FrameTextIndexer | None = None,
    ) -> None:
        """Create an annotation payload builder.

        Preconditions:
            Injected collaborators implement their documented interfaces.
        Postconditions:
            The builder is ready to build annotation payloads.
        """

        from dataset_converter.nymeria.mvnx import MvnxXmlMotionReader

        self._motion_reader = motion_reader or MvnxXmlMotionReader()
        self._narration_reader = narration_reader or NarrationCsvReader()
        self._frame_text_indexer = frame_text_indexer or FrameTextIndexer()

    @property
    def narration_reader(self) -> NarrationReader:
        """Return the injected narration reader.

        Preconditions:
            The builder has been constructed.
        Postconditions:
            Returns the reader used for all narration CSV parsing.
        """

        return self._narration_reader

    @property
    def frame_text_indexer(self) -> FrameTextIndexer:
        """Return the injected frame text indexer.

        Preconditions:
            The builder has been constructed.
        Postconditions:
            Returns the indexer used for per-frame text mapping.
        """

        return self._frame_text_indexer

    def build(
        self,
        sequence_dir: str | Path,
        *,
        start_frame: int = 0,
        end_frame: int = -1,
        stride: int = 1,
        time_zero_ns: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Build the annotation payload for one sequence.

        Preconditions:
            ``sequence_dir/body_xdata_mvnx`` exists and ``stride`` is positive.
        Postconditions:
            Returns a complete annotation payload and summary dictionary.
        """

        from dataset_converter.nymeria.mvnx import MvnxFrameSliceConfig

        sequence_dir = Path(sequence_dir)
        motion = self._motion_reader.read(
            sequence_dir / "body_xdata_mvnx",
            frame_slice=MvnxFrameSliceConfig(start_frame=int(start_frame), end_frame=int(end_frame), stride=int(stride)),
        )
        text_anchor_ms = int(motion.frame_timestamps[0]) if motion.frame_timestamps.size else None
        text_payload, summary = self.build_text_payload(
            sequence_dir=sequence_dir,
            frame_timestamps=motion.frame_timestamps,
            time_zero_ns=time_zero_ns,
            text_anchor_ms=text_anchor_ms,
        )
        frame_timestamps_ns = np.asarray(motion.frame_timestamps, dtype=np.int64) * 1_000_000
        payload = {
            "fps": np.asarray(int(round(float(motion.fps))), dtype=np.int32),
            "num_frames": np.asarray(motion.num_frames, dtype=np.int32),
            "timeline_frame_indices": np.asarray(motion.frame_indices, dtype=np.int32),
            "frame_timestamps": np.asarray(motion.frame_timestamps, dtype=np.int64),
            "raw_frame_timestamps": np.asarray(motion.raw_frame_timestamps, dtype=np.int64),
            "frame_timestamps_ns": frame_timestamps_ns,
            "time_domain": np.asarray("time_code"),
            "mvnx_timestamp_source": np.asarray(motion.timestamp_source),
            "nymeria_time_alignment_version": np.asarray(NYMERIA_TIME_ALIGNMENT_VERSION, dtype=np.int32),
            "text_anchor_ms": np.asarray(-1 if text_anchor_ms is None else text_anchor_ms, dtype=np.int64),
            **text_payload,
        }
        if time_zero_ns is not None:
            time_zero_ns = int(time_zero_ns)
            payload.update(
                {
                    "time_zero_ns": np.asarray(time_zero_ns, dtype=np.int64),
                    "time_zero_source": np.asarray("recording_head/rgb/frame_0"),
                    "time_zero_time_domain": np.asarray("time_code"),
                    "relative_frame_timestamps_ns": frame_timestamps_ns - time_zero_ns,
                }
            )
        return payload, summary

    def build_text_payload(
        self,
        *,
        sequence_dir: str | Path,
        frame_timestamps: np.ndarray,
        time_zero_ns: int | None = None,
        text_anchor_ms: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Build the text portion of ``annotation.npz``.

        Preconditions:
            ``frame_timestamps`` are normalized motion timestamps in ms.
        Postconditions:
            Returns text pools, per-frame indices, segment timing arrays, and a
            summary dictionary.
        """

        sequence_dir = Path(sequence_dir)
        frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
        if text_anchor_ms is None and frame_timestamps.size:
            text_anchor_ms = int(frame_timestamps[0])
        activity_rows = self.narration_reader.read(sequence_dir / "narration" / "activity_summarization.csv", target_anchor_ms=text_anchor_ms)
        atomic_rows = self.narration_reader.read(sequence_dir / "narration" / "atomic_action.csv", target_anchor_ms=text_anchor_ms)
        return _build_text_payload_from_rows(
            activity_rows=activity_rows,
            atomic_rows=atomic_rows,
            frame_timestamps=frame_timestamps,
            frame_text_indexer=self.frame_text_indexer,
            text_anchor_ms=text_anchor_ms,
            time_zero_ns=time_zero_ns,
        )


def _normalize_seconds_to_ms(value: float, *, source_anchor_seconds: float, target_anchor_ms: int) -> int:
    # Nymeria narration timestamps preserve correct relative deltas, but their
    # absolute epoch is not reliable for MVNX-only export. Anchor the first text
    # row to the exported motion start and keep the original segment durations.
    normalized_seconds = float(target_anchor_ms) / 1000.0 + (float(value) - source_anchor_seconds)
    return int(normalized_seconds * 1000.0)


def load_narration_rows(csv_path: str | Path, *, target_anchor_ms: int | None = None) -> list[NarrationRow]:
    """Read Nymeria narration rows from ``csv_path``.

    Preconditions:
        ``target_anchor_ms`` is either ``None`` or a millisecond timestamp.
    Postconditions:
        Returns normalized rows through ``NarrationCsvReader``.
    """

    return NarrationCsvReader().read(csv_path, target_anchor_ms=target_anchor_ms)


def _align_rows_to_frames(rows: list[NarrationRow], frame_timestamps: np.ndarray) -> np.ndarray:
    values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    for row in rows:
        mask = (frame_timestamps >= row.start_time_ms) & (frame_timestamps <= row.end_time_ms)
        values[mask] = row.text
    return values


def build_text_payload(
    *,
    sequence_dir: str | Path,
    frame_timestamps: np.ndarray,
    time_zero_ns: int | None = None,
    text_anchor_ms: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build the text portion of an annotation payload.

    Preconditions:
        ``frame_timestamps`` are normalized motion timestamps in milliseconds.
    Postconditions:
        Returns text payload arrays and diagnostic summary metadata.
    """

    return AnnotationPayloadBuilder().build_text_payload(
        sequence_dir=sequence_dir,
        frame_timestamps=frame_timestamps,
        time_zero_ns=time_zero_ns,
        text_anchor_ms=text_anchor_ms,
    )


def _build_text_payload_from_rows(
    *,
    activity_rows: list[NarrationRow],
    atomic_rows: list[NarrationRow],
    frame_timestamps: np.ndarray,
    frame_text_indexer: FrameTextIndexer,
    text_anchor_ms: int | None,
    time_zero_ns: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build text payload arrays from already-loaded narration rows.

    Preconditions:
        Rows and frame timestamps use the same millisecond timeline.
    Postconditions:
        Returns NPZ-ready arrays and summary metadata.
    """

    main_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    sub_values = frame_text_indexer.align_rows_to_frames(activity_rows, frame_timestamps)
    action_values = frame_text_indexer.align_rows_to_frames(atomic_rows, frame_timestamps)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    main_texts, main_indices = frame_text_indexer.build_pool_and_indices(main_values)
    sub_texts, sub_indices = frame_text_indexer.build_pool_and_indices(sub_values)
    action_texts, action_indices = frame_text_indexer.build_pool_and_indices(action_values)
    interaction_texts, interaction_indices = frame_text_indexer.build_pool_and_indices(interaction_values)
    summary = {
        "mvnx_start_ms": int(frame_timestamps[0]) if frame_timestamps.size else None,
        "mvnx_end_ms": int(frame_timestamps[-1]) if frame_timestamps.size else None,
        "activity_start_ms": min((row.start_time_ms for row in activity_rows), default=None),
        "activity_end_ms": max((row.end_time_ms for row in activity_rows), default=None),
        "atomic_action_start_ms": min((row.start_time_ms for row in atomic_rows), default=None),
        "atomic_action_end_ms": max((row.end_time_ms for row in atomic_rows), default=None),
        "activity_covered_frames": int(np.count_nonzero(sub_indices)),
        "atomic_action_covered_frames": int(np.count_nonzero(action_indices)),
        "text_anchor_ms": text_anchor_ms,
    }
    payload = {
        "main_task_texts": main_texts,
        "sub_task_texts": sub_texts,
        "current_action_texts": action_texts,
        "interaction_texts": interaction_texts,
        "main_task_text_indices": main_indices,
        "sub_task_text_indices": sub_indices,
        "current_action_text_indices": action_indices,
        "interaction_text_indices": interaction_indices,
        **_text_segment_payload("sub_task", activity_rows, time_zero_ns=time_zero_ns),
        **_text_segment_payload("current_action", atomic_rows, time_zero_ns=time_zero_ns),
    }
    return payload, summary


def _text_segment_payload(prefix: str, rows: list[NarrationRow], *, time_zero_ns: int | None) -> dict[str, np.ndarray]:
    start_ns = np.asarray([row.start_time_ms * 1_000_000 for row in rows], dtype=np.int64)
    end_ns = np.asarray([row.end_time_ms * 1_000_000 for row in rows], dtype=np.int64)
    texts = np.asarray([row.text for row in rows], dtype=object)
    payload: dict[str, np.ndarray] = {
        f"{prefix}_segment_texts": texts,
        f"{prefix}_segment_start_timestamps_ns": start_ns,
        f"{prefix}_segment_end_timestamps_ns": end_ns,
    }
    if time_zero_ns is not None:
        time_zero_ns = int(time_zero_ns)
        payload[f"{prefix}_segment_relative_start_timestamps_ns"] = start_ns - time_zero_ns
        payload[f"{prefix}_segment_relative_end_timestamps_ns"] = end_ns - time_zero_ns
    return payload


def build_annotation_payload(
    sequence_dir: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    time_zero_ns: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build a Nymeria annotation payload.

    Preconditions:
        ``sequence_dir`` contains ``body_xdata_mvnx``.
    Postconditions:
        Returns an NPZ-ready payload and summary metadata.
    """

    return AnnotationPayloadBuilder().build(
        sequence_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        time_zero_ns=time_zero_ns,
    )


def save_annotation_payload(payload: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
