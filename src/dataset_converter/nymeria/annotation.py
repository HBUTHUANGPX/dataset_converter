from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_converter.common.text import UNKNOWN_TEXT, build_index_sequence, normalize_text_value
from dataset_converter.nymeria.mvnx import load_mvnx_motion

NYMERIA_TIME_ALIGNMENT_VERSION = 4


@dataclass(frozen=True)
class NarrationRow:
    start_time_ms: int
    end_time_ms: int
    text: str


def _normalize_seconds_to_ms(value: float, *, source_anchor_seconds: float, target_anchor_ms: int) -> int:
    # Nymeria narration timestamps preserve correct relative deltas, but their
    # absolute epoch is not reliable for MVNX-only export. Anchor the first text
    # row to the exported motion start and keep the original segment durations.
    normalized_seconds = float(target_anchor_ms) / 1000.0 + (float(value) - source_anchor_seconds)
    return int(normalized_seconds * 1000.0)


def load_narration_rows(csv_path: str | Path, *, target_anchor_ms: int | None = None) -> list[NarrationRow]:
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
        if target_anchor_ms is None:
            target_anchor_ms = int(source_anchor_seconds * 1000.0)
        for row in raw_rows:
            rows.append(
                NarrationRow(
                    start_time_ms=_normalize_seconds_to_ms(
                        float(row["start_time"]),
                        source_anchor_seconds=source_anchor_seconds,
                        target_anchor_ms=int(target_anchor_ms),
                    ),
                    end_time_ms=_normalize_seconds_to_ms(
                        float(row["end_time"]),
                        source_anchor_seconds=source_anchor_seconds,
                        target_anchor_ms=int(target_anchor_ms),
                    ),
                    text=normalize_text_value(row[text_column]),
                )
            )
    return rows


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
    sequence_dir = Path(sequence_dir)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if text_anchor_ms is None and frame_timestamps.size:
        text_anchor_ms = int(frame_timestamps[0])
    activity_rows = load_narration_rows(sequence_dir / "narration" / "activity_summarization.csv", target_anchor_ms=text_anchor_ms)
    atomic_rows = load_narration_rows(sequence_dir / "narration" / "atomic_action.csv", target_anchor_ms=text_anchor_ms)

    main_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    sub_values = _align_rows_to_frames(activity_rows, frame_timestamps)
    action_values = _align_rows_to_frames(atomic_rows, frame_timestamps)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    main_texts, main_indices = build_index_sequence(main_values)
    sub_texts, sub_indices = build_index_sequence(sub_values)
    action_texts, action_indices = build_index_sequence(action_values)
    interaction_texts, interaction_indices = build_index_sequence(interaction_values)
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
    sequence_dir = Path(sequence_dir)
    motion = load_mvnx_motion(sequence_dir / "body_xdata_mvnx", start_frame=start_frame, end_frame=end_frame, stride=stride)
    text_anchor_ms = int(motion.frame_timestamps[0]) if motion.frame_timestamps.size else None
    text_payload, summary = build_text_payload(
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


def save_annotation_payload(payload: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
