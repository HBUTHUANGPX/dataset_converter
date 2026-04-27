from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dataset_converter.common.text import UNKNOWN_TEXT, build_index_sequence, make_text_array, normalize_text_value
from dataset_converter.hdf5.io import load_body_frame_selection


FRAME_NAME_PATTERN = re.compile(r"^frame[_-](\d+)$")


def load_caption_json(hdf5_path: str | Path) -> dict[str, Any]:
    with h5py.File(Path(hdf5_path), "r") as h5_file:
        raw = h5_file["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def _parse_caption_boundary(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        text = value.strip()
        match = FRAME_NAME_PATTERN.match(text)
        if match:
            return "frame", int(match.group(1))
        return "timestamp", int(text)
    return "timestamp", int(value)


def _caption_range_mask(
    *,
    start_value: Any,
    end_value: Any,
    frame_timestamps: np.ndarray,
    frame_nums: np.ndarray | None,
) -> np.ndarray:
    start_kind, start = _parse_caption_boundary(start_value)
    end_kind, end = _parse_caption_boundary(end_value)
    if start_kind != end_kind:
        raise ValueError(f"Caption range mixes {start_kind!r} start with {end_kind!r} end.")
    if start_kind == "frame":
        if frame_nums is None:
            raise ValueError("Caption uses frame_XXXX ranges, but frame_nums were not provided.")
        return (frame_nums >= start) & (frame_nums <= end)
    return (frame_timestamps >= start) & (frame_timestamps <= end)


def _format_caption_boundary(kind: str, value: int) -> int | str:
    if kind == "frame":
        return f"frame_{value:07d}"
    return int(value)


def align_caption_texts_to_frames(
    *,
    caption: dict[str, Any],
    frame_timestamps: np.ndarray,
    frame_nums: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if frame_nums is not None:
        frame_nums = np.asarray(frame_nums, dtype=np.int64).reshape(-1)
        if frame_nums.shape != frame_timestamps.shape:
            raise ValueError(f"frame_nums and frame_timestamps shape mismatch: {frame_nums.shape} vs {frame_timestamps.shape}.")
    main_task = caption.get("config", {}).get("Main Task", UNKNOWN_TEXT)

    main_values = np.full(frame_timestamps.shape[0], normalize_text_value(main_task), dtype=object)
    sub_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    action_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    for segment in caption.get("segments", []):
        segment_start = segment["start_frame"]
        segment_end = segment["end_frame"]
        segment_mask = _caption_range_mask(
            start_value=segment_start,
            end_value=segment_end,
            frame_timestamps=frame_timestamps,
            frame_nums=frame_nums,
        )
        sub_values[segment_mask] = normalize_text_value(segment.get("Sub Task", UNKNOWN_TEXT))

        for action in segment.get("Current Action", []):
            action_mask = _caption_range_mask(
                start_value=action["start_frame"],
                end_value=action["end_frame"],
                frame_timestamps=frame_timestamps,
                frame_nums=frame_nums,
            )
            action_values[action_mask] = normalize_text_value(action.get("label", UNKNOWN_TEXT))

        interaction_items = sorted(
            (
                (*_parse_caption_boundary(timestamp), normalize_text_value(text))
                for timestamp, text in segment.get("interaction", {}).items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        for item_idx, (interaction_kind, interaction_start, interaction_text) in enumerate(interaction_items):
            if item_idx + 1 < len(interaction_items) and interaction_items[item_idx + 1][0] == interaction_kind:
                interaction_end = interaction_items[item_idx + 1][1] - 1
            else:
                segment_end_kind, interaction_end = _parse_caption_boundary(segment_end)
                if segment_end_kind != interaction_kind:
                    continue
            interaction_mask = _caption_range_mask(
                start_value=_format_caption_boundary(interaction_kind, interaction_start),
                end_value=_format_caption_boundary(interaction_kind, interaction_end),
                frame_timestamps=frame_timestamps,
                frame_nums=frame_nums,
            )
            interaction_values[interaction_mask] = interaction_text

    main_texts, main_indices = build_index_sequence(main_values)
    sub_texts, sub_indices = build_index_sequence(sub_values)
    action_texts, action_indices = build_index_sequence(action_values)
    interaction_texts, interaction_indices = build_index_sequence(interaction_values)
    return {
        "main_task_texts": main_texts,
        "sub_task_texts": sub_texts,
        "current_action_texts": action_texts,
        "interaction_texts": interaction_texts,
        "main_task_text_indices": main_indices,
        "sub_task_text_indices": sub_indices,
        "current_action_text_indices": action_indices,
        "interaction_text_indices": interaction_indices,
    }


def build_annotation_export_payload(
    *,
    fps: float,
    frame_nums: np.ndarray,
    frame_timestamps: np.ndarray,
    text_payload: dict[str, Any],
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    frame_nums = np.asarray(frame_nums, dtype=np.int32).reshape(-1)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    if frame_nums.shape != frame_timestamps.shape:
        raise ValueError(f"frame_nums and frame_timestamps shape mismatch: {frame_nums.shape} vs {frame_timestamps.shape}.")

    payload: dict[str, np.ndarray] = {
        "fps": np.asarray(int(round(float(fps))), dtype=np.int32),
        "num_frames": np.asarray(frame_nums.shape[0], dtype=np.int32),
        "timeline_frame_indices": frame_nums,
        "frame_timestamps": frame_timestamps,
    }
    for key, value in {**text_payload, **(extra_payload or {})}.items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            payload[key] = make_text_array(list(value))
        elif isinstance(value, np.ndarray) and value.dtype == object:
            if value.ndim == 1 and all(isinstance(item, str) for item in value.tolist()):
                payload[key] = make_text_array(value.tolist())
            else:
                payload[key] = value
        else:
            payload[key] = np.asarray(value)
    return payload


def export_hdf5_to_annotation_payload(
    hdf5_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    selection = load_body_frame_selection(hdf5_path, start_frame=start_frame, end_frame=end_frame, stride=stride)
    caption = load_caption_json(hdf5_path)
    text_payload = align_caption_texts_to_frames(
        caption=caption,
        frame_timestamps=selection.frame_timestamps,
        frame_nums=selection.frame_nums,
    )
    return build_annotation_export_payload(
        fps=selection.fps,
        frame_nums=selection.frame_nums,
        frame_timestamps=selection.frame_timestamps,
        text_payload=text_payload,
        extra_payload={"source_caption": np.asarray(json.dumps(caption, ensure_ascii=False))},
    )


def save_annotation_payload(payload: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
