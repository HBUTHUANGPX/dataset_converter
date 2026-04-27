from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dataset_converter.common.text import UNKNOWN_TEXT, build_index_sequence, make_text_array, normalize_text_value
from dataset_converter.hdf5.io import load_body_frame_selection


def load_caption_json(hdf5_path: str | Path) -> dict[str, Any]:
    with h5py.File(Path(hdf5_path), "r") as h5_file:
        raw = h5_file["caption"][()]
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(raw)


def align_caption_texts_to_frames(*, caption: dict[str, Any], frame_timestamps: np.ndarray) -> dict[str, np.ndarray]:
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.int64).reshape(-1)
    main_task = caption.get("config", {}).get("Main Task", UNKNOWN_TEXT)

    main_values = np.full(frame_timestamps.shape[0], normalize_text_value(main_task), dtype=object)
    sub_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    action_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)
    interaction_values = np.full(frame_timestamps.shape[0], UNKNOWN_TEXT, dtype=object)

    for segment in caption.get("segments", []):
        segment_start = int(segment["start_frame"])
        segment_end = int(segment["end_frame"])
        segment_mask = (frame_timestamps >= segment_start) & (frame_timestamps <= segment_end)
        sub_values[segment_mask] = normalize_text_value(segment.get("Sub Task", UNKNOWN_TEXT))

        for action in segment.get("Current Action", []):
            action_start = int(action["start_frame"])
            action_end = int(action["end_frame"])
            action_mask = (frame_timestamps >= action_start) & (frame_timestamps <= action_end)
            action_values[action_mask] = normalize_text_value(action.get("label", UNKNOWN_TEXT))

        interaction_items = sorted(
            ((int(timestamp), normalize_text_value(text)) for timestamp, text in segment.get("interaction", {}).items()),
            key=lambda item: item[0],
        )
        for item_idx, (interaction_start, interaction_text) in enumerate(interaction_items):
            interaction_end = interaction_items[item_idx + 1][0] - 1 if item_idx + 1 < len(interaction_items) else segment_end
            interaction_mask = (frame_timestamps >= interaction_start) & (frame_timestamps <= interaction_end)
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
    text_payload = align_caption_texts_to_frames(caption=caption, frame_timestamps=selection.frame_timestamps)
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
