from __future__ import annotations

from typing import Any

import numpy as np


UNKNOWN_TEXT = "UNKNOWN"


def normalize_text_value(value: Any) -> str:
    text = str(value).strip()
    return text if text else UNKNOWN_TEXT


def make_text_array(values: list[str]) -> np.ndarray:
    if not values:
        values = [UNKNOWN_TEXT]
    max_len = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{max(1, max_len)}")


def build_index_sequence(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    text_to_index = {UNKNOWN_TEXT: 0}
    deduped = [UNKNOWN_TEXT]
    indices = np.zeros(values.shape[0], dtype=np.int32)
    for frame_idx, raw_text in enumerate(values):
        text = normalize_text_value(raw_text)
        if text not in text_to_index:
            text_to_index[text] = len(deduped)
            deduped.append(text)
        indices[frame_idx] = text_to_index[text]
    return make_text_array(deduped), indices
