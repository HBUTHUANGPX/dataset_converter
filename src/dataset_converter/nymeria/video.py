from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from tqdm.auto import tqdm


HEAD_VIDEO_STREAMS = {
    "slam-left": {
        "output_stem": "slam_left",
        "labels": ("camera-slam-left", "slam-front-left"),
        "stream_id": "1201-1",
    },
    "slam-right": {
        "output_stem": "slam_right",
        "labels": ("camera-slam-right", "slam-front-right"),
        "stream_id": "1201-2",
    },
    "rgb": {
        "output_stem": "rgb",
        "labels": ("camera-rgb",),
        "stream_id": "214-1",
    },
}


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class HeadVideoExport:
    video_paths: tuple[Path, ...]
    timestamps_path: Path


def resolve_head_vrs_path(sequence_dir: str | Path, *, prefer_motion: bool = False) -> Path:
    sequence_dir = Path(sequence_dir)
    candidates = (
        (sequence_dir / "recording_head" / "data" / "motion.vrs", sequence_dir / "recording_head" / "data" / "data.vrs")
        if prefer_motion
        else (sequence_dir / "recording_head" / "data" / "data.vrs", sequence_dir / "recording_head" / "data" / "motion.vrs")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No head VRS file found under {sequence_dir / 'recording_head' / 'data'}.")


def _create_projectaria_provider(vrs_path: Path):
    try:
        from projectaria_tools.core import data_provider
    except ImportError as exc:  # pragma: no cover - depends on optional runtime.
        raise ImportError("Nymeria head-video export requires `projectaria-tools`. Install `dataset-converter[video]`.") from exc
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Unable to open VRS file: {vrs_path}")
    return provider


def _stream_id_from_string(stream_id: str):
    try:
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as exc:  # pragma: no cover - depends on optional runtime.
        raise ImportError("Nymeria head-video export requires `projectaria-tools`. Install `dataset-converter[video]`.") from exc
    return StreamId(stream_id)


def _device_time_domain():
    try:
        from projectaria_tools.core.sensor_data import TimeDomain
    except ImportError:
        return None
    return TimeDomain.DEVICE_TIME


def _resolve_stream_id(provider, stream_name: str):
    spec = HEAD_VIDEO_STREAMS[stream_name]
    for label in spec["labels"]:
        try:
            stream_id = provider.get_stream_id_from_label(label)
        except Exception:
            stream_id = None
        if stream_id is not None:
            return stream_id
    return _stream_id_from_string(str(spec["stream_id"]))


def _normalise_image_array(image_data) -> np.ndarray:
    array = image_data.to_numpy_array()
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _rotate_frame(frame: np.ndarray, rotate_degrees: int) -> np.ndarray:
    rotate_degrees = int(rotate_degrees) % 360
    if rotate_degrees == 0:
        return frame
    if rotate_degrees == 90:
        return np.rot90(frame, k=3)
    if rotate_degrees == 180:
        return np.rot90(frame, k=2)
    if rotate_degrees == 270:
        return np.rot90(frame, k=1)
    raise ValueError(f"rotate_degrees must be one of 0, 90, 180, 270; got {rotate_degrees}.")


class FfmpegVideoWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frame_size
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self._path = path
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if self._process.stdin is None:
            raise RuntimeError("Unable to open ffmpeg stdin.")

    def write(self, frame: np.ndarray) -> None:
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr is not None else ""
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self._path}: {stderr.strip()}")


def _create_ffmpeg_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> VideoWriter:
    return FfmpegVideoWriter(path, fps, frame_size)


def _create_opencv_writer(path: Path, fps: float, frame_size: tuple[int, int], codec: str) -> VideoWriter:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime.
        raise ImportError("Nymeria head-video export requires `opencv-python`. Install `dataset-converter[video]`.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open MP4 writer: {path}")
    return writer


def create_default_video_writer(path: Path, fps: float, frame_size: tuple[int, int], codec: str) -> VideoWriter:
    if shutil.which("ffmpeg"):
        try:
            return _create_ffmpeg_writer(path, fps, frame_size)
        except OSError:
            pass
    return _create_opencv_writer(path, fps, frame_size, codec)


def _estimate_stream_fps(provider, stream_id, *, fallback_fps: float | None) -> float:
    if fallback_fps is not None:
        return float(fallback_fps)
    time_domain = _device_time_domain()
    try:
        num_frames = int(provider.get_num_data(stream_id))
        first_ns = int(provider.get_first_time_ns(stream_id, time_domain))
        last_ns = int(provider.get_last_time_ns(stream_id, time_domain))
    except Exception:
        return 30.0
    if num_frames <= 1 or last_ns <= first_ns:
        return 30.0
    return float((num_frames - 1) * 1e9 / (last_ns - first_ns))


def _write_stream_video(
    *,
    provider,
    stream_name: str,
    output_dir: Path,
    writer_factory: Callable[[Path, float, tuple[int, int], str], VideoWriter],
    fps: float | None,
    start_frame: int,
    end_frame: int,
    stride: int,
    max_frames: int | None,
    rotate_degrees: int,
    codec: str,
) -> tuple[Path, dict[str, np.ndarray]]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")

    stream_id = _resolve_stream_id(provider, stream_name)
    output_stem = str(HEAD_VIDEO_STREAMS[stream_name]["output_stem"])
    video_path = output_dir / f"{output_stem}.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = int(provider.get_num_data(stream_id))
    start_frame = max(0, int(start_frame))
    stop = num_frames if end_frame in (-1, None) else min(num_frames, int(end_frame))
    selected_indices = list(range(start_frame, stop, int(stride)))
    if max_frames is not None and max_frames >= 0:
        selected_indices = selected_indices[: int(max_frames)]
    if not selected_indices:
        raise ValueError(f"Stream {stream_name} has no frames to export.")

    stream_fps = _estimate_stream_fps(provider, stream_id, fallback_fps=fps) / float(stride)
    first_image, first_record = provider.get_image_data_by_index(stream_id, selected_indices[0])
    if hasattr(first_image, "is_valid") and not first_image.is_valid():
        raise RuntimeError(f"First image for stream {stream_name} is invalid.")
    first_frame = _rotate_frame(_normalise_image_array(first_image), rotate_degrees)
    height, width = first_frame.shape[:2]
    writer = writer_factory(video_path, stream_fps, (width, height), codec)

    timestamps: list[int] = [int(first_record.capture_timestamp_ns)]
    frame_indices: list[int] = [int(selected_indices[0])]
    try:
        writer.write(first_frame[:, :, ::-1])
        iterator = selected_indices[1:]
        for frame_index in tqdm(iterator, desc=f"Nymeria {output_stem}", unit="frame", leave=False, dynamic_ncols=True):
            image, record = provider.get_image_data_by_index(stream_id, frame_index)
            if hasattr(image, "is_valid") and not image.is_valid():
                continue
            frame = _rotate_frame(_normalise_image_array(image), rotate_degrees)
            writer.write(frame[:, :, ::-1])
            timestamps.append(int(record.capture_timestamp_ns))
            frame_indices.append(int(frame_index))
    finally:
        writer.release()

    return (
        video_path,
        {
            f"{output_stem}_timestamps_ns": np.asarray(timestamps, dtype=np.int64),
            f"{output_stem}_frame_indices": np.asarray(frame_indices, dtype=np.int32),
            f"{output_stem}_stream_fps": np.asarray(stream_fps, dtype=np.float32),
        },
    )


def export_head_videos(
    sequence_dir: str | Path,
    *,
    output_dir: str | Path,
    streams: Iterable[str] = ("slam-left", "slam-right"),
    fps: float | None = None,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    max_frames: int | None = None,
    rotate_degrees: int = 0,
    codec: str = "mp4v",
    prefer_motion_vrs: bool = False,
    provider_factory: Callable[[Path], object] | None = None,
    writer_factory: Callable[[Path, float, tuple[int, int], str], VideoWriter] = create_default_video_writer,
) -> HeadVideoExport:
    output_dir = Path(output_dir)
    vrs_path = resolve_head_vrs_path(sequence_dir, prefer_motion=prefer_motion_vrs)
    provider = (provider_factory or _create_projectaria_provider)(vrs_path)

    timestamp_payload: dict[str, np.ndarray] = {
        "source_vrs_path": np.asarray(str(vrs_path)),
    }
    video_paths: list[Path] = []
    for stream_name in streams:
        if stream_name not in HEAD_VIDEO_STREAMS:
            choices = ", ".join(sorted(HEAD_VIDEO_STREAMS))
            raise ValueError(f"Unknown stream {stream_name!r}. Choices: {choices}.")
        video_path, stream_payload = _write_stream_video(
            provider=provider,
            stream_name=stream_name,
            output_dir=output_dir,
            writer_factory=writer_factory,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            max_frames=max_frames,
            rotate_degrees=rotate_degrees,
            codec=codec,
        )
        video_paths.append(video_path)
        timestamp_payload.update(stream_payload)

    timestamps_path = output_dir / "timestamps.npz"
    np.savez(timestamps_path, **timestamp_payload)
    return HeadVideoExport(tuple(video_paths), timestamps_path)
