from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

from dataset_converter.common.batch import BatchExportResult, run_multiprocess_tasks, run_sequential_tasks
from dataset_converter.common.paths import default_nymeria_output_root, default_nymeria_test_data_root
from dataset_converter.nymeria.annotation import build_annotation_payload, save_annotation_payload
from dataset_converter.nymeria.constants import NYMERIA_TIME_ALIGNMENT_VERSION
from dataset_converter.nymeria.smpl import build_smpl_motion_payload, save_smpl_motion_npz
from dataset_converter.nymeria.soma_bvh import export_nymeria_to_soma_bvh
from dataset_converter.nymeria.video import HEAD_VIDEO_STREAMS, export_head_videos, resolve_head_video_time_zero_ns


DEFAULT_SOMA_BATCH_SIZE = 256


@dataclass(frozen=True)
class NymeriaSequenceTask:
    """One Nymeria sequence export task.

    Responsibilities:
        Carry source and output paths for a sequence through batch exporters.
    Preconditions:
        ``sequence_dir`` points to a sequence directory.
    Postconditions:
        ``task_id`` returns the stable sequence id used in logs and summaries.
    """

    sequence_id: str
    sequence_dir: Path
    output_dir: Path

    @property
    def task_id(self) -> str:
        """Return the stable task identifier.

        Preconditions:
            The task has been constructed.
        Postconditions:
            Returns ``sequence_id`` unchanged.
        """

        return self.sequence_id


class ConvertedOutputGuard:
    """Validate whether converted NPZ outputs are current enough to skip.

    Responsibilities:
        Encapsulate key existence and time-alignment-version checks for
        ``--skip-existing``.
    Preconditions:
        Paths point to NPZ files when they exist.
    Postconditions:
        Returns ``False`` for missing, stale, malformed, or incomplete files.
    """

    def __init__(self, *, required_time_alignment_version: int | None = None) -> None:
        """Create a converted output guard.

        Preconditions:
            ``required_time_alignment_version`` is ``None`` or an integer.
        Postconditions:
            The guard can evaluate NPZ files for skip decisions.
        """

        self._required_time_alignment_version = required_time_alignment_version

    @property
    def required_time_alignment_version(self) -> int | None:
        """Return the required Nymeria time alignment version.

        Preconditions:
            The guard has been constructed.
        Postconditions:
            Returns ``None`` when version checking is disabled.
        """

        return self._required_time_alignment_version

    def has_required_npz_fields(self, path: Path, keys: tuple[str, ...]) -> bool:
        """Return whether ``path`` contains all required NPZ fields.

        Preconditions:
            ``keys`` contains field names expected in the NPZ file.
        Postconditions:
            Returns ``True`` only when all keys exist and the version matches
            when configured.
        """

        if not path.is_file():
            return False
        try:
            import numpy as np

            with np.load(path, allow_pickle=True) as data:
                if not all(key in data.files for key in keys):
                    return False
                if self.required_time_alignment_version is None:
                    return True
                if "nymeria_time_alignment_version" not in data.files:
                    return False
                return int(np.asarray(data["nymeria_time_alignment_version"]).reshape(())) == int(self.required_time_alignment_version)
        except Exception:
            return False


def discover_nymeria_sequence_tasks(
    test_data_root: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> list[NymeriaSequenceTask]:
    test_data_root = Path(test_data_root) if test_data_root is not None else default_nymeria_test_data_root()
    output_root = Path(output_root) if output_root is not None else default_nymeria_output_root()
    tasks: list[NymeriaSequenceTask] = []
    for mvnx_path in sorted(test_data_root.glob("*/body_xdata_mvnx")):
        sequence_dir = mvnx_path.parent
        tasks.append(NymeriaSequenceTask(sequence_dir.name, sequence_dir, output_root / sequence_dir.name))
    return tasks


def _npz_has_keys(path: Path, keys: tuple[str, ...], *, time_alignment_version: int | None = None) -> bool:
    """Compatibility wrapper for checking NPZ skip fields.

    Preconditions:
        ``path`` may or may not exist.
    Postconditions:
        Returns ``True`` only when ``ConvertedOutputGuard`` accepts the file.
    """

    return ConvertedOutputGuard(required_time_alignment_version=time_alignment_version).has_required_npz_fields(path, keys)


def _export_annotation_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    time_zero_ns_by_sequence_id: dict[str, int] | None,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "annotation.npz"
    time_zero_ns = None if time_zero_ns_by_sequence_id is None else time_zero_ns_by_sequence_id.get(task.sequence_id)
    required_keys = ("mvnx_timestamp_source", "nymeria_time_alignment_version") + (
        ("time_zero_ns", "time_zero_time_domain", "relative_frame_timestamps_ns") if time_zero_ns is not None else ()
    )
    if skip_existing and output_path.is_file() and (
        not required_keys or _npz_has_keys(output_path, required_keys, time_alignment_version=NYMERIA_TIME_ALIGNMENT_VERSION)
    ):
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        payload, _ = build_annotation_payload(task.sequence_dir, start_frame=start_frame, end_frame=end_frame, stride=stride, time_zero_ns=time_zero_ns)
        return BatchExportResult(task.task_id, True, (save_annotation_payload(payload, output_path),))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_smpl_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    time_zero_ns_by_sequence_id: dict[str, int] | None,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "smpl" / "nymeria_smpl.npz"
    time_zero_ns = None if time_zero_ns_by_sequence_id is None else time_zero_ns_by_sequence_id.get(task.sequence_id)
    required_keys = ("mvnx_timestamp_source", "nymeria_time_alignment_version") + (
        ("time_zero_ns", "time_zero_time_domain", "relative_timestamps_ns", "timestamps_ns", "frame_indices") if time_zero_ns is not None else ()
    )
    if skip_existing and output_path.is_file() and (
        not required_keys or _npz_has_keys(output_path, required_keys, time_alignment_version=NYMERIA_TIME_ALIGNMENT_VERSION)
    ):
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        payload = build_smpl_motion_payload(task.sequence_dir, start_frame=start_frame, end_frame=end_frame, stride=stride, time_zero_ns=time_zero_ns)
        return BatchExportResult(task.task_id, True, (save_smpl_motion_npz(payload, output_path),))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_soma_bvh_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    device: str,
    batch_size: int | None,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "soma_bvh" / "nymeria_soma.bvh"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        output = export_nymeria_to_soma_bvh(
            task.sequence_dir,
            output_path=output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            device=device,
            batch_size=batch_size,
            soma_assets_root=soma_assets_root,
            smpl_model_path=smpl_model_path,
        )
        return BatchExportResult(task.task_id, True, (output,))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_head_video_task(
    task: NymeriaSequenceTask,
    *,
    skip_existing: bool,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    max_frames: int | None = None,
    fps: float | None = None,
    rotate_degrees: int = 0,
    streams: tuple[str, ...] = ("slam-left", "slam-right"),
) -> BatchExportResult:
    output_dir = task.output_dir / "head_video"
    expected_outputs = tuple(output_dir / f"{HEAD_VIDEO_STREAMS[name]['output_stem']}.mp4" for name in streams) + (output_dir / "timestamps.npz",)
    timestamp_keys = (
        ("nymeria_time_alignment_version", "time_zero_ns", "rgb_relative_timestamps_ns", "rgb_capture_timestamps_ns")
        if "rgb" in streams
        else ("nymeria_time_alignment_version", "time_zero_ns")
    )
    if skip_existing and all(path.is_file() for path in expected_outputs) and _npz_has_keys(
        output_dir / "timestamps.npz",
        timestamp_keys,
        time_alignment_version=NYMERIA_TIME_ALIGNMENT_VERSION,
    ):
        return BatchExportResult(task.task_id, True, expected_outputs)
    try:
        output = export_head_videos(
            task.sequence_dir,
            output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            max_frames=max_frames,
            fps=fps,
            rotate_degrees=rotate_degrees,
            streams=streams,
        )
        return BatchExportResult(task.task_id, True, (*output.video_paths, output.timestamps_path))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def export_batch_annotation(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    time_zero_ns_by_sequence_id: dict[str, int] | None = None,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_annotation_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        time_zero_ns_by_sequence_id=time_zero_ns_by_sequence_id,
        skip_existing=skip_existing,
    )
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="Nymeria annotation", executor_cls=executor_cls)


def export_batch_smpl(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    time_zero_ns_by_sequence_id: dict[str, int] | None = None,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_smpl_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        time_zero_ns_by_sequence_id=time_zero_ns_by_sequence_id,
        skip_existing=skip_existing,
    )
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="Nymeria SMPL", executor_cls=executor_cls)


def resolve_batch_rgb_time_zero(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    start_frame: int = 0,
) -> dict[str, int]:
    values: dict[str, int] = {}
    for task in tasks:
        try:
            values[task.sequence_id] = resolve_head_video_time_zero_ns(task.sequence_dir, stream_name="rgb", start_frame=start_frame)
        except Exception as exc:  # pragma: no cover - depends on local VRS availability.
            print(f"[WARN] {task.task_id}: unable to resolve RGB time zero: {exc!r}", file=sys.stderr)
    return values


def export_batch_soma_bvh(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = DEFAULT_SOMA_BATCH_SIZE,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None = None,
    skip_existing: bool = False,
) -> list[BatchExportResult]:
    worker = partial(
        _export_soma_bvh_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        device=device,
        batch_size=batch_size,
        soma_assets_root=soma_assets_root,
        smpl_model_path=smpl_model_path,
        skip_existing=skip_existing,
    )
    return run_sequential_tasks(tasks, worker=worker, desc="Nymeria SOMA BVH")


def export_batch_head_video(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    skip_existing: bool = False,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    max_frames: int | None = None,
    fps: float | None = None,
    rotate_degrees: int = 0,
    streams: tuple[str, ...] = ("slam-left", "slam-right"),
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_head_video_task,
        skip_existing=skip_existing,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        max_frames=max_frames,
        fps=fps,
        rotate_degrees=rotate_degrees,
        streams=streams,
    )
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="Nymeria head video", executor_cls=executor_cls)
