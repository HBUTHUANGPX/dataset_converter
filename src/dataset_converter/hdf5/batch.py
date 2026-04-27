from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

from dataset_converter.common.batch import BatchExportResult, run_multiprocess_tasks, run_sequential_tasks
from dataset_converter.common.paths import default_hdf5_output_root, default_hdf5_test_data_root
from dataset_converter.hdf5.annotation import export_hdf5_to_annotation_payload, save_annotation_payload
from dataset_converter.hdf5.smpl import export_segmented_smpl_npz
from dataset_converter.hdf5.soma_bvh import export_segmented_soma_bvh


DEFAULT_FILENAME_PREFIX = "annotation"


@dataclass(frozen=True)
class HDF5EpisodeTask:
    subset_id: str
    episode_id: str
    hdf5_path: Path
    output_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.subset_id}/{self.episode_id}"


def discover_hdf5_episode_tasks(
    test_data_root: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> list[HDF5EpisodeTask]:
    test_data_root = Path(test_data_root) if test_data_root is not None else default_hdf5_test_data_root()
    output_root = Path(output_root) if output_root is not None else default_hdf5_output_root()
    tasks: list[HDF5EpisodeTask] = []
    for hdf5_path in sorted(test_data_root.glob("*/*/annotation.hdf5")):
        episode_dir = hdf5_path.parent
        subset_dir = episode_dir.parent
        tasks.append(
            HDF5EpisodeTask(
                subset_id=subset_dir.name,
                episode_id=episode_dir.name,
                hdf5_path=hdf5_path,
                output_dir=output_root / subset_dir.name / episode_dir.name,
            )
        )
    return tasks


def _should_skip(path: Path, pattern: str, *, skip_existing: bool) -> bool:
    return bool(skip_existing and path.is_dir() and any(path.glob(pattern)))


def _export_annotation_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "annotation_soma.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        payload = export_hdf5_to_annotation_payload(task.hdf5_path, start_frame=start_frame, end_frame=end_frame, stride=stride)
        return BatchExportResult(task.task_id, True, (save_annotation_payload(payload, output_path),))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_smpl_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    filename_prefix: str,
    smpl_frame: str,
    skip_existing: bool,
) -> BatchExportResult:
    output_dir = task.output_dir / "smpl"
    if _should_skip(output_dir, "*.npz", skip_existing=skip_existing):
        return BatchExportResult(task.task_id, True, tuple(sorted(output_dir.glob("*.npz"))))
    try:
        outputs = export_segmented_smpl_npz(
            hdf5_path=task.hdf5_path,
            smpl_output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            filename_prefix=filename_prefix,
            smpl_frame=smpl_frame,
        )
        return BatchExportResult(task.task_id, True, tuple(outputs))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_soma_bvh_task(
    task: HDF5EpisodeTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    device: str,
    batch_size: int | None,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None,
    filename_prefix: str,
    skip_existing: bool,
) -> BatchExportResult:
    output_dir = task.output_dir / "soma_bvh"
    if _should_skip(output_dir, "*.bvh", skip_existing=skip_existing):
        return BatchExportResult(task.task_id, True, tuple(sorted(output_dir.glob("*.bvh"))))
    try:
        outputs = export_segmented_soma_bvh(
            hdf5_path=task.hdf5_path,
            soma_bvh_output_dir=output_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            device=device,
            batch_size=batch_size,
            soma_assets_root=soma_assets_root,
            smpl_model_path=smpl_model_path,
            filename_prefix=filename_prefix,
        )
        return BatchExportResult(task.task_id, True, tuple(outputs))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def export_batch_annotation(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(_export_annotation_task, start_frame=start_frame, end_frame=end_frame, stride=stride, skip_existing=skip_existing)
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="HDF5 annotation", executor_cls=executor_cls)


def export_batch_smpl(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    smpl_frame: str = "soma_y_up",
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(
        _export_smpl_task,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        filename_prefix=filename_prefix,
        smpl_frame=smpl_frame,
        skip_existing=skip_existing,
    )
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="HDF5 SMPL", executor_cls=executor_cls)


def export_batch_soma_bvh(
    tasks: Iterable[HDF5EpisodeTask],
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    device: str = "cuda",
    batch_size: int | None = None,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
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
        filename_prefix=filename_prefix,
        skip_existing=skip_existing,
    )
    return run_sequential_tasks(tasks, worker=worker, desc="HDF5 SOMA BVH")
