from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

from dataset_converter.common.batch import BatchExportResult, run_multiprocess_tasks, run_sequential_tasks
from dataset_converter.common.paths import default_nymeria_output_root, default_nymeria_test_data_root
from dataset_converter.nymeria.annotation import build_annotation_payload, save_annotation_payload
from dataset_converter.nymeria.smpl import build_smpl_motion_payload, save_smpl_motion_npz
from dataset_converter.nymeria.soma_bvh import export_nymeria_to_soma_bvh


DEFAULT_SOMA_BATCH_SIZE = 256


@dataclass(frozen=True)
class NymeriaSequenceTask:
    sequence_id: str
    sequence_dir: Path
    output_dir: Path

    @property
    def task_id(self) -> str:
        return self.sequence_id


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


def _export_annotation_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "annotation.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        payload, _ = build_annotation_payload(task.sequence_dir, start_frame=start_frame, end_frame=end_frame, stride=stride)
        return BatchExportResult(task.task_id, True, (save_annotation_payload(payload, output_path),))
    except Exception as exc:  # pragma: no cover
        return BatchExportResult(task.task_id, False, error=repr(exc))


def _export_smpl_task(
    task: NymeriaSequenceTask,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    skip_existing: bool,
) -> BatchExportResult:
    output_path = task.output_dir / "smpl" / "nymeria_smpl.npz"
    if skip_existing and output_path.is_file():
        return BatchExportResult(task.task_id, True, (output_path,))
    try:
        payload = build_smpl_motion_payload(task.sequence_dir, start_frame=start_frame, end_frame=end_frame, stride=stride)
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


def export_batch_annotation(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(_export_annotation_task, start_frame=start_frame, end_frame=end_frame, stride=stride, skip_existing=skip_existing)
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="Nymeria annotation", executor_cls=executor_cls)


def export_batch_smpl(
    tasks: Iterable[NymeriaSequenceTask],
    *,
    workers: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    skip_existing: bool = False,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    worker = partial(_export_smpl_task, start_frame=start_frame, end_frame=end_frame, stride=stride, skip_existing=skip_existing)
    return run_multiprocess_tasks(tasks, worker=worker, workers=workers, desc="Nymeria SMPL", executor_cls=executor_cls)


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
