from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, TypeVar

from tqdm.auto import tqdm


@dataclass(frozen=True)
class BatchExportResult:
    task_id: str
    ok: bool
    outputs: tuple[Path, ...] = ()
    error: str = ""


class BatchTask(Protocol):
    @property
    def task_id(self) -> str: ...


TaskT = TypeVar("TaskT", bound=BatchTask)


def run_multiprocess_tasks(
    tasks: Iterable[TaskT],
    *,
    worker: Callable[[TaskT], BatchExportResult],
    workers: int,
    desc: str,
    executor_cls=ProcessPoolExecutor,
) -> list[BatchExportResult]:
    task_list = list(tasks)
    if int(workers) <= 1:
        return [worker(task) for task in tqdm(task_list, desc=desc, unit="file", dynamic_ncols=True)]
    with executor_cls(max_workers=int(workers)) as executor:
        return list(tqdm(executor.map(worker, task_list), total=len(task_list), desc=desc, unit="file", dynamic_ncols=True))


def run_sequential_tasks(
    tasks: Iterable[TaskT],
    *,
    worker: Callable[[TaskT], BatchExportResult],
    desc: str,
) -> list[BatchExportResult]:
    return [worker(task) for task in tqdm(list(tasks), desc=desc, unit="file", dynamic_ncols=True)]
