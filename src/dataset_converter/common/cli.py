from __future__ import annotations

import json
import sys
from pathlib import Path

from .batch import BatchExportResult


def result_to_json(stage: str, result: BatchExportResult) -> str:
    return json.dumps(
        {
            "stage": stage,
            "task_id": result.task_id,
            "ok": result.ok,
            "outputs": [str(path) for path in result.outputs],
            "error": result.error,
        },
        ensure_ascii=False,
    )


def write_summary(summary_path: Path | None, rows: list[str]) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def print_stage(stage: str, results: list[BatchExportResult]) -> None:
    ok_count = sum(result.ok for result in results)
    print(f"{stage}: {ok_count}/{len(results)} tasks ok")
    for result in results:
        if not result.ok:
            print(f"[FAILED] {result.task_id}: {result.error}", file=sys.stderr)
