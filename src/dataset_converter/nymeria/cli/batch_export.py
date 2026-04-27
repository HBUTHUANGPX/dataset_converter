from __future__ import annotations

import argparse
from pathlib import Path

from dataset_converter.common.cli import print_stage, result_to_json, write_summary
from dataset_converter.common.paths import (
    default_nymeria_output_root,
    default_nymeria_test_data_root,
    require_path,
    resolve_smpl_model_path,
    resolve_soma_assets_root,
)
from dataset_converter.nymeria.batch import (
    DEFAULT_SOMA_BATCH_SIZE,
    discover_nymeria_sequence_tasks,
    export_batch_annotation,
    export_batch_smpl,
    export_batch_soma_bvh,
)


EXPORT_CHOICES = ("annotation", "smpl", "soma-bvh")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch export Nymeria motion assets. SMPL/annotation may use multiprocessing; SOMA BVH is sequential."
    )
    parser.add_argument("--test-data-root", type=Path, default=default_nymeria_test_data_root())
    parser.add_argument("--output-root", type=Path, default=default_nymeria_output_root())
    parser.add_argument("--exports", nargs="+", choices=EXPORT_CHOICES, default=["annotation", "smpl"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SOMA_BATCH_SIZE)
    parser.add_argument("--soma-assets-root", type=Path, default=None, help="Optional. Falls back to SOMA_ASSETS_ROOT or package assets.")
    parser.add_argument("--smpl-model-path", type=Path, default=None, help="Optional. Falls back to SMPL_MODEL_PATH or SOMA assets.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tasks = discover_nymeria_sequence_tasks(args.test_data_root, output_root=args.output_root)
    print(f"Discovered {len(tasks)} Nymeria sequence tasks under {args.test_data_root}")

    summary_rows: list[str] = []
    exit_code = 0

    if "annotation" in args.exports:
        results = export_batch_annotation(
            tasks,
            workers=args.workers,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            skip_existing=args.skip_existing,
        )
        print_stage("annotation", results)
        summary_rows.extend(result_to_json("annotation", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1
            if args.fail_fast:
                write_summary(args.summary_path, summary_rows)
                return exit_code

    if "smpl" in args.exports:
        results = export_batch_smpl(
            tasks,
            workers=args.workers,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            skip_existing=args.skip_existing,
        )
        print_stage("smpl", results)
        summary_rows.extend(result_to_json("smpl", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1
            if args.fail_fast:
                write_summary(args.summary_path, summary_rows)
                return exit_code

    if "soma-bvh" in args.exports:
        soma_assets_root = require_path(resolve_soma_assets_root(args.soma_assets_root), label="SOMA assets root")
        smpl_model_path = require_path(resolve_smpl_model_path(args.smpl_model_path), label="SMPL model")
        results = export_batch_soma_bvh(
            tasks,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
            soma_assets_root=soma_assets_root,
            smpl_model_path=smpl_model_path,
            skip_existing=args.skip_existing,
        )
        print_stage("soma-bvh", results)
        summary_rows.extend(result_to_json("soma-bvh", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1

    write_summary(args.summary_path, summary_rows)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
