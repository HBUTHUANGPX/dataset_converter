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
    export_batch_head_video,
    export_batch_smpl,
    export_batch_soma_bvh,
)
from dataset_converter.nymeria.video import HEAD_VIDEO_STREAMS


EXPORT_CHOICES = ("annotation", "smpl", "soma-bvh", "head-video")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch export Nymeria motion assets. SMPL/annotation/head-video may use multiprocessing; SOMA BVH is sequential.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    common = parser.add_argument_group("Common options")
    common.add_argument("--test-data-root", type=Path, default=default_nymeria_test_data_root(), help="Root containing <sequence>/body_xdata_mvnx files.")
    common.add_argument("--output-root", type=Path, default=default_nymeria_output_root(), help="Root directory for exported files.")
    common.add_argument("--exports", nargs="+", choices=EXPORT_CHOICES, default=["annotation", "smpl"], help="Export stages to run.")
    common.add_argument("--workers", type=int, default=1, help="Multiprocessing worker count for annotation, SMPL, and head-video stages. SOMA BVH stays sequential.")
    common.add_argument("--start-frame", type=int, default=0, help="Start frame index for MVNX/head-video exports.")
    common.add_argument("--end-frame", type=int, default=-1, help="Exclusive end frame index; -1 means all remaining frames.")
    common.add_argument("--stride", type=int, default=1, help="Frame stride for exported motion/video timelines.")
    common.add_argument("--skip-existing", action="store_true", help="Skip a stage output when the expected output file(s) already exist.")
    common.add_argument("--summary-path", type=Path, default=None, help="Optional JSONL summary path with one row per task/stage.")
    common.add_argument("--fail-fast", action="store_true", help="Stop after the first failing stage instead of continuing later stages.")

    soma_group = parser.add_argument_group("SOMA BVH options")
    soma_group.add_argument("--device", default="cuda", help="Torch device used by the SOMA BVH stage.")
    soma_group.add_argument("--batch-size", type=int, default=DEFAULT_SOMA_BATCH_SIZE, help="GPU batch size for SOMA inversion.")
    soma_group.add_argument("--soma-assets-root", type=Path, default=None, help="SOMA assets root. Falls back to SOMA_ASSETS_ROOT or package candidates.")
    soma_group.add_argument("--smpl-model-path", type=Path, default=None, help="SMPL model path. Falls back to SMPL_MODEL_PATH or SOMA assets.")

    video_group = parser.add_argument_group("Nymeria head-video options")
    video_group.add_argument("--video-fps", type=float, default=None, help="Fixed FPS for head-video MP4 files. None estimates FPS from VRS timestamps.")
    video_group.add_argument("--video-max-frames", type=int, default=None, help="Maximum frames per exported head-video stream.")
    video_group.add_argument("--video-rotate-degrees", type=int, choices=(0, 90, 180, 270), default=0, help="Rotate exported video frames clockwise.")
    video_group.add_argument(
        "--video-streams",
        nargs="+",
        choices=tuple(HEAD_VIDEO_STREAMS),
        default=["slam-left", "slam-right"],
        help="Head-video streams to export. SLAM streams are stereo grayscale; rgb is the color camera.",
    )
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

    if "head-video" in args.exports:
        results = export_batch_head_video(
            tasks,
            workers=args.workers,
            skip_existing=args.skip_existing,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
            max_frames=args.video_max_frames,
            fps=args.video_fps,
            rotate_degrees=args.video_rotate_degrees,
            streams=tuple(args.video_streams),
        )
        print_stage("head-video", results)
        summary_rows.extend(result_to_json("head-video", result) for result in results)
        if any(not result.ok for result in results):
            exit_code = 1

    write_summary(args.summary_path, summary_rows)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
