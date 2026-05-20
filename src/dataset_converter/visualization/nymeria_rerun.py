from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation


SMPL_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)
SMPL_PARENT_INDICES = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)
TEXT_KEYS = (
    ("main_task", "main_task_texts", "main_task_text_indices"),
    ("sub_task", "sub_task_texts", "sub_task_text_indices"),
    ("current_action", "current_action_texts", "current_action_text_indices"),
    ("interaction", "interaction_texts", "interaction_text_indices"),
)


@dataclass(frozen=True)
class RgbVideoMetadata:
    video_path: Path
    timestamps_ns: np.ndarray
    frame_indices: np.ndarray
    fps: float
    timestamp_kind: str


@dataclass(frozen=True)
class ConvertedNymeriaSequence:
    sequence_dir: Path
    annotation_path: Path
    smpl_path: Path
    annotation: dict[str, np.ndarray]
    smpl: dict[str, np.ndarray]
    rgb_video: RgbVideoMetadata | None

    @classmethod
    def load(cls, sequence_dir: str | Path) -> "ConvertedNymeriaSequence":
        sequence_dir = Path(sequence_dir)
        annotation_path = sequence_dir / "annotation.npz"
        smpl_path = sequence_dir / "smpl" / "nymeria_smpl.npz"
        if not annotation_path.is_file():
            raise FileNotFoundError(_format_missing_sequence_file_error(sequence_dir, annotation_path, "annotation.npz"))
        if not smpl_path.is_file():
            raise FileNotFoundError(_format_missing_sequence_file_error(sequence_dir, smpl_path, "smpl/nymeria_smpl.npz"))
        annotation = _load_npz(annotation_path, allow_pickle=True)
        smpl = _load_npz(smpl_path, allow_pickle=False)
        return cls(
            sequence_dir=sequence_dir,
            annotation_path=annotation_path,
            smpl_path=smpl_path,
            annotation=annotation,
            smpl=smpl,
            rgb_video=load_rgb_video_metadata(sequence_dir / "head_video"),
        )

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    @property
    def timestamps_ns(self) -> np.ndarray:
        if "relative_frame_timestamps_ns" in self.annotation:
            return np.asarray(self.annotation["relative_frame_timestamps_ns"], dtype=np.int64).reshape(-1)
        if "frame_timestamps_ns" in self.annotation:
            return np.asarray(self.annotation["frame_timestamps_ns"], dtype=np.int64).reshape(-1)
        return np.asarray(self.annotation["frame_timestamps"], dtype=np.int64).reshape(-1) * 1_000_000

    @property
    def frame_indices(self) -> np.ndarray:
        return np.asarray(self.annotation["timeline_frame_indices"], dtype=np.int32).reshape(-1)

    def text_at(self, frame_index: int) -> dict[str, str]:
        result: dict[str, str] = {}
        for label, texts_key, indices_key in TEXT_KEYS:
            texts = np.asarray(self.annotation[texts_key], dtype=object)
            indices = np.asarray(self.annotation[indices_key], dtype=np.int64)
            result[label] = str(texts[int(indices[frame_index])])
        return result


@dataclass(frozen=True)
class SMPLSkeleton:
    joint_names: tuple[str, ...]
    parent_indices: np.ndarray
    joint_positions: np.ndarray

    @property
    def edges(self) -> np.ndarray:
        return np.asarray([(parent, idx) for idx, parent in enumerate(self.parent_indices) if parent >= 0], dtype=np.int32)


def _load_npz(path: Path, *, allow_pickle: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def _format_missing_sequence_file_error(sequence_dir: Path, missing_path: Path, missing_label: str) -> str:
    cwd = Path.cwd()
    return (
        f"Missing converted {missing_label}: {missing_path}\n"
        f"Current working directory: {cwd}\n"
        "Expected converted sequence layout:\n"
        f"  {sequence_dir}/annotation.npz\n"
        f"  {sequence_dir}/smpl/nymeria_smpl.npz\n"
        f"  {sequence_dir}/head_video/rgb.mp4  (optional)\n"
        "Pass --sequence-dir as the directory that directly contains annotation.npz. "
        "For example, if you exported under dataset_converter/test_out/nymeria_batch, "
        "use dataset_converter/test_out/nymeria_batch/<sequence_id> when running from the workspace root."
    )


def nearest_index(sorted_values: np.ndarray, value: int | float) -> int:
    values = np.asarray(sorted_values, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot search an empty timestamp array.")
    insert = int(np.searchsorted(values, int(value), side="left"))
    if insert <= 0:
        return 0
    if insert >= values.size:
        return int(values.size - 1)
    before = values[insert - 1]
    after = values[insert]
    return insert - 1 if abs(int(value) - int(before)) <= abs(int(after) - int(value)) else insert


def load_rgb_video_metadata(video_dir: str | Path) -> RgbVideoMetadata | None:
    video_dir = Path(video_dir)
    video_path = video_dir / "rgb.mp4"
    timestamps_path = video_dir / "timestamps.npz"
    if not video_path.is_file() or not timestamps_path.is_file():
        return None
    with np.load(timestamps_path, allow_pickle=False) as data:
        if "rgb_timestamps_ns" not in data:
            return None
        timestamp_key = "rgb_relative_timestamps_ns" if "rgb_relative_timestamps_ns" in data else "rgb_timestamps_ns"
        timestamp_kind = "relative" if timestamp_key == "rgb_relative_timestamps_ns" else "absolute"
        timestamps_ns = np.asarray(data[timestamp_key], dtype=np.int64).reshape(-1)
        frame_indices = np.asarray(data.get("rgb_frame_indices", np.arange(timestamps_ns.shape[0])), dtype=np.int32).reshape(-1)
        fps = float(np.asarray(data.get("rgb_stream_fps", 30.0)).reshape(()))
    return RgbVideoMetadata(video_path=video_path, timestamps_ns=timestamps_ns, frame_indices=frame_indices, fps=fps, timestamp_kind=timestamp_kind)


def load_smpl_skeleton(
    smpl_payload: dict[str, np.ndarray],
    *,
    smpl_model_path: str | Path,
    frame_slice: slice | None = None,
) -> SMPLSkeleton:
    model = _load_smpl_model_npz(Path(smpl_model_path))
    global_orient = np.asarray(smpl_payload["global_orient"], dtype=np.float32)
    body_pose = np.asarray(smpl_payload["body_pose"], dtype=np.float32).reshape(global_orient.shape[0], 23, 3)
    transl = np.asarray(smpl_payload["transl"], dtype=np.float32)
    betas = np.asarray(smpl_payload["betas"], dtype=np.float32)
    if frame_slice is not None:
        global_orient = global_orient[frame_slice]
        body_pose = body_pose[frame_slice]
        transl = transl[frame_slice]
        betas = betas[frame_slice]

    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None, :], (global_orient.shape[0], betas.shape[0]))
    rotations = Rotation.from_rotvec(np.concatenate([global_orient[:, None, :], body_pose], axis=1).reshape(-1, 3)).as_matrix()
    rotations = rotations.reshape(global_orient.shape[0], 24, 3, 3).astype(np.float32)
    rest_joints = _compute_shaped_rest_joints(model, betas)
    joint_positions = _forward_kinematics(rest_joints, rotations, transl, model["parents"])
    return SMPLSkeleton(tuple(SMPL_JOINT_NAMES), model["parents"], joint_positions)


def _load_smpl_model_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMPL model file: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = ("v_template", "shapedirs", "J_regressor")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"SMPL model is missing required arrays: {', '.join(missing)}")
        parents = _load_smpl_parents(data)
        return {
            "v_template": np.asarray(data["v_template"], dtype=np.float32),
            "shapedirs": np.asarray(data["shapedirs"], dtype=np.float32),
            "J_regressor": _dense_regressor(data["J_regressor"]),
            "parents": parents,
        }


def _dense_regressor(value: np.ndarray) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=np.float32)
    if value.dtype == object and value.shape == ():
        item = value.item()
        if hasattr(item, "toarray"):
            return np.asarray(item.toarray(), dtype=np.float32)
        return np.asarray(item, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_smpl_parents(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "kintree_table" not in data:
        return SMPL_PARENT_INDICES.copy()
    table = np.asarray(data["kintree_table"], dtype=np.int64)
    parents = table[0].copy()
    joint_ids = table[1].copy()
    id_to_index = {int(joint_id): idx for idx, joint_id in enumerate(joint_ids.tolist())}
    parent_indices = np.asarray([id_to_index.get(int(parent_id), -1) for parent_id in parents.tolist()], dtype=np.int32)
    parent_indices[0] = -1
    if parent_indices.shape[0] != 24:
        return SMPL_PARENT_INDICES.copy()
    return parent_indices


def _compute_shaped_rest_joints(model: dict[str, np.ndarray], betas: np.ndarray) -> np.ndarray:
    v_template = model["v_template"]
    shapedirs = model["shapedirs"]
    regressor = model["J_regressor"]
    beta_count = min(betas.shape[1], shapedirs.shape[-1])
    shaped_vertices = v_template[None, :, :] + np.einsum("fb,vcb->fvc", betas[:, :beta_count], shapedirs[:, :, :beta_count])
    return np.einsum("jv,fvc->fjc", regressor, shaped_vertices).astype(np.float32)


def _forward_kinematics(rest_joints: np.ndarray, local_rotations: np.ndarray, transl: np.ndarray, parents: np.ndarray) -> np.ndarray:
    frame_count, joint_count = rest_joints.shape[:2]
    global_rotations = np.empty((frame_count, joint_count, 3, 3), dtype=np.float32)
    global_positions = np.empty((frame_count, joint_count, 3), dtype=np.float32)
    for joint_idx, parent_idx in enumerate(parents.tolist()):
        if parent_idx < 0:
            global_rotations[:, joint_idx] = local_rotations[:, joint_idx]
            global_positions[:, joint_idx] = rest_joints[:, joint_idx] + transl
            continue
        offset = rest_joints[:, joint_idx] - rest_joints[:, parent_idx]
        global_rotations[:, joint_idx] = np.einsum("fij,fjk->fik", global_rotations[:, parent_idx], local_rotations[:, joint_idx])
        global_positions[:, joint_idx] = global_positions[:, parent_idx] + np.einsum("fij,fj->fi", global_rotations[:, parent_idx], offset)
    return global_positions.astype(np.float32)


def run_nymeria_rerun_viewer(
    *,
    sequence_dir: str | Path,
    smpl_model_path: str | Path,
    output_rrd: str | Path | None = None,
    save_rrd: bool = False,
    stride: int = 1,
    video_stride: int = 1,
    spawn: bool = True,
) -> None:
    if stride <= 0 or video_stride <= 0:
        raise ValueError("stride and video_stride must be positive.")

    rr = _import_rerun()
    sequence = ConvertedNymeriaSequence.load(sequence_dir)
    frame_slice = slice(None, None, stride)
    skeleton = load_smpl_skeleton(sequence.smpl, smpl_model_path=smpl_model_path, frame_slice=frame_slice)
    timestamps_ns = sequence.timestamps_ns[frame_slice]
    frame_indices = sequence.frame_indices[frame_slice]
    source_frame_indices = np.arange(sequence.frame_count, dtype=np.int32)[frame_slice]

    if output_rrd is None and save_rrd:
        output_rrd = Path(sequence_dir) / "converted_nymeria.rrd"

    rr.init("dataset-converter nymeria viewer", spawn=spawn and output_rrd is None, recording_id=uuid4())
    if output_rrd is not None:
        rr.save(str(output_rrd))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/body/smpl_joint_names", rr.TextDocument("\n".join(skeleton.joint_names)), static=True)
    rr.log("world/text/time_axis", rr.TextDocument(_format_time_axis_markdown(sequence), media_type="text/markdown"), static=True)
    _log_smpl_skeleton(rr, sequence, skeleton, timestamps_ns, frame_indices, source_frame_indices)
    _log_rgb_video(rr, sequence.rgb_video, stride=video_stride)
    rr.disconnect()


def _log_smpl_skeleton(rr, sequence: ConvertedNymeriaSequence, skeleton: SMPLSkeleton, timestamps_ns: np.ndarray, frame_indices: np.ndarray, source_frame_indices: np.ndarray) -> None:
    edges = skeleton.edges
    colors = np.asarray([[134, 218, 234]], dtype=np.uint8)
    for local_idx, (source_idx, frame_idx, timestamp_ns) in enumerate(zip(source_frame_indices.tolist(), frame_indices.tolist(), timestamps_ns.tolist(), strict=True)):
        rr.set_time("frame", sequence=int(frame_idx))
        rr.set_time("timestamp", timestamp=float(timestamp_ns) * 1e-9)
        joints = skeleton.joint_positions[local_idx]
        rr.log("world/body/smpl_skeleton", rr.LineStrips3D(joints[edges], colors=colors, radii=0.01))
        rr.log("world/body/smpl_joints", rr.Points3D(joints, colors=colors, radii=0.018))
        rr.log("world/text/description", rr.TextDocument(_format_text_markdown(sequence.text_at(int(source_idx))), media_type="text/markdown"))


def _log_rgb_video(rr, metadata: RgbVideoMetadata | None, *, stride: int) -> None:
    if metadata is None:
        return
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional viewer runtime.
        raise ImportError("Nymeria Rerun viewer requires opencv-python for rgb.mp4 sync. Install dataset-converter[viewer].") from exc

    capture = cv2.VideoCapture(str(metadata.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open RGB video: {metadata.video_path}")
    try:
        frame_idx = 0
        timestamp_count = int(metadata.timestamps_ns.shape[0])
        while frame_idx < timestamp_count:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                rr.set_time("rgb_frame", sequence=int(metadata.frame_indices[frame_idx]))
                rr.set_time("timestamp", timestamp=float(metadata.timestamps_ns[frame_idx]) * 1e-9)
                frame_rgb = frame_bgr[:, :, ::-1]
                rr.log("recording_head/rgb", rr.Image(frame_rgb).compress(jpeg_quality=90))
            frame_idx += 1
    finally:
        capture.release()


def _format_text_markdown(text: dict[str, str]) -> str:
    return (
        "# Nymeria Annotation\n\n"
        f"- **Main Task:** {text['main_task']}\n"
        f"- **Sub Task:** {text['sub_task']}\n"
        f"- **Current Action:** {text['current_action']}\n"
        f"- **Interaction:** {text['interaction']}\n"
    )


def _format_time_axis_markdown(sequence: ConvertedNymeriaSequence) -> str:
    annotation = sequence.annotation
    timeline_kind = "relative" if "relative_frame_timestamps_ns" in annotation else "absolute"
    time_domain = str(np.asarray(annotation.get("time_domain", "unknown")).reshape(()))
    time_zero = annotation.get("time_zero_ns")
    time_zero_text = "N/A" if time_zero is None else str(int(np.asarray(time_zero).reshape(())))
    time_zero_source = str(np.asarray(annotation.get("time_zero_source", "N/A")).reshape(()))
    rgb_timeline = "missing"
    if sequence.rgb_video is not None:
        rgb_timeline = sequence.rgb_video.timestamp_kind
    return (
        "# Time Axis\n\n"
        f"- **Motion/Text Timeline:** {timeline_kind}\n"
        f"- **Time Domain:** {time_domain}\n"
        f"- **Time Zero Source:** {time_zero_source}\n"
        f"- **Time Zero ns:** {time_zero_text}\n"
        f"- **RGB Timeline:** {rgb_timeline}\n"
    )


def _import_rerun():
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - optional viewer runtime.
        raise ImportError("Nymeria Rerun viewer requires rerun-sdk. Install dataset-converter[viewer].") from exc
    return rr


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize converted Nymeria SMPL skeleton, text, and RGB video with Rerun.")
    parser.add_argument("--sequence-dir", type=Path, required=True, help="Converted sequence directory containing annotation.npz, smpl/nymeria_smpl.npz, and optional head_video/rgb.mp4.")
    parser.add_argument("--smpl-model-path", type=Path, required=True, help="Path to SMPL_NEUTRAL.npz used to compute the SMPL skeleton joints.")
    parser.add_argument("--output-rrd", type=Path, default=None, help="Optional Rerun .rrd path. If omitted, the interactive Rerun viewer is spawned.")
    parser.add_argument("--save-rrd", action="store_true", help="Save to <sequence-dir>/converted_nymeria.rrd instead of spawning the viewer.")
    parser.add_argument("--stride", type=int, default=1, help="Skeleton/text frame stride.")
    parser.add_argument("--video-stride", type=int, default=1, help="RGB video frame stride.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_nymeria_rerun_viewer(
        sequence_dir=args.sequence_dir,
        smpl_model_path=args.smpl_model_path,
        output_rrd=args.output_rrd,
        save_rrd=args.save_rrd,
        stride=args.stride,
        video_stride=args.video_stride,
    )


if __name__ == "__main__":
    main()
