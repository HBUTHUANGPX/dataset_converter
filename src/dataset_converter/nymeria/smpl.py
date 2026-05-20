from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_converter.nymeria.constants import NYMERIA_TIME_ALIGNMENT_VERSION
from dataset_converter.nymeria.mvnx import MvnxFrameSliceConfig, MvnxMotion, MvnxXmlMotionReader
from dataset_converter.nymeria.xsens_smpl import (
    convert_xsens_root_pos_to_smpl_transl,
    global_to_local_rotations,
    map_xsens_global_rotations_to_smpl,
    matrices_to_rotvec,
)


class SmplMotionConverter:
    """Convert Nymeria Xsens segment motion into SMPL motion arrays.

    Responsibilities:
        Map Xsens global segment rotations to SMPL local axis-angle pose and
        convert Xsens root positions into SMPL translations.
    Preconditions:
        Input ``MvnxMotion`` contains segment quaternions and positions with a
        frame dimension.
    Postconditions:
        Returns SMPL pose arrays without writing files.
    """

    def convert(self, motion: MvnxMotion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert one parsed MVNX motion to SMPL arrays.

        Preconditions:
            ``motion.segment_quat_wxyz`` and ``motion.segment_pos_xyz`` are
            populated and frame-aligned.
        Postconditions:
            Returns ``global_orient``, ``body_pose``, and ``transl`` arrays.
        """

        smpl_global = map_xsens_global_rotations_to_smpl(motion.segment_quat_wxyz)
        smpl_local = global_to_local_rotations(smpl_global)
        smpl_rotvec = matrices_to_rotvec(smpl_local)
        frame_count = motion.num_frames
        transl = convert_xsens_root_pos_to_smpl_transl(motion.segment_pos_xyz)
        return (
            np.asarray(smpl_rotvec[:, 0], dtype=np.float32),
            np.asarray(smpl_rotvec[:, 1:].reshape(frame_count, 69), dtype=np.float32),
            np.asarray(transl, dtype=np.float32),
        )


class SmplPayloadBuilder:
    """Build standard SMPL NPZ payloads for Nymeria sequences.

    Responsibilities:
        Load MVNX motion, convert it to SMPL arrays, and attach timestamp
        metadata used by downstream visualization and training code.
    Preconditions:
        ``sequence_dir`` contains ``body_xdata_mvnx``.
    Postconditions:
        Returns a dictionary suitable for ``np.savez``.
    """

    def __init__(self, *, motion_reader: MvnxXmlMotionReader | None = None, motion_converter: SmplMotionConverter | None = None) -> None:
        """Create a SMPL payload builder.

        Preconditions:
            Injected collaborators satisfy the documented reader/converter
            behavior.
        Postconditions:
            The builder is ready to produce SMPL payloads.
        """

        self._motion_reader = motion_reader or MvnxXmlMotionReader()
        self._motion_converter = motion_converter or SmplMotionConverter()

    @property
    def motion_converter(self) -> SmplMotionConverter:
        """Return the SMPL motion converter.

        Preconditions:
            The builder has been constructed.
        Postconditions:
            Returns the converter used for pose/transl conversion.
        """

        return self._motion_converter

    def build(
        self,
        sequence_dir: str | Path,
        *,
        start_frame: int = 0,
        end_frame: int = -1,
        stride: int = 1,
        time_zero_ns: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Build a SMPL motion payload.

        Preconditions:
            ``sequence_dir/body_xdata_mvnx`` exists and ``stride`` is positive.
        Postconditions:
            Returns standard SMPL fields plus Nymeria timestamp metadata.
        """

        sequence_dir = Path(sequence_dir)
        motion = self._motion_reader.read(
            sequence_dir / "body_xdata_mvnx",
            frame_slice=MvnxFrameSliceConfig(start_frame=int(start_frame), end_frame=int(end_frame), stride=int(stride)),
        )
        global_orient, body_pose, transl = self.motion_converter.convert(motion)
        timestamps_ns = np.asarray(motion.frame_timestamps, dtype=np.int64) * 1_000_000
        payload = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": transl,
            "betas": np.zeros((motion.num_frames, 10), dtype=np.float32),
            "timestamps_ns": timestamps_ns,
            "raw_timestamps_ns": np.asarray(motion.raw_frame_timestamps, dtype=np.int64) * 1_000_000,
            "frame_indices": np.asarray(motion.frame_indices, dtype=np.int32),
            "time_domain": np.asarray("time_code"),
            "mvnx_timestamp_source": np.asarray(motion.timestamp_source),
            "nymeria_time_alignment_version": np.asarray(NYMERIA_TIME_ALIGNMENT_VERSION, dtype=np.int32),
        }
        if time_zero_ns is not None:
            time_zero_ns = int(time_zero_ns)
            payload.update(
                {
                    "time_zero_ns": np.asarray(time_zero_ns, dtype=np.int64),
                    "time_zero_source": np.asarray("recording_head/rgb/frame_0"),
                    "time_zero_time_domain": np.asarray("time_code"),
                    "relative_timestamps_ns": timestamps_ns - time_zero_ns,
                }
            )
        return payload


def build_smpl_motion_payload(
    sequence_dir: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    stride: int = 1,
    time_zero_ns: int | None = None,
) -> dict[str, np.ndarray]:
    """Build a standard SMPL payload for one Nymeria sequence.

    Preconditions:
        ``sequence_dir`` contains ``body_xdata_mvnx``.
    Postconditions:
        Returns an NPZ-ready SMPL payload. This function is a compatibility
        wrapper around ``SmplPayloadBuilder``.
    """

    return SmplPayloadBuilder().build(
        sequence_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
        time_zero_ns=time_zero_ns,
    )


def save_smpl_motion_npz(payload: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return output_path
