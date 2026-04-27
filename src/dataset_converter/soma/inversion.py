from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from dataset_converter.common.smpl import SMPLBodyMotion
from dataset_converter.soma.transforms import pose7_from_transforms


def configure_warp_quiet(quiet: bool = True) -> None:
    import warp as wp

    wp.config.quiet = bool(quiet)


def _infer_model_num_betas(model_path: Path, fallback: int) -> int:
    model_path = Path(model_path)
    if model_path.suffix == ".npz":
        with np.load(model_path, allow_pickle=True) as model_data:
            shapedirs = model_data["shapedirs"]
        return int(max(1, min(int(shapedirs.shape[-1]), int(fallback))))

    if model_path.suffix == ".pkl":
        with model_path.open("rb") as handle:
            model_data = pickle.load(handle, encoding="latin1")
        shapedirs = model_data["shapedirs"]
        return int(max(1, min(int(shapedirs.shape[-1]), int(fallback))))

    return int(max(1, fallback))


def _match_beta_dimension(betas: np.ndarray, target_dim: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas[None, :]
    if betas.shape[-1] == target_dim:
        return betas.astype(np.float32, copy=False)
    if betas.shape[-1] > target_dim:
        return betas[..., :target_dim].astype(np.float32, copy=False)

    padded = np.zeros(betas.shape[:-1] + (target_dim,), dtype=np.float32)
    padded[..., : betas.shape[-1]] = betas
    return padded


def _build_smpl_model(model_path: Path, *, num_betas: int, device: str):
    import smplx
    from smplx.body_models import Struct

    ext = model_path.suffix.lstrip(".")
    if ext == "npz":
        model_data = np.load(model_path, allow_pickle=True)
        data_struct = Struct(**{key: model_data[key] for key in model_data.files})
        return smplx.SMPL(
            str(model_path),
            data_struct=data_struct,
            gender="neutral",
            num_betas=num_betas,
            batch_size=1,
        ).to(device)

    return smplx.create(
        model_path=str(model_path),
        model_type="smpl",
        gender="neutral",
        ext=ext,
        num_betas=num_betas,
        batch_size=1,
    ).to(device)


def _import_soma_runtime():
    try:
        from soma._compat import ensure_legacy_dependency_apis
        from soma.geometry.rig_utils import joint_world_to_local, remove_joint_orient_local
        from soma.geometry.transforms import matrix_to_rotvec
        from soma.pose_inversion import PoseInversion
        from soma.soma import SOMALayer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The SOMA runtime package is not importable. Install or expose the SOMA Python package "
            "inside the active environment; dataset_converter no longer injects an external source-tree path."
        ) from exc

    ensure_legacy_dependency_apis()

    import torch

    return torch, SOMALayer, PoseInversion, joint_world_to_local, remove_joint_orient_local, matrix_to_rotvec


def iter_batch_slices(motion: SMPLBodyMotion, batch_size: int | None) -> list[tuple[int, int]]:
    if motion.num_frames <= 0:
        return []
    active_batch_size = motion.num_frames if batch_size is None else int(batch_size)
    if active_batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    return [(start, min(start + active_batch_size, motion.num_frames)) for start in range(0, motion.num_frames, active_batch_size)]


def _resolve_smpl_model_path(smpl_model_path: str | Path | None, soma_assets_root: Path) -> Path:
    if smpl_model_path is not None:
        return Path(smpl_model_path).expanduser().resolve()
    return (soma_assets_root / "SMPL" / "SMPL_NEUTRAL.npz").expanduser().resolve()


def run_soma_inversion(
    motion: SMPLBodyMotion,
    *,
    device: str = "cuda",
    batch_size: int | None = None,
    body_iters: int = 2,
    finger_iters: int = 0,
    full_iters: int = 1,
    autograd_iters: int = 0,
    autograd_lr: float = 5e-3,
    soma_assets_root: str | Path,
    smpl_model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    if not str(device).startswith("cuda"):
        raise ValueError("This exporter only supports CUDA execution.")

    soma_assets_root = Path(soma_assets_root).expanduser().resolve()
    smpl_model_path = _resolve_smpl_model_path(smpl_model_path, soma_assets_root)
    if not soma_assets_root.exists():
        raise FileNotFoundError(f"SOMA assets root does not exist: {soma_assets_root}")
    if not smpl_model_path.exists():
        raise FileNotFoundError(f"SMPL model does not exist: {smpl_model_path}")

    configure_warp_quiet(True)
    torch, SOMALayer, PoseInversion, joint_world_to_local, remove_joint_orient_local, matrix_to_rotvec = _import_soma_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this exporter, but no CUDA device is available.")

    num_betas = _infer_model_num_betas(smpl_model_path, fallback=int(np.asarray(motion.betas).shape[-1]))
    smpl_model = _build_smpl_model(smpl_model_path, num_betas=num_betas, device=device)
    motion_betas = _match_beta_dimension(motion.betas, num_betas)
    if motion_betas.shape[0] == 1 and motion.num_frames > 1:
        motion_betas = np.broadcast_to(motion_betas, (motion.num_frames, motion_betas.shape[-1])).copy()

    body_pose_np = np.asarray(motion.body_pose, dtype=np.float32)
    global_orient_np = np.asarray(motion.global_orient, dtype=np.float32)
    transl_np = np.asarray(motion.transl, dtype=np.float32)

    soma = SOMALayer(
        soma_assets_root,
        identity_model_type="smpl",
        device=device,
        mode="warp",
        identity_model_kwargs={
            "model_path": str(smpl_model_path),
            "num_betas": num_betas,
        },
    )
    inv = PoseInversion(soma, low_lod=True)
    identity_betas = torch.as_tensor(motion_betas[:1], dtype=torch.float32, device=device)
    inv.prepare_identity(identity_betas)

    with torch.no_grad():
        warmup_out = smpl_model(
            body_pose=torch.as_tensor(body_pose_np[:1], dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(global_orient_np[:1], dtype=torch.float32, device=device),
            betas=identity_betas,
            transl=torch.as_tensor(transl_np[:1], dtype=torch.float32, device=device),
        )
    inv.fit(
        warmup_out.vertices,
        body_iters=body_iters,
        finger_iters=finger_iters,
        full_iters=full_iters,
        autograd_iters=autograd_iters,
        autograd_lr=autograd_lr,
    )

    batch_slices = iter_batch_slices(motion, batch_size)
    all_local_transforms = []
    all_world_transforms = []
    all_soma_poses = []
    all_root_transl = []
    all_errors = []
    for start, end in tqdm(
        batch_slices,
        desc="SOMA inversion",
        unit="batch",
        dynamic_ncols=True,
        disable=len(batch_slices) <= 1,
    ):
        body_pose = torch.as_tensor(body_pose_np[start:end], dtype=torch.float32, device=device)
        global_orient = torch.as_tensor(global_orient_np[start:end], dtype=torch.float32, device=device)
        transl = torch.as_tensor(transl_np[start:end], dtype=torch.float32, device=device)
        betas = torch.as_tensor(motion_betas[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            smpl_out = smpl_model(body_pose=body_pose, global_orient=global_orient, betas=betas, transl=transl)
        result = inv.fit(
            smpl_out.vertices,
            body_iters=body_iters,
            finger_iters=finger_iters,
            full_iters=full_iters,
            autograd_iters=autograd_iters,
            autograd_lr=autograd_lr,
        )
        rotations = result["rotations"]
        root_transl = result["root_translation"]
        all_errors.append(result["per_vertex_error"].detach().cpu())

        active_soma = inv.soma
        bind_transforms = active_soma._cached_bind_transforms_world
        rest_shape = active_soma._cached_rest_shape
        if bind_transforms.shape[0] == 1 and rotations.shape[0] > 1:
            bind_transforms = bind_transforms.expand(rotations.shape[0], -1, -1, -1)
        if rest_shape.shape[0] == 1 and rotations.shape[0] > 1:
            rest_shape = rest_shape.expand(rotations.shape[0], -1, -1)
        active_soma.batched_skinning.rebind(bind_transforms, rest_shape)
        with torch.no_grad():
            _, world_transforms = active_soma.batched_skinning.pose(
                rotations,
                root_transl,
                absolute_pose=True,
                return_transforms=True,
            )
        local_transforms = joint_world_to_local(world_transforms, active_soma.joint_parent_ids)
        relative_rotations = remove_joint_orient_local(
            rotations,
            active_soma._t_pose_orient,
            active_soma._t_pose_orient_parent_T,
        )
        soma_poses = matrix_to_rotvec(relative_rotations.reshape(-1, 3, 3)).reshape(
            relative_rotations.shape[0],
            relative_rotations.shape[1],
            3,
        )

        all_local_transforms.append(pose7_from_transforms(local_transforms.detach().cpu().numpy()))
        all_world_transforms.append(pose7_from_transforms(world_transforms.detach().cpu().numpy()))
        all_soma_poses.append(soma_poses.detach().cpu().numpy().astype(np.float32))
        all_root_transl.append(root_transl.detach().cpu().numpy())

        del body_pose, global_orient, transl, betas, smpl_out, result, rotations, root_transl
        del world_transforms, local_transforms, relative_rotations, soma_poses
        torch.cuda.empty_cache()

    active_soma = inv.soma
    per_vertex_error = torch.cat(all_errors, dim=0).cpu().numpy().astype(np.float32)
    return {
        "joint_names": list(active_soma.rig_data["joint_names"]),
        "parent_indices": active_soma.joint_parent_ids.detach().cpu().numpy().astype(np.int32),
        "reference_local_transforms": pose7_from_transforms(active_soma.t_pose_local.detach().cpu().numpy()),
        "local_transforms": np.concatenate(all_local_transforms, axis=0).astype(np.float32),
        "world_transforms": np.concatenate(all_world_transforms, axis=0).astype(np.float32),
        "soma_poses": np.concatenate(all_soma_poses, axis=0).astype(np.float32),
        "soma_transl": np.concatenate(all_root_transl, axis=0).astype(np.float32),
        "soma_joint_orient": active_soma._t_pose_orient.detach().cpu().numpy().astype(np.float32),
        "per_vertex_error": per_vertex_error,
    }
