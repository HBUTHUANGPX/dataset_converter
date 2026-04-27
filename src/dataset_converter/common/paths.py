from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _looks_like_workspace(path: Path) -> bool:
    return (path / "hdf5_parse").is_dir() and (path / "nymeria_parse").is_dir()


def get_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if _looks_like_workspace(parent):
            return parent
    cwd = Path.cwd().resolve()
    return cwd if _looks_like_workspace(cwd) else cwd


def ensure_workspace_on_sys_path() -> Path:
    workspace = get_repo_root()
    workspace_str = str(workspace)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
    return workspace


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_optional_path(
    *,
    explicit: str | Path | None = None,
    env_var: str | None = None,
    candidates: Iterable[str | Path] = (),
) -> Path | None:
    if explicit is not None:
        return expand_path(explicit)

    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return expand_path(env_value)

    for candidate in candidates:
        path = expand_path(candidate)
        if path.exists():
            return path
    return None


def require_path(path: str | Path | None, *, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"{label} was not found. Pass it explicitly or set the documented environment variable."
        )
    resolved = expand_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def default_hdf5_test_data_root() -> Path:
    return get_repo_root() / "hdf5_parse" / "test_data"


def default_hdf5_output_root() -> Path:
    return get_repo_root() / "hdf5_parse" / "out" / "batch"


def default_nymeria_test_data_root() -> Path:
    return get_repo_root() / "nymeria_parse" / "test_data"


def default_nymeria_output_root() -> Path:
    return get_repo_root() / "nymeria_parse" / "out" / "batch"


def default_package_assets_root() -> Path:
    return get_package_root().parents[1] / "assets"


def resolve_soma_assets_root(explicit: str | Path | None = None) -> Path | None:
    repo_root = get_repo_root()
    return resolve_optional_path(
        explicit=explicit,
        env_var="SOMA_ASSETS_ROOT",
        candidates=[
            default_package_assets_root() / "soma",
            repo_root / "assets" / "soma",
            repo_root / "assets",
        ],
    )


def resolve_smpl_model_path(explicit: str | Path | None = None) -> Path | None:
    soma_assets_root = resolve_soma_assets_root()
    candidates: list[Path] = []
    if soma_assets_root is not None:
        candidates.extend(
            [
                soma_assets_root / "SMPL" / "SMPL_NEUTRAL.npz",
                soma_assets_root / "SMPL" / "SMPL_NEUTRAL.pkl",
            ]
        )
    return resolve_optional_path(explicit=explicit, env_var="SMPL_MODEL_PATH", candidates=candidates)


def resolve_smplh_model_path(explicit: str | Path | None = None) -> Path | None:
    return resolve_optional_path(explicit=explicit, env_var="SMPLH_MODEL_PATH", candidates=())
