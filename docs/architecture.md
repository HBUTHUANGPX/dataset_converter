# Architecture

`dataset_converter` is the new package-level API. The legacy folders remain in place while the package is introduced.

## Package Layout

```text
dataset_converter/
├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── docs/
├── common/
│   ├── batch.py
│   ├── cli.py
│   ├── paths.py
│   ├── rotations.py
│   ├── smpl.py
│   └── text.py
├── hdf5/
│   ├── annotation.py
│   ├── batch.py
│   ├── io.py
│   ├── soma_bvh.py
│   ├── smpl.py
│   └── cli/
└── nymeria/
    ├── annotation.py
    ├── batch.py
    ├── mvnx.py
    ├── soma_bvh.py
    ├── smpl.py
    ├── video.py
    ├── xsens_smpl.py
    └── cli/
└── soma/
    ├── bvh.py
    ├── inversion.py
    └── transforms.py
src/soma/
└── vendored SOMA Python runtime imported as top-level `soma`
```

## Packaging

`pyproject.toml` only declares the PEP 517 build backend. Package metadata, Python version, dependencies, entry points, and extras live in `setup.cfg`.

Dependency profiles:

- Base install: `h5py`, `numpy`, `scipy`, `tqdm`.
- `video` extra: Nymeria Project Aria VRS decoding through `projectaria-tools`, plus OpenCV as the MP4 fallback writer. Runtime export prefers system `ffmpeg` when it is available.
- `soma` extra: GPU/SOMA runtime dependencies such as `torch`, `smplx`, `trimesh`, and `warp-lang`.
- `gpu` extra: alias for `soma`.
- `dev` extra: tests plus SOMA dependencies.

Python 3.11 is the recommended and declared minimum runtime.

## Shared Semantics

`dataset_converter.common.paths` owns path resolution:

- repository-relative defaults for test data and output roots;
- environment variables for external assets;
- no machine-specific absolute paths in the new package.

`dataset_converter.common.batch` owns the shared batch result model and execution helpers:

- `BatchExportResult`
- multiprocess execution for CPU/IO stages;
- sequential execution for CUDA/SOMA stages.

`dataset_converter.common.cli` owns JSONL summaries and stage printing.

`dataset_converter.common.text`, `rotations`, and `smpl` own the shared text-pool, root-frame conversion, and SMPL motion container semantics used by both datasets.

## HDF5 Pipeline

HDF5 tasks are discovered from:

```text
<test-data-root>/<subset_id>/<episode_id>/annotation.hdf5
```

Each task writes under:

```text
<output-root>/<subset_id>/<episode_id>/
```

Current stage ownership:

- `annotation`: native `dataset_converter.hdf5.annotation`.
- `smpl`: native `dataset_converter.hdf5.smpl`.
- `soma-bvh`: native `dataset_converter.hdf5.soma_bvh`, using shared `dataset_converter.soma` modules. Raw HDF5 SMPL root motion is converted to the SOMA Y-up frame before SOMA inversion; SOMA BVH position channels are written in centimeters as first-frame-relative motion on top of the SOMA reference offsets; no post-inversion visualization-frame rotation is applied.

## Nymeria Pipeline

Nymeria tasks are discovered from:

```text
<test-data-root>/<sequence_id>/body_xdata_mvnx
```

Each task writes under:

```text
<output-root>/<sequence_id>/
```

Current stage ownership:

- `annotation`: native `dataset_converter.nymeria.annotation`.
- `smpl`: native `dataset_converter.nymeria.smpl`.
- `soma-bvh`: native `dataset_converter.nymeria.soma_bvh`, using shared `dataset_converter.soma` modules. Nymeria SMPL motion is already prepared for the SOMA Y-up path; SOMA BVH position channels are written in centimeters as first-frame-relative motion on top of the SOMA reference offsets; no post-inversion visualization-frame rotation is applied.
- `head-video`: native `dataset_converter.nymeria.video`, reading `recording_head/data/data.vrs` and exporting selected head-camera streams plus a timestamp sidecar. The default SLAM left/right streams are stereo grayscale; `rgb` exports the color camera.

## Migration Plan

The compatibility direction is:

1. Keep `hdf5_parse` and `nymeria_parse` working while downstream scripts move over.
2. Route new CPU/IO annotation and SMPL export work through `dataset_converter`.
3. Keep the SOMA Python runtime and SOMA assets as explicit environment/runtime dependencies, not source-tree path injections.
4. Turn old scripts into thin wrappers or retire them once downstream users move to the new package.
