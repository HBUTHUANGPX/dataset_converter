# Architecture

`dataset_converter` is the new package-level API. The legacy folders remain in place while the package is introduced.

## Package Layout

```text
dataset_converter/
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
    ├── xsens_smpl.py
    └── cli/
└── soma/
    ├── bvh.py
    ├── inversion.py
    └── transforms.py
```

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
- `soma-bvh`: native `dataset_converter.hdf5.soma_bvh`, using shared `dataset_converter.soma` modules.

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
- `soma-bvh`: native `dataset_converter.nymeria.soma_bvh`, using shared `dataset_converter.soma` modules.

## Migration Plan

The compatibility direction is:

1. Keep `hdf5_parse` and `nymeria_parse` working while downstream scripts move over.
2. Route new CPU/IO annotation and SMPL export work through `dataset_converter`.
3. Keep the SOMA Python runtime and SOMA assets as explicit environment/runtime dependencies, not source-tree path injections.
4. Turn old scripts into thin wrappers or retire them once downstream users move to the new package.
