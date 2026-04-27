# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections import namedtuple

import numpy as np


ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")


def ensure_legacy_inspect_apis():
    """Restore inspect APIs removed in Python 3.11 for legacy dependencies."""
    if hasattr(inspect, "getargspec"):
        return

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec


def ensure_legacy_numpy_aliases():
    """Restore NumPy scalar aliases removed in NumPy 2.x for legacy dependencies."""
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }

    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def ensure_legacy_dependency_apis():
    ensure_legacy_inspect_apis()
    ensure_legacy_numpy_aliases()
