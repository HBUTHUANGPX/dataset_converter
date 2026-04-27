"""Development shim for the src-layout dataset_converter package.

When the package is installed with ``pip install -e dataset_converter``, Python
loads the real package from ``dataset_converter/src``. This shim keeps imports
working when tests are run directly from the larger mimic_baseline workspace.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

_SRC_PACKAGE = Path(__file__).resolve().parent / "src" / "dataset_converter"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))
