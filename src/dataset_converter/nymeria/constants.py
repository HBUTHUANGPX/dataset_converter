from __future__ import annotations

NYMERIA_TIME_ALIGNMENT_VERSION = 4
"""Current schema version for Nymeria motion/text/video time alignment.

Responsibilities:
    Provide one shared version value used by exporters and skip-existing guards.
Preconditions:
    Bump this value whenever output timestamp semantics change.
Postconditions:
    Consumers can reject stale converted files by comparing this value.
"""
