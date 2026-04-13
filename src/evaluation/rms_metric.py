"""
Compatibility shim — re-exports from the rms package.

All logic has been moved to src/evaluation/rms/:
    types.py    — data classes
    distance.py — distance functions
    parser.py   — chart type detection, range extraction, JSON → Mapping
    core.py     — Hungarian matching, compute_rms variants
"""

from .rms import (  # noqa: F401
    AxisRanges,
    BubbleVal,
    Mapping,
    ScatterVal,
    StructuredVal,
    _detect_chart_type,
    compute_rms,
    compute_rms_detailed,
    compute_rms_from_files,
)
