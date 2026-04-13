"""
Relative Mapping Similarity (RMS) metric package.

Public API
----------
compute_rms(predicted, target, tau, theta)           → dict
compute_rms_detailed(predicted, target, tau, theta)  → dict
compute_rms_from_files(pred_path, gt_path, ...)      → dict

Types re-exported for downstream consumers (e.g. report generators):
    Mapping, ScatterVal, BubbleVal, StructuredVal, AxisRanges

Internal helpers exposed for testing:
    _detect_chart_type
"""

from .core import compute_rms, compute_rms_detailed, compute_rms_from_files
from .parser import _detect_chart_type
from .types import AxisRanges, BubbleVal, Mapping, ScatterVal, StructuredVal

__all__ = [
    # public functions
    "compute_rms",
    "compute_rms_detailed",
    "compute_rms_from_files",
    # types
    "Mapping",
    "ScatterVal",
    "BubbleVal",
    "StructuredVal",
    "AxisRanges",
    # internal (exposed for testing)
    "_detect_chart_type",
]
