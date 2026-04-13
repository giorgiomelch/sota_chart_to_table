"""
Data structures shared across all RMS modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScatterVal:
    """Both coordinates of a scatter point."""
    x: float
    y: float


@dataclass
class AxisRanges:
    """
    Per-axis effective numeric ranges extracted from ground-truth axis metadata.

    Linear:      D(p, t) = |p - t| / range          where range = max - min
    Logarithmic: D(p, t) = |log10(p) - log10(t)| / range  where range = log10(max) - log10(min)

    Fields:
        val / val_log  — value axis (categorical and structured charts)
        x   / x_log   — x axis (scatter, bubble numeric position)
        y   / y_log   — y axis (scatter y)
        z   / z_log   — bubble size axis
        w   / w_log   — bubble colour/weight axis
    """
    val:     float | None = None
    val_log: bool         = False
    x:       float | None = None
    x_log:   bool         = False
    y:       float | None = None
    y_log:   bool         = False
    z:       float | None = None
    z_log:   bool         = False
    w:       float | None = None
    w_log:   bool         = False


@dataclass
class BubbleVal:
    """
    Numeric dimensions of a bubble chart point.

    Mandatory : x  (numeric axis position)
    Optional  : z  (bubble size)    — None when absent or null in JSON
                w  (bubble colour)  — None when absent or null in JSON

    The categorical axis value becomes the mapping key and is NOT stored here.
    If a dimension is present in one point but None in the other, distance = 1.0.
    """
    x: float
    z: float | None = None
    w: float | None = None


@dataclass
class StructuredVal:
    """
    Structured numeric value (errorbar or boxplot).

    Errorbar fields : min, median, max
    Boxplot fields  : min, q1, median, q3, max

    Only fields present (not None) in at least one side participate in the distance.
    """
    fields: dict

    FIELD_ORDER = ("min", "q1", "median", "q3", "max")

    @classmethod
    def from_dict(cls, d: dict, to_float_fn) -> "StructuredVal":
        known = {}
        for k, v in d.items():
            if k in cls.FIELD_ORDER and v is not None:
                parsed = to_float_fn(v)
                if parsed is not None:
                    known[k] = parsed
        return cls(fields=known)

    def is_valid(self) -> bool:
        return len(self.fields) > 0


@dataclass
class Mapping:
    """A single (row_header, col_header, value) triple."""
    row: str
    col: str
    val: Any  # float | str | ScatterVal | BubbleVal | StructuredVal
