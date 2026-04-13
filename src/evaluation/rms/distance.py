"""
Distance and similarity functions for the RMS metric.

All numeric distances are range-normalised using axis metadata from the
ground-truth JSON.  Axis ranges must always be provided; this module raises
ValueError if a numeric comparison is attempted without a valid range.
"""

from __future__ import annotations

import math
from typing import Any

from .types import AxisRanges, BubbleVal, Mapping, ScatterVal, StructuredVal


# ---------------------------------------------------------------------------
# String distance
# ---------------------------------------------------------------------------

def normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance in [0, 1]."""
    s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
    if s1 == s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[m] / max(n, m)


def nl_tau(s1: str, s2: str, tau: float) -> float:
    """Normalized Levenshtein clipped at tau."""
    if tau <= 0:
        return float(normalized_levenshtein(s1, s2) > 0)
    return min(1.0, normalized_levenshtein(s1, s2) / tau)


# ---------------------------------------------------------------------------
# Scalar numeric distance
# ---------------------------------------------------------------------------

def d_theta(p: Any, t: Any, theta: float,
            val_range: float | None = None, is_log: bool = False) -> float:
    """
    Range-normalised numeric distance clipped at theta; NL distance for strings.

    D_linear = |p - t| / val_range
    D_log    = |log10(p) - log10(t)| / val_range

    Raises ValueError if val_range is None or ≤ 0 for a numeric comparison.
    Ensure axis metadata (x_axis / y_axis with non-null min and max) is
    present in the ground-truth JSON so _extract_ranges can compute a range.
    """
    from .parser import _is_numeric, _to_float  # local import to avoid circular dep

    if not (_is_numeric(p) and _is_numeric(t)):
        return normalized_levenshtein(str(p), str(t))

    pf, tf = _to_float(p), _to_float(t)
    if pf is None or tf is None or math.isnan(pf) or math.isnan(tf):
        return 1.0

    if val_range is None or val_range <= 0:
        raise ValueError(
            f"val_range must be a positive number, got {val_range!r}. "
            "Ensure axis metadata (x_axis / y_axis with non-null min and max) "
            "is present in the ground-truth JSON."
        )

    if is_log:
        if pf <= 0 or tf <= 0:
            return 1.0
        raw = abs(math.log10(pf) - math.log10(tf)) / val_range
    else:
        raw = abs(pf - tf) / val_range

    return 1.0 if raw > theta else raw


# ---------------------------------------------------------------------------
# Compound value distances
# ---------------------------------------------------------------------------

def d_theta_scatter(p: ScatterVal, t: ScatterVal, theta: float,
                    ranges: AxisRanges | None = None) -> float:
    """Average distance on both axes, using per-axis ranges and log flags."""
    x_range = ranges.x     if ranges else None
    x_log   = ranges.x_log if ranges else False
    y_range = ranges.y     if ranges else None
    y_log   = ranges.y_log if ranges else False
    return (0.5 * d_theta(p.x, t.x, theta, x_range, x_log)
          + 0.5 * d_theta(p.y, t.y, theta, y_range, y_log))


def d_theta_structured(p: StructuredVal, t: StructuredVal, theta: float,
                       ranges: AxisRanges | None = None) -> float:
    """
    Distance between two structured values (errorbar / boxplot).

    All fields share the same value axis range (val / val_log).
    A field present in one but not the other contributes distance = 1.0.
    """
    all_fields = set(p.fields) | set(t.fields)
    if not all_fields:
        return 0.0
    val_range = ranges.val     if ranges else None
    val_log   = ranges.val_log if ranges else False
    total = sum(
        d_theta(p.fields[f], t.fields[f], theta, val_range, val_log)
        if f in p.fields and f in t.fields
        else 1.0
        for f in all_fields
    )
    return total / len(all_fields)


def d_theta_bubble(p: BubbleVal, t: BubbleVal, theta: float,
                   ranges: AxisRanges | None = None) -> float:
    """
    Distance between two bubble values using per-dimension ranges and log flags.

    x uses ranges.x / x_log, z uses ranges.z / z_log, w uses ranges.w / w_log.
    If one side is None and the other is not → distance 1.0.
    Dimensions that are None in both sides are excluded.
    """
    x_range = ranges.x     if ranges else None
    x_log   = ranges.x_log if ranges else False
    z_range = ranges.z     if ranges else None
    z_log   = ranges.z_log if ranges else False
    w_range = ranges.w     if ranges else None
    w_log   = ranges.w_log if ranges else False

    dims: list[float] = [d_theta(p.x, t.x, theta, x_range, x_log)]

    for pv, tv, vr, vl in ((p.z, t.z, z_range, z_log), (p.w, t.w, w_range, w_log)):
        if pv is None and tv is None:
            continue
        elif pv is None or tv is None:
            dims.append(1.0)
        else:
            dims.append(d_theta(pv, tv, theta, vr, vl))

    return sum(dims) / len(dims)


# ---------------------------------------------------------------------------
# Entry-level similarity  (1 - key_dist) × (1 - val_dist)
# ---------------------------------------------------------------------------

def entry_similarity(p: Mapping, t: Mapping, tau: float, theta: float,
                     ranges: AxisRanges | None = None) -> float:
    """
    Similarity for a matched pair of mappings.

    Value distance dispatch:
      ScatterVal    → d_theta_scatter
      BubbleVal     → d_theta_bubble
      StructuredVal → d_theta_structured
      scalar / str  → d_theta (range-normalised numeric or NL string)
    """
    from .parser import _is_numeric, _to_float  # local import

    if isinstance(p.val, ScatterVal) and isinstance(t.val, ScatterVal):
        key_sim  = 1.0 - nl_tau(p.row, t.row, tau)
        val_dist = d_theta_scatter(p.val, t.val, theta, ranges)

    elif isinstance(p.val, BubbleVal) or isinstance(t.val, BubbleVal):
        key_sim = 1.0 - nl_tau(p.row + p.col, t.row + t.col, tau)
        pv = p.val if isinstance(p.val, BubbleVal) else BubbleVal(
            x=_to_float(p.val) if _is_numeric(p.val) else float("nan"))
        tv = t.val if isinstance(t.val, BubbleVal) else BubbleVal(
            x=_to_float(t.val) if _is_numeric(t.val) else float("nan"))
        val_dist = d_theta_bubble(pv, tv, theta, ranges)

    elif isinstance(p.val, StructuredVal) or isinstance(t.val, StructuredVal):
        key_sim = 1.0 - nl_tau(p.row + p.col, t.row + t.col, tau)
        pv = p.val if isinstance(p.val, StructuredVal) else StructuredVal(
            {"median": _to_float(p.val)} if _is_numeric(p.val) else {})
        tv = t.val if isinstance(t.val, StructuredVal) else StructuredVal(
            {"median": _to_float(t.val)} if _is_numeric(t.val) else {})
        val_dist = d_theta_structured(pv, tv, theta, ranges)

    else:
        val_range = ranges.val     if ranges else None
        val_log   = ranges.val_log if ranges else False
        key_sim  = 1.0 - nl_tau(p.row + p.col, t.row + t.col, tau)
        val_dist = d_theta(p.val, t.val, theta, val_range, val_log)

    return key_sim * (1.0 - val_dist)
