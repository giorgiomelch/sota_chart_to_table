"""
Chart type detection, axis range extraction, and JSON → Mapping conversion.

Chart type detection priority
──────────────────────────────
1. Bubble  : presence of z_value / w_value keys in any data point.
             Orientation (bubble_x / bubble_y) determined by whether the
             y_axis metadata has null min/max (categorical).

2. Axis metadata (primary, requires x_axis / y_axis dicts in JSON):
   - x_axis present with null min/max  →  x is categorical  →  categorical_x
   - y_axis present with null min/max  →  y is categorical  →  categorical_y
   - Both axes have numeric ranges     →  scatter

3. categorical_axis field (legacy fallback when axis dicts are absent):
   "x" → categorical_x,  "y" → categorical_y,  None/"none" → scatter

4. Data-point inference (last resort):
   All x and y values numeric  →  scatter,  else  categorical_x
"""

from __future__ import annotations

import math
from typing import Any

from .types import AxisRanges, BubbleVal, Mapping, ScatterVal, StructuredVal


# ---------------------------------------------------------------------------
# Primitive conversion helpers
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _to_float(v: Any) -> float | None:
    """Convert v to float, handling Unicode scientific notation (e.g. 1.5×10⁵)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    if not isinstance(v, str):
        return None
    s = v.strip().translate(_SUPERSCRIPT_MAP)
    s = s.replace("×10", "e").replace("x10", "e")
    try:
        return float(s)
    except ValueError:
        return None


def _is_numeric(v: Any) -> bool:
    if isinstance(v, (ScatterVal, BubbleVal, StructuredVal)):
        return False
    return _to_float(v) is not None


def _parse_value(raw: Any) -> Any:
    """
    Convert a raw JSON value to the appropriate Python type:
      dict with known structured keys → StructuredVal
      anything else                   → returned as-is
    """
    if isinstance(raw, dict):
        sv = StructuredVal.from_dict(raw, _to_float)
        if sv.is_valid():
            return sv
    return raw


def _has_bubble_fields(data: dict) -> bool:
    """True if any data point carries z_value or w_value."""
    return any(
        "z_value" in dp or "w_value" in dp
        for dp in data.get("data_points", [])
    )


# ---------------------------------------------------------------------------
# Chart type detection
# ---------------------------------------------------------------------------

def _detect_chart_type(data: dict) -> str:
    """
    Return one of: 'categorical_x', 'categorical_y', 'scatter',
                   'bubble_x', 'bubble_y'.

    See module docstring for the full detection priority.
    """
    # ── 1. Bubble detection ──────────────────────────────────────────────────
    if _has_bubble_fields(data):
        y_axis_data = data.get("y_axis")
        if y_axis_data is not None:
            y_categorical = (y_axis_data.get("min") is None
                             and y_axis_data.get("max") is None)
            return "bubble_y" if y_categorical else "bubble_x"
        # fallback to legacy field
        cat = data.get("categorical_axis")
        if cat is not None and str(cat).lower().strip() == "y":
            return "bubble_y"
        return "bubble_x"

    # ── 2. Axis metadata ─────────────────────────────────────────────────────
    x_axis_data = data.get("x_axis")
    y_axis_data = data.get("y_axis")

    if x_axis_data is not None or y_axis_data is not None:
        x_has_range = (x_axis_data is not None
                       and (x_axis_data.get("min") is not None
                            or x_axis_data.get("max") is not None))
        y_has_range = (y_axis_data is not None
                       and (y_axis_data.get("min") is not None
                            or y_axis_data.get("max") is not None))
        x_categorical = x_axis_data is not None and not x_has_range
        y_categorical = y_axis_data is not None and not y_has_range

        if x_categorical and y_has_range:
            return "categorical_x"
        if y_categorical and x_has_range:
            return "categorical_y"
        if x_has_range and y_has_range:
            # Both axes numeric, but structured y_values (dicts) mean x acts as
            # a numeric grouping key (dose-response, time series boxplot, etc.).
            # Treat as categorical_x/y based on which side carries the structure.
            dps = data.get("data_points", [])
            if dps:
                if any(isinstance(dp.get("y_value"), dict) for dp in dps):
                    return "categorical_x"
                if any(isinstance(dp.get("x_value"), dict) for dp in dps):
                    return "categorical_y"
            return "scatter"
        # ambiguous (both categorical, or only one axis present) → fall through

    # ── 3. Legacy categorical_axis field ─────────────────────────────────────
    cat = data.get("categorical_axis")
    if cat is not None:
        cat = str(cat).lower().strip()
        if cat == "none":
            return "scatter"
        if cat == "y":
            return "categorical_y"
        return "categorical_x"

    # ── 4. Infer from data point types ───────────────────────────────────────
    dps = data.get("data_points", [])
    if not dps:
        return "categorical_x"
    if all(_is_numeric(dp.get("x_value")) for dp in dps) and \
       all(_is_numeric(dp.get("y_value")) for dp in dps):
        return "scatter"
    return "categorical_x"


# ---------------------------------------------------------------------------
# Axis range extraction
# ---------------------------------------------------------------------------

def _extract_ranges(data: dict, chart_type: str) -> AxisRanges:
    """
    Build an AxisRanges from the axis metadata stored in a ground-truth JSON dict.

    For linear axes : range = |max - min|
    For log axes    : range = |log10(max) - log10(min)|

    Returns None for a given dimension if the axis dict is absent or its
    min / max values are null.
    """
    def _range(axis_key: str) -> tuple[float | None, bool]:
        info = data.get(axis_key)
        if not info:
            return None, False
        mn, mx = info.get("min"), info.get("max")
        is_log = bool(info.get("is_log", False))
        if mn is None or mx is None:
            return None, is_log
        mn, mx = float(mn), float(mx)
        if is_log:
            if mn <= 0 or mx <= 0:
                return None, is_log
            r = abs(math.log10(mx) - math.log10(mn))
        else:
            r = abs(mx - mn)
        return (r if r > 0 else None), is_log

    x_r, x_log = _range("x_axis")
    y_r, y_log = _range("y_axis")
    z_r, z_log = _range("z_axis")
    w_r, w_log = _range("w_axis")

    if chart_type == "categorical_x":
        return AxisRanges(val=y_r, val_log=y_log)
    if chart_type == "categorical_y":
        return AxisRanges(val=x_r, val_log=x_log)
    if chart_type == "scatter":
        return AxisRanges(x=x_r, x_log=x_log, y=y_r, y_log=y_log)
    if chart_type == "bubble_y":
        return AxisRanges(x=x_r, x_log=x_log, z=z_r, z_log=z_log, w=w_r, w_log=w_log)
    if chart_type == "bubble_x":
        return AxisRanges(x=y_r, x_log=y_log, z=z_r, z_log=z_log, w=w_r, w_log=w_log)
    # default
    return AxisRanges(val=y_r, val_log=y_log)


# ---------------------------------------------------------------------------
# JSON → list[Mapping]
# ---------------------------------------------------------------------------

def json_to_mappings(data: dict, transpose: bool = False,
                     chart_type: str | None = None) -> list[Mapping]:
    """
    Convert a chart JSON dict to a list of Mapping triples.

    categorical_x  →  (series_name, x_value_str, y_value_num)
    categorical_y  →  (series_name, y_value_str, x_value_num)
    scatter        →  (series_name, "__scatter__", ScatterVal(x, y))
    bubble_y       →  (series_name, y_value_str, BubbleVal(x, z, w))
    bubble_x       →  (series_name, x_value_str, BubbleVal(y, z, w))

    transpose=True swaps row ↔ col for categorical and bubble charts.
    chart_title is appended as a metadata mapping (never transposed).

    chart_type: if provided, overrides auto-detection (used so that the
    predicted JSON is interpreted using the ground-truth chart type).
    """
    if chart_type is None:
        chart_type = _detect_chart_type(data)

    mappings: list[Mapping] = []

    # --- metadata ---
    title = data.get("chart_title")
    if title is not None:
        mappings.append(Mapping(row="__meta__", col="chart_title", val=str(title)))

    # --- data points ---
    for dp in data.get("data_points", []):
        series = str(dp.get("series_name", ""))
        x_raw  = dp.get("x_value", "")
        y_raw  = dp.get("y_value", "")

        if chart_type == "scatter":
            xf = _to_float(x_raw) if x_raw != "" else float("nan")
            yf = _to_float(y_raw) if y_raw != "" else float("nan")
            if xf is None: xf = float("nan")
            if yf is None: yf = float("nan")
            mappings.append(Mapping(row=series, col="__scatter__",
                                    val=ScatterVal(xf, yf)))

        elif chart_type in ("bubble_x", "bubble_y"):
            z_raw = dp.get("z_value")
            w_raw = dp.get("w_value")
            if chart_type == "bubble_y":
                cat_key = str(y_raw)
                xf = _to_float(x_raw) if x_raw != "" else None
                if xf is None: xf = float("nan")
            else:  # bubble_x
                cat_key = str(x_raw)
                xf = _to_float(y_raw) if y_raw != "" else None
                if xf is None: xf = float("nan")
            bval = BubbleVal(x=xf, z=_to_float(z_raw), w=_to_float(w_raw))
            if transpose:
                mappings.append(Mapping(row=cat_key, col=series, val=bval))
            else:
                mappings.append(Mapping(row=series, col=cat_key, val=bval))

        elif chart_type == "categorical_y":
            cat_key = str(y_raw)
            num_val = _parse_value(x_raw)
            if transpose:
                mappings.append(Mapping(row=cat_key, col=series, val=num_val))
            else:
                mappings.append(Mapping(row=series, col=cat_key, val=num_val))

        else:  # categorical_x (default)
            cat_key = str(x_raw)
            num_val = _parse_value(y_raw)
            if transpose:
                mappings.append(Mapping(row=cat_key, col=series, val=num_val))
            else:
                mappings.append(Mapping(row=series, col=cat_key, val=num_val))

    return mappings
