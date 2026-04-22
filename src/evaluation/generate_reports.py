"""
HTML report generator for the sota_chart_to_table benchmark.

Uses RMS (rms_metric.py) instead of the old table_datapoints_precision_recall.
Prediction loading and normalisation are shared with evaluate.py.

Layout: 3-column grid. Each model block has a unique accent colour.
Tables scroll horizontally to avoid text overlap.
Display values are always raw (un-normalised); metric values use normalised data.
"""

import base64
import copy
import json
from pathlib import Path
from typing import Any

from src.evaluation.evaluate import (
    estrai_basi,
    get_available_models,
    load_prediction,
    normalizza_valori,
)
from src.evaluation.rms_metric import (
    BubbleVal,
    Mapping,
    ScatterVal,
    StructuredVal,
    compute_rms_detailed,
)

# --- CONFIGURAZIONE ---
from src.config import PREDICTIONS_DIR as PREDICTIONS_ROOT
from src.config import GROUNDTRUTH_DIR as GROUNDTRUTH_ROOT
from src.config import IMAGES_DIR as IMAGES_ROOT
from src.config import REPORTS_DIR as REPORTS_ROOT

_MODEL_PALETTE = [
    "#00d4ff",  # cyan
    "#82e0aa",  # green
    "#f0b27a",  # orange
    "#c39bd3",  # purple
    "#f9e79f",  # yellow
    "#85c1e9",  # light blue
    "#f1948a",  # salmon
    "#73c6b6",  # teal
    "#abebc6",  # mint
    "#d7bde2",  # lavender
]


def _model_color(model_name: str, model_names: list[str]) -> str:
    idx = model_names.index(model_name) if model_name in model_names else 0
    return _MODEL_PALETTE[idx % len(_MODEL_PALETTE)]


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

_STRUCTURED_FIELD_ORDER = ("min", "q1", "median", "q3", "max")


def _fmt(v: Any, max_len: int = 28) -> str:
    if isinstance(v, ScatterVal):
        return f"({v.x:.4g}, {v.y:.4g})"
    if isinstance(v, BubbleVal):
        parts = [f"x={v.x:.4g}"]
        if v.z is not None:
            parts.append(f"z={v.z:.4g}")
        if v.w is not None:
            parts.append(f"w={v.w:.4g}")
        return " ".join(parts)
    if isinstance(v, StructuredVal):
        return " | ".join(f"{k}={fv:.3g}" for k, fv in v.fields.items())
    if isinstance(v, dict):
        # Raw structured value (box/errorpoint) stored as dict in the JSON
        parts = []
        for k in _STRUCTURED_FIELD_ORDER:
            if k in v and v[k] is not None:
                try:
                    parts.append(f"{k}={float(v[k]):.3g}")
                except (TypeError, ValueError):
                    pass
        if parts:
            return " | ".join(parts)
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _sim_color(sim: float) -> str:
    if sim >= 0.8:
        return "#82e0aa"
    if sim >= 0.5:
        return "#f9e79f"
    if sim >= 0.2:
        return "#f0b27a"
    return "#e74c3c"


# ---------------------------------------------------------------------------
# Raw-value lookup helpers
# ---------------------------------------------------------------------------

def _build_raw_lookup(raw_data: dict | None, chart_type: str) -> dict:
    """
    Build a (series_name, col_key) → raw_value dict from un-normalised data_points.

    col_key mirrors how json_to_mappings sets Mapping.col:
      categorical_x / bubble_x / scatter : str(x_value)
      categorical_y / bubble_y           : str(y_value)

    For scatter the col is always "__scatter__", so lookup by series+col is not
    unique. In that case we store the first occurrence (good enough for display).
    """
    if not raw_data:
        return {}
    lookup: dict = {}
    for dp in raw_data.get("data_points", []):
        series = str(dp.get("series_name", ""))
        if chart_type == "bubble_y":
            col = str(dp.get("y_value", ""))
            # 3-tuple: (primary, z, w) — distinguished from scatter's 2-tuple
            val = (dp.get("x_value"), dp.get("z_value"), dp.get("w_value"))
        elif chart_type == "bubble_x":
            col = str(dp.get("x_value", ""))
            val = (dp.get("y_value"), dp.get("z_value"), dp.get("w_value"))
        elif chart_type == "categorical_y":
            col = str(dp.get("y_value", ""))
            val = dp.get("x_value", "")
        elif chart_type == "scatter":
            col = "__scatter__"
            # Store a 2-tuple so _fmt_raw can build a ScatterVal
            val = (dp.get("x_value"), dp.get("y_value"))
        else:  # categorical_x
            col = str(dp.get("x_value", ""))
            val = dp.get("y_value", "")
        key = (series, col)
        if key not in lookup:   # keep first match for duplicate keys
            lookup[key] = val
    return lookup


def _fmt_raw(m: Mapping, lookup: dict) -> str:
    """
    Return a display string for m.val using the raw lookup when available.
    Falls back to _fmt(m.val) if the key is not found.
    """
    # Scatter Mappings all share col="__scatter__"; the lookup cannot
    # distinguish between individual points — use the parsed value directly.
    if m.col == "__scatter__":
        return _fmt(m.val)
    raw = lookup.get((m.row, m.col))
    if raw is None:
        return _fmt(m.val)
    if isinstance(raw, tuple):
        if len(raw) == 2:
            # Scatter: (x_raw, y_raw)
            x, y = raw
            try:
                return f"({float(x):.4g}, {float(y):.4g})"
            except (TypeError, ValueError):
                return f"({x}, {y})"
        # Bubble: (primary_raw, z_raw, w_raw)
        primary, z, w = raw
        parts: list[str] = []
        try:
            parts.append(f"{float(primary):.4g}")
        except (TypeError, ValueError):
            parts.append(str(primary))
        if z is not None:
            try:
                parts.append(f"z={float(z):.4g}")
            except (TypeError, ValueError):
                parts.append(f"z={z}")
        if w is not None:
            try:
                parts.append(f"w={float(w):.4g}")
            except (TypeError, ValueError):
                parts.append(f"w={w}")
        return " ".join(parts)
    return _fmt(raw)


# ---------------------------------------------------------------------------
# Table styles
# ---------------------------------------------------------------------------

_TH = (
    "background:#2a2a2a; color:#999; padding:3px 8px; "
    "border-bottom:1px solid #555; text-align:left; white-space:nowrap;"
)
_TD = "padding:2px 8px; border-bottom:1px solid #222; vertical-align:top; white-space:nowrap;"
_SCROLL_WRAP = "overflow-x:auto; max-width:100%;"


# ---------------------------------------------------------------------------
# GT compact table — always uses raw (un-normalised) data
# ---------------------------------------------------------------------------

def _gt_table(gt_data_raw: dict) -> str:
    dps = gt_data_raw.get("data_points", [])
    title = gt_data_raw.get("chart_title", "")
    rows = ""
    if title:
        rows += (
            f"<tr><td colspan='3' style='{_TD} color:#aaa; font-style:italic;'>"
            f"title: {title}</td></tr>"
        )
    for dp in dps:
        series = str(dp.get("series_name", ""))
        x = _fmt(dp.get("x_value", ""))
        y = _fmt(dp.get("y_value", ""))
        rows += (
            f"<tr>"
            f"<td style='{_TD} color:#7fb3d3;'>{series}</td>"
            f"<td style='{_TD}'>{x}</td>"
            f"<td style='{_TD}'>{y}</td>"
            f"</tr>"
        )
    table = (
        f"<table style='border-collapse:collapse; font-size:10px;'>"
        f"<tr>"
        f"<th style='{_TH}'>series</th>"
        f"<th style='{_TH}'>x</th>"
        f"<th style='{_TH}'>y</th>"
        f"</tr>"
        f"{rows}"
        f"</table>"
    )
    return f"<div style='{_SCROLL_WRAP}'>{table}</div>"


# ---------------------------------------------------------------------------
# Match table — display uses raw lookups, Sim comes from normalised metric
# ---------------------------------------------------------------------------

def _match_table(
    detail: dict,
    gt_raw_lookup: dict,
    pred_raw_lookup: dict,
) -> str:
    def _cells(m: Mapping, lookup: dict, color: str = "#e0e0e0") -> str:
        return (
            f"<td style='{_TD} color:#7fb3d3;'>{m.row}</td>"
            f"<td style='{_TD}'>{m.col}</td>"
            f"<td style='{_TD} color:{color};'>{_fmt_raw(m, lookup)}</td>"
        )

    def _empty() -> str:
        return f"<td colspan='3' style='{_TD} color:#444; text-align:center;'>—</td>"

    rows = ""

    for entry in detail["pairs"]:
        sim = entry["similarity"]
        c = _sim_color(sim)
        rows += (
            f"<tr>"
            f"{_cells(entry['gt'], gt_raw_lookup)}"
            f"{_cells(entry['pred'], pred_raw_lookup, color=c)}"
            f"<td style='{_TD} color:{c}; font-weight:bold;'>{sim:.2f}</td>"
            f"</tr>"
        )

    for m in detail["unmatched_gt"]:
        rows += (
            f"<tr>"
            f"{_cells(m, gt_raw_lookup)}"
            f"{_empty()}"
            f"<td style='{_TD} color:#e74c3c;'>0.00</td>"
            f"</tr>"
        )

    for m in detail["unmatched_pred"]:
        rows += (
            f"<tr>"
            f"{_empty()}"
            f"{_cells(m, pred_raw_lookup, color='#e74c3c')}"
            f"<td style='{_TD} color:#e74c3c;'>0.00</td>"
            f"</tr>"
        )

    if not rows:
        return "<div style='color:#888; font-size:10px;'>Nessun dato</div>"

    table = (
        f"<table style='border-collapse:collapse; font-size:10px;'>"
        f"<tr>"
        f"<th style='{_TH}' colspan='3'>GT</th>"
        f"<th style='{_TH}' colspan='3'>Pred</th>"
        f"<th style='{_TH}'>Sim</th>"
        f"</tr>"
        f"<tr>"
        f"<th style='{_TH}'>series</th><th style='{_TH}'>key</th><th style='{_TH}'>val</th>"
        f"<th style='{_TH}'>series</th><th style='{_TH}'>key</th><th style='{_TH}'>val</th>"
        f"<th style='{_TH}'></th>"
        f"</tr>"
        f"{rows}"
        f"</table>"
    )
    return f"<div style='{_SCROLL_WRAP}'>{table}</div>"


# ---------------------------------------------------------------------------
# Metrics bar
# ---------------------------------------------------------------------------

def _metrics_bar(result: dict) -> str:
    f1  = result["f1"]
    pre = result["precision"]
    rec = result["recall"]
    c   = _sim_color(f1)
    return (
        f"<div style='color:{c}; font-size:10px; background:#0a0a0a; "
        f"padding:3px 6px; margin-bottom:6px; border-radius:3px; white-space:nowrap;'>"
        f"F1&nbsp;{f1*100:.1f} &nbsp;|&nbsp; P&nbsp;{pre*100:.1f} &nbsp;|&nbsp; "
        f"R&nbsp;{rec*100:.1f} &nbsp;|&nbsp; "
        f"<span style='color:#777'>{result['chart_type']} / {result['orientation']}</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Block wrapper — 3-column grid
# ---------------------------------------------------------------------------

_BLOCK_FLEX = "flex: 0 0 calc(33.33% - 12px); min-width: 300px; box-sizing: border-box;"


def _block(title: str, body_html: str, accent: str = "#888", bg: str = "#1e1e1e") -> str:
    return (
        f"<div style='{_BLOCK_FLEX} border:1px solid #444; border-top:2px solid {accent}; "
        f"padding:12px; border-radius:6px; background:{bg}; color:#e0e0e0; "
        f"font-family:monospace; font-size:12px; overflow:hidden;'>"
        f"<div style='margin-bottom:8px; color:{accent}; font-size:10px; font-weight:bold; "
        f"border-bottom:1px solid #333; padding-bottom:4px; text-transform:uppercase; "
        f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{title}</div>"
        f"{body_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Per-image section
# ---------------------------------------------------------------------------

def genera_sezione_immagine(
    img_path: Path,
    dataset_type: str,
    chart_class: str,
    model_names: list[str],
) -> str:
    rel_path = img_path.relative_to(IMAGES_ROOT / dataset_type / chart_class)
    gt_json_path = GROUNDTRUTH_ROOT / dataset_type / chart_class / rel_path.with_suffix(".json")

    with open(img_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode("utf-8")
    ext = img_path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    # Load GT: raw for display, normalised for metrics
    gt_data_raw: dict | None = None
    gt_data_norm: dict | None = None
    basi_gt: dict = {}

    if gt_json_path.exists():
        with open(gt_json_path, "r", encoding="utf-8") as fh:
            gt_data_raw = json.load(fh)
        basi_gt = estrai_basi(gt_data_raw)
        gt_data_norm = normalizza_valori(copy.deepcopy(gt_data_raw), basi_gt)

    blocks: list[str] = []

    # Image block
    blocks.append(
        f"<div style='{_BLOCK_FLEX} border:1px solid #444; border-top:2px solid #555; "
        f"padding:10px; border-radius:6px; background:#2a2a2a; overflow:hidden;'>"
        f"<div style='margin-bottom:6px; color:#777; font-size:10px; "
        f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{img_path.name}</div>"
        f"<img src='data:image/{mime};base64,{img_b64}' "
        f"style='width:100%; height:auto; border-radius:4px; display:block;'>"
        f"</div>"
    )

    # GT block — raw values
    gt_body = _gt_table(gt_data_raw) if gt_data_raw else "<div style='color:#888'>GT Assente</div>"
    blocks.append(_block("Ground Truth", gt_body, accent="#5dade2", bg="#1a2535"))

    # Model blocks
    for model in model_names:
        accent = _model_color(model, model_names)
        pred_path = (
            PREDICTIONS_ROOT / model / dataset_type / chart_class / rel_path.with_suffix(".json")
        )

        # Raw prediction (no base subtraction) for display only
        pred_data_raw = load_prediction(pred_path, {})
        # Normalised prediction for metric computation
        pred_data_norm = load_prediction(pred_path, basi_gt)

        if pred_data_norm is None:
            body = "<div style='color:#888; font-size:10px;'>Predizione assente</div>"
            blocks.append(_block(model, body, accent=accent))
            continue

        if gt_data_norm is None:
            body = "<div style='color:#888; font-size:10px;'>GT assente</div>"
            blocks.append(_block(model, body, accent=accent))
            continue

        try:
            detail = compute_rms_detailed(pred_data_norm, gt_data_norm)
            chart_type = detail["chart_type"]
            gt_raw_lookup   = _build_raw_lookup(gt_data_raw,   chart_type)
            pred_raw_lookup = _build_raw_lookup(pred_data_raw, chart_type)
            body = _metrics_bar(detail) + _match_table(detail, gt_raw_lookup, pred_raw_lookup)
        except Exception as exc:
            body = f"<div style='color:#e74c3c; font-size:10px;'>Metric Error: {exc}</div>"

        blocks.append(_block(model, body, accent=accent))

    return (
        f"<div style='margin-bottom:50px; padding-bottom:20px; border-bottom:2px dashed #333;'>"
        f"<h3 style='color:#bbb; font-size:13px; margin-bottom:12px; font-family:monospace;'>"
        f"{img_path.name}</h3>"
        f"<div style='display:flex; flex-wrap:wrap; gap:16px;'>"
        f"{''.join(blocks)}"
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_reports() -> None:
    model_names = get_available_models()

    legend_items = "".join(
        f"<span style='color:{_model_color(m, model_names)}; margin-right:16px;'>"
        f"&#9632; {m}</span>"
        for m in model_names
    )

    for dataset_type in ["arXiv", "PMCharts", "synthetic"]:
        base_img_dir = IMAGES_ROOT / dataset_type
        if not base_img_dir.exists():
            continue

        for chart_class_dir in sorted(base_img_dir.iterdir()):
            if not chart_class_dir.is_dir():
                continue

            chart_class = chart_class_dir.name
            images = sorted(
                (f for f in chart_class_dir.rglob("*")
                 if f.suffix.lower() in (".jpg", ".png", ".jpeg")),
                key=lambda x: x.name.lower(),
            )
            if not images:
                continue

            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>"
                "*, *::before, *::after { box-sizing: border-box; }"
                "body { background:#111; color:#ddd; padding:28px; "
                "       font-family:sans-serif; margin:0; }"
                "h2 { color:#00d4ff; border-bottom:2px solid #00d4ff; padding-bottom:8px; }"
                "</style></head><body>"
                f"<h2>Report Benchmark: {dataset_type.upper()} – {chart_class}</h2>"
                f"<div style='margin-bottom:24px; font-size:11px; font-family:monospace;'>"
                f"{legend_items}"
                f"<span style='color:#555; margin-left:12px;'>RMS tau=0.5 &theta;=0.1</span>"
                f"</div>"
            )

            for img_path in images:
                html += genera_sezione_immagine(
                    img_path, dataset_type, chart_class, model_names
                )

            html += "</body></html>"

            output_file = REPORTS_ROOT / dataset_type / f"report_{chart_class}.html"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html, encoding="utf-8")
            print(f"Creato report: {output_file}")


if __name__ == "__main__":
    generate_reports()
