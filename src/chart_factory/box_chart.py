"""
box_chart.py — Box Plot Generator

Subtypes:
  vertical   – Vertical box plots (one or multiple groups)
  horizontal – Horizontal box plots
  grouped    – Multiple series side-by-side
  notched    – Notched box plots (confidence interval on median)
"""

import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, apply_theme,
    get_axis_limits, build_standard_json, save_outputs,
)

CHART_TYPE = "box"
MAX_GROUPS = 8
N_POINTS   = 80   # samples per distribution (not saved, only statistics matter)

SUBTYPES = ["vertical", "horizontal", "grouped", "notched"]

DATASET_TEMPLATES = [
    {"theme": "Student Test Scores by Class", "x_label": "Class", "y_label": "Score",
     "groups": ["Class A", "Class B", "Class C", "Class D"],
     "value_range": (40, 100), "spread": 15},
    {"theme": "Salary Distribution by Department", "x_label": "Department", "y_label": "Annual Salary (K€)",
     "groups": ["R&D", "Sales", "HR", "IT", "Finance", "Operations"],
     "value_range": (30, 120), "spread": 20},
    {"theme": "Blood Pressure by Treatment Group", "x_label": "Treatment", "y_label": "Systolic BP (mmHg)",
     "groups": ["Placebo", "Drug A", "Drug B", "Drug C"],
     "value_range": (100, 180), "spread": 18},
    {"theme": "Response Time by Server Region", "x_label": "Region", "y_label": "Response Time (ms)",
     "groups": ["EU-West", "EU-East", "US-East", "US-West", "Asia"],
     "value_range": (20, 400), "spread": 60},
    {"theme": "Plant Height by Fertilizer Type", "x_label": "Fertilizer", "y_label": "Plant Height (cm)",
     "groups": ["Control", "Nitrogen", "Phosphorus", "Potassium", "NPK Mix"],
     "value_range": (15, 120), "spread": 22},
    {"theme": "House Price by Neighborhood", "x_label": "Neighborhood", "y_label": "Price (K€)",
     "groups": ["Downtown", "Suburbs", "Rural", "Waterfront", "Industrial"],
     "value_range": (80, 800), "spread": 120},
    {"theme": "Customer Wait Time by Channel", "x_label": "Channel", "y_label": "Wait Time (min)",
     "groups": ["Phone", "Chat", "Email", "In-Person", "App"],
     "value_range": (1, 45), "spread": 10},
    {"theme": "Air Quality Index by City", "x_label": "City", "y_label": "AQI",
     "groups": ["London", "Paris", "Berlin", "Rome", "Madrid", "Athens"],
     "value_range": (20, 180), "spread": 35},
]

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def _gen_distribution(center, spread, n=N_POINTS):
    """Generate a skewed distribution and return sorted samples."""
    base = np.random.normal(center, spread * 0.5, n)
    base = np.clip(base, center - spread * 1.8, center + spread * 1.8)
    return sorted(base.tolist())


def _box_stats(data: list) -> dict:
    """Compute box plot statistics from a list of values."""
    arr = np.array(data)
    return {
        "min":    round(float(np.min(arr)), 4),
        "q1":     round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "q3":     round(float(np.percentile(arr, 75)), 4),
        "max":    round(float(np.max(arr)), 4),
    }

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_vertical(tmpl, theme, idx):
    groups = tmpl["groups"][:MAX_GROUPS]
    lo, hi = tmpl["value_range"]
    centers = [random.uniform(lo + 10, hi - 10) for _ in groups]
    dists   = [_gen_distribution(c, tmpl["spread"]) for c in centers]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    bp = ax.boxplot(dists, labels=groups, patch_artist=True, notch=False,
                    medianprops={"color": theme["title"], "linewidth": 2})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(theme["palette"][i % len(theme["palette"])])
        patch.set_alpha(0.75)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": g, "y_value": _box_stats(d)}
          for g, d in zip(groups, dists)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_horizontal(tmpl, theme, idx):
    groups = tmpl["groups"][:MAX_GROUPS]
    lo, hi = tmpl["value_range"]
    centers = [random.uniform(lo + 10, hi - 10) for _ in groups]
    dists   = [_gen_distribution(c, tmpl["spread"]) for c in centers]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    bp = ax.boxplot(dists, labels=groups, patch_artist=True, notch=False, vert=False,
                    medianprops={"color": theme["title"], "linewidth": 2})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(theme["palette"][i % len(theme["palette"])])
        patch.set_alpha(0.75)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["y_label"], fontsize=11)
    ax.set_ylabel(tmpl["x_label"], fontsize=11)
    plt.tight_layout()
    # vert=False: x is value axis, y is categorical
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y", is_categorical=True)
    dp = [{"series_name": "Main", "x_value": _box_stats(d), "y_value": g}
          for g, d in zip(groups, dists)]
    return fig, build_standard_json(tmpl["theme"], tmpl["y_label"], tmpl["x_label"], x_lim, y_lim, dp)


def _render_grouped(tmpl, theme, idx):
    groups  = tmpl["groups"][:4]
    series  = ["Series 1", "Series 2", "Series 3"][:random.randint(2, 3)]
    lo, hi  = tmpl["value_range"]
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    n_ser   = len(series)
    n_grp   = len(groups)
    positions_all = []
    series_dists  = {s: [] for s in series}
    for gi, g in enumerate(groups):
        for si, s in enumerate(series):
            center = random.uniform(lo + 10, hi - 10)
            dist   = _gen_distribution(center, tmpl["spread"])
            series_dists[s].append((g, dist))

    for si, s in enumerate(series):
        positions = [gi * (n_ser + 1) + si + 1 for gi in range(n_grp)]
        dists     = [d for _, d in series_dists[s]]
        bp = ax.boxplot(dists, positions=positions, widths=0.7, patch_artist=True, notch=False,
                        medianprops={"color": theme["title"], "linewidth": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor(theme["palette"][si % len(theme["palette"])])
            patch.set_alpha(0.75)

    tick_positions = [gi * (n_ser + 1) + n_ser / 2 + 0.5 for gi in range(n_grp)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1,
               color=theme["palette"][si % len(theme["palette"])], alpha=0.75)
               for si, s in enumerate(series)]
    ax.legend(handles, series, facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": s, "x_value": g, "y_value": _box_stats(d)}
          for s in series for g, d in series_dists[s]]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_notched(tmpl, theme, idx):
    groups  = tmpl["groups"][:MAX_GROUPS]
    lo, hi  = tmpl["value_range"]
    centers = [random.uniform(lo + 10, hi - 10) for _ in groups]
    dists   = [_gen_distribution(c, tmpl["spread"]) for c in centers]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    try:
        bp = ax.boxplot(dists, labels=groups, patch_artist=True, notch=True,
                        medianprops={"color": theme["title"], "linewidth": 2})
    except Exception:
        bp = ax.boxplot(dists, labels=groups, patch_artist=True, notch=False,
                        medianprops={"color": theme["title"], "linewidth": 2})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(theme["palette"][i % len(theme["palette"])])
        patch.set_alpha(0.75)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": g, "y_value": _box_stats(d)}
          for g, d in zip(groups, dists)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


RENDERERS = {
    "vertical":   _render_vertical,
    "horizontal": _render_horizontal,
    "grouped":    _render_grouped,
    "notched":    _render_notched,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 13 + 3)
        np.random.seed(i * 13 + 3)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
