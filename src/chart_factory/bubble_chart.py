"""
bubble_chart.py — Bubble Chart Generator

Subtypes:
  numeric_xy   – Both X and Y are numeric
  log_scale    – Logarithmic X or Y axis
  categorical_x – Categorical X axis with numeric Y and Z
"""

import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
warnings.filterwarnings = lambda *a, **kw: None
import warnings
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, rv, apply_theme,
    get_axis_limits, build_bubble_json, save_outputs,
)

CHART_TYPE = "bubble"
MAX_PTS    = 12

SUBTYPES = ["numeric_xy", "log_scale", "categorical_x"]

NUMERIC_TEMPLATES = [
    {
        "name": "GDP vs Life Expectancy",
        "title": "GDP per Capita vs Life Expectancy",
        "x_label": "GDP per Capita (USD)", "y_label": "Life Expectancy (years)",
        "z_label": "Population (M)", "z_range": (1, 1500),
        "w_meaning": "HDI Index", "w_range": (0.3, 1.0),
        "series": [
            {"name": "Africa",   "n": 6, "x": (500, 8000),    "y": (45, 72)},
            {"name": "Asia",     "n": 7, "x": (1000, 55000),  "y": (65, 85)},
            {"name": "Europe",   "n": 8, "x": (15000, 80000), "y": (75, 85)},
            {"name": "Americas", "n": 6, "x": (3000, 65000),  "y": (68, 82)},
        ],
    },
    {
        "name": "Tech Revenue vs R&D",
        "title": "Tech Companies: Revenue vs R&D Spend",
        "x_label": "Annual Revenue (B$)", "y_label": "R&D Expenditure (B$)",
        "z_label": "Market Cap (B$)", "z_range": (10, 5000),
        "w_meaning": "Employees (K)", "w_range": (1, 500),
        "series": [
            {"name": "Software",  "n": 5, "x": (5, 200),  "y": (1, 40)},
            {"name": "Hardware",  "n": 5, "x": (10, 300), "y": (2, 20)},
            {"name": "Cloud",     "n": 4, "x": (20, 400), "y": (5, 80)},
        ],
    },
    {
        "name": "CO₂ Emissions vs Renewables",
        "title": "CO₂ Emissions vs Renewable Energy Share",
        "x_label": "Renewable Energy Share (%)", "y_label": "CO₂ Emissions per Capita (t)",
        "z_label": "Total Energy (TWh)", "z_range": (5, 2000),
        "w_meaning": "Coal Dependency (%)", "w_range": (0, 80),
        "series": [
            {"name": "High Income", "n": 7, "x": (10, 80), "y": (3, 15)},
            {"name": "Mid Income",  "n": 8, "x": (5, 50),  "y": (2, 8)},
            {"name": "Low Income",  "n": 6, "x": (20, 90), "y": (0.2, 2)},
        ],
    },
    {
        "name": "Social Media Metrics",
        "title": "Social Media: Followers vs Engagement Rate",
        "x_label": "Followers (M)", "y_label": "Engagement Rate (%)",
        "z_label": "Avg Likes per Post (K)", "z_range": (0.5, 8000),
        "w_meaning": "Post Frequency/week", "w_range": (1, 21),
        "series": [
            {"name": "Instagram", "n": 6, "x": (0.5, 250), "y": (0.5, 8)},
            {"name": "TikTok",    "n": 5, "x": (0.1, 150), "y": (2, 15)},
            {"name": "YouTube",   "n": 5, "x": (0.5, 100), "y": (1, 6)},
        ],
    },
    {
        "name": "Research Output",
        "title": "Universities: Publications vs Citations",
        "x_label": "Annual Publications", "y_label": "Average Citations per Paper",
        "z_label": "Research Budget (M$)", "z_range": (10, 2000),
        "w_meaning": "International Collab (%)", "w_range": (20, 90),
        "series": [
            {"name": "STEM",       "n": 8, "x": (500, 5000),  "y": (10, 80)},
            {"name": "Medical",    "n": 7, "x": (200, 3000),  "y": (15, 120)},
            {"name": "Humanities", "n": 5, "x": (100, 1000),  "y": (5, 30)},
        ],
    },
]

CATEGORICAL_TEMPLATES = [
    {
        "name": "Product Category Performance",
        "title": "Sales Volume vs Margin by Product Category",
        "x_label": "Category", "y_label": "Profit Margin (%)",
        "z_label": "Units Sold (K)", "z_range": (10, 500),
        "w_meaning": "Return Rate (%)", "w_range": (1, 20),
        "categories": ["Electronics", "Clothing", "Food", "Books", "Toys", "Sports"],
        "y_range": (5, 55), "series": ["Main"],
    },
    {
        "name": "Department Efficiency",
        "title": "Department: Cost vs Efficiency Score",
        "x_label": "Department", "y_label": "Efficiency Score",
        "z_label": "Staff Count", "z_range": (5, 200),
        "w_meaning": "Satisfaction (%)", "w_range": (50, 100),
        "categories": ["Sales", "IT", "HR", "Finance", "Ops", "R&D", "Marketing"],
        "y_range": (40, 95), "series": ["Main"],
    },
    {
        "name": "Country Indicators",
        "title": "Country Metrics by Region",
        "x_label": "Region", "y_label": "Human Development Index",
        "z_label": "Population (M)", "z_range": (1, 500),
        "w_meaning": "GDP Growth (%)", "w_range": (-2, 10),
        "categories": ["N. America", "S. America", "Europe", "Africa", "Asia", "Oceania"],
        "y_range": (0.3, 1.0), "series": ["Main"],
    },
]

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_numeric_xy(tmpl, theme, idx):
    fig, ax = plt.subplots(figsize=(12, 8))
    apply_theme(fig, ax, theme)
    cmap = cm.get_cmap("viridis")
    all_z, all_w = [], []
    series_data = []
    for ser in tmpl["series"]:
        n = min(ser["n"], MAX_PTS)
        xs = rv(n, *ser["x"])
        ys = rv(n, *ser["y"])
        zs = rv(n, *tmpl["z_range"])
        ws = rv(n, *tmpl["w_range"])
        series_data.append((ser["name"], xs, ys, zs, ws))
        all_z.extend(zs)
        all_w.extend(ws)

    z_min, z_max = min(all_z), max(all_z)
    w_min, w_max = min(all_w), max(all_w)
    z_norm = mcolors.Normalize(vmin=z_min, vmax=z_max)

    for i, (name, xs, ys, zs, ws) in enumerate(series_data):
        sizes = [100 + 1800 * (z - z_min) / max(z_max - z_min, 1) for z in zs]
        color = theme["palette"][i % len(theme["palette"])]
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.65, edgecolors=theme["bg"],
                   linewidths=0.8, label=name, zorder=3)

    ax.set_title(tmpl["title"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    z_lim = {"min": round(z_min, 4), "max": round(z_max, 4), "is_log": False}
    w_lim = {"min": round(w_min, 4), "max": round(w_max, 4), "is_log": False}
    dp = [
        {"series_name": name, "x_value": x, "y_value": y, "z_value": z, "w_value": w}
        for name, xs, ys, zs, ws in series_data
        for x, y, z, w in zip(xs, ys, zs, ws)
    ]
    return fig, build_bubble_json(tmpl["title"], tmpl["x_label"], tmpl["y_label"],
                                   x_lim, y_lim, z_lim, w_lim, dp)


def _render_log_scale(tmpl, theme, idx):
    fig, ax = plt.subplots(figsize=(12, 8))
    apply_theme(fig, ax, theme)
    series_data = []
    all_z, all_w = [], []
    for ser in tmpl["series"]:
        n = min(ser["n"], MAX_PTS)
        xs = rv(n, *ser["x"])
        ys = rv(n, *ser["y"])
        zs = rv(n, *tmpl["z_range"])
        ws = rv(n, *tmpl["w_range"])
        series_data.append((ser["name"], xs, ys, zs, ws))
        all_z.extend(zs)
        all_w.extend(ws)

    z_min, z_max = min(all_z), max(all_z)
    w_min, w_max = min(all_w), max(all_w)

    for i, (name, xs, ys, zs, ws) in enumerate(series_data):
        sizes = [80 + 1500 * (z - z_min) / max(z_max - z_min, 1) for z in zs]
        color = theme["palette"][i % len(theme["palette"])]
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.65, edgecolors=theme["bg"],
                   linewidths=0.8, label=name, zorder=3)

    ax.set_xscale("log")
    ax.set_title(tmpl["title"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")   # is_log=True auto-detected
    y_lim = get_axis_limits(ax, "y")
    z_lim = {"min": round(z_min, 4), "max": round(z_max, 4), "is_log": False}
    w_lim = {"min": round(w_min, 4), "max": round(w_max, 4), "is_log": False}
    dp = [
        {"series_name": name, "x_value": x, "y_value": y, "z_value": z, "w_value": w}
        for name, xs, ys, zs, ws in series_data
        for x, y, z, w in zip(xs, ys, zs, ws)
    ]
    return fig, build_bubble_json(tmpl["title"], tmpl["x_label"], tmpl["y_label"],
                                   x_lim, y_lim, z_lim, w_lim, dp)


def _render_categorical_x(tmpl_cat, theme, idx):
    tmpl = tmpl_cat
    cats = tmpl["categories"]
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_theme(fig, ax, theme)
    ys   = rv(len(cats), *tmpl["y_range"])
    zs   = rv(len(cats), *tmpl["z_range"])
    ws   = rv(len(cats), *tmpl["w_range"])
    z_min, z_max = min(zs), max(zs)
    w_min, w_max = min(ws), max(ws)
    sizes = [80 + 1800 * (z - z_min) / max(z_max - z_min, 1) for z in zs]
    scatter = ax.scatter(cats, ys, s=sizes, c=ws, cmap="viridis",
                         alpha=0.75, edgecolors=theme["bg"], linewidths=0.8, zorder=3)
    plt.colorbar(scatter, ax=ax, label=tmpl["w_meaning"])
    ax.set_title(tmpl["title"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    z_lim = {"min": round(z_min, 4), "max": round(z_max, 4), "is_log": False}
    w_lim = {"min": round(w_min, 4), "max": round(w_max, 4), "is_log": False}
    dp = [
        {"series_name": "Main", "x_value": c, "y_value": y, "z_value": z, "w_value": w}
        for c, y, z, w in zip(cats, ys, zs, ws)
    ]
    return fig, build_bubble_json(tmpl["title"], tmpl["x_label"], tmpl["y_label"],
                                   x_lim, y_lim, z_lim, w_lim, dp)


# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 11 + 2)
        subtype = random.choice(SUBTYPES)
        theme   = random.choice(CHART_THEMES)
        if subtype == "categorical_x":
            tmpl = random.choice(CATEGORICAL_TEMPLATES)
            fig, js = _render_categorical_x(tmpl, theme, i)
        elif subtype == "log_scale":
            tmpl = random.choice(NUMERIC_TEMPLATES)
            fig, js = _render_log_scale(tmpl, theme, i)
        else:
            tmpl = random.choice(NUMERIC_TEMPLATES)
            fig, js = _render_numeric_xy(tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
