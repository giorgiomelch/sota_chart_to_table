"""
Line Chart Generator
Genera grafici a linee diversificati con immagini PNG e JSON corrispondenti.
Massimo 15 punti per serie.

Sottotipologie supportate:
  1.  simple_line          – Linea singola classica
  2.  multi_line           – Multi-serie su stesso asse
  3.  smooth_line          – Linea con interpolazione smooth (spline)
  4.  area_line            – Area riempita sotto la linea
  5.  stacked_area         – Aree impilate multi-serie
  6.  step_line            – Linea a gradini (step)
  7.  dot_line             – Linea con marker prominenti (lollipop-style)
  8.  dual_axis            – Due assi Y (scala differente)
  9.  annotated_line       – Linea con annotazioni su punti chiave
 10.  band_line            – Linea con banda di confidenza (±σ)
 11.  slope_chart          – Slope chart (solo 2 punti temporali, multi-serie)
 12.  sparkline_grid       – Griglia di sparkline (mini-grafici)
"""

import os, json, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings("ignore")

IMG_OUTPUT_DIR = "data/synthetic/line"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/line"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 15   # hard cap per serie

# ══════════════════════════════════════════════════════
#  DATASET TEMPLATES  (12 temi distinti)
# ══════════════════════════════════════════════════════

DATASET_TEMPLATES = [
    {
        "theme": "Temperatura Media Mensile",
        "x_label": "Mese", "y_label": "Temperatura (°C)",
        "x_values": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "series": {
            "Roma":    {"base": 12, "amplitude": 10, "noise": 1.2},
            "Milano":  {"base": 8,  "amplitude": 12, "noise": 1.5},
            "Palermo": {"base": 16, "amplitude": 8,  "noise": 0.8},
            "Torino":  {"base": 7,  "amplitude": 13, "noise": 1.8},
        },
        "y_range": (-2, 38),
    },
    {
        "theme": "Prezzo Azioni Tech (Indice Normalizzato)",
        "x_label": "Settimana", "y_label": "Prezzo (€)",
        "x_values": [f"W{i+1}" for i in range(12)],
        "series": {
            "TechAlpha":  {"base": 120, "amplitude": 20, "noise": 8,  "trend": 2.5},
            "NovaByte":   {"base": 85,  "amplitude": 15, "noise": 6,  "trend": 1.8},
            "CloudCore":  {"base": 200, "amplitude": 30, "noise": 12, "trend": -1.5},
            "DataStream": {"base": 60,  "amplitude": 10, "noise": 5,  "trend": 3.2},
        },
        "y_range": (40, 280),
    },
    {
        "theme": "Consumo Energetico Giornaliero",
        "x_label": "Ora del Giorno", "y_label": "Consumo (kWh)",
        "x_values": ["00","02","04","06","08","10","12","14","16","18","20","22"],
        "series": {
            "Lunedì":  {"base": 30, "amplitude": 25, "noise": 3, "shape": "bimodal"},
            "Sabato":  {"base": 20, "amplitude": 20, "noise": 4, "shape": "flat"},
            "Domenica":{"base": 18, "amplitude": 18, "noise": 3, "shape": "flat"},
        },
        "y_range": (5, 80),
    },
    {
        "theme": "Crescita Utenti Piattaforma",
        "x_label": "Mese", "y_label": "Utenti Attivi (milioni)",
        "x_values": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "series": {
            "App Mobile":  {"base": 1.2,  "amplitude": 0, "noise": 0.1, "trend": 0.4},
            "Web Desktop": {"base": 2.5,  "amplitude": 0, "noise": 0.2, "trend": 0.15},
            "API Partner": {"base": 0.3,  "amplitude": 0, "noise": 0.05,"trend": 0.08},
        },
        "y_range": (0.5, 8),
    },
    {
        "theme": "Precipitazioni Mensili",
        "x_label": "Mese", "y_label": "Precipitazioni (mm)",
        "x_values": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "series": {
            "Nord Italia": {"base": 70, "amplitude": 40, "noise": 15},
            "Centro":      {"base": 55, "amplitude": 30, "noise": 12},
            "Sud Italia":  {"base": 40, "amplitude": 35, "noise": 18},
        },
        "y_range": (0, 180),
    },
    {
        "theme": "Performance CPU / GPU Server",
        "x_label": "Minuto", "y_label": "Utilizzo (%)",
        "x_values": [str(i) for i in range(0, 15)],
        "series": {
            "CPU Core 0": {"base": 45, "amplitude": 30, "noise": 8},
            "CPU Core 1": {"base": 38, "amplitude": 25, "noise": 9},
            "GPU":        {"base": 70, "amplitude": 20, "noise": 5},
            "RAM":        {"base": 60, "amplitude": 10, "noise": 3},
        },
        "y_range": (0, 100),
    },
    {
        "theme": "Vendite E-commerce per Categoria",
        "x_label": "Trimestre", "y_label": "Vendite (k€)",
        "x_values": ["Q1 2022","Q2 2022","Q3 2022","Q4 2022",
                     "Q1 2023","Q2 2023","Q3 2023","Q4 2023",
                     "Q1 2024","Q2 2024","Q3 2024","Q4 2024"],
        "series": {
            "Elettronica": {"base": 120, "amplitude": 0, "noise": 15, "trend": 8},
            "Abbigliamento":{"base": 80, "amplitude": 20, "noise": 10, "trend": 5},
            "Casa & Giardino":{"base": 45,"amplitude": 15, "noise": 8, "trend": 3},
            "Sport":       {"base": 35, "amplitude": 10, "noise": 7,  "trend": 4},
        },
        "y_range": (20, 300),
    },
    {
        "theme": "Indice di Soddisfazione Cliente (NPS)",
        "x_label": "Mese", "y_label": "NPS Score",
        "x_values": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "series": {
            "Prodotto A": {"base": 42, "amplitude": 0, "noise": 6, "trend": 1.2},
            "Prodotto B": {"base": 28, "amplitude": 0, "noise": 8, "trend": 2.5},
            "Prodotto C": {"base": 55, "amplitude": 0, "noise": 4, "trend": 0.5},
        },
        "y_range": (10, 80),
    },
    {
        "theme": "Tasso di Conversione Campagne Marketing",
        "x_label": "Settimana", "y_label": "Conversione (%)",
        "x_values": [f"S{i+1}" for i in range(10)],
        "series": {
            "Email":        {"base": 3.2, "amplitude": 0, "noise": 0.5, "trend":  0.1},
            "Social Ads":   {"base": 1.8, "amplitude": 0, "noise": 0.4, "trend":  0.15},
            "Search (SEM)": {"base": 5.1, "amplitude": 0, "noise": 0.6, "trend": -0.05},
            "Display":      {"base": 0.9, "amplitude": 0, "noise": 0.2, "trend":  0.08},
        },
        "y_range": (0, 8),
    },
    {
        "theme": "Livello CO2 Atmosferico (ppm)",
        "x_label": "Anno", "y_label": "CO2 (ppm)",
        "x_values": [str(y) for y in range(2010, 2025)],
        "series": {
            "Stazione Nord": {"base": 390, "amplitude": 2, "noise": 1, "trend": 2.2},
            "Stazione Sud":  {"base": 388, "amplitude": 1.5,"noise": 0.8,"trend": 2.1},
            "Media Globale": {"base": 389, "amplitude": 1.8,"noise": 0.5,"trend": 2.15},
        },
        "y_range": (385, 430),
    },
    {
        "theme": "Frequenza Cardiaca durante Attività Fisica",
        "x_label": "Minuto", "y_label": "BPM",
        "x_values": [str(i) for i in range(0, 13)],
        "series": {
            "Atleta A": {"base": 68, "amplitude": 0, "noise": 5, "trend": 7, "peak_at": 8},
            "Atleta B": {"base": 72, "amplitude": 0, "noise": 6, "trend": 6, "peak_at": 9},
            "Atleta C": {"base": 65, "amplitude": 0, "noise": 4, "trend": 5, "peak_at": 7},
        },
        "y_range": (55, 185),
    },
    {
        "theme": "Latenza API per Endpoint",
        "x_label": "Ora", "y_label": "Latenza (ms)",
        "x_values": [f"{h:02d}:00" for h in range(0, 13)],
        "series": {
            "/api/search":   {"base": 120, "amplitude": 60, "noise": 20},
            "/api/checkout": {"base": 85,  "amplitude": 40, "noise": 15},
            "/api/auth":     {"base": 35,  "amplitude": 20, "noise": 8},
            "/api/feed":     {"base": 200, "amplitude": 80, "noise": 30},
        },
        "y_range": (10, 380),
    },
]

# ══════════════════════════════════════════════════════
#  VISUAL THEMES  (12 temi distinti)
# ══════════════════════════════════════════════════════

CHART_THEMES = [
    {
        "name": "corporate_blue",
        "bg": "#FFFFFF", "fig_bg": "#F4F7FB",
        "grid": "#E0E7F1", "title": "#1A237E", "label": "#37474F", "tick": "#546E7A",
        "spine": True,
        "palette": ["#1565C0","#E53935","#2E7D32","#F57F17","#6A1B9A","#00838F","#4E342E"],
        "linewidth": 2.2, "alpha_fill": 0.15,
    },
    {
        "name": "dark_pro",
        "bg": "#1E1E2E", "fig_bg": "#181825",
        "grid": "#313244", "title": "#CDD6F4", "label": "#BAC2DE", "tick": "#A6ADC8",
        "spine": False,
        "palette": ["#89B4FA","#A6E3A1","#FAB387","#F38BA8","#CBA6F7","#94E2D5","#F9E2AF"],
        "linewidth": 2.4, "alpha_fill": 0.18,
    },
    {
        "name": "vibrant_pop",
        "bg": "#FFFFFF", "fig_bg": "#F8F9FA",
        "grid": "#DEE2E6", "title": "#212529", "label": "#495057", "tick": "#6C757D",
        "spine": True,
        "palette": ["#E63946","#F4A261","#2A9D8F","#8338EC","#3A86FF","#FB5607","#06D6A0"],
        "linewidth": 2.5, "alpha_fill": 0.12,
    },
    {
        "name": "pastel_notebook",
        "bg": "#FFFEF7", "fig_bg": "#FFF9E6",
        "grid": "#EDE8D0", "title": "#4A3728", "label": "#6B5744", "tick": "#8D7566",
        "spine": True,
        "palette": ["#E07B54","#5B8DB8","#7DB87D","#B87DB8","#B8A45B","#5BB8B8","#B85B5B"],
        "linewidth": 2.0, "alpha_fill": 0.14,
    },
    {
        "name": "neon_dark",
        "bg": "#0A0A14", "fig_bg": "#06060E",
        "grid": "#14142A", "title": "#00FF88", "label": "#AAAACC", "tick": "#777799",
        "spine": False,
        "palette": ["#00FF88","#FF006E","#3A86FF","#FFBE0B","#FB5607","#8338EC","#00F5D4"],
        "linewidth": 2.6, "alpha_fill": 0.10,
    },
    {
        "name": "earth_tones",
        "bg": "#FFF8F0", "fig_bg": "#FEF0DC",
        "grid": "#EDD9BE", "title": "#5D3A1A", "label": "#7B4F2E", "tick": "#8B6347",
        "spine": True,
        "palette": ["#C84B31","#2D6A4F","#E9C46A","#264653","#A8763E","#8B5E3C","#457B9D"],
        "linewidth": 2.2, "alpha_fill": 0.16,
    },
    {
        "name": "ocean_depth",
        "bg": "#EAF4FB", "fig_bg": "#D6EAF8",
        "grid": "#AED6F1", "title": "#0B3D6B", "label": "#154360", "tick": "#1A5276",
        "spine": True,
        "palette": ["#0B3D6B","#C0392B","#1ABC9C","#D68910","#7D3C98","#2874A6","#1E8449"],
        "linewidth": 2.2, "alpha_fill": 0.13,
    },
    {
        "name": "sunset_fire",
        "bg": "#12000A", "fig_bg": "#0A0006",
        "grid": "#220010", "title": "#FF6B35", "label": "#FFB347", "tick": "#CC7722",
        "spine": False,
        "palette": ["#FF0000","#FF8C00","#FFD700","#FF69B4","#FF4500","#ADFF2F","#00CED1"],
        "linewidth": 2.4, "alpha_fill": 0.12,
    },
    {
        "name": "mint_clean",
        "bg": "#F0FFF4", "fig_bg": "#E2F5E9",
        "grid": "#B7DFC5", "title": "#1B5E20", "label": "#2E7D32", "tick": "#388E3C",
        "spine": True,
        "palette": ["#1B5E20","#B71C1C","#0D47A1","#F57F17","#4A148C","#006064","#BF360C"],
        "linewidth": 2.2, "alpha_fill": 0.14,
    },
    {
        "name": "mono_editorial",
        "bg": "#FFFFFF", "fig_bg": "#F5F5F5",
        "grid": "#E0E0E0", "title": "#000000", "label": "#333333", "tick": "#666666",
        "spine": True,
        "palette": ["#000000","#555555","#999999","#CC0000","#004499","#007700","#AA5500"],
        "linewidth": 2.0, "alpha_fill": 0.10,
    },
    {
        "name": "aurora_night",
        "bg": "#0D1117", "fig_bg": "#080C10",
        "grid": "#161B22", "title": "#79C0FF", "label": "#8B949E", "tick": "#6E7681",
        "spine": False,
        "palette": ["#79C0FF","#56D364","#FF7B72","#D2A8FF","#E3B341","#39C5CF","#F78166"],
        "linewidth": 2.5, "alpha_fill": 0.14,
    },
    {
        "name": "retro_warm",
        "bg": "#F5ECD7", "fig_bg": "#EDE0C8",
        "grid": "#D6C9A8", "title": "#2C1810", "label": "#4A3020", "tick": "#6B4C30",
        "spine": True,
        "palette": ["#8B2500","#1A4A6B","#2D5A1B","#7B4F00","#5B1A8B","#006B5B","#8B6B00"],
        "linewidth": 2.1, "alpha_fill": 0.15,
    },
]

SUBTYPES = [
    "simple_line", "multi_line", "smooth_line", "area_line",
    "stacked_area", "step_line", "dot_line", "dual_axis",
    "annotated_line", "band_line", "slope_chart", "sparkline_grid",
]

# ══════════════════════════════════════════════════════
#  DATA GENERATORS
# ══════════════════════════════════════════════════════

def gen_series(cfg, n_points, seed=None):
    """Generate a realistic time-series from a config dict."""
    if seed is not None:
        np.random.seed(seed)
    base      = cfg.get("base", 50)
    amplitude = cfg.get("amplitude", 20)
    noise     = cfg.get("noise", 5)
    trend     = cfg.get("trend", 0)
    shape     = cfg.get("shape", "sine")
    peak_at   = cfg.get("peak_at", None)

    t = np.linspace(0, 2 * np.pi, n_points)
    vals = base + np.random.normal(0, noise, n_points)

    if shape == "bimodal":
        vals += amplitude * (np.exp(-((t - 1.8)**2) / 0.5) +
                             np.exp(-((t - 4.8)**2) / 0.5))
    elif shape == "flat":
        vals += amplitude * np.exp(-((t - np.pi)**2) / 2.5)
    else:
        vals += amplitude * np.sin(t - np.pi / 2)

    if trend != 0:
        vals += np.linspace(0, trend * n_points, n_points)

    if peak_at is not None and peak_at < n_points:
        peak_curve = np.exp(-((np.arange(n_points) - peak_at)**2) / (2 * 2**2))
        vals += 80 * peak_curve

    return vals.tolist()

def pick_series(tmpl, n_series=None, max_pts=MAX_POINTS):
    """Return (x_labels, {name: [values]}) respecting MAX_POINTS."""
    xs = tmpl["x_values"][:max_pts]
    n  = len(xs)
    all_series = list(tmpl["series"].keys())
    if n_series is None:
        n_series = random.randint(1, len(all_series))
    chosen = random.sample(all_series, min(n_series, len(all_series)))
    result = {}
    for i, name in enumerate(chosen):
        result[name] = gen_series(tmpl["series"][name], n, seed=random.randint(0, 9999))
    return xs, result

def palette_n(t, n):
    p = t["palette"]
    return [p[i % len(p)] for i in range(n)]

# ══════════════════════════════════════════════════════
#  THEME APPLIER
# ══════════════════════════════════════════════════════

def apply_theme(fig, ax, t, secondary_ax=None):
    fig.patch.set_facecolor(t["fig_bg"])
    ax.set_facecolor(t["bg"])
    ax.title.set_color(t["title"])
    ax.xaxis.label.set_color(t["label"])
    ax.yaxis.label.set_color(t["label"])
    ax.tick_params(colors=t["tick"], which="both")
    ax.grid(True, color=t["grid"], linewidth=0.75, alpha=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(t["spine"])
        if t["spine"]: sp.set_edgecolor(t["grid"])
    if secondary_ax:
        secondary_ax.set_facecolor(t["bg"])
        secondary_ax.yaxis.label.set_color(t["label"])
        secondary_ax.tick_params(colors=t["tick"])
        for sp in secondary_ax.spines.values():
            sp.set_visible(t["spine"])
            if t["spine"]: sp.set_edgecolor(t["grid"])

def mk_json(title, xl, yl, dp):
    return {"chart_title": title, "x_axis_label": xl, "y_axis_label": yl, "data_points": dp}

def dp_from(series_dict, xs):
    dp = []
    for name, vals in series_dict.items():
        for x, v in zip(xs, vals):
            dp.append({"series_name": name, "x_value": str(x), "y_value": round(float(v), 4)})
    return dp

# ══════════════════════════════════════════════════════
#  MARKER STYLES
# ══════════════════════════════════════════════════════

MARKERS = ["o","s","^","D","v","P","X","*","h","8"]

def rand_linestyle():
    return random.choice(["-", "--", "-.", ":"])

def rand_marker():
    return random.choice(MARKERS)

# ══════════════════════════════════════════════════════
#  RENDERERS
# ══════════════════════════════════════════════════════

# ── 1. SIMPLE LINE ───────────────────────────────────
def render_simple_line(tmpl, t, _):
    xs, series = pick_series(tmpl, n_series=1)
    name = list(series.keys())[0]
    vals = series[name]
    color = t["palette"][0]
    n = len(xs)

    fig, ax = plt.subplots(figsize=(11, 5))
    apply_theme(fig, ax, t)

    marker = rand_marker()
    ms = random.choice([5, 6, 7])
    ax.plot(xs, vals, color=color, linewidth=t["linewidth"], marker=marker,
            markersize=ms, markerfacecolor=t["bg"], markeredgecolor=color,
            markeredgewidth=1.8, zorder=4)

    # Highlight min/max
    lo_i, hi_i = int(np.argmin(vals)), int(np.argmax(vals))
    for idx, label, va in [(lo_i, f"Min\n{vals[lo_i]:.1f}", "top"),
                            (hi_i, f"Max\n{vals[hi_i]:.1f}", "bottom")]:
        ax.annotate(label, xy=(xs[idx], vals[idx]),
                    xytext=(0, 18 if va == "bottom" else -18),
                    textcoords="offset points",
                    ha="center", va=va, fontsize=8.5,
                    color=t["title"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

    ax.set_title(f"{tmpl['theme']} — {name}", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.xticks(rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    plt.tight_layout()

    dp = [{"series_name": name, "x_value": str(x), "y_value": round(float(v), 4)}
          for x, v in zip(xs, vals)]
    return fig, mk_json(f"{tmpl['theme']} — {name}", tmpl["x_label"], tmpl["y_label"], dp)


# ── 2. MULTI LINE ────────────────────────────────────
def render_multi_line(tmpl, t, _):
    n_s = random.randint(2, min(4, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    for (name, vals), color in zip(series.items(), colors):
        marker = rand_marker()
        ls = rand_linestyle()
        ax.plot(xs, vals, color=color, linewidth=t["linewidth"],
                linestyle=ls, marker=marker, markersize=5,
                markerfacecolor=t["bg"], markeredgecolor=color,
                markeredgewidth=1.6, label=name, zorder=4)

    ax.set_title(tmpl["theme"], fontsize=13, fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.xticks(rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"],
                        dp_from(series, xs))


# ── 3. SMOOTH LINE (Spline) ──────────────────────────
def render_smooth_line(tmpl, t, _):
    n_s = random.randint(1, min(3, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    for (name, vals), color in zip(series.items(), colors):
        x_num = np.arange(len(xs))
        if len(x_num) >= 4:
            x_smooth = np.linspace(x_num[0], x_num[-1], 300)
            spl = make_interp_spline(x_num, vals, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x_num, vals

        ax.plot(x_smooth, y_smooth, color=color, linewidth=t["linewidth"] + 0.4,
                label=name, zorder=4)
        # Scatter original points
        ax.scatter(x_num, vals, color=color, s=40, zorder=5,
                   edgecolors=t["bg"], linewidths=1.4)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(tmpl["theme"] + " (Smooth)", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"] + " (Smooth)", tmpl["x_label"], tmpl["y_label"],
                        dp_from(series, xs))


# ── 4. AREA LINE ─────────────────────────────────────
def render_area_line(tmpl, t, _):
    n_s = random.randint(1, min(3, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)
    x_num = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    for (name, vals), color in zip(series.items(), colors):
        ax.plot(x_num, vals, color=color, linewidth=t["linewidth"], label=name, zorder=4)
        ax.fill_between(x_num, vals, alpha=t["alpha_fill"] + 0.08,
                        color=color, zorder=2)

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(tmpl["theme"] + " (Area)", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"] + " (Area)", tmpl["x_label"], tmpl["y_label"],
                        dp_from(series, xs))


# ── 5. STACKED AREA ──────────────────────────────────
def render_stacked_area(tmpl, t, _):
    n_s = random.randint(2, min(4, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)
    x_num = np.arange(n)

    # Force positive values
    names = list(series.keys())
    matrix = np.array([np.abs(series[n_]) for n_ in names])

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    ax.stackplot(x_num, matrix, labels=names, colors=colors,
                 alpha=0.85, zorder=3)

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(tmpl["theme"] + " (Stacked Area)", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["label"],
              loc="upper left", reverse=True)
    plt.tight_layout()

    dp = [{"series_name": name, "x_value": str(x), "y_value": round(float(v), 4)}
          for name, row in zip(names, matrix) for x, v in zip(xs, row)]
    return fig, mk_json(tmpl["theme"] + " (Stacked Area)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 6. STEP LINE ─────────────────────────────────────
def render_step_line(tmpl, t, _):
    n_s = random.randint(1, min(3, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)
    x_num = np.arange(n)
    step_where = random.choice(["pre", "post", "mid"])

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    for (name, vals), color in zip(series.items(), colors):
        ax.step(x_num, vals, where=step_where, color=color,
                linewidth=t["linewidth"] + 0.3, label=name, zorder=4)
        ax.scatter(x_num, vals, color=color, s=30, zorder=5,
                   edgecolors=t["bg"], linewidths=1.2)

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(tmpl["theme"] + f" (Step — {step_where})", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"] + f" (Step)", tmpl["x_label"], tmpl["y_label"],
                        dp_from(series, xs))


# ── 7. DOT LINE (Lollipop-style) ─────────────────────
def render_dot_line(tmpl, t, _):
    n_s = random.randint(1, min(2, len(tmpl["series"])))
    xs, series = pick_series(tmpl, n_series=n_s)
    colors = palette_n(t, len(series))
    n = len(xs)
    x_num = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    for (name, vals), color in zip(series.items(), colors):
        ax.plot(x_num, vals, color=color, linewidth=1.2,
                alpha=0.5, linestyle="--", zorder=3, label=name)
        ax.scatter(x_num, vals, color=color, s=90, zorder=5,
                   edgecolors=t["bg"], linewidths=1.8)
        # Vertical stems
        for xi, v in zip(x_num, vals):
            base_y = min(vals) - (max(vals) - min(vals)) * 0.05
            ax.plot([xi, xi], [base_y, v], color=color,
                    linewidth=1.0, alpha=0.35, zorder=2)
        # Value labels on top
        for xi, v in zip(x_num, vals):
            ax.text(xi, v + (max(vals) - min(vals)) * 0.025, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=t["tick"])

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(tmpl["theme"] + " (Dot-Line)", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"] + " (Dot-Line)", tmpl["x_label"], tmpl["y_label"],
                        dp_from(series, xs))


# ── 8. DUAL AXIS ─────────────────────────────────────
def render_dual_axis(tmpl, t, _):
    all_keys = list(tmpl["series"].keys())
    if len(all_keys) < 2:
        all_keys = all_keys * 2
    k1, k2 = random.sample(all_keys, 2)
    xs = tmpl["x_values"][:MAX_POINTS]
    n  = len(xs)
    x_num = np.arange(n)

    v1 = gen_series(tmpl["series"][k1], n, seed=random.randint(0, 9999))
    v2 = gen_series(tmpl["series"][k2], n, seed=random.randint(0, 9999))
    c1, c2 = t["palette"][0], t["palette"][1 % len(t["palette"])]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    apply_theme(fig, ax1, t, secondary_ax=ax2)

    ax1.plot(x_num, v1, color=c1, linewidth=t["linewidth"], marker="o",
             markersize=5, markerfacecolor=t["bg"], markeredgecolor=c1,
             markeredgewidth=1.6, label=k1, zorder=4)
    ax1.fill_between(x_num, v1, alpha=t["alpha_fill"], color=c1, zorder=2)
    ax2.plot(x_num, v2, color=c2, linewidth=t["linewidth"], marker="s",
             linestyle="--", markersize=5, markerfacecolor=t["bg"],
             markeredgecolor=c2, markeredgewidth=1.6, label=k2, zorder=4)

    ax1.set_xticks(x_num)
    ax1.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax1.set_title(tmpl["theme"] + " (Dual Axis)", fontsize=13, fontweight="bold",
                  color=t["title"], pad=14)
    ax1.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax1.set_ylabel(k1, fontsize=10, color=c1)
    ax2.set_ylabel(k2, fontsize=10, color=c2)
    ax1.tick_params(axis="y", colors=c1)
    ax2.tick_params(axis="y", colors=c2)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, frameon=False,
               fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    dp = ([{"series_name": k1, "x_value": str(x), "y_value": round(float(v), 4)}
            for x, v in zip(xs, v1)] +
          [{"series_name": k2, "x_value": str(x), "y_value": round(float(v), 4)}
            for x, v in zip(xs, v2)])
    return fig, mk_json(tmpl["theme"] + " (Dual Axis)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 9. ANNOTATED LINE ────────────────────────────────
def render_annotated_line(tmpl, t, _):
    xs, series = pick_series(tmpl, n_series=1)
    name = list(series.keys())[0]
    vals = series[name]
    color = t["palette"][0]
    n = len(xs)
    x_num = np.arange(n)

    # Pick 2–3 interesting points to annotate (local extrema)
    vals_arr = np.array(vals)
    candidates = []
    for i in range(1, n - 1):
        if vals_arr[i] > vals_arr[i-1] and vals_arr[i] > vals_arr[i+1]:
            candidates.append((i, "peak"))
        elif vals_arr[i] < vals_arr[i-1] and vals_arr[i] < vals_arr[i+1]:
            candidates.append((i, "valley"))
    # Always include global min/max
    candidates.append((int(np.argmax(vals_arr)), "peak"))
    candidates.append((int(np.argmin(vals_arr)), "valley"))
    # Deduplicate and pick up to 3
    seen = set()
    ann_pts = []
    for idx, kind in candidates:
        if idx not in seen:
            seen.add(idx)
            ann_pts.append((idx, kind))
    ann_pts = ann_pts[:3]

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    ax.plot(x_num, vals, color=color, linewidth=t["linewidth"] + 0.3,
            marker="o", markersize=4.5, markerfacecolor=t["bg"],
            markeredgecolor=color, markeredgewidth=1.5, zorder=4)
    ax.fill_between(x_num, vals, alpha=t["alpha_fill"], color=color, zorder=2)

    for idx, kind in ann_pts:
        yoff = 22 if kind == "peak" else -28
        va   = "bottom" if kind == "peak" else "top"
        ax.annotate(f"{vals[idx]:.1f}",
                    xy=(x_num[idx], vals[idx]),
                    xytext=(0, yoff), textcoords="offset points",
                    ha="center", va=va, fontsize=9, color=t["title"],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=t["fig_bg"],
                              ec=color, lw=1.2, alpha=0.9),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(f"{tmpl['theme']} — {name} (Annotata)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.tight_layout()

    dp = [{"series_name": name, "x_value": str(x), "y_value": round(float(v), 4)}
          for x, v in zip(xs, vals)]
    return fig, mk_json(f"{tmpl['theme']} — {name} (Annotata)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 10. BAND LINE (Confidence interval) ──────────────
def render_band_line(tmpl, t, _):
    xs, series = pick_series(tmpl, n_series=1)
    name = list(series.keys())[0]
    vals = np.array(series[name])
    color = t["palette"][0]
    n = len(xs)
    x_num = np.arange(n)

    # Simulate ±1σ band with variable width
    noise_scale = (vals.max() - vals.min()) * random.uniform(0.04, 0.12)
    sigma = np.abs(np.random.normal(noise_scale, noise_scale * 0.3, n))
    upper = vals + sigma
    lower = vals - sigma

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, t)

    ax.fill_between(x_num, lower, upper, color=color,
                    alpha=t["alpha_fill"] + 0.12, label="±1σ", zorder=2)
    ax.plot(x_num, upper, color=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(x_num, lower, color=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
    ax.plot(x_num, vals,  color=color, linewidth=t["linewidth"] + 0.3,
            marker="o", markersize=5, markerfacecolor=t["bg"],
            markeredgecolor=color, markeredgewidth=1.6, label=name, zorder=5)

    ax.set_xticks(x_num)
    ax.set_xticklabels(xs, rotation=30 if n > 8 else 0, ha="right", fontsize=9)
    ax.set_title(f"{tmpl['theme']} — {name} (Banda di Confidenza)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    dp = ([{"series_name": name, "x_value": str(x), "y_value": round(float(v), 4)}
            for x, v in zip(xs, vals)] +
          [{"series_name": f"{name} upper", "x_value": str(x), "y_value": round(float(v), 4)}
            for x, v in zip(xs, upper)] +
          [{"series_name": f"{name} lower", "x_value": str(x), "y_value": round(float(v), 4)}
            for x, v in zip(xs, lower)])
    return fig, mk_json(f"{tmpl['theme']} — {name} (Banda)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 11. SLOPE CHART ──────────────────────────────────
def render_slope_chart(tmpl, t, _):
    n_s = random.randint(3, min(6, len(tmpl["series"])))
    xs_all = tmpl["x_values"][:MAX_POINTS]
    # Pick just 2 time points for slope
    idx_a = 0
    idx_b = random.randint(max(1, len(xs_all) // 2), len(xs_all) - 1)
    x_left, x_right = xs_all[idx_a], xs_all[idx_b]

    all_keys = list(tmpl["series"].keys())
    chosen = random.sample(all_keys, min(n_s, len(all_keys)))
    colors = palette_n(t, len(chosen))

    fig, ax = plt.subplots(figsize=(8, 8))
    apply_theme(fig, ax, t)
    ax.grid(False)
    ax.set_facecolor(t["bg"])

    left_vals, right_vals = [], []
    for name, cfg in [(k, tmpl["series"][k]) for k in chosen]:
        full = gen_series(cfg, len(xs_all), seed=random.randint(0, 9999))
        left_vals.append(full[idx_a])
        right_vals.append(full[idx_b])

    for i, (name, lv, rv) in enumerate(zip(chosen, left_vals, right_vals)):
        color = colors[i]
        ax.plot([0, 1], [lv, rv], color=color, linewidth=2.0, zorder=4,
                marker="o", markersize=7, markerfacecolor=color,
                markeredgecolor=t["bg"], markeredgewidth=1.5)
        # Labels
        ax.text(-0.04, lv, f"{name}  {lv:.1f}", ha="right", va="center",
                fontsize=9, color=color, fontweight="bold")
        delta = rv - lv
        arrow = "▲" if delta >= 0 else "▼"
        ax.text(1.04, rv, f"{rv:.1f}  {arrow}{abs(delta):.1f}", ha="left",
                va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([x_left, x_right], fontsize=12, fontweight="bold",
                       color=t["label"])
    ax.set_yticks([])
    ax.axvline(0, color=t["grid"], linewidth=1.5)
    ax.axvline(1, color=t["grid"], linewidth=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(tmpl["theme"] + " (Slope Chart)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    plt.tight_layout()

    dp = ([{"series_name": name, "x_value": str(x_left),  "y_value": round(float(lv), 4)}
            for name, lv in zip(chosen, left_vals)] +
          [{"series_name": name, "x_value": str(x_right), "y_value": round(float(rv), 4)}
            for name, rv in zip(chosen, right_vals)])
    return fig, mk_json(tmpl["theme"] + " (Slope Chart)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 12. SPARKLINE GRID ───────────────────────────────
def render_sparkline_grid(tmpl, t, _):
    all_keys = list(tmpl["series"].keys())
    n_series = min(len(all_keys), random.randint(3, 4))
    chosen = random.sample(all_keys, n_series)
    xs = tmpl["x_values"][:MAX_POINTS]
    n  = len(xs)
    x_num = np.arange(n)
    colors = palette_n(t, n_series)

    ncols = 2
    nrows = math.ceil(n_series / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
    if n_series == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor(t["fig_bg"])

    fig.suptitle(tmpl["theme"] + " — Sparklines", fontsize=13,
                 fontweight="bold", color=t["title"], y=1.01)

    all_dp = []
    for idx, (name, color) in enumerate(zip(chosen, colors)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor(t["bg"])

        vals = gen_series(tmpl["series"][name], n, seed=random.randint(0, 9999))
        ax.plot(x_num, vals, color=color, linewidth=2.0, zorder=4)
        ax.fill_between(x_num, vals, alpha=t["alpha_fill"] + 0.1, color=color)
        ax.scatter([int(np.argmin(vals)), int(np.argmax(vals))],
                   [min(vals), max(vals)],
                   color=[t["palette"][3 % len(t["palette"])],
                          t["palette"][2 % len(t["palette"])]],
                   s=50, zorder=6, edgecolors=t["bg"], linewidths=1.2)

        ax.set_xticks(x_num[::max(1, n // 5)])
        ax.set_xticklabels(xs[::max(1, n // 5)], fontsize=7.5,
                            rotation=30, ha="right", color=t["tick"])
        ax.tick_params(colors=t["tick"], labelsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold", color=color, pad=6)
        ax.set_ylabel(tmpl["y_label"], fontsize=8, color=t["label"])
        ax.grid(True, color=t["grid"], linewidth=0.6, alpha=0.6)
        for sp in ax.spines.values():
            sp.set_visible(t["spine"])
            if t["spine"]: sp.set_edgecolor(t["grid"])

        # Current value callout
        last = vals[-1]
        ax.text(x_num[-1], last, f"  {last:.1f}",
                ha="left", va="center", fontsize=8.5,
                color=color, fontweight="bold")

        for x, v in zip(xs, vals):
            all_dp.append({"series_name": name, "x_value": str(x),
                            "y_value": round(float(v), 4)})

    # Hide spare axes
    for idx in range(n_series, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    return fig, mk_json(tmpl["theme"] + " (Sparklines)", tmpl["x_label"],
                        tmpl["y_label"], all_dp)


# ══════════════════════════════════════════════════════
#  RENDERER MAP
# ══════════════════════════════════════════════════════

RENDERERS = {
    "simple_line":     render_simple_line,
    "multi_line":      render_multi_line,
    "smooth_line":     render_smooth_line,
    "area_line":       render_area_line,
    "stacked_area":    render_stacked_area,
    "step_line":       render_step_line,
    "dot_line":        render_dot_line,
    "dual_axis":       render_dual_axis,
    "annotated_line":  render_annotated_line,
    "band_line":       render_band_line,
    "slope_chart":     render_slope_chart,
    "sparkline_grid":  render_sparkline_grid,
}

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def generate_charts(n: int):
    used = set()
    print(f"\nGenerazione di {n} grafici a linee...\n")
    for i in range(1, n + 1):
        for _ in range(400):
            st    = random.choice(SUBTYPES)
            tmpl  = random.choice(DATASET_TEMPLATES)
            theme = random.choice(CHART_THEMES)
            key   = (st, tmpl["theme"], theme["name"])
            if key not in used:
                used.add(key)
                break

        print(f"  [{i:>3}/{n}] {st:20s} | {tmpl['theme'][:36]:36s} | {theme['name']}")
        try:
            fig, data = RENDERERS[st](tmpl, theme, i)
        except Exception as e:
            print(f"          Errore: {e} — skip")
            continue

        base  = f"line_{i:03d}_{st}"
        img_p = os.path.join(IMG_OUTPUT_DIR, base+".png")
        jsn_p = os.path.join(JSON_OUTPUT_DIR, base+".json")

        fig.savefig(img_p, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        with open(jsn_p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"          {img_p}  +  {jsn_p}")


if __name__ == "__main__":
    while True:
        try:
            n = int(input("Quanti grafici a linee vuoi generare? "))
            if n > 0:
                break
            print("Inserisci un numero maggiore di 0.")
        except ValueError:
            print("Inserisci un numero intero valido.")
    generate_charts(n)
