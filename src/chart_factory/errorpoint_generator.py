#!/usr/bin/env python3
"""
Generatore di grafici Errorpoint — solo punti con barre d'errore esplicite.
Nessuna barra, nessuna area. I punti possono essere collegati da linee per gruppo.
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────── CARTELLE DI OUTPUT ─────────────────────────────
IMG_OUTPUT_DIR = "data/synthetic/errorpoint"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/errorpoint"

# ─────────────────────────── PALETTE & STILI ────────────────────────────────
PALETTES = [
    ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"],
    ["#6A0572", "#AB83A1", "#C77DFF", "#E0AAFF", "#9D4EDD"],
    ["#1B4332", "#40916C", "#74C69D", "#52B788", "#95D5B2"],
    ["#03045E", "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
    ["#370617", "#9D0208", "#DC2F02", "#F48C06", "#FAA307"],
    ["#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF"],
    ["#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0", "#F72585"],
    ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
    ["#D62828", "#F77F00", "#FCBF49", "#EAE2B7", "#003049"],
    ["#606C38", "#DDA15E", "#BC6C25", "#8B9D77", "#3A5A40"],
]

BACKGROUNDS = [
    "#FFFFFF", "#F8F9FA", "#0D0D0D", "#1A1A2E", "#0F3460",
    "#16213E", "#F0EAD6", "#E8F4F8", "#1C1C1E", "#FAF0E6",
    "#0A0A0A", "#F5F0EB", "#111827", "#FDFCFB",
]

GRID_STYLES = [
    {"color": "#CCCCCC", "linestyle": "--", "alpha": 0.45},
    {"color": "#555555", "linestyle": ":",  "alpha": 0.40},
    {"color": "#BBBBBB", "linestyle": "-",  "alpha": 0.18},
    {"color": "#88AAFF", "linestyle": "--", "alpha": 0.28},
    {"color": "#FFAA88", "linestyle": ":",  "alpha": 0.28},
    {"color": "#AAAAAA", "linestyle": (0, (5, 10)), "alpha": 0.35},
    None,
    None,   # doppia probabilità "no grid"
]

MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8", "<", ">"]
FONTS   = ["DejaVu Sans", "monospace", "serif", "DejaVu Serif"]

# ─────────────────────────── TIPOLOGIE DI DATI ──────────────────────────────
DATASET_THEMES = [
    {
        "name": "Temperatura mensile",
        "x_label": "Mese", "y_label": "Temperatura (°C)",
        "x_type": "categorical",
        "categories": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "base_range": (2, 38), "spread": (2, 9),
        "series": ["Nord", "Centro", "Sud"],
    },
    {
        "name": "Pressione sanguigna per età",
        "x_label": "Fascia d'età", "y_label": "Pressione sistolica (mmHg)",
        "x_type": "categorical",
        "categories": ["20-30","30-40","40-50","50-60","60-70","70+"],
        "base_range": (108, 160), "spread": (4, 18),
        "series": ["Uomini", "Donne"],
    },
    {
        "name": "Latenza API per endpoint",
        "x_label": "Endpoint", "y_label": "Latenza (ms)",
        "x_type": "categorical",
        "categories": ["/auth","/users","/data","/search","/export","/upload","/sync"],
        "base_range": (15, 900), "spread": (8, 220),
        "series": ["Prod", "Staging", "Dev"],
    },
    {
        "name": "Rendimento colture agricole",
        "x_label": "Anno", "y_label": "Resa (ton/ha)",
        "x_type": "numeric", "x_start": 2014, "x_step": 1,
        "n_points_range": (7, 11),
        "base_range": (2.0, 9.0), "spread": (0.3, 1.8),
        "series": ["Grano", "Mais", "Orzo"],
    },
    {
        "name": "Consumi energetici giornalieri",
        "x_label": "Ora del giorno", "y_label": "Consumo (kWh)",
        "x_type": "numeric", "x_start": 0, "x_step": 2,
        "n_points_range": (12, 13),
        "base_range": (80, 950), "spread": (25, 160),
        "series": ["Residenziale", "Industriale", "Commerciale"],
    },
    {
        "name": "Velocità del vento per stazione",
        "x_label": "Stazione meteo", "y_label": "Velocità (km/h)",
        "x_type": "categorical",
        "categories": ["Alpi","Pianura","Costa","Isole","Appennino","Valle"],
        "base_range": (8, 90), "spread": (4, 28),
        "series": ["Inverno", "Primavera", "Estate", "Autunno"],
    },
    {
        "name": "Punteggio test cognitivi",
        "x_label": "Test", "y_label": "Punteggio standardizzato",
        "x_type": "categorical",
        "categories": ["Memoria","Attenzione","Ragionamento","Linguaggio","Velocità"],
        "base_range": (35, 105), "spread": (4, 22),
        "series": ["Gruppo A", "Gruppo B", "Gruppo C"],
    },
    {
        "name": "Volatilità azionaria trimestrale",
        "x_label": "Trimestre", "y_label": "Volatilità (%)",
        "x_type": "categorical",
        "categories": ["Q1-22","Q2-22","Q3-22","Q4-22","Q1-23","Q2-23","Q3-23","Q4-23"],
        "base_range": (4, 48), "spread": (1.5, 14),
        "series": ["Tech", "Finance", "Energy"],
    },
    {
        "name": "Emissioni CO₂ per settore",
        "x_label": "Settore", "y_label": "Emissioni (Mt CO₂)",
        "x_type": "categorical",
        "categories": ["Trasporti","Industria","Energia","Agricoltura","Edilizia"],
        "base_range": (40, 420), "spread": (15, 85),
        "series": ["2019", "2021", "2023"],
    },
    {
        "name": "Risposta immunitaria vaccini",
        "x_label": "Giorni post-vaccinazione", "y_label": "Titolo anticorpale (UI/mL)",
        "x_type": "numeric", "x_start": 7, "x_step": 7,
        "n_points_range": (8, 12),
        "base_range": (40, 1400), "spread": (25, 350),
        "series": ["Vaccino A", "Vaccino B", "Placebo"],
    },
    {
        "name": "Precipitazioni mensili",
        "x_label": "Mese", "y_label": "Precipitazioni (mm)",
        "x_type": "categorical",
        "categories": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
        "base_range": (10, 200), "spread": (5, 50),
        "series": ["Milano", "Roma", "Palermo"],
    },
    {
        "name": "Frequenza cardiaca durante esercizio",
        "x_label": "Minuto", "y_label": "BPM",
        "x_type": "numeric", "x_start": 0, "x_step": 5,
        "n_points_range": (8, 13),
        "base_range": (60, 185), "spread": (3, 18),
        "series": ["Atleta A", "Atleta B", "Atleta C"],
    },
    {
        "name": "Concentrazione PM2.5",
        "x_label": "Giorno della settimana", "y_label": "PM2.5 (μg/m³)",
        "x_type": "categorical",
        "categories": ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"],
        "base_range": (5, 80), "spread": (2, 20),
        "series": ["Centro città", "Periferia", "Zona industriale"],
    },
    {
        "name": "Spessore ghiacciai alpini",
        "x_label": "Anno", "y_label": "Spessore medio (m)",
        "x_type": "numeric", "x_start": 1990, "x_step": 5,
        "n_points_range": (7, 8),
        "base_range": (15, 120), "spread": (2, 12),
        "series": ["Ortles", "Adamello", "Monte Rosa"],
    },
]

# ─────────────────────────── TIPOLOGIE DI CHART ─────────────────────────────
CHART_TYPES = [
    "points_only",
    "points_only_jitter",
    "points_capsize_heavy",
    "points_no_cap",
    "points_asymmetric",
    "points_sized",
    "points_horizontal",
    "points_dual_axis",
    "connected_line",
    "connected_dashed",
    "connected_smooth",
    "connected_ranged",
    "connected_markers_prominent",
]

# ─────────────────────────────── HELPERS ────────────────────────────────────

def is_dark(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.299*r + 0.587*g + 0.114*b < 128

def contrast_text(bg):
    return "#EEEEEE" if is_dark(bg) else "#111111"

def generate_data(theme, n_series=None):
    low, high = theme["base_range"]
    slo, shi  = theme["spread"]
    if theme["x_type"] == "categorical":
        x_vals = theme["categories"]
        n_pts  = len(x_vals)
    else:
        n_pts  = random.randint(*theme["n_points_range"])
        x_vals = [theme["x_start"] + i*theme["x_step"] for i in range(n_pts)]
    series_pool = theme.get("series", ["Main"])
    if n_series is not None:
        series_pool = series_pool[:n_series]
    all_series = []
    for sname in series_pool:
        off     = random.uniform(-0.18*(high-low), 0.18*(high-low))
        medians = [random.uniform(low+off, high+off) for _ in range(n_pts)]
        spreads = [random.uniform(slo, shi)           for _ in range(n_pts)]
        asym    = random.uniform(0.25, 0.75)
        mins_   = [max(0.0, m - s*asym)      for m, s in zip(medians, spreads)]
        maxs_   = [max(m, m + s*(1-asym))   for m, s in zip(medians, spreads)]
        all_series.append({"name": sname, "x": x_vals,
                           "median": medians, "min": mins_, "max": maxs_})
    return all_series, x_vals

def build_json(theme, all_series, chart_title, categorical_axis):
    dp = []
    for s in all_series:
        for i, xv in enumerate(s["x"]):
            dp.append({
                "series_name": s["name"],
                "x_value": str(xv),
                "y_value": {
                    "min":    round(s["min"][i],    4),
                    "median": round(s["median"][i], 4),
                    "max":    round(s["max"][i],    4),
                },
            })
    return {
        "chart_title":  chart_title,
        "x_axis_label": theme.get("x_label") or None,
        "y_axis_label": theme.get("y_label") or None,
        "categorical_axis": categorical_axis,
        "data_points":  dp,
    }

def _setup_figure(bg, text_color, font):
    fig_w = random.uniform(9, 16)
    fig_h = random.uniform(5, 9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(random.uniform(0.4, 1.6))
    ax.tick_params(colors=text_color, labelsize=random.randint(8, 12))
    plt.rcParams["font.family"] = font
    return fig, ax

def _apply_grid(ax, grid_style):
    if grid_style is None:
        ax.grid(False)
    else:
        ax.grid(True, **grid_style)
        ax.set_axisbelow(True)

def _set_labels(ax, theme, title, text_color):
    fs_title = random.randint(12, 19)
    fs_label = random.randint(9, 13)
    if title:
        ax.set_title(title, color=text_color, fontsize=fs_title,
                     fontweight=random.choice(["bold","normal"]),
                     pad=random.uniform(8, 22))
    if theme.get("x_label"):
        ax.set_xlabel(theme["x_label"], color=text_color, fontsize=fs_label)
    if theme.get("y_label"):
        ax.set_ylabel(theme["y_label"], color=text_color, fontsize=fs_label)

def _set_xticks(ax, all_series, rotation=None):
    ax.set_xticks(range(len(all_series[0]["x"])))
    ax.set_xticklabels(all_series[0]["x"],
                       rotation=rotation if rotation is not None
                       else random.choice([0, 30, 45]))

def _legend(ax, n_series, bg, text_color):
    if n_series < 2:
        return
    locs = ["best","upper right","upper left","lower right",
            "lower left","center right","center left"]
    leg = ax.legend(loc=random.choice(locs),
                    fontsize=random.randint(7, 11),
                    framealpha=random.uniform(0.25, 0.88),
                    edgecolor=text_color)
    for t in leg.get_texts():
        t.set_color(text_color)
    leg.get_frame().set_facecolor(bg)

def _eb_kwargs(color, cap):
    return dict(ecolor=color, capsize=cap,
                capthick=random.uniform(1.0, 2.5),
                linewidth=random.uniform(0.8, 2.2),
                alpha=random.uniform(0.7, 1.0))

def _safe_elo(series_median, series_min):
    return [max(0.0, m - mn) for m, mn in zip(series_median, series_min)]

def _safe_ehi(series_median, series_max):
    return [max(0.0, mx - m) for m, mx in zip(series_median, series_max)]


# ─────────────────────────── RENDERER ───────────────────────────────────────

def render_points_only(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], **_eb_kwargs(c, random.randint(4, 10)))
    _set_xticks(ax, series)

def render_points_only_jitter(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        n   = len(s["x"])
        jw  = 0.12 * max(1, len(series))
        off = (idx - (len(series)-1)/2) * jw
        xs  = [i + off + random.uniform(-0.04, 0.04) for i in range(n)]
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 12),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], **_eb_kwargs(c, random.randint(3, 8)))
    ax.set_xticks(range(len(series[0]["x"])))
    ax.set_xticklabels(series[0]["x"], rotation=random.choice([0, 30, 45]))

def render_points_capsize_heavy(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        cap = random.randint(14, 24)
        kw  = _eb_kwargs(c, cap)
        kw["capthick"] = random.uniform(2.5, 4.5)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 10),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], **kw)
    _set_xticks(ax, series)

def render_points_no_cap(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        kw  = _eb_kwargs(c, 0)
        kw["linewidth"] = random.uniform(1.8, 3.2)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(7, 13),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=1.0,
                    label=s["name"], **kw)
    _set_xticks(ax, series)

def render_points_asymmetric(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = [max(0.0, (m - mn) * random.uniform(0.05, 0.95))
               for m, mn in zip(s["median"], s["min"])]
        ehi = [max(0.0, (mx - m) * random.uniform(0.05, 0.95))
               for m, mx in zip(s["median"], s["max"])]
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], **_eb_kwargs(c, random.randint(4, 9)))
    _set_xticks(ax, series)

def render_points_sized(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        tot = [a + b for a, b in zip(elo, ehi)]
        mx_e = max(tot) if max(tot) > 0 else 1
        sizes = [5 + 15*(e/mx_e) for e in tot]
        kw  = _eb_kwargs(c, random.randint(4, 8))
        mk  = random.choice(MARKERS)
        for i, (x, med, el, eh, ms) in enumerate(
                zip(xs, s["median"], elo, ehi, sizes)):
            ax.errorbar([x], [med], yerr=[[el],[eh]],
                        fmt=mk, color=c, markersize=ms,
                        markerfacecolor=c, markeredgecolor=bg,
                        markeredgewidth=0.8,
                        label=s["name"] if i == 0 else "_nolegend_", **kw)
    _set_xticks(ax, series)

def render_points_horizontal(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        ys  = list(range(len(s["x"])))
        off = (idx - (len(series)-1)/2) * 0.25
        ys_off = [y + off for y in ys]
        
        # Sostituzione con le funzioni sicure
        xlo = _safe_elo(s["median"], s["min"])
        xhi = _safe_ehi(s["median"], s["max"])
        
        ax.errorbar(s["median"], ys_off, xerr=[xlo, xhi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], **_eb_kwargs(c, random.randint(4, 9)))
    ax.set_yticks(range(len(series[0]["x"])))
    ax.set_yticklabels(series[0]["x"])

def render_points_dual_axis(ax, series, palette, bg):
    text_color = contrast_text(bg)
    axes = [ax]
    if len(series) > 1:
        ax2 = ax.twinx()
        ax2.set_facecolor(bg)
        ax2.tick_params(colors=text_color)
        for sp in ax2.spines.values():
            sp.set_color(text_color)
        axes.append(ax2)
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        cur = axes[min(idx, 1)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        cur.errorbar(xs, s["median"], yerr=[elo, ehi],
                     fmt=random.choice(MARKERS), color=c,
                     markersize=random.randint(6, 11),
                     markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                     label=s["name"], **_eb_kwargs(c, random.randint(4, 9)))
    _set_xticks(ax, series)
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    if handles:
        ax.legend(handles, labels, fontsize=9, loc="upper left",
                  framealpha=0.6, edgecolor=text_color)

def render_connected_line(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.plot(xs, s["median"], color=c,
                linewidth=random.uniform(1.4, 2.8),
                linestyle="-", alpha=random.uniform(0.5, 0.8), zorder=2)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], zorder=3, **_eb_kwargs(c, random.randint(3,8)))
    _set_xticks(ax, series)

def render_connected_dashed(ax, series, palette, bg):
    ls_pool = ["--", "-.", (0,(3,5,1,5)), (0,(5,2))]
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.plot(xs, s["median"], color=c,
                linewidth=random.uniform(1.2, 2.5),
                linestyle=random.choice(ls_pool),
                alpha=random.uniform(0.5, 0.85), zorder=2)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], zorder=3, **_eb_kwargs(c, random.randint(3,8)))
    _set_xticks(ax, series)

def render_connected_smooth(ax, series, palette, bg):
    try:
        from scipy.interpolate import make_interp_spline
        has_scipy = True
    except ImportError:
        has_scipy = False
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        lw  = random.uniform(1.5, 2.8)
        if has_scipy and len(xs) >= 4:
            x_new = np.linspace(min(xs), max(xs), 300)
            spl   = make_interp_spline(xs, s["median"], k=min(3, len(xs)-1))
            ax.plot(x_new, spl(x_new), color=c, linewidth=lw,
                    alpha=random.uniform(0.5, 0.8), zorder=2)
        else:
            ax.plot(xs, s["median"], color=c, linewidth=lw,
                    alpha=random.uniform(0.5, 0.8), zorder=2)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(6, 11),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=0.8,
                    label=s["name"], zorder=3, **_eb_kwargs(c, random.randint(3,8)))
    _set_xticks(ax, series)

def render_connected_ranged(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.plot(xs, s["median"], color=c,
                linewidth=random.uniform(1.8, 3.0),
                alpha=random.uniform(0.6, 0.9), zorder=3)
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt="none", ecolor=c, capsize=0,
                    linewidth=random.uniform(1.0, 2.0), alpha=0.55, zorder=2)
        ax.scatter(xs, s["median"], color=c,
                   s=random.randint(7, 13)**2, zorder=4,
                   marker=random.choice(MARKERS),
                   edgecolors=bg, linewidths=0.8, label=s["name"])
    _set_xticks(ax, series)

def render_connected_markers_prominent(ax, series, palette, bg):
    for idx, s in enumerate(series):
        c   = palette[idx % len(palette)]
        xs  = list(range(len(s["x"])))
        elo = _safe_elo(s["median"], s["min"])
        ehi = _safe_ehi(s["median"], s["max"])
        ax.plot(xs, s["median"], color=c, linewidth=0.8,
                alpha=0.3, zorder=2, linestyle="--")
        ax.errorbar(xs, s["median"], yerr=[elo, ehi],
                    fmt=random.choice(MARKERS), color=c,
                    markersize=random.randint(10, 16),
                    markerfacecolor=c, markeredgecolor=bg, markeredgewidth=1.2,
                    label=s["name"], zorder=3,
                    **_eb_kwargs(c, random.randint(4, 9)))
    _set_xticks(ax, series)

RENDERERS = {
    "points_only":                render_points_only,
    "points_only_jitter":         render_points_only_jitter,
    "points_capsize_heavy":       render_points_capsize_heavy,
    "points_no_cap":              render_points_no_cap,
    "points_asymmetric":          render_points_asymmetric,
    "points_sized":               render_points_sized,
    "points_horizontal":          render_points_horizontal,
    "points_dual_axis":           render_points_dual_axis,
    "connected_line":             render_connected_line,
    "connected_dashed":           render_connected_dashed,
    "connected_smooth":           render_connected_smooth,
    "connected_ranged":           render_connected_ranged,
    "connected_markers_prominent":render_connected_markers_prominent,
}

# ───────────────────────── GENERATORE PRINCIPALE ────────────────────────────

def generate_chart(idx, img_dir, json_dir, used_combos):
    for _ in range(40):
        theme      = random.choice(DATASET_THEMES)
        chart_type = random.choice(CHART_TYPES)
        key = (theme["name"], chart_type)
        if key not in used_combos:
            used_combos.add(key)
            break

    palette    = random.choice(PALETTES)
    random.shuffle(palette)
    bg         = random.choice(BACKGROUNDS)
    text_color = contrast_text(bg)
    grid_style = random.choice(GRID_STYLES)
    font       = random.choice(FONTS)

    max_s    = len(theme.get("series", ["Main"]))
    n_series = random.randint(1, min(3, max_s))
    series, _ = generate_data(theme, n_series=n_series)

    categorical_axis = "y" if chart_type == "points_horizontal" else "x"

    chart_title = theme["name"] if random.random() > 0.15 else None
    data_json   = build_json(theme, series, chart_title, categorical_axis)

    fig, ax = _setup_figure(bg, text_color, font)
    _apply_grid(ax, grid_style)

    RENDERERS[chart_type](ax, series, palette, bg)

    if chart_type != "points_dual_axis":
        _legend(ax, n_series, bg, text_color)

    _set_labels(ax, theme, chart_title, text_color)
    ax.margins(x=random.uniform(0.04, 0.14), y=random.uniform(0.06, 0.18))
    plt.tight_layout(pad=random.uniform(0.9, 2.5))

    base      = f"chart_{idx:03d}_{chart_type}"
    
    # Salvataggio nei percorsi separati
    img_path  = os.path.join(img_dir, base + ".png")
    json_path = os.path.join(json_dir, base + ".json")

    fig.savefig(img_path, dpi=random.choice([100,120,150]),
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    print(f"  [{idx:03d}] ✓  {chart_type:<30} | {theme['name']:<38} | bg {bg}")

# ────────────────────────────── MAIN ────────────────────────────────────────

def main():
    print("=" * 80)
    print("      GENERATORE ERRORPOINT  —  punti espliciti con barre d'errore")
    print("=" * 80)
    while True:
        try:
            n = int(input("\nQuanti grafici vuoi generare? "))
            if n >= 1:
                break
            print("Inserisci un numero >= 1.")
        except ValueError:
            print("Numero intero valido, per favore.")

    # Creazione delle directory specifiche se non esistono
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    print(f"\nDirectory Immagini: {IMG_OUTPUT_DIR}")
    print(f"Directory JSON: {JSON_OUTPUT_DIR}")
    print(f"\nGenerazione di {n} grafici...\n")

    used_combos: set = set()
    for i in range(1, n+1):
        generate_chart(i, IMG_OUTPUT_DIR, JSON_OUTPUT_DIR, used_combos)

    print(f"\n{'='*80}")
    print(f"  ✓  {n} grafici salvati correttamente nelle rispettive cartelle.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()