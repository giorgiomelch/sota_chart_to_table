#!/usr/bin/env python3
"""
Generatore di grafici Box Plot con alta diversificazione.
Nessuna barra piena. Ogni grafico mostra distribuzione con: min, Q1, mediana, Q3, max.
Salva immagine PNG + JSON strutturato.
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
from datetime import datetime

# ─────────────────────────── PALETTE & STILI ────────────────────────────────
PALETTES = [
    ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"],
    ["#6A0572", "#C77DFF", "#9D4EDD", "#7B2FBE", "#48CAE4"],
    ["#1B4332", "#40916C", "#74C69D", "#52B788", "#95D5B2"],
    ["#03045E", "#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"],
    ["#370617", "#9D0208", "#DC2F02", "#F48C06", "#FAA307"],
    ["#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF"],
    ["#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0", "#F72585"],
    ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
    ["#D62828", "#F77F00", "#FCBF49", "#003049", "#119DA4"],
    ["#606C38", "#DDA15E", "#BC6C25", "#3A5A40", "#A3B18A"],
    ["#2B2D42", "#8D99AE", "#EDF2F4", "#EF233C", "#D90429"],
    ["#05668D", "#028090", "#00B4D8", "#F0F3BD", "#02C39A"],
]

BACKGROUNDS = [
    "#FFFFFF", "#F8F9FA", "#0D0D0D", "#1A1A2E", "#0F3460",
    "#16213E", "#F0EAD6", "#E8F4F8", "#1C1C1E", "#FAF0E6",
    "#0A0A0A", "#F5F0EB", "#111827", "#FDFCFB", "#0D1117",
    "#1E1E2E", "#2D2D2D", "#FFFFF0", "#FFF8F0",
]

GRID_STYLES = [
    {"color": "#CCCCCC", "linestyle": "--", "alpha": 0.45},
    {"color": "#555555", "linestyle": ":",  "alpha": 0.40},
    {"color": "#BBBBBB", "linestyle": "-",  "alpha": 0.18},
    {"color": "#88AAFF", "linestyle": "--", "alpha": 0.28},
    {"color": "#FFAA88", "linestyle": ":",  "alpha": 0.28},
    {"color": "#AAAAAA", "linestyle": (0, (5, 10)), "alpha": 0.35},
    None, None,
]

FONTS = ["DejaVu Sans", "monospace", "serif", "DejaVu Serif"]

# ─────────────────────────── DATASET TEMATICI ───────────────────────────────
DATASET_THEMES = [
    {
        "name": "Stipendi per settore lavorativo",
        "x_label": "Settore", "y_label": "Stipendio annuo (k€)",
        "x_type": "categorical",
        "categories": ["Tech", "Finance", "Healthcare", "Education", "Retail", "Legal", "Engineering"],
        "base_range": (22, 120), "spread": (8, 35),
        "series": ["Junior", "Mid", "Senior"],
    },
    {
        "name": "Punteggi esami universitari",
        "x_label": "Materia", "y_label": "Voto (su 30)",
        "x_type": "categorical",
        "categories": ["Matematica", "Fisica", "Chimica", "Informatica", "Statistica", "Economia"],
        "base_range": (18, 30), "spread": (2, 8),
        "series": ["Anno 1", "Anno 2", "Anno 3"],
    },
    {
        "name": "Tempi di risposta server",
        "x_label": "Servizio", "y_label": "Latenza (ms)",
        "x_type": "categorical",
        "categories": ["Auth", "API Gateway", "Database", "Cache", "CDN", "Microservice A", "Microservice B"],
        "base_range": (5, 800), "spread": (10, 250),
        "series": ["Peak", "Off-peak"],
    },
    {
        "name": "Pressione arteriosa per fascia d'età",
        "x_label": "Fascia d'età", "y_label": "Pressione (mmHg)",
        "x_type": "categorical",
        "categories": ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        "base_range": (100, 165), "spread": (5, 22),
        "series": ["Sistolica", "Diastolica"],
    },
    {
        "name": "Temperatura giornaliera per città",
        "x_label": "Città", "y_label": "Temperatura (°C)",
        "x_type": "categorical",
        "categories": ["Milano", "Roma", "Napoli", "Palermo", "Torino", "Bologna", "Venezia"],
        "base_range": (0, 38), "spread": (3, 14),
        "series": ["Inverno", "Estate"],
    },
    {
        "name": "Consumo energetico edifici",
        "x_label": "Tipo edificio", "y_label": "kWh/m²/anno",
        "x_type": "categorical",
        "categories": ["Residenziale A", "Residenziale B", "Uffici", "Industria", "Commerciale", "Ospedale"],
        "base_range": (50, 400), "spread": (20, 100),
        "series": ["Pre-2000", "Post-2010"],
    },
    {
        "name": "Rendimento portfolio investimenti",
        "x_label": "Asset class", "y_label": "Rendimento annuo (%)",
        "x_type": "categorical",
        "categories": ["Azionario", "Obbligazionario", "Real Estate", "Commodities", "Crypto", "ETF"],
        "base_range": (-30, 80), "spread": (5, 40),
        "series": ["2020-2022", "2022-2024"],
    },
    {
        "name": "Indice di qualità dell'aria",
        "x_label": "Città", "y_label": "AQI",
        "x_type": "categorical",
        "categories": ["Pechino", "Delhi", "Londra", "Berlino", "Oslo", "Città del Messico", "Tokyo"],
        "base_range": (10, 300), "spread": (15, 80),
        "series": ["Invernale", "Estivo"],
    },
    {
        "name": "Durata batteria smartphone",
        "x_label": "Modello", "y_label": "Ore di utilizzo",
        "x_type": "categorical",
        "categories": ["Model A", "Model B", "Model C", "Model D", "Model E", "Model F"],
        "base_range": (4, 20), "spread": (1, 5),
        "series": ["Schermo acceso", "Standby parziale"],
    },
    {
        "name": "Velocità vento per regione",
        "x_label": "Regione", "y_label": "Velocità (km/h)",
        "x_type": "categorical",
        "categories": ["Sardegna", "Sicilia", "Puglia", "Liguria", "Toscana", "Lombardia"],
        "base_range": (5, 95), "spread": (4, 30),
        "series": ["Media", "Raffiche"],
    },
    {
        "name": "Tempi di completamento task",
        "x_label": "Task", "y_label": "Minuti",
        "x_type": "categorical",
        "categories": ["Onboarding", "Form A", "Checkout", "Ricerca", "Upload", "Report"],
        "base_range": (1, 60), "spread": (1, 18),
        "series": ["Utenti esperti", "Utenti novizi"],
    },
    {
        "name": "Precipitazioni per stagione",
        "x_label": "Stagione", "y_label": "Precipitazioni (mm)",
        "x_type": "categorical",
        "categories": ["Primavera", "Estate", "Autunno", "Inverno"],
        "base_range": (10, 280), "spread": (10, 80),
        "series": ["Nord Italia", "Centro", "Sud Italia"],
    },
    {
        "name": "Frequenza cardiaca a riposo per sport",
        "x_label": "Disciplina", "y_label": "BPM a riposo",
        "x_type": "categorical",
        "categories": ["Ciclismo", "Nuoto", "Corsa", "Calcio", "Tennis", "Basket", "Sedentario"],
        "base_range": (38, 85), "spread": (3, 15),
        "series": ["Atleti professionisti", "Amatori"],
    },
    {
        "name": "Peso neonati per settimana gestazionale",
        "x_label": "Settimana", "y_label": "Peso (g)",
        "x_type": "categorical",
        "categories": ["34w", "35w", "36w", "37w", "38w", "39w", "40w"],
        "base_range": (1800, 4500), "spread": (150, 500),
        "series": ["Maschi", "Femmine"],
    },
    {
        "name": "Errori per build nel CI/CD",
        "x_label": "Pipeline", "y_label": "N° errori",
        "x_type": "categorical",
        "categories": ["Frontend", "Backend", "DB Migration", "E2E Test", "Deploy", "Security Scan"],
        "base_range": (0, 45), "spread": (1, 15),
        "series": ["main branch", "feature branches"],
    },
    {
        "name": "Indice di soddisfazione clienti",
        "x_label": "Canale", "y_label": "NPS score",
        "x_type": "categorical",
        "categories": ["App mobile", "Web", "Telefono", "Chat", "Email", "In negozio"],
        "base_range": (-20, 90), "spread": (5, 25),
        "series": ["2023", "2024"],
    },
]

# ─────────────────────────── TIPOLOGIE DI BOX CHART ─────────────────────────
CHART_TYPES = [
    # ── Box classici ────────────────────────────────────────────────────────
    "classic_box",             # Box plot tradizionale con whisker
    "classic_box_notched",     # Box con intaglio alla mediana
    "classic_box_no_outlier",  # Box senza outlier mostrati
    "classic_box_flier_star",  # Box con outlier a stella
    # ── Box orizzontali ─────────────────────────────────────────────────────
    "horizontal_box",          # Box plot ruotato orizzontalmente
    "horizontal_box_colored",  # Box orizzontale con fill per IQR
    # ── Varianti stilizzate verticali ───────────────────────────────────────
    "thin_box",                # Box molto sottili, whisker prominenti
    "fat_box",                 # Box molto larghi quasi touching
    "rounded_box",             # Box con angoli arrotondati (FancyBbox)
    "outlined_box",            # Box trasparenti solo con bordo colorato
    "gradient_fill_box",       # Box con fill alpha variabile per IQR/whisker
    # ── Multi-serie ─────────────────────────────────────────────────────────
    "grouped_box",             # Box affiancati per categoria
    "grouped_box_jitter",      # Box affiancati + punti jitter sovrapposti
    "colored_per_series",      # Ogni serie ha palette propria
    # ── Ibridi con punti ────────────────────────────────────────────────────
    "box_with_mean_dot",       # Box + punto per la media
    "box_with_scatter_overlay",# Box + scatter dei punti simulati
    "letter_value_style",      # Stile letter-value: box multipli per code
    # ── Orientazione e layout ───────────────────────────────────────────────
    "violin_box_hybrid",       # Contorno a violino + box interno
]


# ─────────────────────────────── HELPERS ────────────────────────────────────

def is_dark(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return 0.299*r + 0.587*g + 0.114*b < 128

def contrast_text(bg): return "#EEEEEE" if is_dark(bg) else "#111111"

def generate_series_stats(theme, n_series=None):
    """Genera statistiche (min,Q1,median,Q3,max) per ogni (serie, categoria)."""
    low, high = theme["base_range"]
    slo, shi  = theme["spread"]
    cats = theme["categories"]
    pool = theme.get("series", ["Main"])
    if n_series:
        pool = pool[:n_series]

    all_series = []
    for sname in pool:
        off  = random.uniform(-0.15*(high-low), 0.15*(high-low))
        data = []
        for _ in cats:
            center = random.uniform(low+off, high+off)
            spread = random.uniform(slo, shi)
            # Genera 5 statistiche ordinate
            mn  = max(low - spread, center - spread * random.uniform(1.2, 2.5))
            q1  = center - spread * random.uniform(0.4, 0.9)
            med = center + random.uniform(-spread*0.15, spread*0.15)
            q3  = center + spread * random.uniform(0.4, 0.9)
            mx  = center + spread * random.uniform(1.2, 2.5)
            # Garantisci ordine
            vals = sorted([mn, q1, med, q3, mx])
            data.append({
                "min": round(vals[0], 4),
                "q1":  round(vals[1], 4),
                "median": round(vals[2], 4),
                "q3":  round(vals[3], 4),
                "max": round(vals[4], 4),
            })
        all_series.append({"name": sname, "categories": cats, "stats": data})
    return all_series

def build_json(theme, all_series, chart_title, categorical_axis="x"):
    dp = []
    for s in all_series:
        for i, cat in enumerate(s["categories"]):
            st = s["stats"][i]
            dp.append({
                "series_name": s["name"],
                "x_value": str(cat),
                "y_value": {
                    "min":    st["min"],
                    "q1":     st["q1"],
                    "median": st["median"],
                    "q3":     st["q3"],
                    "max":    st["max"],
                },
            })
    return {
        "chart_title":      chart_title,
        "x_axis_label":     theme.get("x_label") or None,
        "y_axis_label":     theme.get("y_label") or None,
        "categorical_axis": categorical_axis,
        "data_points":      dp,
    }

def _setup_figure(bg, text_color, font, w=None, h=None):
    w = w or random.uniform(10, 17)
    h = h or random.uniform(5.5, 9)
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_color(text_color)
        sp.set_linewidth(random.uniform(0.4, 1.6))
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
    if title:
        ax.set_title(title, color=text_color,
                     fontsize=random.randint(12, 19),
                     fontweight=random.choice(["bold","normal"]),
                     pad=random.uniform(8, 22))
    if theme.get("x_label"):
        ax.set_xlabel(theme["x_label"], color=text_color,
                      fontsize=random.randint(9, 13))
    if theme.get("y_label"):
        ax.set_ylabel(theme["y_label"], color=text_color,
                      fontsize=random.randint(9, 13))

def _legend(ax, n_series, bg, text_color, handles=None, labels=None):
    if n_series < 2:
        return
    locs = ["best","upper right","upper left","lower right","lower left"]
    if handles and labels:
        leg = ax.legend(handles, labels, loc=random.choice(locs),
                        fontsize=random.randint(7,11),
                        framealpha=random.uniform(0.25, 0.88),
                        edgecolor=text_color)
    else:
        leg = ax.legend(loc=random.choice(locs),
                        fontsize=random.randint(7,11),
                        framealpha=random.uniform(0.25, 0.88),
                        edgecolor=text_color)
    for t in leg.get_texts(): t.set_color(text_color)
    leg.get_frame().set_facecolor(bg)

def simulate_points(st, n=40):
    """Simula punti grezzi approssimati dalle 5 statistiche."""
    pts  = np.random.uniform(st["min"], st["q1"], n//5)
    pts  = np.append(pts, np.random.uniform(st["q1"], st["median"], n//5*2))
    pts  = np.append(pts, np.random.uniform(st["median"], st["q3"], n//5*2))
    pts  = np.append(pts, np.random.uniform(st["q3"], st["max"], n//5))
    return pts

def _draw_single_box(ax, x, st, color, box_w, bg,
                     show_mean=False, flier_marker="o",
                     rounded=False, outlined_only=False,
                     fill_alpha=0.45, median_lw=2.0,
                     whisker_lw=1.4, cap_lw=1.4, cap_size=0.4,
                     notch=False):
    """Disegna un singolo box da statistiche pre-calcolate."""
    q1, med, q3 = st["q1"], st["median"], st["q3"]
    mn, mx       = st["min"], st["max"]
    iqr          = q3 - q1

    face_alpha = 0.0 if outlined_only else fill_alpha

    if notch and iqr > 0:
        # Intaglio: 1.58*IQR/sqrt(n) — usiamo un valore fisso estetico
        notch_h = iqr * 0.15
        n_lo = med - notch_h
        n_hi = med + notch_h
        hw   = box_w / 2
        in_w = hw * 0.65
        verts = [
            (x - hw,  q1),
            (x - hw,  n_lo),
            (x - in_w, med),
            (x - hw,  n_hi),
            (x - hw,  q3),
            (x + hw,  q3),
            (x + hw,  n_hi),
            (x + in_w, med),
            (x + hw,  n_lo),
            (x + hw,  q1),
            (x - hw,  q1),
        ]
        codes = ([Path.MOVETO] + [Path.LINETO]*9 + [Path.CLOSEPOLY])
        path  = Path(verts, codes)
        patch = PathPatch(path,
                          facecolor=color if not outlined_only else "none",
                          edgecolor=color,
                          alpha=fill_alpha if not outlined_only else 1.0,
                          linewidth=random.uniform(1.2, 2.0))
        ax.add_patch(patch)
    elif rounded:
        style = f"round,pad={random.uniform(0.01,0.04):.3f}"
        patch = FancyBboxPatch(
            (x - box_w/2, q1), box_w, iqr,
            boxstyle=style,
            facecolor=color if not outlined_only else "none",
            edgecolor=color,
            alpha=fill_alpha if not outlined_only else 1.0,
            linewidth=random.uniform(1.2, 2.2))
        ax.add_patch(patch)
    else:
        rect = plt.Rectangle(
            (x - box_w/2, q1), box_w, iqr,
            facecolor=color if not outlined_only else "none",
            edgecolor=color,
            alpha=fill_alpha if not outlined_only else 1.0,
            linewidth=random.uniform(1.0, 2.2))
        ax.add_patch(rect)

    # Mediana
    ax.plot([x - box_w/2, x + box_w/2], [med, med],
            color=color if outlined_only else contrast_text(bg),
            linewidth=median_lw, zorder=5, solid_capstyle="round")

    # Whisker
    ax.plot([x, x], [q3, mx], color=color, linewidth=whisker_lw,
            alpha=0.85, zorder=3)
    ax.plot([x, x], [mn, q1], color=color, linewidth=whisker_lw,
            alpha=0.85, zorder=3)

    # Caps
    cap_hw = box_w * cap_size
    for end in [mn, mx]:
        ax.plot([x - cap_hw, x + cap_hw], [end, end],
                color=color, linewidth=cap_lw, alpha=0.9, zorder=4)

    if show_mean:
        mean_val = (mn + q1 + med + q3 + mx) / 5
        ax.plot(x, mean_val, marker="D", color=color,
                markersize=random.randint(4, 7),
                markeredgecolor=bg, markeredgewidth=0.8, zorder=6)


# ─────────────────────────── RENDERER ────────────────────────────────────────

def render_classic_box(ax, all_series, palette, bg, **kw):
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    spacing = 1.0
    box_w   = random.uniform(0.35, 0.6) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i * spacing + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.35, 0.65),
                             median_lw=random.uniform(1.8, 3.0),
                             whisker_lw=random.uniform(1.0, 2.0))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks([i*spacing for i in range(n_cats)])
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_classic_box_notched(ax, all_series, palette, bg, **kw):
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    box_w   = random.uniform(0.3, 0.55) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             notch=True,
                             fill_alpha=random.uniform(0.3, 0.6),
                             median_lw=random.uniform(2.0, 3.2))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_classic_box_no_outlier(ax, all_series, palette, bg, **kw):
    """Box senza cappelli, whisker terminano senza ornamenti."""
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.35, 0.6) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             cap_size=0.0,
                             fill_alpha=random.uniform(0.35, 0.65),
                             whisker_lw=random.uniform(1.2, 2.2))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_classic_box_flier_star(ax, all_series, palette, bg, **kw):
    """Box con simbolo a stella su min e max."""
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.3, 0.55) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             cap_size=0.0,
                             fill_alpha=random.uniform(0.3, 0.6))
            # Stelle su estremi
            mk = random.choice(["*", "P", "X", "D"])
            ms = random.randint(7, 12)
            ax.plot(x, st["min"], marker=mk, color=c, markersize=ms,
                    markeredgecolor=bg, markeredgewidth=0.6, zorder=6)
            ax.plot(x, st["max"], marker=mk, color=c, markersize=ms,
                    markeredgecolor=bg, markeredgewidth=0.6, zorder=6)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_horizontal_box(ax, all_series, palette, bg, **kw):
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_h  = random.uniform(0.25, 0.45) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_h/2, (n_s-1)*box_h/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            y    = i + off
            q1, med, q3 = st["q1"], st["median"], st["q3"]
            mn, mx      = st["min"], st["max"]
            # Box
            rect = plt.Rectangle((q1, y - box_h/2), q3-q1, box_h,
                                  facecolor=c, edgecolor=c,
                                  alpha=random.uniform(0.35, 0.65),
                                  linewidth=1.5)
            ax.add_patch(rect)
            ax.plot([med, med], [y - box_h/2, y + box_h/2],
                    color=contrast_text(bg), linewidth=2.5, zorder=5)
            # Whisker orizzontali
            ax.plot([mn, q1], [y, y], color=c, linewidth=1.5, alpha=0.8)
            ax.plot([q3, mx], [y, y], color=c, linewidth=1.5, alpha=0.8)
            cap_h = box_h * 0.45
            for end in [mn, mx]:
                ax.plot([end, end], [y - cap_h, y + cap_h],
                        color=c, linewidth=1.5, alpha=0.9)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(all_series[0]["categories"])


def render_horizontal_box_colored(ax, all_series, palette, bg, **kw):
    """Orizzontale con IQR colorato e whisker sottili."""
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_h  = random.uniform(0.3, 0.5) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_h*1.4, (n_s-1)*box_h*1.4, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            y    = i + off
            q1, med, q3 = st["q1"], st["median"], st["q3"]
            mn, mx      = st["min"], st["max"]
            iqr  = q3 - q1
            # Fill gradiente Q1-Q3
            n_bands = 12
            for b in range(n_bands):
                frac    = b / n_bands
                bq1_val = q1 + frac * iqr
                bw      = iqr / n_bands
                alp     = 0.15 + 0.5 * (1 - abs(frac - 0.5) * 2)
                ax.add_patch(plt.Rectangle(
                    (bq1_val, y - box_h/2), bw, box_h,
                    facecolor=c, edgecolor="none", alpha=alp))
            ax.add_patch(plt.Rectangle(
                (q1, y - box_h/2), iqr, box_h,
                facecolor="none", edgecolor=c, linewidth=1.5))
            ax.plot([med, med], [y - box_h/2, y + box_h/2],
                    color=contrast_text(bg), linewidth=2.8, zorder=5)
            ax.plot([mn, q1], [y, y], color=c, linewidth=1.2, alpha=0.7)
            ax.plot([q3, mx], [y, y], color=c, linewidth=1.2, alpha=0.7)
            cap_h = box_h * 0.4
            for end in [mn, mx]:
                ax.plot([end, end], [y - cap_h, y + cap_h],
                        color=c, linewidth=1.5, alpha=0.8)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(all_series[0]["categories"])


def render_thin_box(ax, all_series, palette, bg, **kw):
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.06, 0.14)
    offsets = np.linspace(-(n_s-1)*0.22, (n_s-1)*0.22, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.55, 0.85),
                             cap_size=random.uniform(0.6, 1.2),
                             whisker_lw=random.uniform(1.5, 2.5),
                             cap_lw=random.uniform(1.5, 2.5))
        ax.plot([], [], color=c, linewidth=4, alpha=0.6, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_fat_box(ax, all_series, palette, bg, **kw):
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.6, 0.82) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.2, 0.5),
                             cap_size=0.35,
                             median_lw=random.uniform(2.5, 4.0))
        ax.plot([], [], color=c, linewidth=6, alpha=0.4, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_rounded_box(ax, all_series, palette, bg, **kw):
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.3, 0.55) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             rounded=True,
                             fill_alpha=random.uniform(0.3, 0.6),
                             median_lw=random.uniform(2.0, 3.5))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_outlined_box(ax, all_series, palette, bg, **kw):
    """Box trasparenti, solo contorno."""
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.35, 0.6) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             outlined_only=True,
                             median_lw=random.uniform(2.2, 3.5),
                             cap_size=random.uniform(0.3, 0.5))
        ax.plot([], [], color=c, linewidth=3, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_gradient_fill_box(ax, all_series, palette, bg, **kw):
    """Fill progressivo più scuro verso la mediana."""
    n_cats = len(all_series[0]["categories"])
    n_s    = len(all_series)
    box_w  = random.uniform(0.32, 0.58) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]
    n_bands = 16

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x    = i + off
            q1, med, q3 = st["q1"], st["median"], st["q3"]
            mn, mx      = st["min"], st["max"]
            iqr  = max(q3 - q1, 1e-6)
            # Bands Q1→Q3
            for b in range(n_bands):
                frac = b / n_bands
                yb   = q1 + frac * iqr
                bh   = iqr / n_bands
                alp  = 0.1 + 0.55 * (1 - abs(frac - 0.5) * 2)
                ax.add_patch(plt.Rectangle(
                    (x - box_w/2, yb), box_w, bh,
                    facecolor=c, edgecolor="none", alpha=alp))
            ax.add_patch(plt.Rectangle(
                (x - box_w/2, q1), box_w, iqr,
                facecolor="none", edgecolor=c, linewidth=1.4))
            ax.plot([x - box_w/2, x + box_w/2], [med, med],
                    color=contrast_text(bg), linewidth=2.5, zorder=5)
            # Whisker
            for seg in [(mn, q1), (q3, mx)]:
                ax.plot([x, x], seg, color=c, linewidth=1.4, alpha=0.8)
            cap_hw = box_w * 0.38
            for end in [mn, mx]:
                ax.plot([x - cap_hw, x + cap_hw], [end, end],
                        color=c, linewidth=1.5, alpha=0.9)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_grouped_box(ax, all_series, palette, bg, **kw):
    """Box strettamente affiancati per gruppo."""
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    group_w = 0.8
    box_w   = group_w / max(n_s, 1)
    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = (idx - (n_s-1)/2) * box_w
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w*0.88, bg,
                             fill_alpha=random.uniform(0.35, 0.65),
                             median_lw=random.uniform(1.8, 3.0))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_grouped_box_jitter(ax, all_series, palette, bg, **kw):
    """Box + punti jitter sovrapposti."""
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    group_w = 0.75
    box_w   = group_w / max(n_s, 1)

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = (idx - (n_s-1)/2) * box_w
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w*0.82, bg,
                             fill_alpha=random.uniform(0.2, 0.45),
                             median_lw=random.uniform(1.8, 2.8))
            pts = simulate_points(st, n=random.randint(25, 50))
            jitter = np.random.uniform(-box_w*0.22, box_w*0.22, len(pts))
            ax.scatter(x + jitter, pts, color=c, alpha=random.uniform(0.3, 0.55),
                       s=random.randint(8, 20), zorder=4)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_colored_per_series(ax, all_series, palette, bg, **kw):
    """Ogni serie usa un colore esclusivo anche per whisker e caps."""
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    box_w   = random.uniform(0.3, 0.52) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.4, 0.7),
                             median_lw=random.uniform(2.0, 3.5),
                             whisker_lw=random.uniform(1.5, 2.5),
                             cap_lw=random.uniform(1.5, 2.5))
        ax.plot([], [], color=c, linewidth=8, alpha=0.6, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_box_with_mean_dot(ax, all_series, palette, bg, **kw):
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    box_w   = random.uniform(0.32, 0.55) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.3, 0.6),
                             show_mean=True)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])
        ax.plot([], [], marker="D", color=c, markersize=6,
                markeredgecolor=bg, label=f"{s['name']} (media)")

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_box_with_scatter_overlay(ax, all_series, palette, bg, **kw):
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    box_w   = random.uniform(0.3, 0.5) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x = i + off
            _draw_single_box(ax, x, st, c, box_w, bg,
                             fill_alpha=random.uniform(0.18, 0.4))
            pts    = simulate_points(st, n=random.randint(30, 60))
            jitter = np.random.uniform(-box_w*0.35, box_w*0.35, len(pts))
            ax.scatter(x + jitter, pts, color=c,
                       alpha=random.uniform(0.35, 0.6),
                       s=random.randint(10, 25), zorder=5,
                       edgecolors="none")
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


def render_letter_value_style(ax, all_series, palette, bg, **kw):
    """Stile letter-value: box concentrici che si restringono verso le code."""
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    base_w  = random.uniform(0.35, 0.55) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*base_w/2, (n_s-1)*base_w/2, n_s) if n_s > 1 else [0]

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x    = i + off
            q1, med, q3 = st["q1"], st["median"], st["q3"]
            mn, mx      = st["min"], st["max"]
            # Box centrale (IQR) — più largo
            _draw_single_box(ax, x, st, c, base_w, bg,
                             fill_alpha=0.5, cap_size=0.0)
            # Whisker mid (Q1→min, Q3→max) come box più stretto
            mid_lo = (mn + q1) / 2
            mid_hi = (q3 + mx) / 2
            for (y0, y1), w_frac in [((mn, q1), 0.55), ((q3, mx), 0.55)]:
                ax.add_patch(plt.Rectangle(
                    (x - base_w*w_frac/2, y0),
                    base_w*w_frac, y1-y0,
                    facecolor=c, edgecolor=c,
                    alpha=0.25, linewidth=1.0))
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))



def render_violin_box_hybrid(ax, all_series, palette, bg, **kw):
    """Contorno a campana simulato + box interno."""
    n_cats  = len(all_series[0]["categories"])
    n_s     = len(all_series)
    box_w   = random.uniform(0.22, 0.38) / max(n_s, 1)
    offsets = np.linspace(-(n_s-1)*box_w/2, (n_s-1)*box_w/2, n_s) if n_s > 1 else [0]
    max_vw  = box_w * 1.8

    for idx, s in enumerate(all_series):
        c   = palette[idx % len(palette)]
        off = offsets[idx]
        for i, st in enumerate(s["stats"]):
            x    = i + off
            mn, q1, med, q3, mx = (st["min"], st["q1"], st["median"],
                                    st["q3"], st["max"])
            rng = max(mx - mn, 1e-6)
            # Campana: profilo gaussiano approssimato
            y_vals = np.linspace(mn, mx, 60)
            sigma  = rng / 5
            mu     = med
            density = np.exp(-0.5 * ((y_vals - mu)/sigma)**2)
            density = density / density.max() * max_vw / 2
            # Contorno violino sinistro e destro
            left  = [x - d for d in density]
            right = [x + d for d in density]
            ax.fill_betweenx(y_vals, left, right, color=c,
                             alpha=random.uniform(0.15, 0.3))
            ax.plot(left,  y_vals, color=c, linewidth=0.8, alpha=0.5)
            ax.plot(right, y_vals, color=c, linewidth=0.8, alpha=0.5)
            # Box interno
            _draw_single_box(ax, x, st, c, box_w*0.7, bg,
                             fill_alpha=0.6, cap_size=0.25,
                             median_lw=2.5)
        ax.plot([], [], color=c, linewidth=6, alpha=0.5, label=s["name"])

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(all_series[0]["categories"],
                       rotation=random.choice([0, 30, 45]))


RENDERERS = {
    "classic_box":               render_classic_box,
    "classic_box_notched":       render_classic_box_notched,
    "classic_box_no_outlier":    render_classic_box_no_outlier,
    "classic_box_flier_star":    render_classic_box_flier_star,
    "horizontal_box":            render_horizontal_box,
    "horizontal_box_colored":    render_horizontal_box_colored,
    "thin_box":                  render_thin_box,
    "fat_box":                   render_fat_box,
    "rounded_box":               render_rounded_box,
    "outlined_box":              render_outlined_box,
    "gradient_fill_box":         render_gradient_fill_box,
    "grouped_box":               render_grouped_box,
    "grouped_box_jitter":        render_grouped_box_jitter,
    "colored_per_series":        render_colored_per_series,
    "box_with_mean_dot":         render_box_with_mean_dot,
    "box_with_scatter_overlay":  render_box_with_scatter_overlay,
    "letter_value_style":        render_letter_value_style,
    "violin_box_hybrid":         render_violin_box_hybrid,
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

    # Per dual_axis limitiamo a 2 serie
    if chart_type == "dual_axis_box":
        n_series = min(n_series, 2)

    all_series = generate_series_stats(theme, n_series=n_series)

    # Tipi orizzontali: le categorie stanno sull'asse Y
    HORIZONTAL_TYPES = {"horizontal_box", "horizontal_box_colored"}
    categorical_axis = "y" if chart_type in HORIZONTAL_TYPES else "x"

    chart_title = theme["name"] if random.random() > 0.15 else None
    data_json   = build_json(theme, all_series, chart_title, categorical_axis)

    fig, ax = _setup_figure(bg, text_color, font)
    _apply_grid(ax, grid_style)

    RENDERERS[chart_type](ax, all_series, palette, bg)

    if chart_type != "dual_axis_box":
        _legend(ax, n_series, bg, text_color)

    _set_labels(ax, theme, chart_title, text_color)
    ax.margins(x=random.uniform(0.04, 0.14), y=random.uniform(0.06, 0.16))
    plt.tight_layout(pad=random.uniform(0.9, 2.5))

    base      = f"chart_{idx:03d}_{chart_type}"
    img_path  = os.path.join(img_dir, base + ".png")
    json_path = os.path.join(json_dir, base + ".json")

    fig.savefig(img_path, dpi=random.choice([100, 120, 150]),
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    print(f"  [{idx:03d}] ✓  {chart_type:<30} | {theme['name']:<40} | bg {bg}")


# ────────────────────────────── MAIN ────────────────────────────────────────


IMG_OUTPUT_DIR = "data/synthetic/box"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/box"
def main():
    print("=" * 82)
    print("         GENERATORE BOX PLOT  —  max diversificazione template + dati")
    print("=" * 82)
    while True:
        try:
            n = int(input("\nQuanti grafici vuoi generare? "))
            if n >= 1:
                break
            print("Inserisci un numero >= 1.")
        except ValueError:
            print("Numero intero valido, per favore.")

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
