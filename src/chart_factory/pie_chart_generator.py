"""
Pie Chart Generator
Genera grafici a torta diversificati con immagini PNG e JSON corrispondenti.

Sottotipologie supportate:
  1. classic_pie         – Torta classica con etichette esterne
  2. exploded_pie        – Fetta esplosa (evidenziata)
  3. donut               – Anello (donut chart)
  4. donut_nested        – Anelli concentrici (nested donut)
  5. semi_donut          – Semicerchio (gauge-style)
  6. multi_pie           – Griglia di torte per confronto
  7. rose_nightingale    – Rosa/Nightingale (raggi proporzionali)
  8. waffle_pie          – Waffle chart (griglia 10×10)
  9. annotated_donut     – Donut con annotazioni centrali e callout
 10. pie_with_bar        – Torta + barra di dettaglio della fetta principale
"""

import os, json, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore")

IMG_OUTPUT_DIR = "data/synthetic/pie"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/pie"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
#  DATASET TEMPLATES  (10 temi distinti)
# ══════════════════════════════════════════════════════

DATASET_TEMPLATES = [
    {
        "theme": "Quota di Mercato Smartphone",
        "x_label": "Produttore", "y_label": "Quota (%)",
        "categories": ["Samsung", "Apple", "Xiaomi", "OPPO", "Vivo", "Altri"],
        "weights":    [3, 3, 2, 2, 1.5, 2],
        "value_range": (3, 35),
        "series": ["Main"],
    },
    {
        "theme": "Distribuzione Budget Aziendale",
        "x_label": "Voce di Costo", "y_label": "Budget (%)",
        "categories": ["R&D", "Marketing", "Personale", "IT", "Operations", "Legale", "Altro"],
        "weights":    [2, 2, 4, 1.5, 2, 1, 1],
        "value_range": (3, 40),
        "series": ["Main"],
    },
    {
        "theme": "Fonti di Energia Rinnovabile",
        "x_label": "Fonte", "y_label": "Produzione (%)",
        "categories": ["Solare", "Eolico", "Idroelettrico", "Biomasse", "Geotermico", "Maree"],
        "weights":    [3, 3, 2.5, 1.5, 1, 0.5],
        "value_range": (2, 40),
        "series": ["Main"],
    },
    {
        "theme": "Composizione Portfolio Investimenti",
        "x_label": "Asset Class", "y_label": "Allocazione (%)",
        "categories": ["Azioni", "Obbligazioni", "Immobili", "Commodities", "Cripto", "Liquidità"],
        "weights":    [3, 2.5, 2, 1.5, 1, 1.5],
        "value_range": (5, 45),
        "series": ["Main"],
    },
    {
        "theme": "Traffico Web per Canale",
        "x_label": "Canale", "y_label": "Visite (%)",
        "categories": ["Organico", "Diretto", "Social", "Email", "Referral", "PPC"],
        "weights":    [3.5, 2, 2.5, 1.5, 1, 2],
        "value_range": (4, 38),
        "series": ["Main"],
    },
    {
        "theme": "Distribuzione Età della Popolazione",
        "x_label": "Fascia d'Età", "y_label": "Popolazione (%)",
        "categories": ["0-14", "15-24", "25-39", "40-54", "55-69", "70+"],
        "weights":    [2, 1.5, 2.5, 2.5, 2, 1.5],
        "value_range": (8, 30),
        "series": ["Main"],
    },
    {
        "theme": "Cause di Insoddisfazione Cliente",
        "x_label": "Causa", "y_label": "Frequenza (%)",
        "categories": ["Tempi di Consegna", "Qualità Prodotto", "Assistenza", "Prezzo", "UX", "Altro"],
        "weights":    [3, 2.5, 2, 2, 1.5, 1],
        "value_range": (5, 35),
        "series": ["Main"],
    },
    {
        "theme": "Utilizzo Dispositivi per Accesso Internet",
        "x_label": "Dispositivo", "y_label": "Sessioni (%)",
        "categories": ["Smartphone", "Laptop", "Desktop", "Tablet", "Smart TV", "Wearable"],
        "weights":    [4, 2.5, 2, 1.5, 1, 0.5],
        "value_range": (2, 50),
        "series": ["Main"],
    },
    {
        "theme": "Generi Musicali più Ascoltati",
        "x_label": "Genere", "y_label": "Stream (%)",
        "categories": ["Pop", "Hip-Hop", "Rock", "Elettronica", "R&B", "Classica", "Jazz"],
        "weights":    [3, 3, 2, 2, 2, 0.8, 0.7],
        "value_range": (2, 32),
        "series": ["Main"],
    },
    {
        "theme": "Spese Familiari Mensili",
        "x_label": "Categoria", "y_label": "Spesa (%)",
        "categories": ["Abitazione", "Cibo", "Trasporti", "Salute", "Svago", "Abbigliamento", "Risparmio"],
        "weights":    [3, 2.5, 2, 1.5, 1.5, 1, 1.5],
        "value_range": (4, 38),
        "series": ["Main"],
    },
]

# ══════════════════════════════════════════════════════
#  THEMES  (10 temi visivi distinti)
# ══════════════════════════════════════════════════════

CHART_THEMES = [
    {
        "name": "corporate_blue",
        "bg": "#FFFFFF", "fig_bg": "#F4F7FB",
        "title": "#1A237E", "label": "#37474F", "tick": "#546E7A",
        "palette": ["#1565C0","#1E88E5","#42A5F5","#90CAF9","#BBDEFB","#0D47A1","#1976D2"],
        "text_center": "#1A237E", "edge": "#FFFFFF",
    },
    {
        "name": "dark_pro",
        "bg": "#1E1E2E", "fig_bg": "#181825",
        "title": "#CDD6F4", "label": "#BAC2DE", "tick": "#A6ADC8",
        "palette": ["#89B4FA","#A6E3A1","#FAB387","#F38BA8","#CBA6F7","#94E2D5","#F9E2AF"],
        "text_center": "#CDD6F4", "edge": "#1E1E2E",
    },
    {
        "name": "vibrant_pop",
        "bg": "#FFFFFF", "fig_bg": "#F8F9FA",
        "title": "#212529", "label": "#495057", "tick": "#6C757D",
        "palette": ["#E63946","#F4A261","#2A9D8F","#264653","#E9C46A","#8338EC","#3A86FF"],
        "text_center": "#212529", "edge": "#FFFFFF",
    },
    {
        "name": "pastel_dream",
        "bg": "#FEFEFE", "fig_bg": "#F9F4FF",
        "title": "#4A4A4A", "label": "#6A6A6A", "tick": "#8A8A8A",
        "palette": ["#FFB3BA","#FFDFBA","#FFFFBA","#BAFFC9","#BAE1FF","#E8BAFF","#FFD6E7"],
        "text_center": "#555555", "edge": "#FFFFFF",
    },
    {
        "name": "neon_dark",
        "bg": "#0D0D0D", "fig_bg": "#050505",
        "title": "#00FF88", "label": "#CCCCCC", "tick": "#888888",
        "palette": ["#00FF88","#FF006E","#3A86FF","#FFBE0B","#FB5607","#8338EC","#00F5D4"],
        "text_center": "#00FF88", "edge": "#0D0D0D",
    },
    {
        "name": "earth_tones",
        "bg": "#FFF8F0", "fig_bg": "#FDF0E0",
        "title": "#5D3A1A", "label": "#7B4F2E", "tick": "#8B6347",
        "palette": ["#C84B31","#E07B39","#F4A460","#DEB887","#8FBC8F","#556B2F","#A0522D"],
        "text_center": "#5D3A1A", "edge": "#FFF8F0",
    },
    {
        "name": "ocean_depth",
        "bg": "#EAF4FB", "fig_bg": "#D6EAF8",
        "title": "#0B3D6B", "label": "#154360", "tick": "#1A5276",
        "palette": ["#0B3D6B","#1A6E9C","#2E86C1","#5DADE2","#85C1E9","#AED6F1","#D6EAF8"],
        "text_center": "#0B3D6B", "edge": "#EAF4FB",
    },
    {
        "name": "sunset_fire",
        "bg": "#1A0005", "fig_bg": "#0D0002",
        "title": "#FF6B35", "label": "#FFB347", "tick": "#FF8C42",
        "palette": ["#FF0000","#FF4500","#FF6B35","#FF8C42","#FFA500","#FFD700","#FFEC8B"],
        "text_center": "#FFD700", "edge": "#1A0005",
    },
    {
        "name": "mint_fresh",
        "bg": "#F0FFF4", "fig_bg": "#E8F8F0",
        "title": "#1B5E20", "label": "#2E7D32", "tick": "#388E3C",
        "palette": ["#00695C","#00897B","#26A69A","#4DB6AC","#80CBC4","#B2DFDB","#E0F2F1"],
        "text_center": "#1B5E20", "edge": "#F0FFF4",
    },
    {
        "name": "mono_editorial",
        "bg": "#FFFFFF", "fig_bg": "#F5F5F5",
        "title": "#000000", "label": "#333333", "tick": "#666666",
        "palette": ["#000000","#1A1A1A","#333333","#4D4D4D","#666666","#808080","#999999"],
        "text_center": "#000000", "edge": "#FFFFFF",
    },
]

SUBTYPES = [
    "classic_pie", "exploded_pie", "donut", "donut_nested",
    "semi_donut", "multi_pie", "rose_nightingale",
    "annotated_donut",
]

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def weighted_values(tmpl, n=None):
    cats = tmpl["categories"]
    if n is not None:
        cats = cats[:n]
    w = tmpl["weights"][:len(cats)]
    lo, hi = tmpl["value_range"]
    raw = [random.uniform(lo * wi, hi * wi) for wi in w]
    total = sum(raw)
    return cats, [round(v / total * 100, 2) for v in raw]

def apply_fig_theme(fig, t):
    fig.patch.set_facecolor(t["fig_bg"])

def title_style(ax, title, t):
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=t["title"], pad=16)

def mk_json(chart_title, x_label, y_label, dp):
    return {
        "chart_title": chart_title,
        "data_points": dp,
    }

def palette_cycle(t, n):
    p = t["palette"]
    return [p[i % len(p)] for i in range(n)]

def pct_fmt(pct):
    return f"{pct:.1f}%" if pct > 4 else ""

def autopct_fn(pct):
    return pct_fmt(pct)

# ══════════════════════════════════════════════════════
#  RENDERERS
# ══════════════════════════════════════════════════════

# ── 1. CLASSIC PIE ───────────────────────────────────
def render_classic_pie(tmpl, t, _):
    cats, vals = weighted_values(tmpl)
    colors = palette_cycle(t, len(cats))

    fig, ax = plt.subplots(figsize=(9, 7))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    wedges, texts, autotexts = ax.pie(
        vals, labels=None, autopct=autopct_fn,
        colors=colors, startangle=random.randint(0, 360),
        pctdistance=0.78, wedgeprops=dict(edgecolor=t["edge"], linewidth=1.5),
    )
    for at in autotexts:
        at.set_color(t["bg"]); at.set_fontsize(9); at.set_fontweight("bold")

    ax.legend(wedges, [f"{c}  ({v:.1f}%)" for c, v in zip(cats, vals)],
              loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=9,
              labelcolor=t["label"])
    title_style(ax, tmpl["theme"], t)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], dp)


# ── 2. EXPLODED PIE ──────────────────────────────────
def render_exploded_pie(tmpl, t, _):
    cats, vals = weighted_values(tmpl)
    colors = palette_cycle(t, len(cats))

    # Explode the largest slice
    max_idx = vals.index(max(vals))
    explode = [0.0] * len(cats)
    explode[max_idx] = 0.12

    fig, ax = plt.subplots(figsize=(9, 7))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    wedges, texts, autotexts = ax.pie(
        vals, explode=explode, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=colors, startangle=random.randint(0, 360),
        pctdistance=0.80,
        wedgeprops=dict(edgecolor=t["edge"], linewidth=1.8),
        shadow=True,
    )
    for at in autotexts:
        at.set_color(t["bg"]); at.set_fontsize(8.5); at.set_fontweight("bold")

    # Annotate exploded slice — compute (x, y) from mid-angle
    mid_angle_rad = math.radians((wedges[max_idx].theta1 + wedges[max_idx].theta2) / 2)
    r_tip = 0.72 + explode[max_idx]   # tip of the exploded wedge
    xy_tip = (r_tip * math.cos(mid_angle_rad), r_tip * math.sin(mid_angle_rad))
    xy_txt = (1.35 * math.cos(mid_angle_rad), 1.35 * math.sin(mid_angle_rad))
    ha = "left" if xy_txt[0] >= 0 else "right"
    ax.annotate(f"▲ {cats[max_idx]}\n{vals[max_idx]:.1f}%",
                xy=xy_tip, xytext=xy_txt, textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=t["title"], lw=1.3),
                fontsize=10, color=t["title"], fontweight="bold", ha=ha, va="center")

    ax.legend(wedges, cats, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=9, labelcolor=t["label"])
    title_style(ax, tmpl["theme"] + " (Esplosa)", t)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Esplosa)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 3. DONUT ─────────────────────────────────────────
def render_donut(tmpl, t, _):
    cats, vals = weighted_values(tmpl)
    colors = palette_cycle(t, len(cats))
    hole_ratio = random.choice([0.50, 0.55, 0.60, 0.65])

    fig, ax = plt.subplots(figsize=(9, 7))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    wedges, _, autotexts = ax.pie(
        vals, labels=None, autopct=autopct_fn,
        colors=colors, startangle=random.randint(0, 360),
        pctdistance=0.82,
        wedgeprops=dict(width=1 - hole_ratio, edgecolor=t["edge"], linewidth=1.5),
    )
    for at in autotexts:
        at.set_color(t["bg"]); at.set_fontsize(8.5); at.set_fontweight("bold")

    # Centre label
    total_label = f"{sum(vals):.0f}" if random.random() > 0.5 else "TOTALE"
    ax.text(0, 0.08, total_label, ha="center", va="center",
            fontsize=18, fontweight="bold", color=t["text_center"])
    ax.text(0, -0.18, tmpl["y_label"], ha="center", va="center",
            fontsize=9, color=t["label"])

    ax.legend(wedges, [f"{c} ({v:.1f}%)" for c, v in zip(cats, vals)],
              loc="lower center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=9, labelcolor=t["label"])
    title_style(ax, tmpl["theme"] + " (Donut)", t)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Donut)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 4. NESTED DONUT ──────────────────────────────────
def render_donut_nested(tmpl, t, _):
    # Two rings: outer = full dataset, inner = aggregated macro-groups
    cats_outer, vals_outer = weighted_values(tmpl)
    n_inner = random.randint(2, 3)
    # Build inner by summing consecutive slices
    inner_cats, inner_vals = [], []
    step = max(1, len(cats_outer) // n_inner)
    for i in range(n_inner):
        chunk = cats_outer[i*step:(i+1)*step]
        v = sum(vals_outer[i*step:(i+1)*step])
        inner_cats.append(f"Gruppo {i+1}")
        inner_vals.append(round(v, 2))

    palette_outer = palette_cycle(t, len(cats_outer))
    palette_inner = [t["palette"][i * 2 % len(t["palette"])] for i in range(n_inner)]

    fig, ax = plt.subplots(figsize=(9, 8))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    ax.pie(vals_outer, radius=1.0, colors=palette_outer,
           startangle=90, pctdistance=0.88,
           autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
           wedgeprops=dict(width=0.38, edgecolor=t["edge"], linewidth=1.2))
    ax.pie(inner_vals, radius=0.58, colors=palette_inner,
           startangle=90,
           autopct=lambda p: f"{p:.0f}%",
           pctdistance=0.70,
           wedgeprops=dict(width=0.38, edgecolor=t["edge"], linewidth=1.2))

    ax.text(0, 0, "NESTED", ha="center", va="center",
            fontsize=11, fontweight="bold", color=t["text_center"])

    outer_patches = [mpatches.Patch(color=c, label=f"{la} ({v:.1f}%)")
                     for c, la, v in zip(palette_outer, cats_outer, vals_outer)]
    inner_patches = [mpatches.Patch(color=c, label=f"{la}")
                     for c, la in zip(palette_inner, inner_cats)]
    ax.legend(handles=outer_patches + inner_patches,
              loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=t["label"])
    title_style(ax, tmpl["theme"] + " (Nested Donut)", t)
    plt.tight_layout()

    dp = ([{"series_name": "Esterno", "x_value": c, "y_value": v}
            for c, v in zip(cats_outer, vals_outer)] +
          [{"series_name": "Interno", "x_value": c, "y_value": v}
            for c, v in zip(inner_cats, inner_vals)])
    return fig, mk_json(tmpl["theme"] + " (Nested Donut)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 5. SEMI-DONUT (Gauge) ────────────────────────────
def render_semi_donut(tmpl, t, _):
    cats, vals = weighted_values(tmpl, n=random.randint(3, 5))
    colors = palette_cycle(t, len(cats))

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    # Only upper half
    wedges, _, autotexts = ax.pie(
        vals + [sum(vals)],          # mirror slice to close semicircle
        colors=colors + [t["bg"]],
        startangle=180,
        pctdistance=0.82,
        autopct=lambda p: f"{p:.1f}%" if p < 45 and p > 4 else "",
        wedgeprops=dict(width=0.42, edgecolor=t["edge"], linewidth=1.5),
        counterclock=False,
    )
    for at in autotexts:
        at.set_color(t["bg"]); at.set_fontsize(9); at.set_fontweight("bold")

    ax.set_ylim(-0.15, 1.05)

    # Centre annotation
    ax.text(0, -0.05, f"{max(vals):.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=t["text_center"])
    ax.text(0, -0.25, f"max · {cats[vals.index(max(vals))]}",
            ha="center", va="center", fontsize=10, color=t["label"])

    ax.legend(wedges[:len(cats)],
              [f"{c}  {v:.1f}%" for c, v in zip(cats, vals)],
              loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(cats), frameon=False, fontsize=9, labelcolor=t["label"])
    title_style(ax, tmpl["theme"] + " (Semi-Donut)", t)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Semi-Donut)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 6. MULTI PIE ─────────────────────────────────────
def render_multi_pie(tmpl, t, _):
    # 4 mini-torte, ognuna con dati leggermente diversi (simulano anni/regioni)
    labels = ["2021", "2022", "2023", "2024"]
    n_cats = min(5, len(tmpl["categories"]))
    cats = tmpl["categories"][:n_cats]
    colors = palette_cycle(t, n_cats)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    apply_fig_theme(fig, t)
    fig.suptitle(tmpl["theme"] + " — Confronto Annuale",
                 fontsize=13, fontweight="bold", color=t["title"], y=1.0)

    all_dp = []
    for ax, yr in zip(axes, labels):
        ax.set_facecolor(t["bg"])
        _, vals = weighted_values(tmpl, n=n_cats)
        wedges, _, autotexts = ax.pie(
            vals, colors=colors, startangle=random.randint(0, 360),
            autopct=lambda p: f"{p:.0f}%" if p > 7 else "",
            pctdistance=0.75,
            wedgeprops=dict(edgecolor=t["edge"], linewidth=1.2),
        )
        for at in autotexts:
            at.set_color(t["bg"]); at.set_fontsize(8); at.set_fontweight("bold")
        ax.set_title(yr, fontsize=11, fontweight="bold", color=t["title"])
        for c, v in zip(cats, vals):
            all_dp.append({"series_name": yr, "x_value": c, "y_value": v})

    patches = [mpatches.Patch(color=c, label=la) for c, la in zip(colors, cats)]
    fig.legend(handles=patches, loc="lower center", ncol=n_cats,
               frameon=False, fontsize=9, labelcolor=t["label"],
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"] + " (Confronto Annuale)",
                        tmpl["x_label"], tmpl["y_label"], all_dp)


# ── 7. ROSE / NIGHTINGALE ────────────────────────────
def render_rose_nightingale(tmpl, t, _):
    cats, vals = weighted_values(tmpl)
    n = len(cats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    colors = palette_cycle(t, n)
    radii = np.array(vals)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    width = 2 * np.pi / n * 0.85
    bars = ax.bar(angles, radii, width=width, bottom=0.0,
                  color=colors, edgecolor=t["edge"], linewidth=1.2, alpha=0.92)

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=9, color=t["label"])
    ax.yaxis.set_visible(False)
    ax.grid(color=t["fig_bg"], linewidth=0.6)
    ax.spines["polar"].set_visible(False)

    # Value labels
    for angle, r, cat, val in zip(angles, radii, cats, vals):
        ax.text(angle, r + max(radii) * 0.07, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7.5,
                color=t["tick"], fontweight="bold")

    ax.set_title(tmpl["theme"] + " (Rose Chart)",
                 fontsize=13, fontweight="bold", color=t["title"], pad=22)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Rose Chart)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 8. WAFFLE PIE ────────────────────────────────────
def render_waffle_pie(tmpl, t, _):
    cats, vals = weighted_values(tmpl, n=random.randint(4, 6))
    colors = palette_cycle(t, len(cats))

    # Build 10×10 grid
    grid = np.zeros(100, dtype=int)
    idx = 0
    for i, v in enumerate(vals):
        count = round(v)
        grid[idx:idx + count] = i
        idx += count
    grid = grid[:100]
    np.random.shuffle(grid)
    grid = grid.reshape(10, 10)

    fig, ax = plt.subplots(figsize=(9, 8))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    for row in range(10):
        for col in range(10):
            cat_idx = grid[row, col]
            color = colors[cat_idx] if cat_idx < len(colors) else t["palette"][-1]
            rect = FancyBboxPatch((col, 9 - row), 0.88, 0.88,
                                  boxstyle="round,pad=0.04",
                                  facecolor=color, edgecolor=t["bg"], linewidth=1.5)
            ax.add_patch(rect)

    ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.1, 10.1)
    ax.set_aspect("equal"); ax.axis("off")

    patches = [mpatches.Patch(color=colors[i], label=f"{cats[i]}  {vals[i]:.1f}%")
               for i in range(len(cats))]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=False, fontsize=9.5, labelcolor=t["label"])
    ax.set_title(tmpl["theme"] + " (Waffle Chart)",
                 fontsize=13, fontweight="bold", color=t["title"], pad=12)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Waffle)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 9. ANNOTATED DONUT ───────────────────────────────
def render_annotated_donut(tmpl, t, _):
    cats, vals = weighted_values(tmpl, n=random.randint(4, 6))
    colors = palette_cycle(t, len(cats))

    fig, ax = plt.subplots(figsize=(10, 8))
    apply_fig_theme(fig, t)
    ax.set_facecolor(t["bg"])

    wedges, _ = ax.pie(
        vals, labels=None, autopct=None,
        colors=colors, startangle=random.randint(0, 360),
        wedgeprops=dict(width=0.48, edgecolor=t["edge"], linewidth=1.8),
    )

    # Callout annotations for each slice
    for i, (wedge, cat, val) in enumerate(zip(wedges, cats, vals)):
        theta = (wedge.theta1 + wedge.theta2) / 2
        theta_rad = math.radians(theta)
        r_mid = 0.78
        x_mid = r_mid * math.cos(theta_rad)
        y_mid = r_mid * math.sin(theta_rad)
        x_far = 1.28 * math.cos(theta_rad)
        y_far = 1.28 * math.sin(theta_rad)
        ha = "left" if x_far > 0 else "right"
        ax.annotate(f"{cat}\n{val:.1f}%",
                    xy=(x_mid, y_mid), xytext=(x_far, y_far),
                    arrowprops=dict(arrowstyle="-", color=colors[i],
                                   lw=1.4, connectionstyle="arc3,rad=0.0"),
                    fontsize=9, ha=ha, va="center",
                    color=t["label"], fontweight="bold")

    # Centre circle info
    ax.text(0, 0.10, f"{len(cats)}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=t["text_center"])
    ax.text(0, -0.18, "categorie", ha="center", va="center",
            fontsize=10, color=t["label"])

    title_style(ax, tmpl["theme"] + " (Annotated Donut)", t)
    plt.tight_layout()

    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, mk_json(tmpl["theme"] + " (Annotated Donut)", tmpl["x_label"], tmpl["y_label"], dp)


# ── 10. PIE + BAR BREAKDOWN ──────────────────────────
def render_pie_with_bar(tmpl, t, _):
    cats, vals = weighted_values(tmpl)
    colors = palette_cycle(t, len(cats))

    # The largest slice gets a sub-breakdown bar chart
    max_idx = vals.index(max(vals))
    n_sub = random.randint(3, 5)
    sub_cats = [f"Sub-{i+1}" for i in range(n_sub)]
    sub_raw = [random.uniform(5, 40) for _ in range(n_sub)]
    sub_tot = sum(sub_raw)
    sub_vals = [round(v / sub_tot * vals[max_idx], 2) for v in sub_raw]

    fig = plt.figure(figsize=(13, 7))
    apply_fig_theme(fig, t)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)
    ax_pie = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    for ax in [ax_pie, ax_bar]:
        ax.set_facecolor(t["bg"])

    # Pie
    explode = [0.0] * len(cats)
    explode[max_idx] = 0.08
    wedges, _, autotexts = ax_pie.pie(
        vals, explode=explode, labels=None,
        autopct=autopct_fn, colors=colors,
        startangle=random.randint(0, 360), pctdistance=0.78,
        wedgeprops=dict(edgecolor=t["edge"], linewidth=1.5),
    )
    for at in autotexts:
        at.set_color(t["bg"]); at.set_fontsize(8); at.set_fontweight("bold")
    ax_pie.legend(wedges, [f"{c} ({v:.1f}%)" for c, v in zip(cats, vals)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.14),
                  ncol=2, frameon=False, fontsize=8.5, labelcolor=t["label"])
    title_style(ax_pie, tmpl["theme"], t)

    # Breakdown bar
    sub_color = colors[max_idx]
    sub_shades = [to_rgba(sub_color, alpha=0.5 + 0.5 * (i / n_sub)) for i in range(n_sub)]
    bars = ax_bar.barh(sub_cats, sub_vals, color=sub_shades,
                       edgecolor=t["edge"], height=0.55, zorder=3)
    ax_bar.set_facecolor(t["bg"])
    ax_bar.tick_params(colors=t["tick"])
    ax_bar.xaxis.label.set_color(t["label"])
    ax_bar.yaxis.label.set_color(t["label"])
    ax_bar.grid(axis="x", color=t["fig_bg"], linewidth=0.8, zorder=0)
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    for bar, v in zip(bars, sub_vals):
        ax_bar.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}%", va="center", ha="left", fontsize=8, color=t["tick"])
    ax_bar.set_title(f"Dettaglio: {cats[max_idx]}",
                     fontsize=11, fontweight="bold", color=t["title"], pad=10)
    ax_bar.set_xlabel("Quota (%)", fontsize=9, color=t["label"])
    ax_bar.invert_yaxis()

    plt.tight_layout()

    dp = ([{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)] +
          [{"series_name": f"Dettaglio {cats[max_idx]}", "x_value": c, "y_value": v}
           for c, v in zip(sub_cats, sub_vals)])
    return fig, mk_json(tmpl["theme"] + " (Pie+Bar)", tmpl["x_label"], tmpl["y_label"], dp)


RENDERERS = {
    "classic_pie":      render_classic_pie,
    "exploded_pie":     render_exploded_pie,
    "donut":            render_donut,
    "donut_nested":     render_donut_nested,
    "semi_donut":       render_semi_donut,
    "multi_pie":        render_multi_pie,
    "rose_nightingale": render_rose_nightingale,
    "annotated_donut":  render_annotated_donut,
}

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def generate_charts(n: int):
    used = set()
    print(f"\nGenerazione di {n} grafici a torta...\n")
    for i in range(1, n + 1):
        for _ in range(300):
            st    = random.choice(SUBTYPES)
            tmpl  = random.choice(DATASET_TEMPLATES)
            theme = random.choice(CHART_THEMES)
            key   = (st, tmpl["theme"], theme["name"])
            if key not in used:
                used.add(key)
                break

        print(f"  [{i:>3}/{n}] {st:20s} | {tmpl['theme'][:34]:34s} | {theme['name']}")
        fig, data = RENDERERS[st](tmpl, theme, i)

        base  = f"pie_{i:03d}_{st}"
        img_p = os.path.join(IMG_OUTPUT_DIR, base+".png")
        jsn_p = os.path.join(JSON_OUTPUT_DIR, base+".json")

        fig.savefig(img_p, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        with open(jsn_p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"         {img_p}  +  {jsn_p}")


if __name__ == "__main__":
    while True:
        try:
            n = int(input("Quanti grafici a torta vuoi generare? "))
            if n > 0:
                break
            print("Inserisci un numero maggiore di 0.")
        except ValueError:
            print("Inserisci un numero intero valido.")

    generate_charts(n)