"""
Scatter Chart Generator
Genera grafici scatter diversificati con immagini PNG e JSON corrispondenti.
Massimo 15 punti per classe/serie.

Sottotipologie supportate:
  1.  simple_scatter        – Scatter classico singola classe
  2.  multi_class_scatter   – Scatter multi-classe con colori distinti
  3.  scatter_with_regression – Scatter + linea di regressione lineare
  5.  connected_scatter     – Punti connessi in sequenza temporale
  6.  marginal_scatter      – Scatter + istogrammi marginali su X e Y
  7.  annotated_scatter     – Scatter con etichette testuali sui punti
  8.  quadrant_scatter      – Scatter diviso in 4 quadranti con linee di riferimento
"""

import os, json, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

IMG_OUTPUT_DIR = "data/synthetic/scatter"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/scatter"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

MAX_PTS = 15   # hard cap per classe

# ══════════════════════════════════════════════════════
#  DATASET TEMPLATES  (14 temi distinti)
# ══════════════════════════════════════════════════════

DATASET_TEMPLATES = [
    {
        "theme": "Altezza vs Peso per Gruppo Demografico",
        "x_label": "Altezza (cm)", "y_label": "Peso (kg)",
        "x_range": (155, 195), "y_range": (50, 110),
        "classes": {
            "Uomini 20-30":   {"x_shift": 8,   "y_shift": 12,  "correlation": 0.75},
            "Donne 20-30":    {"x_shift": -5,  "y_shift": -10, "correlation": 0.70},
            "Uomini 50-60":   {"x_shift": 5,   "y_shift": 8,   "correlation": 0.65},
            "Donne 50-60":    {"x_shift": -8,  "y_shift": -8,  "correlation": 0.60},
        },
        "third_var": "BMI", "third_range": (18, 32),
        "variables": ["Altezza (cm)", "Peso (kg)", "BMI", "Età"],
        "var_ranges": [(155,195), (50,110), (18,32), (20,70)],
    },
    {
        "theme": "PIL pro capite vs Aspettativa di Vita",
        "x_label": "PIL pro capite (k$)", "y_label": "Aspettativa di vita (anni)",
        "x_range": (1, 65), "y_range": (55, 85),
        "classes": {
            "Europa":        {"x_shift": 20,  "y_shift": 8,   "correlation": 0.70},
            "Asia":          {"x_shift": 5,   "y_shift": 2,   "correlation": 0.65},
            "Africa":        {"x_shift": -10, "y_shift": -15, "correlation": 0.50},
            "Americhe":      {"x_shift": 15,  "y_shift": 5,   "correlation": 0.68},
        },
        "third_var": "Popolazione (M)", "third_range": (1, 200),
        "variables": ["PIL pro capite (k$)", "Aspettativa di vita (anni)",
                      "Tasso di Alfabetizzazione (%)", "Densità (ab/km²)"],
        "var_ranges": [(1,65), (55,85), (50,100), (5,500)],
    },
    {
        "theme": "Spesa Pubblicitaria vs Ricavi",
        "x_label": "Spesa Ads (k€)", "y_label": "Ricavi (k€)",
        "x_range": (5, 150), "y_range": (20, 800),
        "classes": {
            "E-commerce":    {"x_shift": 20,  "y_shift": 100, "correlation": 0.80},
            "SaaS":          {"x_shift": 10,  "y_shift": 80,  "correlation": 0.72},
            "Retail":        {"x_shift": 30,  "y_shift": 60,  "correlation": 0.65},
        },
        "third_var": "ROI (%)", "third_range": (50, 800),
        "variables": ["Spesa Ads (k€)", "Ricavi (k€)", "ROI (%)", "Clienti Acquisiti"],
        "var_ranges": [(5,150), (20,800), (50,800), (10,500)],
    },
    {
        "theme": "Temperatura vs Vendite Gelato",
        "x_label": "Temperatura (°C)", "y_label": "Vendite (unità/giorno)",
        "x_range": (5, 40), "y_range": (20, 500),
        "classes": {
            "Centro Città":  {"x_shift": 0,   "y_shift": 80,  "correlation": 0.88},
            "Periferia":     {"x_shift": 0,   "y_shift": 20,  "correlation": 0.82},
            "Spiaggia":      {"x_shift": 5,   "y_shift": 150, "correlation": 0.91},
        },
        "third_var": "Affluenza", "third_range": (50, 1000),
        "variables": ["Temperatura (°C)", "Vendite (unità)", "Affluenza", "Ora di Punta"],
        "var_ranges": [(5,40), (20,500), (50,1000), (10,22)],
    },
    {
        "theme": "Ore di Studio vs Voto d'Esame",
        "x_label": "Ore di Studio", "y_label": "Voto (su 30)",
        "x_range": (0, 50), "y_range": (18, 30),
        "classes": {
            "Ingegneria":    {"x_shift": 10,  "y_shift": 1,   "correlation": 0.78},
            "Economia":      {"x_shift": 5,   "y_shift": 0,   "correlation": 0.72},
            "Lettere":       {"x_shift": 0,   "y_shift": 2,   "correlation": 0.65},
            "Medicina":      {"x_shift": 15,  "y_shift": 0,   "correlation": 0.85},
        },
        "third_var": "Frequenza lezioni (%)", "third_range": (40, 100),
        "variables": ["Ore Studio", "Voto", "Frequenza (%)", "Anni di Corso"],
        "var_ranges": [(0,50), (18,30), (40,100), (1,6)],
    },
    {
        "theme": "Latenza vs Throughput Sistema",
        "x_label": "Throughput (req/s)", "y_label": "Latenza (ms)",
        "x_range": (10, 1000), "y_range": (2, 500),
        "classes": {
            "Server A":      {"x_shift": 200, "y_shift": -50, "correlation": -0.70},
            "Server B":      {"x_shift": 100, "y_shift": -30, "correlation": -0.65},
            "Server C (SSD)":{"x_shift": 300, "y_shift": -80, "correlation": -0.80},
        },
        "third_var": "CPU (%)", "third_range": (10, 100),
        "variables": ["Throughput (req/s)", "Latenza (ms)", "CPU (%)", "RAM (GB)"],
        "var_ranges": [(10,1000), (2,500), (10,100), (4,64)],
    },
    {
        "theme": "Qualità del Suolo vs Resa Agricola",
        "x_label": "Indice Qualità Suolo", "y_label": "Resa (t/ha)",
        "x_range": (20, 95), "y_range": (1, 12),
        "classes": {
            "Grano":         {"x_shift": 0,   "y_shift": 0,   "correlation": 0.72},
            "Mais":          {"x_shift": 5,   "y_shift": 2,   "correlation": 0.68},
            "Soia":          {"x_shift": -5,  "y_shift": 1,   "correlation": 0.65},
            "Pomodori":      {"x_shift": 10,  "y_shift": 3,   "correlation": 0.75},
        },
        "third_var": "Precipitazioni (mm)", "third_range": (200, 900),
        "variables": ["Qualità Suolo", "Resa (t/ha)", "Precipitazioni (mm)", "T° media (°C)"],
        "var_ranges": [(20,95), (1,12), (200,900), (8,28)],
    },
    {
        "theme": "Prezzo Immobile vs Superficie",
        "x_label": "Superficie (m²)", "y_label": "Prezzo (k€)",
        "x_range": (30, 250), "y_range": (60, 900),
        "classes": {
            "Centro Storico":{"x_shift": -20, "y_shift": 200, "correlation": 0.82},
            "Semi-Centro":   {"x_shift": 0,   "y_shift": 50,  "correlation": 0.78},
            "Periferia":     {"x_shift": 20,  "y_shift": -80, "correlation": 0.75},
            "Provincia":     {"x_shift": 30,  "y_shift": -150,"correlation": 0.70},
        },
        "third_var": "Anno costruzione", "third_range": (1950, 2024),
        "variables": ["Superficie (m²)", "Prezzo (k€)", "Distanza Centro (km)", "Piano"],
        "var_ranges": [(30,250), (60,900), (0,30), (0,15)],
    },
    {
        "theme": "Velocità vs Consumo Carburante",
        "x_label": "Velocità (km/h)", "y_label": "Consumo (L/100km)",
        "x_range": (40, 200), "y_range": (3, 18),
        "classes": {
            "Berlina":       {"x_shift": 0,   "y_shift": 0,   "correlation": 0.75},
            "SUV":           {"x_shift": -10, "y_shift": 3,   "correlation": 0.72},
            "Sportiva":      {"x_shift": 20,  "y_shift": 2,   "correlation": 0.80},
            "Elettrica":     {"x_shift": 5,   "y_shift": -5,  "correlation": 0.60},
        },
        "third_var": "Emissioni CO2 (g/km)", "third_range": (0, 220),
        "variables": ["Velocità (km/h)", "Consumo (L/100km)", "Emissioni CO2", "Cilindrata (cc)"],
        "var_ranges": [(40,200), (0,18), (0,220), (500,4000)],
    },
    {
        "theme": "Età vs Reddito Annuo",
        "x_label": "Età (anni)", "y_label": "Reddito (k€/anno)",
        "x_range": (22, 65), "y_range": (15, 120),
        "classes": {
            "Tech":          {"x_shift": -5,  "y_shift": 20,  "correlation": 0.68},
            "Sanità":        {"x_shift": 5,   "y_shift": 15,  "correlation": 0.72},
            "Istruzione":    {"x_shift": 0,   "y_shift": -10, "correlation": 0.60},
            "Finanza":       {"x_shift": -8,  "y_shift": 30,  "correlation": 0.70},
        },
        "third_var": "Anni di esperienza", "third_range": (1, 40),
        "variables": ["Età (anni)", "Reddito (k€)", "Anni esperienza", "Titolo di studio"],
        "var_ranges": [(22,65), (15,120), (1,40), (1,5)],
    },
    {
        "theme": "Luminosità vs Distanza Stelle",
        "x_label": "Distanza (anni luce)", "y_label": "Luminosità relativa",
        "x_range": (1, 1000), "y_range": (0.001, 100),
        "classes": {
            "Nane Bianche":  {"x_shift": -200,"y_shift": -20, "correlation": -0.50},
            "Giganti Rosse": {"x_shift": 200, "y_shift": 40,  "correlation": 0.40},
            "Sequenza Princ":{"x_shift": 0,   "y_shift": 0,   "correlation": 0.55},
            "Supergiganti":  {"x_shift": 300, "y_shift": 60,  "correlation": 0.45},
        },
        "third_var": "Temperatura (K)", "third_range": (3000, 30000),
        "variables": ["Distanza (al)", "Luminosità", "Temperatura (K)", "Raggio (R☉)"],
        "var_ranges": [(1,1000), (0.01,100), (3000,30000), (0.1,100)],
    },
    {
        "theme": "Soddisfazione Cliente vs Tempo di Risposta",
        "x_label": "Tempo Risposta (ore)", "y_label": "Soddisfazione (1-10)",
        "x_range": (0.1, 72), "y_range": (1, 10),
        "classes": {
            "Chat Live":     {"x_shift": -20, "y_shift": 3,   "correlation": -0.82},
            "Email":         {"x_shift": 10,  "y_shift": -1,  "correlation": -0.75},
            "Telefono":      {"x_shift": -15, "y_shift": 2,   "correlation": -0.78},
        },
        "third_var": "Numero Interazioni", "third_range": (1, 15),
        "variables": ["Tempo Risposta (h)", "Soddisfazione", "N. Interazioni", "Costo Ticket (€)"],
        "var_ranges": [(0.1,72), (1,10), (1,15), (5,200)],
    },
    {
        "theme": "Frequenza Allenamento vs Performance Atletica",
        "x_label": "Allenamenti/settimana", "y_label": "Punteggio Performance",
        "x_range": (1, 14), "y_range": (30, 100),
        "classes": {
            "Nuoto":         {"x_shift": 0,   "y_shift": 5,   "correlation": 0.78},
            "Corsa":         {"x_shift": 1,   "y_shift": 0,   "correlation": 0.75},
            "Ciclismo":      {"x_shift": 2,   "y_shift": -5,  "correlation": 0.72},
            "Palestra":      {"x_shift": -1,  "y_shift": 0,   "correlation": 0.68},
        },
        "third_var": "VO2max (ml/kg/min)", "third_range": (30, 80),
        "variables": ["Allenamenti/sett.", "Performance", "VO2max", "Anni di pratica"],
        "var_ranges": [(1,14), (30,100), (30,80), (0,20)],
    },
    {
        "theme": "Investimento R&D vs Brevetti Depositati",
        "x_label": "Investimento R&D (M€)", "y_label": "Brevetti/anno",
        "x_range": (0.5, 200), "y_range": (1, 150),
        "classes": {
            "Farmaceutico":  {"x_shift": 20,  "y_shift": 10,  "correlation": 0.82},
            "Tecnologia":    {"x_shift": 50,  "y_shift": 30,  "correlation": 0.85},
            "Automotive":    {"x_shift": 30,  "y_shift": 20,  "correlation": 0.78},
            "Energia":       {"x_shift": 10,  "y_shift": 5,   "correlation": 0.70},
        },
        "third_var": "Fatturato (B€)", "third_range": (0.5, 50),
        "variables": ["R&D (M€)", "Brevetti/anno", "Fatturato (B€)", "Dipendenti (k)"],
        "var_ranges": [(0.5,200), (1,150), (0.5,50), (0.5,100)],
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
        "palette": ["#1565C0","#E53935","#2E7D32","#F57F17","#6A1B9A","#00838F","#BF360C"],
        "marker_edge": "#FFFFFF", "alpha": 0.82,
    },
    {
        "name": "dark_pro",
        "bg": "#1E1E2E", "fig_bg": "#181825",
        "grid": "#313244", "title": "#CDD6F4", "label": "#BAC2DE", "tick": "#A6ADC8",
        "spine": False,
        "palette": ["#89B4FA","#A6E3A1","#FAB387","#F38BA8","#CBA6F7","#94E2D5","#F9E2AF"],
        "marker_edge": "#1E1E2E", "alpha": 0.88,
    },
    {
        "name": "vibrant_pop",
        "bg": "#FFFFFF", "fig_bg": "#F8F9FA",
        "grid": "#DEE2E6", "title": "#212529", "label": "#495057", "tick": "#6C757D",
        "spine": True,
        "palette": ["#E63946","#F4A261","#2A9D8F","#8338EC","#3A86FF","#FB5607","#06D6A0"],
        "marker_edge": "#FFFFFF", "alpha": 0.80,
    },
    {
        "name": "pastel_notebook",
        "bg": "#FFFEF7", "fig_bg": "#FFF9E6",
        "grid": "#EDE8D0", "title": "#4A3728", "label": "#6B5744", "tick": "#8D7566",
        "spine": True,
        "palette": ["#E07B54","#5B8DB8","#7DB87D","#B87DB8","#B8A45B","#5BB8B8","#B85B5B"],
        "marker_edge": "#FFFEF7", "alpha": 0.78,
    },
    {
        "name": "neon_dark",
        "bg": "#0A0A14", "fig_bg": "#06060E",
        "grid": "#14142A", "title": "#00FF88", "label": "#AAAACC", "tick": "#666688",
        "spine": False,
        "palette": ["#00FF88","#FF006E","#3A86FF","#FFBE0B","#FB5607","#8338EC","#00F5D4"],
        "marker_edge": "#0A0A14", "alpha": 0.90,
    },
    {
        "name": "earth_tones",
        "bg": "#FFF8F0", "fig_bg": "#FEF0DC",
        "grid": "#EDD9BE", "title": "#5D3A1A", "label": "#7B4F2E", "tick": "#8B6347",
        "spine": True,
        "palette": ["#C84B31","#2D6A4F","#E9C46A","#264653","#A8763E","#8B5E3C","#457B9D"],
        "marker_edge": "#FFF8F0", "alpha": 0.82,
    },
    {
        "name": "ocean_depth",
        "bg": "#EAF4FB", "fig_bg": "#D6EAF8",
        "grid": "#AED6F1", "title": "#0B3D6B", "label": "#154360", "tick": "#1A5276",
        "spine": True,
        "palette": ["#0B3D6B","#C0392B","#1ABC9C","#D68910","#7D3C98","#2874A6","#1E8449"],
        "marker_edge": "#EAF4FB", "alpha": 0.80,
    },
    {
        "name": "aurora_night",
        "bg": "#0D1117", "fig_bg": "#080C10",
        "grid": "#161B22", "title": "#79C0FF", "label": "#8B949E", "tick": "#6E7681",
        "spine": False,
        "palette": ["#79C0FF","#56D364","#FF7B72","#D2A8FF","#E3B341","#39C5CF","#F78166"],
        "marker_edge": "#0D1117", "alpha": 0.88,
    },
    {
        "name": "sunset_fire",
        "bg": "#12000A", "fig_bg": "#0A0006",
        "grid": "#220010", "title": "#FF6B35", "label": "#FFB347", "tick": "#CC7722",
        "spine": False,
        "palette": ["#FF4500","#FFD700","#FF69B4","#ADFF2F","#00CED1","#FF8C00","#DA70D6"],
        "marker_edge": "#12000A", "alpha": 0.88,
    },
    {
        "name": "mint_clean",
        "bg": "#F0FFF4", "fig_bg": "#E2F5E9",
        "grid": "#B7DFC5", "title": "#1B5E20", "label": "#2E7D32", "tick": "#388E3C",
        "spine": True,
        "palette": ["#1B5E20","#B71C1C","#0D47A1","#F57F17","#4A148C","#006064","#BF360C"],
        "marker_edge": "#F0FFF4", "alpha": 0.80,
    },
    {
        "name": "mono_editorial",
        "bg": "#FFFFFF", "fig_bg": "#F5F5F5",
        "grid": "#E0E0E0", "title": "#000000", "label": "#333333", "tick": "#666666",
        "spine": True,
        "palette": ["#000000","#CC0000","#0044AA","#007700","#AA5500","#770077","#005555"],
        "marker_edge": "#FFFFFF", "alpha": 0.75,
    },
    {
        "name": "retro_warm",
        "bg": "#F5ECD7", "fig_bg": "#EDE0C8",
        "grid": "#D6C9A8", "title": "#2C1810", "label": "#4A3020", "tick": "#6B4C30",
        "spine": True,
        "palette": ["#8B2500","#1A4A6B","#2D5A1B","#7B4F00","#5B1A8B","#006B5B","#8B6B00"],
        "marker_edge": "#F5ECD7", "alpha": 0.80,
    },
]

SUBTYPES = [
    "simple_scatter", "multi_class_scatter", 
    "scatter_with_regression",
    "connected_scatter", "marginal_scatter",
    "annotated_scatter", "quadrant_scatter", "sized_scatter",
]

MARKER_POOL = ["o","s","^","D","v","P","X","*","h","8","<",">"]

# ══════════════════════════════════════════════════════
#  DATA GENERATORS
# ══════════════════════════════════════════════════════

def gen_class_points(tmpl, class_name, n=None, seed=None):
    """Generate correlated (x, y) points for a class."""
    if seed is not None:
        np.random.seed(seed)
    if n is None:
        n = random.randint(8, MAX_PTS)
    cfg = tmpl["classes"][class_name]
    corr = cfg.get("correlation", 0.6)
    x_shift = cfg.get("x_shift", 0)
    y_shift = cfg.get("y_shift", 0)

    xlo, xhi = tmpl["x_range"]
    ylo, yhi = tmpl["y_range"]
    cx = (xlo + xhi) / 2 + x_shift
    cy = (ylo + yhi) / 2 + y_shift
    sx = (xhi - xlo) / 5
    sy = (yhi - ylo) / 5

    cov = [[sx**2, corr * sx * sy],
           [corr * sx * sy, sy**2]]
    pts = np.random.multivariate_normal([cx, cy], cov, n)
    xs = np.clip(pts[:, 0], xlo, xhi)
    ys = np.clip(pts[:, 1], ylo, yhi)
    return xs.tolist(), ys.tolist()

def gen_third_var(tmpl, n):
    lo, hi = tmpl["third_range"]
    return [round(random.uniform(lo, hi), 2) for _ in range(n)]

def palette_n(t, n):
    p = t["palette"]
    return [p[i % len(p)] for i in range(n)]

def apply_theme(fig, ax, t):
    fig.patch.set_facecolor(t["fig_bg"])
    ax.set_facecolor(t["bg"])
    ax.title.set_color(t["title"])
    ax.xaxis.label.set_color(t["label"])
    ax.yaxis.label.set_color(t["label"])
    ax.tick_params(colors=t["tick"])
    ax.grid(True, color=t["grid"], linewidth=0.7, alpha=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(t["spine"])
        if t["spine"]: sp.set_edgecolor(t["grid"])

def mk_json(title, xl, yl, dp):
    return {"chart_title": title, "x_axis_label": xl,
            "y_axis_label": yl, "data_points": dp}

# ══════════════════════════════════════════════════════
#  RENDERERS
# ══════════════════════════════════════════════════════

# ── 1. SIMPLE SCATTER ────────────────────────────────
def render_simple_scatter(tmpl, t, _):
    cls = random.choice(list(tmpl["classes"].keys()))
    n   = random.randint(10, MAX_PTS)
    xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
    color  = t["palette"][0]
    marker = random.choice(MARKER_POOL)
    size   = random.choice([40, 55, 70, 90])

    fig, ax = plt.subplots(figsize=(9, 7))
    apply_theme(fig, ax, t)

    ax.scatter(xs, ys, c=color, s=size, marker=marker, alpha=t["alpha"],
               edgecolors=t["marker_edge"], linewidths=0.9, zorder=4)

    # Annotate extreme points
    arr = np.array(list(zip(xs, ys)))
    top = arr[np.argmax(arr[:,1])]
    bot = arr[np.argmin(arr[:,1])]
    for pt, lbl in [(top, f"Max Y\n{top[1]:.1f}"), (bot, f"Min Y\n{bot[1]:.1f}")]:
        ax.annotate(lbl, xy=pt, xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color=t["title"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["fig_bg"],
                              ec=color, lw=1, alpha=0.85))

    ax.set_title(f"{tmpl['theme']} — {cls}", fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.tight_layout()

    dp = [{"series_name": cls, "x_value": round(float(x),4), "y_value": round(float(y),4)}
          for x, y in zip(xs, ys)]
    return fig, mk_json(f"{tmpl['theme']} — {cls}", tmpl["x_label"], tmpl["y_label"], dp)


# ── 2. MULTI-CLASS SCATTER ───────────────────────────
def render_multi_class_scatter(tmpl, t, _):
    n_cls   = random.randint(2, min(4, len(tmpl["classes"])))
    classes = random.sample(list(tmpl["classes"].keys()), n_cls)
    colors  = palette_n(t, n_cls)
    markers = random.sample(MARKER_POOL, n_cls)
    size    = random.choice([45, 60, 75])

    fig, ax = plt.subplots(figsize=(10, 7))
    apply_theme(fig, ax, t)

    dp = []
    for cls, color, marker in zip(classes, colors, markers):
        n = random.randint(8, MAX_PTS)
        xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
        ax.scatter(xs, ys, c=color, s=size, marker=marker, alpha=t["alpha"],
                   edgecolors=t["marker_edge"], linewidths=0.9,
                   label=cls, zorder=4)
        for x, y in zip(xs, ys):
            dp.append({"series_name": cls,
                       "x_value": round(float(x),4), "y_value": round(float(y),4)})

    ax.set_title(tmpl["theme"], fontsize=13, fontweight="bold",
                 color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], dp)



# ── 4. SCATTER + REGRESSION ──────────────────────────
def render_scatter_with_regression(tmpl, t, _):
    n_cls   = random.randint(1, min(3, len(tmpl["classes"])))
    classes = random.sample(list(tmpl["classes"].keys()), n_cls)
    colors  = palette_n(t, n_cls)
    markers = random.sample(MARKER_POOL, n_cls)

    fig, ax = plt.subplots(figsize=(10, 7))
    apply_theme(fig, ax, t)

    dp = []
    for cls, color, marker in zip(classes, colors, markers):
        n  = random.randint(10, MAX_PTS)
        xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
        ax.scatter(xs, ys, c=color, s=55, marker=marker, alpha=t["alpha"],
                   edgecolors=t["marker_edge"], linewidths=0.9,
                   label=cls, zorder=4)
        # Regression line
        coeffs = np.polyfit(xs, ys, 1)
        xfit   = np.linspace(min(xs), max(xs), 100)
        yfit   = np.polyval(coeffs, xfit)
        ax.plot(xfit, yfit, color=color, linewidth=1.8, linestyle="--",
                alpha=0.75, zorder=3)
        # R² annotation
        y_mean = np.mean(ys)
        ss_tot = sum((y - y_mean)**2 for y in ys)
        ss_res = sum((y - np.polyval(coeffs, x))**2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.02, 0.97 - classes.index(cls) * 0.07,
                f"{cls}  R²={r2:.2f}",
                transform=ax.transAxes, fontsize=8.5,
                color=color, va="top", fontweight="bold")
        for x, y in zip(xs, ys):
            dp.append({"series_name": cls, "x_value": round(float(x),4),
                       "y_value": round(float(y),4)})

    ax.set_title(f"{tmpl['theme']} (+ Regressione)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.tight_layout()

    return fig, mk_json(f"{tmpl['theme']} (Regressione)", tmpl["x_label"],
                        tmpl["y_label"], dp)



# ── 7. CONNECTED SCATTER ─────────────────────────────
def render_connected_scatter(tmpl, t, _):
    cls    = random.choice(list(tmpl["classes"].keys()))
    n      = random.randint(8, MAX_PTS)
    xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
    color  = t["palette"][0]
    cmap   = plt.cm.get_cmap("plasma", n)

    fig, ax = plt.subplots(figsize=(10, 7))
    apply_theme(fig, ax, t)

    # Draw connecting lines
    ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.4,
            linestyle="-", zorder=2)
    # Color points by sequence
    for i, (x, y) in enumerate(zip(xs, ys)):
        c_pt = cmap(i / max(n-1, 1))
        ax.scatter(x, y, c=[c_pt], s=80, zorder=5,
                   edgecolors=t["marker_edge"], linewidths=1.2)
        ax.text(x, y, f" {i+1}", fontsize=7.5, color=t["tick"], va="center")

    # Colorbar for sequence
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("Sequenza temporale", fontsize=9, color=t["label"])
    cbar.ax.yaxis.set_tick_params(color=t["tick"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=t["tick"])

    ax.set_title(f"{tmpl['theme']} — {cls} (Connected)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.tight_layout()

    dp = [{"series_name": cls, "x_value": round(float(x),4),
           "y_value": round(float(y),4)} for x, y in zip(xs, ys)]
    return fig, mk_json(f"{tmpl['theme']} — {cls} (Connected)",
                        tmpl["x_label"], tmpl["y_label"], dp)



# ── 9. MARGINAL SCATTER ──────────────────────────────
def render_marginal_scatter(tmpl, t, _):
    n_cls   = random.randint(1, min(3, len(tmpl["classes"])))
    classes = random.sample(list(tmpl["classes"].keys()), n_cls)
    colors  = palette_n(t, n_cls)

    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor(t["fig_bg"])
    gs  = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,4],
                             hspace=0.04, wspace=0.04)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for ax in [ax_main, ax_top, ax_right]:
        ax.set_facecolor(t["bg"])
        ax.tick_params(colors=t["tick"])
        ax.grid(True, color=t["grid"], linewidth=0.6, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_visible(t["spine"])
            if t["spine"]: sp.set_edgecolor(t["grid"])

    dp = []
    all_xs, all_ys = [], []
    for cls, color in zip(classes, colors):
        n = random.randint(8, MAX_PTS)
        xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
        marker = random.choice(MARKER_POOL[:8])
        ax_main.scatter(xs, ys, c=color, s=60, marker=marker,
                        alpha=t["alpha"], edgecolors=t["marker_edge"],
                        linewidths=0.9, label=cls, zorder=4)
        ax_top.hist(xs, bins=6, color=color, alpha=0.55,
                    edgecolor=t["bg"], orientation="vertical")
        ax_right.hist(ys, bins=6, color=color, alpha=0.55,
                      edgecolor=t["bg"], orientation="horizontal")
        all_xs += xs; all_ys += ys
        for x, y in zip(xs, ys):
            dp.append({"series_name": cls, "x_value": round(float(x),4),
                       "y_value": round(float(y),4)})

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_main.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax_main.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax_top.set_ylabel("Count", fontsize=8, color=t["label"])
    ax_right.set_xlabel("Count", fontsize=8, color=t["label"])
    if n_cls > 1:
        ax_main.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    fig.suptitle(f"{tmpl['theme']} (Marginale)", fontsize=13,
                 fontweight="bold", color=t["title"], y=0.98)
    return fig, mk_json(f"{tmpl['theme']} (Marginale)", tmpl["x_label"],
                        tmpl["y_label"], dp)


# ── 10. ANNOTATED SCATTER ────────────────────────────
def render_annotated_scatter(tmpl, t, _):
    cls    = random.choice(list(tmpl["classes"].keys()))
    n      = random.randint(7, 12)
    xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
    color  = t["palette"][0]

    # Label each point with a short identifier
    label_pool = (
        [f"P{i+1}" for i in range(n)] if random.random() > 0.5
        else [chr(65+i) for i in range(min(n, 26))]
    )
    labels = label_pool[:n]

    fig, ax = plt.subplots(figsize=(10, 8))
    apply_theme(fig, ax, t)

    ax.scatter(xs, ys, c=color, s=70, alpha=t["alpha"],
               edgecolors=t["marker_edge"], linewidths=1.0, zorder=4)

    # Simple collision-avoidance: alternate offset directions
    offsets = [(10,10),(-10,10),(10,-14),(-10,-14),(14,0),(-14,0),(0,14),(0,-14)]
    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(lbl, xy=(x, y), xytext=(ox, oy),
                    textcoords="offset points",
                    fontsize=8.5, color=t["title"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.9, alpha=0.6),
                    bbox=dict(boxstyle="round,pad=0.2", fc=t["fig_bg"],
                              ec=color, lw=0.8, alpha=0.88))

    ax.set_title(f"{tmpl['theme']} — {cls} (Annotato)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    plt.tight_layout()

    dp = [{"series_name": lbl, "x_value": round(float(x),4),
           "y_value": round(float(y),4)} for lbl, x, y in zip(labels, xs, ys)]
    return fig, mk_json(f"{tmpl['theme']} — {cls} (Annotato)",
                        tmpl["x_label"], tmpl["y_label"], dp)


# ── 11. QUADRANT SCATTER ─────────────────────────────
def render_quadrant_scatter(tmpl, t, _):
    n_cls   = random.randint(2, min(4, len(tmpl["classes"])))
    classes = random.sample(list(tmpl["classes"].keys()), n_cls)
    colors  = palette_n(t, n_cls)
    markers = random.sample(MARKER_POOL, n_cls)

    # Quadrant thresholds = midpoints of ranges
    x_mid = sum(tmpl["x_range"]) / 2
    y_mid = sum(tmpl["y_range"]) / 2
    quadrant_labels = {
        (True,  True):  "Alta X / Alta Y",
        (True,  False): "Alta X / Bassa Y",
        (False, True):  "Bassa X / Alta Y",
        (False, False): "Bassa X / Bassa Y",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    apply_theme(fig, ax, t)

    # Quadrant dividers
    ax.axvline(x_mid, color=t["tick"], linewidth=1.3, linestyle="--",
               alpha=0.6, zorder=2)
    ax.axhline(y_mid, color=t["tick"], linewidth=1.3, linestyle="--",
               alpha=0.6, zorder=2)

    # Quadrant background tints
    xlo, xhi = tmpl["x_range"]
    ylo, yhi = tmpl["y_range"]
    quad_colors = [t["palette"][i % len(t["palette"])] for i in range(4)]
    for (qx, qy), qc in zip([(x_mid,y_mid),(xlo,y_mid),(x_mid,ylo),(xlo,ylo)],
                              quad_colors):
        ax.fill_between([qx, xhi if qx==x_mid else x_mid],
                        y_mid if qy==y_mid else ylo,
                        yhi if qy==y_mid else y_mid,
                        color=qc, alpha=0.05, zorder=1)

    # Quadrant corner labels
    pad_x = (xhi - xlo) * 0.02
    pad_y = (yhi - ylo) * 0.02
    for (lbl, xp, yp, ha_, va_) in [
        ("Alta X / Alta Y",    xhi - pad_x*8, yhi - pad_y*2, "right", "top"),
        ("Bassa X / Alta Y",   xlo + pad_x,   yhi - pad_y*2, "left",  "top"),
        ("Alta X / Bassa Y",   xhi - pad_x*8, ylo + pad_y*2, "right", "bottom"),
        ("Bassa X / Bassa Y",  xlo + pad_x,   ylo + pad_y*2, "left",  "bottom"),
    ]:
        ax.text(xp, yp, lbl, ha=ha_, va=va_, fontsize=7.5,
                color=t["tick"], alpha=0.65, fontstyle="italic")

    dp = []
    for cls, color, marker in zip(classes, colors, markers):
        n  = random.randint(8, MAX_PTS)
        xs, ys = gen_class_points(tmpl, cls, n, seed=random.randint(0,9999))
        ax.scatter(xs, ys, c=color, s=65, marker=marker, alpha=t["alpha"],
                   edgecolors=t["marker_edge"], linewidths=0.9,
                   label=cls, zorder=5)
        for x, y in zip(xs, ys):
            dp.append({"series_name": cls, "x_value": round(float(x),4),
                       "y_value": round(float(y),4)})

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_title(f"{tmpl['theme']} (Quadranti)", fontsize=13,
                 fontweight="bold", color=t["title"], pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11, color=t["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=11, color=t["label"])
    ax.legend(frameon=False, fontsize=9, labelcolor=t["label"])
    plt.tight_layout()

    return fig, mk_json(f"{tmpl['theme']} (Quadranti)", tmpl["x_label"],
                        tmpl["y_label"], dp)




# ══════════════════════════════════════════════════════
#  RENDERER MAP
# ══════════════════════════════════════════════════════

RENDERERS = {
    "simple_scatter":         render_simple_scatter,
    "multi_class_scatter":    render_multi_class_scatter,
    "scatter_with_regression":render_scatter_with_regression,
    "connected_scatter":      render_connected_scatter,
    "marginal_scatter":       render_marginal_scatter,
    "annotated_scatter":      render_annotated_scatter,
    "quadrant_scatter":       render_quadrant_scatter,
}

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def generate_charts(n: int):
    used = set()
    print(f"\nGenerazione di {n} grafici scatter...\n")
    for i in range(1, n + 1):
        for _ in range(400):
            st    = random.choice(SUBTYPES)
            tmpl  = random.choice(DATASET_TEMPLATES)
            theme = random.choice(CHART_THEMES)
            key   = (st, tmpl["theme"], theme["name"])
            if key not in used:
                used.add(key)
                break

        print(f"  [{i:>3}/{n}] {st:26s} | {tmpl['theme'][:36]:36s} | {theme['name']}")
        try:
            fig, data = RENDERERS[st](tmpl, theme, i)
        except Exception as e:
            print(f"          Errore: {e} — skip")
            import traceback; traceback.print_exc()
            continue

        base  = f"scatter_{i:03d}_{st}"
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
            n = int(input("Quanti grafici scatter vuoi generare? "))
            if n > 0:
                break
            print("Inserisci un numero maggiore di 0.")
        except ValueError:
            print("Inserisci un numero intero valido.")
    generate_charts(n)
