"""
Bar Chart Generator
Genera grafici a barre diversificati con immagini PNG e JSON corrispondenti.

Sottotipologie supportate:
  1. simple          – Barre verticali semplici
  2. horizontal      – Barre orizzontali
  3. grouped         – Raggruppato (multi-serie)
  4. stacked         – Impilato (stacked)
  5. stacked_percent – 100% stacked
  6. diverging       – Divergente (valori positivi/negativi)
  7. lollipop        – Lollipop chart
"""

import os, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

IMG_OUTPUT_DIR = "data/synthetic/bar"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/bar"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  DATASET TEMPLATES
# ─────────────────────────────────────────

DATASET_TEMPLATES = [
    {"theme": "Vendite Trimestrali", "x_label": "Trimestre", "y_label": "Vendite (€)",
     "categories": ["Q1","Q2","Q3","Q4"], "value_range": (50000,300000),
     "series": ["Nord","Centro","Sud"]},
    {"theme": "Consumo Energetico Mensile", "x_label": "Mese", "y_label": "kWh",
     "categories": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"],
     "value_range": (100,900), "series": ["2022","2023","2024"]},
    {"theme": "Popolarità Linguaggi di Programmazione", "x_label": "Linguaggio",
     "y_label": "Indice (%)", "categories": ["Python","JavaScript","Java","C++","Rust","Go","TypeScript"],
     "value_range": (2,35), "series": ["Main"]},
    {"theme": "Punteggi Medi per Materia", "x_label": "Materia", "y_label": "Punteggio Medio",
     "categories": ["Matematica","Fisica","Chimica","Biologia","Storia","Inglese"],
     "value_range": (55,98), "series": ["Classe A","Classe B"]},
    {"theme": "Produzione Agricola Regionale", "x_label": "Regione",
     "y_label": "Tonnellate (k)", "categories": ["Sicilia","Puglia","Campania","Lombardia","Veneto","Emilia-Romagna"],
     "value_range": (10,500), "series": ["Grano","Mais","Pomodori"]},
    {"theme": "Utenti Attivi per Piattaforma", "x_label": "Piattaforma",
     "y_label": "Milioni di Utenti", "categories": ["Mobile","Desktop","Tablet","Smart TV","Console"],
     "value_range": (5,200), "series": ["2023","2024"]},
    {"theme": "Emissioni CO2 per Settore", "x_label": "Settore", "y_label": "MtCO2e",
     "categories": ["Trasporti","Industria","Agricoltura","Energia","Edilizia","Rifiuti"],
     "value_range": (20,450), "series": ["Main"]},
    {"theme": "Budget Dipartimentale", "x_label": "Dipartimento", "y_label": "Budget (M€)",
     "categories": ["R&D","Marketing","HR","IT","Operations","Legal","Finance"],
     "value_range": (1,80), "series": ["Previsto","Effettivo"]},
    {"theme": "Temperatura Media Mensile", "x_label": "Mese", "y_label": "Temperatura (°C)",
     "categories": ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago"],
     "value_range": (0,38), "series": ["Roma","Milano","Palermo"]},
    {"theme": "Tasso di Occupazione per Fascia d'Età", "x_label": "Fascia d'Età",
     "y_label": "Tasso (%)", "categories": ["18-24","25-34","35-44","45-54","55-64","65+"],
     "value_range": (20,85), "series": ["Uomini","Donne"]},
]

# ─────────────────────────────────────────
#  THEMES
# ─────────────────────────────────────────

CHART_THEMES = [
    {"name":"corporate_blue","bg":"#FFFFFF","grid":"#E5E5E5","title":"#1A237E",
     "label":"#37474F","tick":"#546E7A","spine":True,
     "palette":["#1565C0","#1976D2","#1E88E5","#42A5F5","#90CAF9","#64B5F6"]},
    {"name":"dark_pro","bg":"#1E1E2E","grid":"#313244","title":"#CDD6F4",
     "label":"#BAC2DE","tick":"#A6ADC8","spine":False,
     "palette":["#89B4FA","#A6E3A1","#FAB387","#F38BA8","#CBA6F7","#94E2D5"]},
    {"name":"pastel_soft","bg":"#FAFAFA","grid":"#F0F0F0","title":"#4A4A4A",
     "label":"#6A6A6A","tick":"#8A8A8A","spine":False,
     "palette":["#FFB3BA","#FFDFBA","#FFFFBA","#BAFFC9","#BAE1FF","#E8BAFF"]},
    {"name":"vibrant_modern","bg":"#F8F9FA","grid":"#DEE2E6","title":"#212529",
     "label":"#495057","tick":"#6C757D","spine":True,
     "palette":["#E63946","#F4A261","#2A9D8F","#264653","#E9C46A","#8338EC"]},
    {"name":"minimal_mono","bg":"#FFFFFF","grid":"#F0F0F0","title":"#000000",
     "label":"#333333","tick":"#666666","spine":False,
     "palette":["#000000","#333333","#555555","#777777","#999999","#BBBBBB"]},
    {"name":"sunset_gradient","bg":"#FFF8F0","grid":"#FFE0CC","title":"#7B2D00",
     "label":"#8B4513","tick":"#A0522D","spine":True,
     "palette":["#FF6B35","#F7C59F","#004E89","#1A936F","#88D498","#C84B31"]},
    {"name":"neon_dark","bg":"#0D0D0D","grid":"#1A1A1A","title":"#00FF88",
     "label":"#CCCCCC","tick":"#888888","spine":False,
     "palette":["#00FF88","#FF006E","#3A86FF","#FFBE0B","#FB5607","#8338EC"]},
    {"name":"nature_green","bg":"#F1F8E9","grid":"#DCEDC8","title":"#1B5E20",
     "label":"#2E7D32","tick":"#388E3C","spine":True,
     "palette":["#2E7D32","#388E3C","#43A047","#66BB6A","#A5D6A7","#81C784"]},
]

SUBTYPES = ["simple","horizontal","grouped","stacked","stacked_percent",
            "diverging","lollipop"]

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def rv(n, lo, hi):
    return [round(random.uniform(lo, hi), 2) for _ in range(n)]

def apply_theme(fig, ax, t):
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.title.set_color(t["title"])
    ax.xaxis.label.set_color(t["label"])
    ax.yaxis.label.set_color(t["label"])
    ax.tick_params(colors=t["tick"])
    ax.grid(True, color=t["grid"], linewidth=0.8, alpha=0.7, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(t["spine"])
        if t["spine"]:
            sp.set_edgecolor(t["grid"])

def val_label(ax, bars, tick_color, horizontal=False):
    for b in bars:
        if horizontal:
            w = b.get_width()
            ax.text(w * 1.01 + 0.5, b.get_y() + b.get_height()/2,
                    f"{w:,.0f}", va="center", ha="left", fontsize=7.5, color=tick_color)
        else:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h * 1.01 + 0.5,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=7.5, color=tick_color)

def mk_json(title, xl, yl, dp):
    return {"chart_title": title, "x_axis_label": xl, "y_axis_label": yl, "data_points": dp}

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def render_simple(tmpl, t, _):
    cats, vals = tmpl["categories"], rv(len(tmpl["categories"]), *tmpl["value_range"])
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, t)
    bars = ax.bar(cats, vals, color=t["palette"][0], edgecolor=t["bg"], width=0.6, zorder=3)
    val_label(ax, bars, t["tick"])
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    dp = [{"series_name":"Main","x_value":c,"y_value":v} for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], dp)


def render_horizontal(tmpl, t, _):
    cats, vals = tmpl["categories"], rv(len(tmpl["categories"]), *tmpl["value_range"])
    colors = [t["palette"][i % len(t["palette"])] for i in range(len(cats))]
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_theme(fig, ax, t)
    bars = ax.barh(cats, vals, color=colors, edgecolor=t["bg"], height=0.6, zorder=3)
    val_label(ax, bars, t["tick"], horizontal=True)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["y_label"], fontsize=11)
    ax.set_ylabel(tmpl["x_label"], fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    dp = [{"series_name":"Main","x_value":c,"y_value":v} for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], dp)


def render_grouped(tmpl, t, _):
    cats = tmpl["categories"]
    series_list = tmpl["series"][:3]
    x = np.arange(len(cats))
    w = 0.8 / len(series_list)
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_theme(fig, ax, t)
    all_data = []
    for i, s in enumerate(series_list):
        vals = rv(len(cats), *tmpl["value_range"])
        ax.bar(x + i*w - (len(series_list)-1)*w/2, vals, width=w*0.9,
               color=t["palette"][i % len(t["palette"])], edgecolor=t["bg"],
               label=s, zorder=3)
        all_data.append((s, vals))
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.legend(facecolor=t["bg"], labelcolor=t["label"], framealpha=0.8)
    plt.tight_layout()
    dp = [{"series_name":s,"x_value":c,"y_value":v} for s,vals in all_data for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], dp)


def render_stacked(tmpl, t, _):
    cats = tmpl["categories"]
    series_list = tmpl["series"][:4]
    fig, ax = plt.subplots(figsize=(11, 7))
    apply_theme(fig, ax, t)
    bottoms = np.zeros(len(cats))
    all_data = []
    for i, s in enumerate(series_list):
        vals = np.array(rv(len(cats), *tmpl["value_range"]))
        ax.bar(cats, vals, bottom=bottoms, color=t["palette"][i % len(t["palette"])],
               edgecolor=t["bg"], label=s, zorder=3)
        all_data.append((s, vals)); bottoms += vals
    ax.set_title(tmpl["theme"] + " (Stacked)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11); ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    ax.legend(facecolor=t["bg"], labelcolor=t["label"], framealpha=0.8)
    plt.tight_layout()
    dp = [{"series_name":s,"x_value":c,"y_value":round(float(v),2)}
          for s,vals in all_data for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"]+" (Stacked)", tmpl["x_label"], tmpl["y_label"], dp)


def render_stacked_percent(tmpl, t, _):
    cats = tmpl["categories"]
    series_list = tmpl["series"][:3]
    M = np.array([rv(len(cats), *tmpl["value_range"]) for _ in series_list])
    P = M / M.sum(axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 7))
    apply_theme(fig, ax, t)
    bottoms = np.zeros(len(cats))
    for i, s in enumerate(series_list):
        ax.bar(cats, P[i], bottom=bottoms, color=t["palette"][i % len(t["palette"])],
               edgecolor=t["bg"], label=s, zorder=3)
        bottoms += P[i]
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(tmpl["theme"] + " (100% Stacked)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11); ax.set_ylabel("Percentuale (%)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    ax.legend(facecolor=t["bg"], labelcolor=t["label"], framealpha=0.8)
    plt.tight_layout()
    dp = [{"series_name":s,"x_value":c,"y_value":round(float(v),2)}
          for i,s in enumerate(series_list) for c,v in zip(cats,P[i])]
    return fig, mk_json(tmpl["theme"]+" (100% Stacked)", tmpl["x_label"], "Percentuale (%)", dp)


def render_diverging(tmpl, t, _):
    cats = tmpl["categories"]
    lo, hi = tmpl["value_range"]
    center = (lo + hi) / 2
    vals = [round(random.uniform(lo, hi) - center, 2) for _ in cats]
    colors = [t["palette"][0] if v >= 0 else t["palette"][1 % len(t["palette"])] for v in vals]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, t)
    ax.bar(cats, vals, color=colors, edgecolor=t["bg"], width=0.65, zorder=3)
    ax.axhline(0, color=t["label"], linewidth=1.2, zorder=4)
    for xi, v in enumerate(vals):
        off = max(abs(v)*0.03, 1)
        ax.text(xi, v + (off if v >= 0 else -off*4), f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color=t["tick"])
    ax.set_title(tmpl["theme"] + " (Divergente)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(f"Δ {tmpl['y_label']}", fontsize=11)
    plt.xticks(range(len(cats)), cats, rotation=30, ha="right")
    pos_p = mpatches.Patch(color=t["palette"][0], label="Positivo")
    neg_p = mpatches.Patch(color=t["palette"][1 % len(t["palette"])], label="Negativo")
    ax.legend(handles=[pos_p, neg_p], facecolor=t["bg"], labelcolor=t["label"], framealpha=0.8)
    plt.tight_layout()
    dp = [{"series_name":"Main","x_value":c,"y_value":v} for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"]+" (Divergente)", tmpl["x_label"], f"Δ {tmpl['y_label']}", dp)


def render_lollipop(tmpl, t, _):
    cats = tmpl["categories"]
    vals = rv(len(cats), *tmpl["value_range"])
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, t)
    for xi, v in zip(x, vals):
        ax.plot([xi, xi], [0, v], color=t["palette"][0], linewidth=2.5, zorder=3)
        ax.plot(xi, v, "o", color=t["palette"][min(2, len(t["palette"])-1)],
                markersize=11, zorder=4)
        ax.text(xi, v + max(abs(v)*0.025, 1), f"{v:,.1f}",
                ha="center", va="bottom", fontsize=8, color=t["tick"])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(cats)-0.4); ax.set_ylim(0, max(vals)*1.18)
    ax.set_title(tmpl["theme"] + " (Lollipop)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(tmpl["x_label"], fontsize=11); ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    dp = [{"series_name":"Main","x_value":c,"y_value":v} for c,v in zip(cats,vals)]
    return fig, mk_json(tmpl["theme"]+" (Lollipop)", tmpl["x_label"], tmpl["y_label"], dp)




RENDERERS = {
    "simple": render_simple, "horizontal": render_horizontal,
    "grouped": render_grouped, "stacked": render_stacked,
    "stacked_percent": render_stacked_percent, "diverging": render_diverging,
    "lollipop": render_lollipop, 
}

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def generate_charts(n: int):
    used = set()
    print(f"\nGenerazione di {n} grafici...\n")
    for i in range(1, n+1):
        for _ in range(200):
            st = random.choice(SUBTYPES)
            tmpl = random.choice(DATASET_TEMPLATES)
            theme = random.choice(CHART_THEMES)
            key = (st, tmpl["theme"], theme["name"])
            if key not in used:
                used.add(key); break
        print(f"  [{i:>3}/{n}] {st:18s} | {tmpl['theme'][:34]:34s} | {theme['name']}")
        fig, data = RENDERERS[st](tmpl, theme, i)
        base = f"chart_{i:03d}_{st}"
        img_p = os.path.join(IMG_OUTPUT_DIR, base+".png")
        jsn_p = os.path.join(JSON_OUTPUT_DIR, base+".json")
        fig.savefig(img_p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        with open(jsn_p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"          {img_p}  +  {jsn_p}")


if __name__ == "__main__":
    while True:
        try:
            n = int(input("Quanti grafici vuoi generare? "))
            if n > 0: break
            print("Inserisci un numero maggiore di 0.")
        except ValueError:
            print("Inserisci un numero intero valido.")
    generate_charts(n)
