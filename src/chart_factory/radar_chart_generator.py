"""
Radar Chart Generator — full version written directly to disk
"""
import os, json, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore")


IMG_OUTPUT_DIR = "data/synthetic/radar"
JSON_OUTPUT_DIR = "data_groundtruth/synthetic/radar"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

DATASET_TEMPLATES = [
    {
        "theme": "Competenze Professionali",
        "axes": ["Python","Machine Learning","SQL","Comunicazione","Problem Solving","Leadership","Visualizzazione"],
        "entities": {
            "Data Scientist Sr.":[88,92,78,72,90,65,85],
            "Data Analyst":      [70,55,90,80,75,60,88],
            "ML Engineer":       [92,95,65,60,88,55,70],
            "BI Developer":      [55,40,95,75,70,58,92],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Performance Atleti",
        "axes": ["Velocità","Resistenza","Forza","Agilità","Tecnica","Mentale","Recupero"],
        "entities": {
            "Sprinter":   [98,55,85,80,78,82,88],
            "Maratoneta": [75,98,60,65,72,90,70],
            "Nuotatore":  [82,85,75,88,90,80,78],
            "Ciclista":   [80,95,70,72,82,85,75],
        },
        "value_range":(0,100),"unit":"pt",
    },
    {
        "theme": "Qualità del Vino",
        "axes": ["Acidità","Tannini","Fruttato","Corpo","Persistenza","Equilibrio"],
        "entities": {
            "Barolo 2018":   [78,90,72,92,88,85],
            "Chianti 2020":  [82,70,80,75,72,78],
            "Amarone 2017":  [70,88,78,95,92,88],
            "Brunello 2016": [75,92,68,90,95,90],
        },
        "value_range":(0,100),"unit":"pt",
    },
    {
        "theme": "Benchmark Smartphone",
        "axes": ["CPU","GPU","RAM","Fotocamera","Batteria","Display","Connettività"],
        "entities": {
            "Modello Alpha": [95,92,88,90,78,95,88],
            "Modello Beta":  [88,85,92,85,90,88,82],
            "Modello Gamma": [78,80,85,92,95,82,90],
            "Modello Delta": [82,88,78,88,85,90,95],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Indici ESG Aziendale",
        "axes": ["CO2","Acqua","Riciclo","Parità Genere","Stipendi","Governance","Fornitura"],
        "entities": {
            "Azienda A": [82,75,88,72,80,85,70],
            "Azienda B": [65,88,72,85,75,78,82],
            "Azienda C": [90,80,78,78,88,80,75],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Valutazione Ristoranti",
        "axes": ["Cucina","Servizio","Ambiente","Prezzo/Qualità","Carta Vini","Posizione"],
        "entities": {
            "Ristorante Sole": [92,85,88,75,80,90],
            "Trattoria Luna":  [88,92,80,90,72,75],
            "Osteria Mare":    [85,78,92,85,88,82],
            "Pizzeria Fuoco":  [80,90,75,95,65,88],
        },
        "value_range":(0,100),"unit":"★",
    },
    {
        "theme": "Big Five Personalità",
        "axes": ["Apertura","Coscienziosità","Estroversione","Gradevolezza","Nevroticismo"],
        "entities": {
            "Profilo A": [82,75,68,80,35],
            "Profilo B": [65,88,85,70,25],
            "Profilo C": [90,60,50,88,45],
            "Profilo D": [72,80,78,65,30],
        },
        "value_range":(0,100),"unit":"pt",
    },
    {
        "theme": "Analisi Competitiva di Mercato",
        "axes": ["Quota Mercato","Innovazione","Brand","Customer Service","Prezzo","Distribuzione","Digital"],
        "entities": {
            "Leader":     [88,82,92,78,65,90,85],
            "Challenger": [72,90,75,85,80,78,92],
            "Follower":   [55,68,65,90,92,70,75],
            "Niche":      [40,95,60,88,70,55,80],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Caratteristiche Auto Sportive",
        "axes": ["Velocità Max","Accelerazione","Frenata","Tenuta","Comfort","Efficienza"],
        "entities": {
            "Supercar A":  [98,96,92,95,60,45],
            "Sport GT B":  [88,85,90,88,78,65],
            "Turismo C":   [75,72,85,80,92,80],
            "Elettrica D": [90,98,88,82,85,95],
        },
        "value_range":(0,100),"unit":"pt",
    },
    {
        "theme": "Qualità della Vita per Città",
        "axes": ["Trasporti","Sanità","Istruzione","Sicurezza","Verde","Cultura","Costo Vita"],
        "entities": {
            "Milano": [82,80,85,72,70,88,55],
            "Roma":   [65,78,80,68,75,95,58],
            "Bologna":[78,82,88,80,78,85,70],
            "Torino": [80,78,82,75,80,80,72],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Nutrizione per 100g",
        "axes": ["Proteine","Carboidrati","Grassi","Fibre","Vitamine","Minerali"],
        "entities": {
            "Salmone":    [85,5,65,0,70,75],
            "Pollo":      [90,0,35,0,45,60],
            "Lenticchie": [70,80,10,90,50,70],
            "Uova":       [75,5,60,0,80,65],
            "Spinaci":    [25,20,5,85,90,80],
        },
        "value_range":(0,100),"unit":"%DV",
    },
    {
        "theme": "Skill Personaggio RPG",
        "axes": ["Forza","Destrezza","Intelligenza","Costituzione","Saggezza","Carisma","Fortuna"],
        "entities": {
            "Guerriero": [95,70,45,90,55,60,50],
            "Mago":      [30,65,98,55,80,70,65],
            "Ladro":     [60,95,70,60,65,75,85],
            "Paladino":  [82,65,70,85,88,90,60],
        },
        "value_range":(0,100),"unit":"pt",
    },
    {
        "theme": "Metriche DevOps per Team",
        "axes": ["Deploy Freq.","Lead Time","MTTR","Change Fail Rate","Test Coverage","Doc Quality"],
        "entities": {
            "Team Alpha": [88,75,82,90,85,70],
            "Team Beta":  [72,88,75,80,78,85],
            "Team Gamma": [80,80,90,75,90,80],
        },
        "value_range":(0,100),"unit":"%",
    },
    {
        "theme": "Valutazione Università",
        "axes": ["Didattica","Ricerca","Internazionaliz.","Placement","Strutture","Sostenibilità","Reputazione"],
        "entities": {
            "Università A": [88,92,85,80,78,75,90],
            "Università B": [82,78,90,88,85,80,85],
            "Università C": [90,85,78,85,90,85,88],
            "Università D": [78,80,82,92,88,90,80],
        },
        "value_range":(0,100),"unit":"pt",
    },
]

CHART_THEMES = [
    {"name":"corporate_blue","bg":"#FFFFFF","fig_bg":"#F4F7FB","grid":"#D0DCF0","title":"#1A237E","label":"#37474F","tick":"#546E7A","spine":True,"alpha_fill":0.20,"alpha_line":0.90,"palette":["#1565C0","#E53935","#2E7D32","#F57F17","#6A1B9A","#00838F","#BF360C"],"gridstyle":"solid"},
    {"name":"dark_pro","bg":"#1E1E2E","fig_bg":"#181825","grid":"#45475A","title":"#CDD6F4","label":"#BAC2DE","tick":"#A6ADC8","spine":False,"alpha_fill":0.22,"alpha_line":0.92,"palette":["#89B4FA","#A6E3A1","#FAB387","#F38BA8","#CBA6F7","#94E2D5","#F9E2AF"],"gridstyle":"dashed"},
    {"name":"vibrant_pop","bg":"#FFFFFF","fig_bg":"#F8F9FA","grid":"#CED4DA","title":"#212529","label":"#495057","tick":"#6C757D","spine":True,"alpha_fill":0.18,"alpha_line":0.88,"palette":["#E63946","#F4A261","#2A9D8F","#8338EC","#3A86FF","#FB5607","#06D6A0"],"gridstyle":"solid"},
    {"name":"pastel_soft","bg":"#FEFEFE","fig_bg":"#FDF6FF","grid":"#E8D5F5","title":"#4A235A","label":"#6C3483","tick":"#884EA0","spine":False,"alpha_fill":0.28,"alpha_line":0.85,"palette":["#C39BD3","#82E0AA","#F9E79F","#F1948A","#85C1E9","#F0B27A","#A9CCE3"],"gridstyle":"dashed"},
    {"name":"neon_dark","bg":"#0A0A14","fig_bg":"#06060E","grid":"#1A1A2E","title":"#00FF88","label":"#AAAACC","tick":"#6666AA","spine":False,"alpha_fill":0.15,"alpha_line":0.95,"palette":["#00FF88","#FF006E","#3A86FF","#FFBE0B","#FB5607","#8338EC","#00F5D4"],"gridstyle":"dashed"},
    {"name":"earth_tones","bg":"#FFF8F0","fig_bg":"#FEF0DC","grid":"#D4B896","title":"#5D3A1A","label":"#7B4F2E","tick":"#8B6347","spine":True,"alpha_fill":0.22,"alpha_line":0.88,"palette":["#C84B31","#2D6A4F","#E9C46A","#264653","#A8763E","#8B5E3C","#457B9D"],"gridstyle":"solid"},
    {"name":"ocean_depth","bg":"#EAF4FB","fig_bg":"#D6EAF8","grid":"#7FB3D3","title":"#0B3D6B","label":"#154360","tick":"#1A5276","spine":True,"alpha_fill":0.20,"alpha_line":0.88,"palette":["#0B3D6B","#C0392B","#1ABC9C","#D68910","#7D3C98","#2874A6","#1E8449"],"gridstyle":"solid"},
    {"name":"aurora_night","bg":"#0D1117","fig_bg":"#080C10","grid":"#21262D","title":"#79C0FF","label":"#8B949E","tick":"#6E7681","spine":False,"alpha_fill":0.18,"alpha_line":0.92,"palette":["#79C0FF","#56D364","#FF7B72","#D2A8FF","#E3B341","#39C5CF","#F78166"],"gridstyle":"dashed"},
    {"name":"sunset_fire","bg":"#12000A","fig_bg":"#0A0006","grid":"#3A0020","title":"#FF6B35","label":"#FFB347","tick":"#CC7722","spine":False,"alpha_fill":0.16,"alpha_line":0.92,"palette":["#FF4500","#FFD700","#FF69B4","#ADFF2F","#00CED1","#FF8C00","#DA70D6"],"gridstyle":"dashed"},
    {"name":"mint_clean","bg":"#F0FFF4","fig_bg":"#E2F5E9","grid":"#88D8A8","title":"#1B5E20","label":"#2E7D32","tick":"#388E3C","spine":True,"alpha_fill":0.20,"alpha_line":0.88,"palette":["#1B5E20","#B71C1C","#0D47A1","#F57F17","#4A148C","#006064","#BF360C"],"gridstyle":"solid"},
    {"name":"mono_editorial","bg":"#FFFFFF","fig_bg":"#F0F0F0","grid":"#BBBBBB","title":"#000000","label":"#333333","tick":"#666666","spine":True,"alpha_fill":0.12,"alpha_line":0.85,"palette":["#000000","#CC0000","#0044AA","#007700","#AA5500","#770077","#005555"],"gridstyle":"dashed"},
    {"name":"retro_warm","bg":"#F5ECD7","fig_bg":"#EDE0C8","grid":"#B8A882","title":"#2C1810","label":"#4A3020","tick":"#6B4C30","spine":True,"alpha_fill":0.22,"alpha_line":0.85,"palette":["#8B2500","#1A4A6B","#2D5A1B","#7B4F00","#5B1A8B","#006B5B","#8B6B00"],"gridstyle":"solid"},
]

SUBTYPES = ["simple_radar","multi_radar","filled_radar","radar_small_multiples",
            "radar_with_dots",
            "radar_polar_bar","radar_delta"]

# ─── helpers ───────────────────────────────────────────────

def palette_n(t, n):
    p = t["palette"]; return [p[i % len(p)] for i in range(n)]

def mk_json(title, xl, yl, dp):
    return {"chart_title":title, "data_points":dp}

def pick_entities(tmpl, n=None):
    keys = list(tmpl["entities"].keys())
    if n is None: n = random.randint(1, min(4,len(keys)))
    chosen = random.sample(keys, min(n,len(keys)))
    return {k: tmpl["entities"][k] for k in chosen}

def jitter(vals, tmpl, s=0.07):
    lo,hi = tmpl["value_range"]; span=hi-lo
    return [round(min(hi,max(lo, v+random.uniform(-s*span,s*span))),1) for v in vals]

def angles_for(n):
    a = np.linspace(0,2*np.pi,n,endpoint=False).tolist(); return a+a[:1]

def radar_ax(fig, pos, t):
    ax = fig.add_subplot(pos, polar=True)
    ax.set_facecolor(t["bg"])
    ax.tick_params(colors=t["tick"], labelsize=8)
    ax.grid(color=t["grid"], linewidth=0.7, linestyle=t["gridstyle"], alpha=0.8)
    ax.spines["polar"].set_color(t["grid"])
    return ax

def style_axes(ax, labels, vmin, vmax, t, n_rings=5):
    n=len(labels); angs=angles_for(n)
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(labels, color=t["label"], fontsize=8.5, fontweight="bold")
    ax.set_ylim(vmin,vmax)
    ticks=np.linspace(vmin,vmax,n_rings+1)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.0f}" for v in ticks], color=t["tick"], fontsize=7, alpha=0.7)
    ax.set_rlabel_position(random.choice([15,30,45,60]))

def draw_poly(ax, angs, vals, color, lw, af, al, ls="-", mk=None, mks=6):
    v2=vals+vals[:1]
    ax.plot(angs,v2,color=color,linewidth=lw,linestyle=ls,zorder=4,alpha=al,
            marker=mk,markersize=mks if mk else 0,
            markerfacecolor=color,markeredgecolor="white",markeredgewidth=0.8)
    ax.fill(angs,v2,color=color,alpha=af)

def dp_ents(entities, axes):
    dp=[]
    for ent,vals in entities.items():
        for ax_n,v in zip(axes,vals):
            dp.append({"series_name":ent,"x_value":ax_n,"y_value":round(float(v),2)})
    return dp

# ─── renderers ─────────────────────────────────────────────

def render_simple_radar(tmpl,t,_):
    entities=pick_entities(tmpl,n=1); name,raw=list(entities.items())[0]
    vals=jitter(raw,tmpl,0.07); axes_lbl=tmpl["axes"]; n=len(axes_lbl)
    angs=angles_for(n); vmin,vmax=tmpl["value_range"]; color=t["palette"][0]
    fig=plt.figure(figsize=(8,8)); fig.patch.set_facecolor(t["fig_bg"])
    ax=radar_ax(fig,111,t); style_axes(ax,axes_lbl,vmin,vmax,t)
    draw_poly(ax,angs,vals,color,2.5,t["alpha_fill"]+0.10,t["alpha_line"])
    for ang,v in zip(angs[:-1],vals):
        ax.text(ang,v+(vmax-vmin)*0.08,f"{v:.0f}{tmpl['unit']}",
                ha="center",va="center",fontsize=8,color=color,fontweight="bold")
    ax.set_title(f"{tmpl['theme']}\n{name}",fontsize=13,fontweight="bold",color=t["title"],pad=20)
    plt.tight_layout()
    dp=[{"series_name":name,"x_value":a,"y_value":round(float(v),2)} for a,v in zip(axes_lbl,vals)]
    return fig,mk_json(f"{tmpl['theme']} — {name}","Asse","Valore",dp)

def render_multi_radar(tmpl,t,_):
    n_ent=random.randint(2,min(4,len(tmpl["entities"]))); entities=pick_entities(tmpl,n=n_ent)
    axes_lbl=tmpl["axes"]; n=len(axes_lbl); angs=angles_for(n)
    vmin,vmax=tmpl["value_range"]; colors=palette_n(t,n_ent)
    lss=["-","--","-.",":"]; fig=plt.figure(figsize=(9,9)); fig.patch.set_facecolor(t["fig_bg"])
    ax=radar_ax(fig,111,t); style_axes(ax,axes_lbl,vmin,vmax,t)
    jittered={}
    for (name,raw),color,ls in zip(entities.items(),colors,lss):
        vals=jitter(raw,tmpl,0.05); jittered[name]=vals
        draw_poly(ax,angs,vals,color,2.2,t["alpha_fill"],t["alpha_line"],ls=ls)
    patches=[mpatches.Patch(color=c,label=n,alpha=0.85) for n,c in zip(entities.keys(),colors)]
    ax.legend(handles=patches,loc="upper right",bbox_to_anchor=(1.35,1.15),
              frameon=False,fontsize=9,labelcolor=t["label"])
    ax.set_title(tmpl["theme"],fontsize=13,fontweight="bold",color=t["title"],pad=22)
    plt.tight_layout()
    return fig,mk_json(tmpl["theme"],"Asse","Valore",dp_ents(jittered,axes_lbl))

def render_filled_radar(tmpl,t,_):
    n_ent=random.randint(2,min(3,len(tmpl["entities"]))); entities=pick_entities(tmpl,n=n_ent)
    axes_lbl=tmpl["axes"]; n=len(axes_lbl); angs=angles_for(n)
    vmin,vmax=tmpl["value_range"]; colors=palette_n(t,n_ent)
    items=[(name,jitter(raw,tmpl,0.05)) for name,raw in entities.items()]
    items.sort(key=lambda x:sum(x[1]))
    fig=plt.figure(figsize=(9,9)); fig.patch.set_facecolor(t["fig_bg"])
    ax=radar_ax(fig,111,t); style_axes(ax,axes_lbl,vmin,vmax,t,n_rings=4)
    for (name,vals),color in zip(items,colors):
        draw_poly(ax,angs,vals,color,2.0,t["alpha_fill"]+0.12,t["alpha_line"])
    patches=[mpatches.Patch(color=colors[i],label=items[i][0],alpha=0.85) for i in range(n_ent)]
    ax.legend(handles=patches,loc="upper right",bbox_to_anchor=(1.35,1.15),
              frameon=False,fontsize=9,labelcolor=t["label"])
    ax.set_title(f"{tmpl['theme']} (Filled)",fontsize=13,fontweight="bold",color=t["title"],pad=22)
    plt.tight_layout()
    return fig,mk_json(f"{tmpl['theme']} (Filled)","Asse","Valore",
                       dp_ents({n:v for n,v in items},axes_lbl))

def render_radar_small_multiples(tmpl,t,_):
    n_ent=random.randint(3,min(4,len(tmpl["entities"]))); entities=pick_entities(tmpl,n=n_ent)
    axes_lbl=tmpl["axes"]; n=len(axes_lbl); angs=angles_for(n)
    vmin,vmax=tmpl["value_range"]; colors=palette_n(t,n_ent)
    ncols=2; nrows=math.ceil(n_ent/ncols)
    fig=plt.figure(figsize=(12,5*nrows)); fig.patch.set_facecolor(t["fig_bg"])
    fig.suptitle(tmpl["theme"]+" — Confronto",fontsize=14,fontweight="bold",color=t["title"],y=1.01)
    dp=[]
    for idx,((name,raw),color) in enumerate(zip(entities.items(),colors)):
        vals=jitter(raw,tmpl,0.06)
        ax=radar_ax(fig,int(f"{nrows}{ncols}{idx+1}"),t)
        style_axes(ax,axes_lbl,vmin,vmax,t,n_rings=4)
        draw_poly(ax,angs,vals,color,2.4,t["alpha_fill"]+0.15,t["alpha_line"])
        ax.set_title(name,fontsize=11,fontweight="bold",color=color,pad=18)
        score=sum(vals)/len(vals)
        ax.text(0.5,-0.08,f"Media: {score:.1f}{tmpl['unit']}",
                transform=ax.transAxes,ha="center",fontsize=9,color=t["tick"])
        for a,v in zip(axes_lbl,vals): dp.append({"series_name":name,"x_value":a,"y_value":round(float(v),2)})
    for idx in range(n_ent,nrows*ncols):
        fig.add_subplot(int(f"{nrows}{ncols}{idx+1}"),polar=True).set_visible(False)
    plt.tight_layout()
    return fig,mk_json(f"{tmpl['theme']} (Small Multiples)","Asse","Valore",dp)

def render_radar_with_dots(tmpl,t,_):
    n_ent=random.randint(1,min(3,len(tmpl["entities"]))); entities=pick_entities(tmpl,n=n_ent)
    axes_lbl=tmpl["axes"]; n=len(axes_lbl); angs=angles_for(n)
    vmin,vmax=tmpl["value_range"]; colors=palette_n(t,n_ent)
    markers=["o","s","^","D"]
    fig=plt.figure(figsize=(9,9)); fig.patch.set_facecolor(t["fig_bg"])
    ax=radar_ax(fig,111,t); style_axes(ax,axes_lbl,vmin,vmax,t)
    dp=[]
    for (name,raw),color,mk in zip(entities.items(),colors,markers):
        vals=jitter(raw,tmpl,0.06)
        draw_poly(ax,angs,vals,color,2.3,t["alpha_fill"],t["alpha_line"],mk=mk,mks=9)
        for ang,v in zip(angs[:-1],vals):
            ax.text(ang,v+(vmax-vmin)*0.10,f"{v:.0f}",ha="center",va="center",
                    fontsize=7.5,color=color,fontweight="bold")
        for a,v in zip(axes_lbl,vals): dp.append({"series_name":name,"x_value":a,"y_value":round(float(v),2)})
    patches=[mpatches.Patch(color=c,label=n,alpha=0.85) for n,c in zip(entities.keys(),colors)]
    ax.legend(handles=patches,loc="upper right",bbox_to_anchor=(1.38,1.15),
              frameon=False,fontsize=9,labelcolor=t["label"])
    ax.set_title(f"{tmpl['theme']} (Dots & Values)",fontsize=13,fontweight="bold",color=t["title"],pad=22)
    plt.tight_layout()
    return fig,mk_json(f"{tmpl['theme']} (Dots)","Asse","Valore",dp)

def render_radar_polar_bar(tmpl,t,_):
    entity_name,raw=random.choice(list(tmpl["entities"].items()))
    vals=jitter(raw,tmpl,0.08); axes_lbl=tmpl["axes"]; n=len(axes_lbl)
    angles=np.linspace(0,2*np.pi,n,endpoint=False)
    vmin,vmax=tmpl["value_range"]; colors=palette_n(t,n)
    bar_width=2*np.pi/n*0.82
    fig=plt.figure(figsize=(9,9)); fig.patch.set_facecolor(t["fig_bg"])
    ax=fig.add_subplot(111,polar=True); ax.set_facecolor(t["bg"])
    ax.grid(color=t["grid"],linewidth=0.6,linestyle=t["gridstyle"],alpha=0.7)
    ax.spines["polar"].set_color(t["grid"])
    ax.bar(angles,vals,width=bar_width,bottom=vmin,color=colors,alpha=0.80,
           edgecolor=t["bg"],linewidth=1.2)
    for ang,v,color in zip(angles,vals,colors):
        ax.text(ang,vmin+(v-vmin)*0.5+(vmax-vmin)*0.05,f"{v:.0f}",
                ha="center",va="center",fontsize=8,color=t["bg"],fontweight="bold")
    ax.set_xticks(angles); ax.set_xticklabels(axes_lbl,color=t["label"],fontsize=9,fontweight="bold")
    ax.set_ylim(vmin,vmax)
    ticks=np.linspace(vmin,vmax,5)[1:]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v:.0f}" for v in ticks],color=t["tick"],fontsize=7)
    ax.set_title(f"{tmpl['theme']}\n{entity_name} (Polar Bar)",fontsize=13,fontweight="bold",color=t["title"],pad=22)
    plt.tight_layout()
    dp=[{"series_name":entity_name,"x_value":a,"y_value":round(float(v),2)} for a,v in zip(axes_lbl,vals)]
    return fig,mk_json(f"{tmpl['theme']} — {entity_name} (Polar Bar)","Asse",tmpl.get("unit","Valore"),dp)

def render_radar_delta(tmpl,t,_):
    keys=list(tmpl["entities"].keys())
    if len(keys)<2: keys=keys*2
    bname,tname=random.sample(keys,2)
    baseline=jitter(tmpl["entities"][bname],tmpl,0.04)
    target  =jitter(tmpl["entities"][tname],tmpl,0.04)
    axes_lbl=tmpl["axes"]; n=len(axes_lbl); angs=angles_for(n)
    vmin,vmax=tmpl["value_range"]
    cb=t["palette"][0]; ct=t["palette"][1%len(t["palette"])]; cd=t["palette"][2%len(t["palette"])]
    fig=plt.figure(figsize=(9,9)); fig.patch.set_facecolor(t["fig_bg"])
    ax=radar_ax(fig,111,t); style_axes(ax,axes_lbl,vmin,vmax,t)
    draw_poly(ax,angs,baseline,cb,1.8,0.08,0.75,ls="--")
    draw_poly(ax,angs,target,ct,2.4,t["alpha_fill"]+0.08,t["alpha_line"])
    for ang,bv,tv in zip(angs[:-1],baseline,target):
        if abs(tv-bv)>(vmax-vmin)*0.04:
            ax.annotate("",xy=(ang,tv),xytext=(ang,bv),
                        arrowprops=dict(arrowstyle="-|>" if tv>bv else "<|-",color=cd,lw=1.5))
    patches=[mpatches.Patch(color=cb,label=f"Baseline: {bname}",alpha=0.75),
             mpatches.Patch(color=ct,label=f"Target: {tname}",alpha=0.85),
             mpatches.Patch(color=cd,label="Δ Variazione",alpha=0.85)]
    ax.legend(handles=patches,loc="upper right",bbox_to_anchor=(1.42,1.15),
              frameon=False,fontsize=9,labelcolor=t["label"])
    ax.set_title(f"{tmpl['theme']} (Delta)",fontsize=13,fontweight="bold",color=t["title"],pad=22)
    plt.tight_layout()
    dp=([{"series_name":bname,"x_value":a,"y_value":round(float(v),2)} for a,v in zip(axes_lbl,baseline)]+
        [{"series_name":tname,"x_value":a,"y_value":round(float(v),2)} for a,v in zip(axes_lbl,target)])
    return fig,mk_json(f"{tmpl['theme']} (Delta)","Asse","Valore",dp)


RENDERERS = {
    "simple_radar":render_simple_radar,"multi_radar":render_multi_radar,
    "filled_radar":render_filled_radar,"radar_small_multiples":render_radar_small_multiples,
    "radar_with_dots":render_radar_with_dots,"radar_polar_bar":render_radar_polar_bar,
    "radar_delta":render_radar_delta,
}

def generate_charts(n:int):
    used=set()
    print(f"\nGenerazione di {n} grafici radar...\n")
    for i in range(1,n+1):
        for _ in range(400):
            st=random.choice(SUBTYPES); tmpl=random.choice(DATASET_TEMPLATES); theme=random.choice(CHART_THEMES)
            key=(st,tmpl["theme"],theme["name"])
            if key not in used: used.add(key); break
        print(f"  [{i:>3}/{n}] {st:26s} | {tmpl['theme'][:36]:36s} | {theme['name']}")
        try:
            fig,data=RENDERERS[st](tmpl,theme,i)
        except Exception as e:
            print(f"         Errore: {e}"); import traceback; traceback.print_exc(); continue
        base=f"radar_{i:03d}_{st}"
        img_p = os.path.join(IMG_OUTPUT_DIR, base+".png")
        jsn_p = os.path.join(JSON_OUTPUT_DIR, base+".json")
        fig.savefig(img_p,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.close(fig)
        with open(jsn_p,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        print(f"          {img_p}  +  {jsn_p}")

if __name__=="__main__":
    while True:
        try:
            n=int(input("Quanti grafici radar vuoi generare? "))
            if n>0: break
            print("Inserisci un numero maggiore di 0.")
        except ValueError: print("Inserisci un numero intero valido.")
    generate_charts(n)
