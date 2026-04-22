import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json
import math
from collections import defaultdict
from typing import Any
from src.evaluation.rms_metric import compute_rms

# --- CONFIGURAZIONE ---
from src.config import PREDICTIONS_DIR as PREDICTIONS_ROOT
from src.config import GROUNDTRUTH_DIR as GROUNDTRUTH_ROOT
from src.config import IMAGES_DIR as IMAGES_ROOT
from src.config import METRICS_DIR as METRICS_OUTPUT

# --- UTILS ---

def get_available_models():
    """Rileva dinamicamente i modelli presenti in ordine alfabetico."""
    if not PREDICTIONS_ROOT.exists():
        return []
    return sorted([d.name for d in PREDICTIONS_ROOT.iterdir() if d.is_dir()])

def sottrai_valore_base(valore, base):
    if isinstance(valore, (int, float)) and not isinstance(valore, bool):
        return valore - base
    elif isinstance(valore, dict):
        return {
            chiave: (sub_valore - base if isinstance(sub_valore, (int, float)) and not isinstance(sub_valore, bool) else sub_valore)
            for chiave, sub_valore in valore.items()
        }
    return valore

def estrai_basi(dati_json):
    """Estrae i valori base da un dizionario JSON e li restituisce in un dizionario."""
    return {
        "x_base": dati_json.get("x_base"),
        "y_base": dati_json.get("y_base"),
        "w_base": dati_json.get("w_base"),
        "z_base": dati_json.get("z_base")
    }

def normalizza_valori(dati_json, basi):
    """Applica il dizionario delle basi ai data_points forniti."""
    if "data_points" not in dati_json or not isinstance(dati_json["data_points"], list):
        return dati_json

    for punto in dati_json["data_points"]:
        if basi.get("x_base") is not None and "x_value" in punto:
            punto["x_value"] = sottrai_valore_base(punto["x_value"], basi["x_base"])
            
        if basi.get("y_base") is not None and "y_value" in punto:
            punto["y_value"] = sottrai_valore_base(punto["y_value"], basi["y_base"])
            
        if basi.get("w_base") is not None and "w_value" in punto:
            punto["w_value"] = sottrai_valore_base(punto["w_value"], basi["w_base"])
            
        if basi.get("z_base") is not None and "z_value" in punto:
            punto["z_value"] = sottrai_valore_base(punto["z_value"], basi["z_base"])

    return dati_json

def _merge_chart_list(charts: list) -> dict:
    """
    Converte una lista di chart in un unico dict.
    Se tutti i data_points hanno series_name == 'Main', usa chart_title come series_name.
    Altrimenti restituisce il primo elemento della lista.
    """
    if not charts:
        return {}
    all_main = all(
        dp.get("series_name", "Main") == "Main"
        for chart in charts
        for dp in chart.get("data_points", [])
    )
    if not all_main:
        return charts[0]
    merged_points = []
    for chart in charts:
        title = chart.get("chart_title") or "Main"
        for dp in chart.get("data_points", []):
            new_dp = dict(dp)
            new_dp["series_name"] = title
            merged_points.append(new_dp)
    return {"data_points": merged_points}


def deplot_txt_to_json(txt: str) -> dict:
    """
    Parse a DePlot markdown-table .txt output into a standard chart JSON dict.

    DePlot format
    -------------
    TITLE | <title>
    <x_axis_label> | <series1> | <series2> | ...
    <x_value>      | <y1>      | <y2>      | ...
    ...

    Returns a categorical_x dict compatible with compute_rms.
    Numeric cells are converted to float; non-numeric are kept as strings.
    """
    lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
    if not lines:
        return {"data_points": []}

    chart_title = None
    header_idx = 0

    # Optional TITLE row
    if lines[0].upper().startswith("TITLE"):
        parts = lines[0].split("|", 1)
        raw_title = parts[1].strip() if len(parts) > 1 else ""
        chart_title = raw_title if raw_title else None
        header_idx = 1

    if header_idx >= len(lines):
        return {"data_points": [], **({"chart_title": chart_title} if chart_title else {})}

    # Header row: first cell = x-axis label (discarded), rest = series names
    header_parts = [p.strip() for p in lines[header_idx].split("|")]
    series_names = header_parts[1:]  # may be empty

    data_points = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue
        x_val = parts[0]
        for i, series in enumerate(series_names):
            raw_y = parts[i + 1] if i + 1 < len(parts) else ""
            try:
                y_val: Any = float(raw_y)
            except (ValueError, TypeError):
                y_val = raw_y
            data_points.append({
                "series_name": series,
                "x_value": x_val,
                "y_value": y_val,
            })

    result: dict = {"categorical_axis": "x", "data_points": data_points}
    if chart_title is not None:
        result["chart_title"] = chart_title
    return result


def load_prediction(pred_path, basi_gt):
    """
    Carica la predizione e applica la normalizzazione dei valori base.

    Supporta:
      - .json  → formato chart JSON standard
      - .txt   → formato DePlot markdown table (convertito via deplot_txt_to_json)

    Se pred_path ha estensione .json ma il file non esiste, prova automaticamente
    il corrispondente .txt (utile quando la valutazione itera su file .json del GT).
    """
    # Fallback .json → .txt (DePlot salva in .txt)
    if not pred_path.exists() and pred_path.suffix == '.json':
        pred_path = pred_path.with_suffix('.txt')

    if not pred_path.exists():
        return None

    if pred_path.suffix == '.json':
        try:
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            if isinstance(pred_data, list):
                pred_data = _merge_chart_list(pred_data)
            if not isinstance(pred_data, dict):
                return None
            return normalizza_valori(pred_data, basi_gt)
        except Exception:
            return None

    if pred_path.suffix == '.txt':
        try:
            txt = pred_path.read_text(encoding='utf-8')
            pred_data = deplot_txt_to_json(txt)
            return normalizza_valori(pred_data, basi_gt)
        except Exception:
            return None

    return None

# --- CORE CALCULATION ---

def compute_metrics_for_class(model_name, dataset_type, chart_class):
    """Calcola l'F1 per una specifica classe e restituisce tuple: (numero_elementi, f1_score)."""
    pred_dir = PREDICTIONS_ROOT / model_name / dataset_type / chart_class
    gt_dir = GROUNDTRUTH_ROOT / dataset_type / chart_class

    if not pred_dir.exists() or not gt_dir.exists():
        return []

    f1_data = []

    for gt_file in gt_dir.rglob("*.json"):
        rel_path = gt_file.relative_to(gt_dir)
        pred_file = pred_dir / rel_path

        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        num_elementi = len(gt_data.get("data_points", []))

        basi_gt = estrai_basi(gt_data)
        gt_data_norm = normalizza_valori(gt_data, basi_gt)

        pred_data = load_prediction(pred_file, basi_gt)
        if pred_data is None:
            continue

        try:
            result = compute_rms(pred_data, gt_data_norm)
        except Exception as e:
            print(f"[WARN] compute_rms fallito su {gt_file.name}: {e}")
            continue

        f1_data.append((num_elementi, result['f1'] * 100))

    return f1_data

# --- VISUALIZATION ---

def salva_grafico_comparativo(dati_f1, dataset_label):
    """Genera il grafico a barre classico con deviazione standard."""
    if not dati_f1: return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    num_classes = len(chart_classes)
    num_models = len(model_names)

    fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')

    x_positions = np.arange(num_classes)
    global_avg_x = num_classes + 1  # posizione colonna media globale (con gap)
    bar_width = 0.8 / num_models
    colors = plt.cm.tab10.colors

    for i, model in enumerate(model_names):
        means = []
        stdevs = []
        all_scores = []  # accumulo per media globale

        for cc in chart_classes:
            # Estrae solo i valori F1 dalla tupla (num_elementi, f1)
            scores_tuples = dati_f1[cc].get(model, [])
            scores = [f1 for _, f1 in scores_tuples]
            all_scores.extend(scores)

            if len(scores) > 1:
                means.append(statistics.mean(scores))
                stdevs.append(statistics.stdev(scores))
            elif len(scores) == 1:
                means.append(scores[0])
                stdevs.append(0.0)
            else:
                means.append(0.0)
                stdevs.append(0.0)

        offset = (i - num_models/2 + 0.5) * bar_width

        lower_errors = [min(m, s) for m, s in zip(means, stdevs)]
        upper_errors = [min(100.0 - m, s) for m, s in zip(means, stdevs)]
        asymmetric_error = [lower_errors, upper_errors]

        rects = ax.bar(x_positions + offset, means, bar_width, label=model,
                       color=colors[i % len(colors)], alpha=0.8)

        ax.errorbar(x_positions + offset, means, yerr=asymmetric_error,
                    fmt='none', capsize=4, ecolor='black', elinewidth=1)

        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + 2,
                    f'{height:.1f}',
                    ha='center',
                    va='center',
                    fontsize=8,
                )

        # --- Barra media globale ---
        global_mean = statistics.mean(all_scores) if all_scores else 0.0
        ax.bar(global_avg_x + offset, global_mean, bar_width,
               color=colors[i % len(colors)], alpha=0.95,
               edgecolor='white', linewidth=0.8)
        if global_mean > 0:
            ax.text(
                global_avg_x + offset,
                global_mean + 2,
                f'{global_mean:.1f}',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
            )

    # Linea separatrice verticale tra classi e media globale
    ax.axvline(x=num_classes + 0.5, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_ylabel('F1 Score Medio')
    ax.set_title(f'Performance F1 Score Globale (Media ± Dev. Std.) - Dataset {dataset_label.upper()}')

    all_x_positions = list(x_positions) + [global_avg_x]
    all_x_labels = chart_classes + ['Media\nGlobale']
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_x_labels, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_ylim(bottom=0, top=105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = METRICS_OUTPUT / f"f1_barplot_stdev_{dataset_label.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafico a barre salvato: {output_path}")

def salva_grafico_facet_elementi(dati_f1, dataset_label):
    if not dati_f1: return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    num_classes = len(chart_classes)
    cols = min(3, num_classes)
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), layout=None)
    
    if num_classes == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.tab10.colors

    for idx, chart_class in enumerate(chart_classes):
        ax = axes[idx]
        
        for i, model in enumerate(model_names):
            scores_data = dati_f1[chart_class].get(model, [])
            
            raggruppamento = defaultdict(list)
            for num_elem, f1 in scores_data:
                raggruppamento[num_elem].append(f1)
                
            if not raggruppamento:
                continue
                
            x_vals = sorted(raggruppamento.keys())
            y_means = [statistics.mean(raggruppamento[x]) for x in x_vals]
            
            ax.plot(x_vals, y_means, marker='o', linestyle='-', linewidth=2, 
                    color=colors[i % len(colors)], label=model)

        ax.set_title(chart_class.replace('_', ' ').upper())
        ax.set_xlabel('Numero di elementi (data_points)')
        ax.set_ylabel('F1 Score Medio')
        ax.set_ylim(bottom=0, top=105)
        ax.grid(True, linestyle='--', alpha=0.6)

    for idx in range(num_classes, len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    title_str = f'Andamento F1 Score per numero di elementi - Dataset {dataset_label.upper()}'
    fig.suptitle(title_str, fontsize=16, y=0.975)

    handles, labels = ax.get_legend_handles_labels()
    
    fig.legend(handles, labels, 
               loc='upper left', 
               bbox_to_anchor=(1.02, 1.0), 
               ncol=1, 
               fontsize=12, 
               title="Modelli", 
               title_fontsize=13,
               frameon=True, 
               shadow=False)

    output_path = METRICS_OUTPUT / f"f1_facet_elements_{dataset_label.lower()}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico facet salvato con legenda esterna: {output_path}")

def stampa_risultati_f1(dati_f1, dataset_label):
    if not dati_f1:
        return

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    col_w = 10
    label_w = max(20, max(len(cc) for cc in chart_classes + ["Media Globale"]) + 2)

    header = "Classe".ljust(label_w) + "".join(m[:col_w-1].rjust(col_w) for m in model_names)
    sep = "-" * len(header)

    print(f"\n=== F1 Score — Dataset: {dataset_label.upper()} ===\n")
    print(header)
    print(sep)

    global_scores = {m: [] for m in model_names}

    for cc in chart_classes:
        row = cc.ljust(label_w)
        for model in model_names:
            scores = [f1 for _, f1 in dati_f1[cc].get(model, [])]
            global_scores[model].extend(scores)
            mean = statistics.mean(scores) if scores else 0.0
            row += f"{mean:>{col_w}.1f}"
        print(row)

    print(sep)
    row = "Media Globale".ljust(label_w)
    for model in model_names:
        mean = statistics.mean(global_scores[model]) if global_scores[model] else 0.0
        row += f"{mean:>{col_w}.1f}"
    print(row)
    print()

# --- MAIN ---

def run_evaluation():
    models = get_available_models()
    if not models:
        print("Errore: Nessun modello trovato in outputs/predictions")
        return

    print(f"Modelli rilevati per il benchmark: {', '.join(models)}")

    for dataset_type in ["arXiv","PMCharts", "synthetic"]:
        print(f"\nAnalisi dataset: {dataset_type.upper()}...")
        
        img_base_dir = IMAGES_ROOT / dataset_type
        if not img_base_dir.exists():
            print(f"Skip: Directory {img_base_dir} non trovata.")
            continue

        chart_classes = sorted([d.name for d in img_base_dir.iterdir() if d.is_dir()])
        results_f1 = {cc: {} for cc in chart_classes}

        for chart_class in chart_classes:
            for model in models:
                f1_data = compute_metrics_for_class(model, dataset_type, chart_class)
                results_f1[chart_class][model] = f1_data
                
        # Richiama entrambe le funzioni di plotting passando lo stesso dizionario dati
        salva_grafico_comparativo(results_f1, dataset_type)
        salva_grafico_facet_elementi(results_f1, dataset_type)
        stampa_risultati_f1(results_f1, dataset_type)

if __name__ == "__main__":
    run_evaluation()