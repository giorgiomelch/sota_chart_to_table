import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from src.evaluation.metric import table_datapoints_precision_recall
from src.utils.annotation2markdown import (
    json_to_markdown, 
    json_to_markdown_scatter, 
    json_to_markdown_errorpoint, 
    json_to_markdown_box
)

# --- CONFIGURAZIONE ---
PREDICTIONS_ROOT = Path("outputs/predictions")
GROUNDTRUTH_ROOT = Path("data/groundtruth")
IMAGES_ROOT = Path("data/images")
METRICS_OUTPUT = Path("outputs/metrics")

# --- UTILS ---

def get_available_models():
    """Rileva dinamicamente i modelli presenti in ordine alfabetico."""
    if not PREDICTIONS_ROOT.exists():
        return []
    return sorted([d.name for d in PREDICTIONS_ROOT.iterdir() if d.is_dir()])

def get_markdown_formatter(chart_type):
    conversion_map = {
        "scatter": json_to_markdown_scatter,
        "errorpoint": json_to_markdown_errorpoint,
        "box": json_to_markdown_box
    }
    return conversion_map.get(chart_type, json_to_markdown)

def load_prediction(pred_path, markdown_fn):
    """Carica la predizione supportando JSON (VLM) e TXT (DePlot)."""
    if not pred_path.exists():
        return None
    
    if pred_path.suffix == '.json':
        try:
            result = markdown_fn(pred_path)
            return list(result) if isinstance(result, tuple) else [result, result]
        except Exception:
            return None
    
    if pred_path.suffix == '.txt':
        content = pred_path.read_text(encoding='utf-8')
        return [content, content]
    
    return None

# --- CORE CALCULATION ---

def compute_metrics_for_class(model_name, dataset_type, chart_class):
    """Calcola la media F1 per una specifica classe di grafico e modello."""
    pred_dir = PREDICTIONS_ROOT / model_name / dataset_type / chart_class
    gt_dir = GROUNDTRUTH_ROOT / dataset_type / chart_class
    
    if not pred_dir.exists() or not gt_dir.exists():
        return 0.0

    markdown_fn = get_markdown_formatter(chart_class)
    f1_scores = []

    # Scansione ricorsiva per supportare la struttura di PMC (con sottocartelle difficulty)
    for gt_file in gt_dir.rglob("*.json"):
        # Ricostruiamo il path della predizione relativo alla classe
        rel_path = gt_file.relative_to(gt_dir)
        pred_file = pred_dir / rel_path
        
        # Prova con .json, se non esiste prova con .txt
        if not pred_file.exists():
            pred_file = pred_file.with_suffix('.txt')
            
        pred_table = load_prediction(pred_file, markdown_fn)
        if not pred_table:
            continue

        result_gt = markdown_fn(gt_file)
        gt_table = list(result_gt) if isinstance(result_gt, tuple) else [result_gt, result_gt]
        metrics = table_datapoints_precision_recall(
            [[pred_table[0]], [pred_table[1]]], 
            [gt_table[0], gt_table[1]]
        )
        '''dacanc
        print(pred_table[0])
        print(pred_table[1])
        print(gt_table[0])
        print(gt_table[1])
        print(metrics['table_datapoints_f1'])
        print(crusha il programma)
        '''
        f1_scores.append(metrics['table_datapoints_f1'])
    return statistics.mean(f1_scores) if f1_scores else 0.0

# --- VISUALIZATION ---

def salva_grafico_comparativo(dati_f1, dataset_label):
    """Genera un grafico a barre raggruppate per tutti i modelli rilevati."""
    if not dati_f1: return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    x = np.arange(len(chart_classes))
    width = 0.8 / len(model_names) # Spaziatura dinamica in base al numero di modelli
    
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')

    for i, model in enumerate(model_names):
        scores = [dati_f1[cc].get(model, 0.0) for cc in chart_classes]
        offset = (i - len(model_names)/2 + 0.5) * width
        rects = ax.bar(x + offset, scores, width, label=model)
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

    ax.set_ylabel('F1 Score (%)')
    ax.set_title(f'Benchmark Risultati - Dataset {dataset_label.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(chart_classes, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = METRICS_OUTPUT / f"f1_comparison_{dataset_label.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafico salvato: {output_path}")

# --- MAIN ---

def run_evaluation():
    models = get_available_models()
    if not models:
        print("Errore: Nessun modello trovato in outputs/predictions")
        return

    print(f"Modelli rilevati per il benchmark: {', '.join(models)}")

    for dataset_type in ["pmc", "synthetic"]:
        print(f"\nAnalisi dataset: {dataset_type.upper()}...")
        
        img_base_dir = IMAGES_ROOT / dataset_type
        if not img_base_dir.exists():
            print(f"Skip: Directory {img_base_dir} non trovata.")
            continue

        chart_classes = sorted([d.name for d in img_base_dir.iterdir() if d.is_dir()])
        results_f1 = {cc: {} for cc in chart_classes}

        for chart_class in chart_classes:
            for model in models:
                f1_val = compute_metrics_for_class(model, dataset_type, chart_class)
                print(f"{chart_class} {model} f1: {f1_val}")
                results_f1[chart_class][model] = f1_val

        salva_grafico_comparativo(results_f1, dataset_type)

if __name__ == "__main__":
    run_evaluation()