import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.evaluation.metric import table_datapoints_precision_recall
from src.utils.annotation2markdown import (
    json_to_markdown, 
    json_to_markdown_scatter, 
    json_to_markdown_errorpoint, 
    json_to_markdown_box
)

# --- FUNZIONI DI SUPPORTO ---

def safe_mean(values_list):
    """Restituisce la media di una lista, o 0.0 se la lista è vuota per evitare errori."""
    return statistics.mean(values_list) if values_list else 0.0

def load_prediction(pred_json_path, pred_txt_path, markdown_fn):
    """Carica la predizione (Gemini o DePlot) e garantisce un formato [originale, trasposta]."""
    if pred_json_path.exists():
        result = markdown_fn(pred_json_path)
        return list(result) if isinstance(result, tuple) else [result, result]
    
    if pred_txt_path.exists():
        content = pred_txt_path.read_text(encoding='utf-8')
        return [content, content]
    
    return None

def get_markdown_formatter(chart_type):
    """Mappa il tipo di grafico alla funzione di parsing JSON appropriata."""
    conversion_map = {
        "scatter": json_to_markdown_scatter,
        "errorpoint": json_to_markdown_errorpoint,
        "box": json_to_markdown_box
    }
    return conversion_map.get(chart_type, json_to_markdown)

def salva_grafico_f1(dati_f1, nome_dataset, directory_output="outputs/metrics"):
    """
    Genera un grafico a barre raggruppate per i punteggi F1 e lo salva su disco.
    """
    if not dati_f1:
        print(f"Nessun dato disponibile per generare il grafico di {nome_dataset}.")
        return

    out_dir = Path(directory_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tipi_grafico = list(dati_f1.keys())
    modelli = list(next(iter(dati_f1.values())).keys())

    x = np.arange(len(tipi_grafico))
    width = 0.35
    moltiplicatore = 0

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    for modello in modelli:
        punteggi = [dati_f1[tg].get(modello, 0) for tg in tipi_grafico]
        offset = width * moltiplicatore
        rects = ax.bar(x + offset, punteggi, width, label=modello)
        ax.bar_label(rects, padding=3, fmt='%.1f')
        moltiplicatore += 1

    ax.set_ylabel('F1 Score (%)')
    ax.set_title(f'Confronto F1 Score - Dataset {nome_dataset.upper()}')
    ax.set_xticks(x + width / 2, tipi_grafico)
    ax.legend(loc='upper left', ncols=len(modelli))
    ax.set_ylim(0, 100)

    percorso_salvataggio = out_dir / f"f1_score_{nome_dataset.lower()}.png"
    plt.savefig(percorso_salvataggio, dpi=300)
    plt.close()
    
    print(f"Grafico salvato in: {percorso_salvataggio}")


# --- FUNZIONI DI CALCOLO METRICHE ---

def compute_pmc_metrics(path_to_preds, path_to_gt):
    results_by_diff = {}
    global_accum = {'p': [], 'r': [], 'f1': []}

    for diff_class in path_to_gt.iterdir():
        if not diff_class.is_dir(): continue
            
        class_name = diff_class.name
        results_by_diff[class_name] = {'p': [], 'r': [], 'f1': []}
        
        for gt_file in diff_class.glob("*.json"):
            pred_path_json = path_to_preds / class_name / gt_file.name
            pred_path_txt = pred_path_json.with_suffix('.txt')
            
            pred_table = load_prediction(pred_path_json, pred_path_txt, json_to_markdown)
            if not pred_table: continue

            result_gt = json_to_markdown(gt_file)
            gt_table = list(result_gt) if isinstance(result_gt, tuple) else [result_gt, result_gt]
            
            metrics = table_datapoints_precision_recall(
                [[pred_table[0]], [pred_table[1]]], 
                [gt_table[0], gt_table[1]]
            )

            results_by_diff[class_name]['p'].append(metrics['table_datapoints_precision'])
            results_by_diff[class_name]['r'].append(metrics['table_datapoints_recall'])
            results_by_diff[class_name]['f1'].append(metrics['table_datapoints_f1'])

            global_accum['p'].append(metrics['table_datapoints_precision'])
            global_accum['r'].append(metrics['table_datapoints_recall'])
            global_accum['f1'].append(metrics['table_datapoints_f1'])

    final_metrics = {'classes': {}, 'global': {}}
    for cn, m in results_by_diff.items():
        if m['f1']:
            final_metrics['classes'][cn] = {
                'p': safe_mean(m['p']), 'r': safe_mean(m['r']), 'f1': safe_mean(m['f1'])
            }

    if global_accum['f1']:
        final_metrics['global'] = {
            'p': safe_mean(global_accum['p']), 'r': safe_mean(global_accum['r']), 'f1': safe_mean(global_accum['f1'])
        }
    return final_metrics


def compute_synthetic_metrics(path_to_preds, path_to_gt, chart_type):
    markdown_fn = get_markdown_formatter(chart_type)
    global_accum = {'p': [], 'r': [], 'f1': []}

    for gt_file in path_to_gt.glob("*.json"):
        pred_path_json = path_to_preds / gt_file.name
        pred_path_txt = pred_path_json.with_suffix('.txt')
        
        pred_table = load_prediction(pred_path_json, pred_path_txt, markdown_fn)
        if not pred_table: continue

        result_gt = markdown_fn(gt_file)
        gt_table = list(result_gt) if isinstance(result_gt, tuple) else [result_gt, result_gt]
        
        metrics = table_datapoints_precision_recall(
            [[pred_table[0]], [pred_table[1]]], 
            [gt_table[0], gt_table[1]]
        )

        global_accum['p'].append(metrics['table_datapoints_precision'])
        global_accum['r'].append(metrics['table_datapoints_recall'])
        global_accum['f1'].append(metrics['table_datapoints_f1'])

    if not global_accum['f1']:
        return None

    return {
        'p': safe_mean(global_accum['p']),
        'r': safe_mean(global_accum['r']),
        'f1': safe_mean(global_accum['f1'])
    }


# --- ORCHESTRATORE ---

def run_evaluation():
    """Funzione principale che esegue l'intera pipeline di valutazione."""
    
    # 1. Analisi PMC
    print("\nInizio analisi dataset PMC...")
    pmc_data_dir = Path("data/images/pmc/")
    gt_pmc = Path("data/groundtruth/pmc/")
    pmc_models = {
        "Gemini": "outputs/predictions/Gemini/pmc/", 
        "DePlot": "outputs/predictions/DePlot/pmc/"
    }

    if pmc_data_dir.exists():
        pmc_charts = sorted([d.name for d in pmc_data_dir.iterdir() if d.is_dir()])
        dati_f1_pmc = {chart: {} for chart in pmc_charts}

        for model_name, pred_base in pmc_models.items():
            for chart in pmc_charts:
                percorso_pred = Path(pred_base) / chart
                percorso_gt = gt_pmc / chart
                
                if percorso_pred.exists() and percorso_gt.exists():
                    res = compute_pmc_metrics(percorso_pred, percorso_gt)
                    f1_score = res.get('global', {}).get('f1', 0.0) if res else 0.0
                else:
                    f1_score = 0.0
                    
                dati_f1_pmc[chart][model_name] = f1_score

        salva_grafico_f1(dati_f1_pmc, "PMC")
    else:
        print(f"Cartella dati PMC non trovata in {pmc_data_dir}")

    # 2. Analisi SINTETICI
    print("\nInizio analisi dataset Synthetic...\n")
    synth_data_dir = Path("data/images/synthetic/")
    gt_synth = Path("data/groundtruth/synthetic/")
    synth_models = {
        "Gemini": "outputs/predictions/Gemini/synthetic/", 
        "DePlot": "outputs/predictions/DePlot/synthetic/"
    }

    if synth_data_dir.exists():
        synth_charts = sorted([d.name for d in synth_data_dir.iterdir() if d.is_dir()])
        dati_f1_synth = {chart: {} for chart in synth_charts}

        for model_name, pred_base in synth_models.items():
            for chart in synth_charts:
                percorso_pred = Path(pred_base) / chart
                percorso_gt = gt_synth / chart
                
                if percorso_pred.exists() and percorso_gt.exists():
                    res = compute_synthetic_metrics(percorso_pred, percorso_gt, chart)
                    f1_score = res.get('f1', 0.0) if res else 0.0
                else:
                    f1_score = 0.0
                    
                dati_f1_synth[chart][model_name] = f1_score

        salva_grafico_f1(dati_f1_synth, "Synthetic")
    else:
        print(f"Cartella dati Sintetici non trovata in {synth_data_dir}")


if __name__ == "__main__":
    run_evaluation()