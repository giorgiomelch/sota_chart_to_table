import base64
import json
import os
from pathlib import Path
from IPython.display import display, HTML
import random
from src.utils.annotation2markdown import *
from src.evaluation.metric import table_datapoints_precision_recall

def genera_riga_html(img_path, txt_path, gemini_json_path, gt_json_path, chart_type):
    conversion_map = {
        "scatter": json_to_markdown_scatter,
        "errorpoint": json_to_markdown_errorpoint,
        "box": json_to_markdown_box,
        "default": json_to_markdown 
    }
    to_markdown_fn = conversion_map.get(chart_type, conversion_map["default"])

    img_path = Path(img_path)
    txt_path = Path(txt_path)
    gemini_json_path = Path(gemini_json_path)
    gt_json_path = Path(gt_json_path)
    
    if not img_path.exists():
        return f"<div style='color: red;'>Immagine non trovata: {img_path}</div>"
        
    # Lettura immagine
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Lettura testo DePlot
    testo_deplot = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    
    # Generazione di un unico valore casuale per allineare GT e Gemini
    is_originale = random.random() > 0.5
    label_tipo = "Originale" if is_originale else "Trasposta"
    
    # Inizializzazione sicura delle variabili
    gt_original, gt_transposed = "", ""
    gemini_original, gemini_transposed = "", ""
    
    # Lettura e conversione Ground Truth
    if gt_json_path.exists():
        gt_original, gt_transposed = to_markdown_fn(gt_json_path)
        testo_gt = gt_original if is_originale else gt_transposed
    else:
        testo_gt = f"Ground Truth non trovata in {gt_json_path}."

    # Lettura e conversione JSON Gemini
    if gemini_json_path.exists():
        gemini_original, gemini_transposed = to_markdown_fn(gemini_json_path)
        testo_gemini = gemini_original if is_originale else gemini_transposed
    else:
        testo_gemini = "JSON Gemini non trovato."

    # Calcolo Metriche
    gemini_metrics_html = ""
    deplot_metrics_html = ""
    
    # Procedi con il calcolo solo se la Ground Truth è presente
    if gt_original and gt_transposed:
        # Metriche Gemini
        if gemini_original and gemini_transposed:
            g_metrics = table_datapoints_precision_recall(
                [[gemini_original], [gemini_transposed]], 
                [gt_original, gt_transposed]
            )
            gemini_metrics_html = (
                f"<div style='margin-bottom: 10px; color: #82e0aa; font-size: 12px; background: #111; padding: 6px; border-radius: 4px; font-weight: bold;'>"
                f"Precision: {g_metrics['table_datapoints_precision']:.1f} | "
                f"Recall: {g_metrics['table_datapoints_recall']:.1f} | "
                f"F1: {g_metrics['table_datapoints_f1']:.1f}"
                f"</div>"
            )
        
        # Metriche DePlot
        if testo_deplot:
            d_metrics = table_datapoints_precision_recall(
                [[testo_deplot], [testo_deplot]], 
                [gt_original, gt_transposed]
            )
            deplot_metrics_html = (
                f"<div style='margin-bottom: 10px; color: #82e0aa; font-size: 12px; background: #111; padding: 6px; border-radius: 4px; font-weight: bold;'>"
                f"Precision: {d_metrics['table_datapoints_precision']:.1f} | "
                f"Recall: {d_metrics['table_datapoints_recall']:.1f} | "
                f"F1: {d_metrics['table_datapoints_f1']:.1f}"
                f"</div>"
            )
            
    if not testo_deplot:
        testo_deplot = "Testo DePlot non trovato."

    # Costruzione HTML della singola riga
    html_code = f"""
    <div style="display: flex; gap: 20px; align-items: stretch; margin-bottom: 40px; padding-bottom: 40px; border-bottom: 2px solid #444;">
        <div style="flex: 1; border: 1px solid #444; padding: 10px; border-radius: 8px; background-color: #2d2d2d;">
            <div style="margin-bottom: 10px; color: #aaa; font-family: sans-serif; font-size: 12px;">{img_path.name}</div>
            <img src="data:image/jpeg;base64,{img_b64}" style="max-width: 100%; height: auto; border-radius: 4px;">
        </div>
        
        <div style="flex: 1; border: 1px solid #444; padding: 15px; border-radius: 8px; background-color: #2c3e50; color: #e0e0e0; white-space: pre-wrap; font-family: monospace; overflow-x: auto; font-size: 13px;">
            <div style="margin-bottom: 10px; color: #888; font-size: 11px;">{gt_json_path.name} (Ground Truth - {label_tipo})</div>
{testo_gt}
        </div>
        
        <div style="flex: 1; border: 1px solid #444; padding: 15px; border-radius: 8px; background-color: #1a2b3c; color: #e0e0e0; white-space: pre-wrap; font-family: monospace; overflow-x: auto; font-size: 13px;">
            <div style="margin-bottom: 10px; color: #888; font-size: 11px;">{gemini_json_path.name} (Gemini - {label_tipo})</div>
            {gemini_metrics_html}
{testo_gemini}
        </div>
        
        <div style="flex: 1; border: 1px solid #444; padding: 15px; border-radius: 8px; background-color: #1e1e1e; color: #d4d4d4; white-space: pre-wrap; font-family: monospace; overflow-x: auto; font-size: 13px;">
            <div style="margin-bottom: 10px; color: #888; font-size: 11px;">{txt_path.name} (DePlot)</div>
            {deplot_metrics_html}
{testo_deplot}
        </div>
    </div>
    """
    return html_code

def crea_report_completo(lista_file, output_filename="report_completo.html", chart_class="default"):
    """
    lista_file: una lista di tuple o liste, dove ogni elemento contiene i 4 percorsi:
    [(img_path1, txt_path1, gemini_path1, gt_path1), (img_path2, txt_path2, ...)]
    """
    
    # Inizio del documento HTML con lo stile base
    html_completo = """
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body { background-color: #121212; color: #fff; padding: 20px; font-family: sans-serif; }
        </style>
    </head>
    <body>
        <h2>Report Confronto Modelli</h2>
    """
    
    # Genera e aggiunge l'HTML per ogni elemento nella lista
    for img, txt, gemini, gt in lista_file:
        riga_html = genera_riga_html(img, txt, gemini, gt, chart_class)
        html_completo += riga_html
        
    # Chiusura del documento HTML
    html_completo += """
    </body>
    </html>
    """
    
    # Salvataggio su file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_completo)
        
    print(f"Report completato e salvato in: {output_filename}")


def generate_chartclass_report(pmc_synt, chart_class_name):
    data_dir = Path(f"data/images/{pmc_synt}/{chart_class_name}")
    pred_dir = Path(f"outputs/predictions/DePlot/{pmc_synt}/{chart_class_name}")
    gemini_dir = Path(f"outputs/predictions/Gemini/{pmc_synt}/{chart_class_name}")
    gt_dir = Path(f"data/groundtruth/{pmc_synt}/{chart_class_name}")

    valid_extensions = ('.jpg', '.png', '.jpeg')
    file_list = [f for f in data_dir.rglob('*') if f.suffix.lower() in valid_extensions]

    lista_file_da_analizzare = []

    for img_file in file_list:
        percorso_relativo = img_file.relative_to(data_dir)
        
        txt_file = pred_dir / percorso_relativo.with_suffix('.txt')
        gemini_file = gemini_dir / percorso_relativo.with_suffix('.json')
        gt_file = gt_dir / percorso_relativo.with_suffix('.json')
        
        lista_file_da_analizzare.append((img_file, txt_file, gemini_file, gt_file))

    output_html_name = Path(f"outputs/reports/{pmc_synt}/report_{chart_class_name}.html")
    output_html_name.parent.mkdir(parents=True, exist_ok=True)
    crea_report_completo(lista_file_da_analizzare, output_html_name, chart_class_name)

def generate_reports():
    data_path = Path(f"data/images/")
    categories = ["pmc", "synthetic"]
    for c in categories:
        data_path_c = data_path / c
        if not data_path_c.exists(): continue
        chart_classes = data_path_c.iterdir()
        for chart_class in chart_classes:
            generate_chartclass_report(c, chart_class.name)

if __name__ == "__main__":
    generate_reports()