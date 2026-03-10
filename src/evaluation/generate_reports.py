import base64
import json
import random
from pathlib import Path
from src.utils.annotation2markdown import *
from src.evaluation.metric import table_datapoints_precision_recall

# --- CONFIGURAZIONE ---
PREDICTIONS_ROOT = Path("outputs/predictions")
GROUNDTRUTH_ROOT = Path("data/groundtruth")
IMAGES_ROOT = Path("data/images")
REPORTS_ROOT = Path("outputs/reports")

def get_available_models():
    if not PREDICTIONS_ROOT.exists():
        return []
    models = [d.name for d in PREDICTIONS_ROOT.iterdir() if d.is_dir()]
    return sorted(models)

def genera_blocco_html(titolo, contenuto, metrics_html="", bg_color="#1e1e1e"):
    """
    Genera un singolo blocco della griglia. 
    Larghezza impostata per occupare 1/5 dello spazio (circa 20%).
    """
    return f"""
    <div style="flex: 0 0 calc(20% - 16px); min-width: 250px; border: 1px solid #444; padding: 12px; border-radius: 8px; background-color: {bg_color}; color: #e0e0e0; font-family: monospace; overflow-x: auto; font-size: 12px; box-sizing: border-box;">
        <div style="margin-bottom: 8px; color: #888; font-size: 10px; font-weight: bold; border-bottom: 1px solid #444; padding-bottom: 4px; text-transform: uppercase;">{titolo}</div>
        {metrics_html}
        <div style="white-space: pre-wrap; word-break: break-all;">{contenuto}</div>
    </div>
    """

def genera_sezione_immagine(img_path, dataset_type, chart_class, model_names):
    conversion_map = {
        "scatter": json_to_markdown_scatter,
        "errorpoint": json_to_markdown_errorpoint,
        "box": json_to_markdown_box,
        "default": json_to_markdown 
    }
    to_markdown_fn = conversion_map.get(chart_class, conversion_map["default"])
    rel_path = img_path.relative_to(IMAGES_ROOT / dataset_type / chart_class)
    gt_json_path = GROUNDTRUTH_ROOT / dataset_type / chart_class / rel_path.with_suffix('.json')
    
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    is_originale = random.random() > 0.5
    idx = 0 if is_originale else 1
    label_tipo = "Orig" if is_originale else "Trasp"

    # 1. Ground Truth
    gt_content, gt_data = "GT Assente", (None, None)
    if gt_json_path.exists():
        gt_data = to_markdown_fn(gt_json_path)
        gt_content = gt_data[idx]

    # 2. Raccolta Blocchi (Immagine + GT + Modelli)
    blocchi_html = []
    
    # Blocco Immagine
    img_html = f"""
    <div style="flex: 0 0 calc(20% - 16px); min-width: 250px; border: 1px solid #444; padding: 10px; border-radius: 8px; background-color: #2d2d2d; box-sizing: border-box;">
        <div style="margin-bottom: 8px; color: #aaa; font-size: 10px;">{img_path.name}</div>
        <img src="data:image/jpeg;base64,{img_b64}" style="width: 100%; height: auto; border-radius: 4px;">
    </div>
    """
    blocchi_html.append(img_html)
    
    # Blocco GT
    blocchi_html.append(genera_blocco_html(f"Ground Truth ({label_tipo})", gt_content, bg_color="#2c3e50"))
    
    # Blocchi Modelli
    for model in model_names:
        pred_path = PREDICTIONS_ROOT / model / dataset_type / chart_class / rel_path.with_suffix('.json')
        if not pred_path.exists():
            pred_path = pred_path.with_suffix('.txt')

        pred_content = "Assente"
        metrics_html = ""
        
        if pred_path.exists():
            if pred_path.suffix == '.json':
                try:
                    p_orig, p_trans = to_markdown_fn(pred_path)
                    pred_content = p_orig if is_originale else p_trans
                    if gt_data[0]:
                        m = table_datapoints_precision_recall([[p_orig], [p_trans]], [gt_data[0], gt_data[1]])
                        metrics_html = f"<div style='color: #82e0aa; font-size: 10px; background: #000; padding: 3px; margin-bottom: 5px;'>F1: {m['table_datapoints_f1']:.2f}</div>"
                except:
                    pred_content = "JSON Error"
            else:
                pred_content = pred_path.read_text(encoding="utf-8")

        blocchi_html.append(genera_blocco_html(f"{model} ({label_tipo})", pred_content, metrics_html))

    # Wrapper della singola immagine che permette il wrap dei blocchi interni
    return f"""
    <div style="margin-bottom: 60px; padding-bottom: 20px; border-bottom: 2px dashed #555;">
        <h3 style="color: #ccc; font-size: 14px; margin-bottom: 15px;">Analisi Immagine: {img_path.name}</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: flex-start;">
            {''.join(blocchi_html)}
        </div>
    </div>
    """

def generate_reports():
    model_names = get_available_models()
    for dataset_type in ["pmc", "synthetic"]:
        base_img_dir = IMAGES_ROOT / dataset_type
        if not base_img_dir.exists(): continue
        
        for chart_class_dir in base_img_dir.iterdir():
            if not chart_class_dir.is_dir(): continue
            
            chart_class = chart_class_dir.name
            images = [f for f in chart_class_dir.rglob('*') if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
            if not images: continue
            
            # Inizializza la stringa HTML
            html_content = f"""
            <html><head><meta charset='utf-8'><style>
                body {{ background-color: #121212; color: #fff; padding: 30px; font-family: sans-serif; }}
                h2 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
            </style></head>
            <body>
                <h2>Report Benchmark: {dataset_type.upper()} - {chart_class}</h2>
                <div style="margin-bottom: 30px; color: #888;">Modelli rilevati: <b>{', '.join(model_names)}</b></div>
            """
            
            # Accumula le righe dei grafici
            for img_path in images:
                html_content += genera_sezione_immagine(img_path, dataset_type, chart_class, model_names)
            
            # Chiudi i tag HTML nella stringa
            html_content += "</body></html>"
            
            # Definisci il percorso di output
            output_file = REPORTS_ROOT / dataset_type / f"report_{chart_class}.html"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # SCRITTURA CORRETTA: Chiama write_text sull'oggetto Path passandogli la stringa
            output_file.write_text(html_content, encoding="utf-8")
            
            print(f"Creato report: {output_file}")
            
if __name__ == "__main__":
    generate_reports()