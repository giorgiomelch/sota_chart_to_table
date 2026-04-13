"""
Punto di ingresso principale del benchmark sota_chart_to_table.

Uso:
    python run_benchmark.py --model qwen --dataset PMCharts
    python run_benchmark.py --model qwen,internvl --dataset all --drive-path /content/drive/MyDrive/MioProgetto
    python run_benchmark.py --model deplot --evaluate --report

Modelli disponibili (open-weight):
    qwen       Qwen2-VL (2B o 7B, default 2B)
    internvl   InternVL2.5 (2.5-2B o 2.5-8B, default 2.5-2B)
    phi        Phi-3.5-Vision (ottimizzato per Colab)
    deplot     DePlot / Pix2Struct

Argomenti:
    --model       Modelli da eseguire (virgola-separati o "all")
    --dataset     Dataset da usare: PMCharts, synthetic, arXiv, all (default: PMCharts)
    --tier        Variante del modello (es. 7B per Qwen, 2.5-8B per InternVL)
    --drive-path  Path su Google Drive dove salvare pesi e output
    --evaluate    Esegui la valutazione dopo le predizioni
    --report      Genera i report HTML dopo la valutazione
"""

import argparse
import os
import sys


AVAILABLE_MODELS = ["qwen", "internvl", "phi", "deplot"]
AVAILABLE_DATASETS = ["PMCharts", "synthetic", "arXiv"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark chart-to-table con modelli VLM open-weight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="",
        help='Modelli da eseguire, virgola-separati o "all". Es: qwen,internvl',
    )
    parser.add_argument(
        "--dataset",
        default="PMCharts",
        help='Dataset da usare, virgola-separati o "all". Es: PMCharts,synthetic',
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Variante del modello (es. 7B per Qwen, 2.5-8B per InternVL). "
             "Usato solo se si esegue un singolo modello.",
    )
    parser.add_argument(
        "--drive-path",
        dest="drive_path",
        default=None,
        help="Path base su Google Drive per pesi e output. "
             "Equivalente a impostare la variabile d'ambiente DRIVE_BASE_DIR.",
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        default=None,
        help="Path della cartella data/ contenente images/ e groundtruth/. "
             "Utile su Colab quando i dati sono su Drive. "
             "Equivalente a impostare la variabile d'ambiente DATA_DIR.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Esegui la valutazione RMS dopo le predizioni.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Genera i report HTML.",
    )
    return parser.parse_args()


def resolve_models(model_arg: str) -> list[str]:
    if not model_arg or model_arg.lower() == "all":
        return AVAILABLE_MODELS
    models = [m.strip().lower() for m in model_arg.split(",") if m.strip()]
    unknown = [m for m in models if m not in AVAILABLE_MODELS]
    if unknown:
        print(f"[ERRORE] Modelli non riconosciuti: {unknown}. Disponibili: {AVAILABLE_MODELS}")
        sys.exit(1)
    return models


def resolve_datasets(dataset_arg: str) -> list[str]:
    if not dataset_arg or dataset_arg.lower() == "all":
        return AVAILABLE_DATASETS
    datasets = [d.strip() for d in dataset_arg.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in AVAILABLE_DATASETS]
    if unknown:
        print(f"[ERRORE] Dataset non riconosciuti: {unknown}. Disponibili: {AVAILABLE_DATASETS}")
        sys.exit(1)
    return datasets


def main():
    args = parse_args()

    # Queste env var devono essere settate PRIMA di importare src.config
    if args.drive_path:
        os.environ["DRIVE_BASE_DIR"] = args.drive_path
        print(f"Drive path (output/pesi): {args.drive_path}")
    if args.data_path:
        os.environ["DATA_DIR"] = args.data_path
        print(f"Data path (immagini/GT):  {args.data_path}")

    # Import differiti: src.config legge DRIVE_BASE_DIR al momento dell'import
    from src.config import IMAGES_DIR

    models = resolve_models(args.model)
    datasets = resolve_datasets(args.dataset)

    for model_name in models:
        for dataset in datasets:
            dataset_path = IMAGES_DIR / dataset
            if not dataset_path.exists():
                print(f"[SKIP] Dataset {dataset} non trovato in {dataset_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Modello: {model_name.upper()} | Dataset: {dataset}")
            print(f"{'='*60}")

            if model_name == "qwen":
                from src.models.qwen import ask_qwen
                tier = args.tier or "2B"
                ask_qwen(tier=tier)

            elif model_name == "internvl":
                from src.models.internVL import ask_internvl
                tier = args.tier or "2.5-2B"
                ask_internvl(tier=tier)

            elif model_name == "phi":
                import importlib
                phi_module = importlib.import_module("src.models.phi_35")
                tier = args.tier or "3.5-Vision"
                phi_module.ask_phi(tier=tier)

            elif model_name == "deplot":
                from src.models.deplot import DePlot_predict
                DePlot_predict()

    if args.evaluate:
        print("\nAvvio valutazione...")
        from src.evaluation.evaluate import run_evaluation
        run_evaluation()

    if args.report:
        print("\nGenerazione report HTML...")
        from src.evaluation.generate_reports import generate_reports
        generate_reports()


if __name__ == "__main__":
    main()
