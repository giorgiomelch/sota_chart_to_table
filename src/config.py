"""
Configurazione centralizzata dei path per il benchmark sota_chart_to_table.

Variabili d'ambiente:
  DRIVE_BASE_DIR   Path Drive per pesi e output (predictions, metrics, reports).
  DATA_DIR         Path della cartella data/ con images/ e groundtruth/.
                   Se non impostata, usa PROJECT_ROOT/data.

Esempio Colab:
    python run_benchmark.py \\
        --drive-path /content/drive/MyDrive/MioProgetto \\
        --data-path  /content/drive/MyDrive/Progetti_GitHub/data_ChartToTable
"""

import os
from pathlib import Path

# --- Rilevamento ambiente ---

_is_colab = Path("/content").exists() and Path("/content/sota_chart_to_table").exists()

if _is_colab:
    PROJECT_ROOT = Path("/content/sota_chart_to_table")
else:
    # Due livelli sopra src/config.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Drive base per output e pesi (opzionale) ---

_drive_env = os.environ.get("DRIVE_BASE_DIR")
DRIVE_BASE_DIR: Path | None = Path(_drive_env) if _drive_env else None

# --- Path dati di input ---
# DATA_DIR può puntare a una cartella su Drive che contiene images/ e groundtruth/

_data_env = os.environ.get("DATA_DIR")
_data_base: Path = Path(_data_env) if _data_env else PROJECT_ROOT / "data"

IMAGES_DIR = _data_base / "images"
GROUNDTRUTH_DIR = _data_base / "groundtruth"

# --- Path di output e pesi (su Drive se disponibile, altrimenti locali) ---

_out_base: Path = DRIVE_BASE_DIR if DRIVE_BASE_DIR is not None else PROJECT_ROOT

WEIGHTS_DIR = _out_base / "weights"
PREDICTIONS_DIR = _out_base / "outputs" / "predictions"
METRICS_DIR = _out_base / "outputs" / "metrics"
REPORTS_DIR = _out_base / "outputs" / "reports"
