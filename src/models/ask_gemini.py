import os
import io
import json
from PIL import Image
from pathlib import Path
from google import genai
from google.genai import types
from src.utils.prompts import *

MAX_SIZE = 768
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_KEY_ENV_VAR = "GEMINI_API_KEY"

PROMPT2CHARTCLASS = {
    "area": PROMPT_AreaLineBarHistogram,
    "line": PROMPT_AreaLineBarHistogram,
    "bar": PROMPT_AreaLineBarHistogram,
    "histogram": PROMPT_AreaLineBarHistogram,      
    "scatter": PROMPT_Scatter,
    "radar": PROMPT_Radar,
    "pie": PROMPT_Pie,
    "venn": PROMPT_Venn,
    "box": PROMPT_Box,
    "errorpoint": PROMPT_Errorpoint,
    "violin": PROMPT_Violin,
    "bubble": PROMPT_Bubble                 
}


def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key non trovata. Imposta la variabile d'ambiente GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)
from PIL import Image

def prepara_immagine_per_api(percorso_immagine, max_size=768):
    """Legge un'immagine dal disco, la ridimensiona mantenendo l'aspect ratio
    e restituisce i byte in formato PNG."""
    with Image.open(percorso_immagine) as immagine:
        if max(immagine.size) > max_size:
            immagine.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        immagine.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

def estrai_dati_con_gemini(client, modello, prompt_testo, image_bytes):
    """Invia l'immagine e il prompt all'API di Gemini richiedendo un output JSON.
    Restituisce la stringa della risposta."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )
    
    response = client.models.generate_content(
        model=modello,
        contents=[
            prompt_testo,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ],
        config=config
    )
    return response.text

def salva_json_su_disco(testo_json, file_output):
    """Converte la stringa JSON in un dizionario Python e lo salva su file."""
    dati_json = json.loads(testo_json)
    file_output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_output, 'w', encoding='utf-8') as f:
        json.dump(dati_json, f, ensure_ascii=False, indent=4)

def processa_singolo_grafico(client, percorso_immagine, prompt_testo, file_output):
    """Orchestra l'intero flusso di lavoro per un singolo grafico gestendo le eccezioni."""
    try:
        image_bytes = prepara_immagine_per_api(percorso_immagine, MAX_SIZE)
        risposta_testuale = estrai_dati_con_gemini(client, DEFAULT_MODEL, prompt_testo, image_bytes)
        salva_json_su_disco(risposta_testuale, file_output)
        print(f"Completato: {percorso_immagine.name} -> {file_output.name}")
        
    except json.JSONDecodeError:
        print(f"Errore di parsing JSON per {percorso_immagine.name}. La risposta del modello non era un JSON valido.")
    except Exception as e:
        print(f"Errore di sistema o API per {percorso_immagine.name}: {e}")

def pred_pmc():
    client = setup_gemini()
    path_to_pmc_data = Path("data/images/pmc")
    output_path = Path("outputs/predictions/Gemini/pmc")

    if not path_to_pmc_data.exists():
        print(f"Errore: Percorso {path_to_pmc_data} non trovato.")
        return

    for chart_class in path_to_pmc_data.iterdir():
        if not chart_class.is_dir() or chart_class.name not in PROMPT2CHARTCLASS:
            continue
            
        prompt = PROMPT2CHARTCLASS[chart_class.name]
        
        for difficulty in chart_class.iterdir():
            if not difficulty.is_dir():
                continue
            
            immagini = list(difficulty.iterdir())
            print(f"\n--- Elaborazione: {chart_class.name}/{difficulty.name} ({len(immagini)} file) ---")
            
            for image_path in immagini:
                if not image_path.is_file() or image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                file_output = output_path / chart_class.name / difficulty.name / f"{image_path.stem}.json"
                if file_output.exists():
                    print(f"Skip: {image_path.name} (Già processato)")
                    continue
                    
                processa_singolo_grafico(client, image_path, prompt, file_output)

def pred_synthetic():
    client = setup_gemini()
    path_to_synthetic_data = Path("data/images/synthetic")
    output_path = Path("outputs/predictions/Gemini/synthetic")

    if not path_to_synthetic_data.exists():
        print(f"Errore: Percorso {path_to_synthetic_data} non trovato.")
        return

    for chart_class in path_to_synthetic_data.iterdir():

        if not chart_class.is_dir() or chart_class.name not in PROMPT2CHARTCLASS:
            continue
        immagini = list(chart_class.iterdir())
        print(f"\n--- Elaborazione: {chart_class.name} ({len(immagini)} file) ---")
            
        prompt = PROMPT2CHARTCLASS[chart_class.name]
        
        for image_path in chart_class.iterdir():    
            if not image_path.is_file() or image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            file_output = output_path / chart_class.name / f"{image_path.stem}.json"
            if file_output.exists():
                print(f"Skip: {image_path.name} (Già processato)")
                continue
                    
            processa_singolo_grafico(client, image_path, prompt, file_output)

def gemini_predict():
    pred_pmc()
    pred_synthetic()

if __name__ == "__main__":
    pred_pmc()
    pred_synthetic()
        