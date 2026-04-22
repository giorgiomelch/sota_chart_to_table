from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch
from src.config import IMAGES_DIR, PREDICTIONS_DIR

def load_model_and_processor():
    model_name = "google/deplot"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Device in uso: {device}")
    return model, processor, device

def process_image(percorso_immagine, file_output, model, processor, device):
    img_path = Path(percorso_immagine)
    
    if not img_path.exists():
        print(f"Salto: {img_path} (File non trovato)")
        return

    try:
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to(device)
        with torch.no_grad():
            predictions = model.generate(**inputs, max_new_tokens=512)
        result = processor.decode(predictions[0], skip_special_tokens=True).replace("<0x0A>", "\n")
        
        file_output.parent.mkdir(parents=True, exist_ok=True)
        file_output.write_text(result, encoding="utf-8")
        print(f"Processato con successo: {img_path.name} -> {file_output.name}")
        
    except Exception as e:
        print(f"Errore durante l'elaborazione di {img_path.name}: {e}")



def pred_dir(dir, model, processor, device):
    path_to_dir = Path(dir)
    output_path = PREDICTIONS_DIR / "DePlot" / path_to_dir.name

    if not path_to_dir.exists():
        print(f"Errore: Percorso {path_to_dir} non trovato.")
        return

    for chart_class in path_to_dir.iterdir():
        if not chart_class.is_dir():
            continue
        immagini = list(chart_class.iterdir())
        print(f"\n--- Elaborazione: {chart_class.name} ({len(immagini)} file) ---")
        
        for image_path in chart_class.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            file_output = output_path / chart_class.name / f"{image_path.stem}.txt"
            if file_output.exists():
                print(f"Skip: {image_path.name} (Già processato)")
                continue
            process_image(image_path, file_output, model, processor, device)

def DePlot_predict(dataset):
    modello, proc, disp = load_model_and_processor()
    pred_dir(IMAGES_DIR / dataset, modello, proc, disp)

    

if __name__ == "__main__":
    DePlot_predict("PMCharts")