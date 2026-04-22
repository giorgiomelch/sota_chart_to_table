import os
import glob
import torch
import json
import math
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from huggingface_hub import snapshot_download

from src.utils.prompts import PROMPT2CHARTCLASS, PROMPT_AreaLineBarHistogram
from src.config import WEIGHTS_DIR, IMAGES_DIR, PREDICTIONS_DIR

INTERNVL_MODELS = {
    "2.5-2B": "OpenGVLab/InternVL2_5-2B",
    "2.5-8B": "OpenGVLab/InternVL2_5-8B"
}

# --- GESTIONE IMMAGINI INTERNVL ---

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- GESTIONE MODELLO ---

def setup_internvl(tier="2.5-2B"):
    model_id = download_internvl_model(tier)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"\nCaricamento di InternVL ({tier}) a precisione {dtype} in corso...")

    # Monkey-patch runtime per bug torch.linspace senza device='cpu' nel codice del modello
    _real_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return _real_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace

    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False
        ).eval().cuda()
    finally:
        torch.linspace = _real_linspace  # ripristina sempre, anche in caso di errore

    return model, tokenizer

def get_local_path(tier):
    safe_tier = tier.replace(".", "_").replace("-", "_")
    return WEIGHTS_DIR / f"InternVL2_5_{safe_tier}"

def download_internvl_model(tier="2.5-2B"):
    if tier not in INTERNVL_MODELS:
        raise ValueError(f"Taglia non supportata. Scegli tra: {list(INTERNVL_MODELS.keys())}")
        
    repo_id = INTERNVL_MODELS[tier]
    output_path = get_local_path(tier)
    
    if output_path.exists() and any(output_path.glob("*.safetensors")):
        print(f"Modello {tier} già presente in {output_path}. Skip download.")
        return str(output_path)
        
    print(f"Inizio download del modello {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_path),
        ignore_patterns=["*.pt", "*.bin"]
    )
    print(f"Download completato e salvato in {output_path}")
    return str(output_path)


# --- INFERENZA ---

def extract_table_internvl(model, tokenizer, image_path, prompt_text, max_num_blocks=12):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pixel_values = load_image(image_path, max_num=max_num_blocks).to(dtype).to(model.device)
    
    generation_config = dict(
        max_new_tokens=10_000,
        do_sample=False
    )
    
    question = f'<image>\n{prompt_text}'
    
    with torch.no_grad():
        # Assegna il risultato direttamente a response senza ", _"
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return response

def run_batch_inference(model, tokenizer, tier="2.5-2B"):
    input_base_dir = IMAGES_DIR
    output_base_dir = PREDICTIONS_DIR / f"InternVL{tier}"
    
    if not input_base_dir.exists():
        print(f"Errore: La directory di input {input_base_dir} non esiste.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for img_path in input_base_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            
            relative_path = img_path.relative_to(input_base_dir)
            json_output_path = output_base_dir / relative_path.with_suffix('.json')
            
            if json_output_path.exists():
                print(f"Skip: {json_output_path.name} (già processato).")
                continue
            
            if len(relative_path.parts) >= 2:
                chart_class = relative_path.parts[1]
            else:
                print(f"Attenzione: Impossibile determinare la classe per {relative_path}. Salto.")
                continue
            
            try:
                prompt_scelto = PROMPT2CHARTCLASS.get(chart_class, PROMPT_AreaLineBarHistogram)
            except NameError:
                prompt_scelto = "Extract the data from this chart into a JSON format."
            
            print(f"Elaborazione: {relative_path} [Classe: {chart_class}]...")
            json_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                raw_result = extract_table_internvl(model, tokenizer, img_path, prompt_scelto, max_num_blocks=12)
                
                clean_json = raw_result.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                try:
                    json.loads(clean_json)
                except json.JSONDecodeError as e:
                    print(f"  [ATTENZIONE] Il modello ha generato un JSON non valido per {img_path.name}: {e}")
                
                json_output_path.write_text(clean_json, encoding="utf-8")
                
            except Exception as e:
                print(f"Errore critico durante l'elaborazione di {img_path}: {e}")

def ask_internvl(tier="2.5-2B"):
    # Chiamata aggiornata senza il parametro quantizzazione
    modello, tokenizer = setup_internvl(tier=tier)
    run_batch_inference(modello, tokenizer, tier=tier)
    print(f"\nInternVL {tier}: inferenza batch completata.")

if __name__ == "__main__":
    ask_internvl()