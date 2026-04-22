import os
import io
import json
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from openai import OpenAI
from src.utils.prompts import PROMPT2CHARTCLASS
from src.utils.schema_json import SCHEMA2CHARTCLASS
from src.config import PREDICTIONS_DIR

MAX_SIZE = 768

class BaseLLMClient(ABC):
    @abstractmethod
    def extract_data(self, prompt: str, image_bytes: bytes, schema=None) -> str:
        pass

class GeminiClient(BaseLLMClient):
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY mancante")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def extract_data(self, prompt, image_bytes, schema=None):
        config = types.GenerateContentConfig(response_mime_type="application/json")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            ],
            config=config
        )
        return response.text

class OpenAIClient(BaseLLMClient):
    def __init__(self, model_name="gpt-4o"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY mancante")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def extract_data(self, prompt, image_bytes, schema=None):
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        params = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        }
        if schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True
                }
            }
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content


class ChartToTableProcessor:
    def __init__(self, client: BaseLLMClient, output_base_path: Path):
        self.client = client
        self.output_base_path = output_base_path

    @staticmethod
    def _prepara_immagine(path, max_size=MAX_SIZE):
        with Image.open(path) as img:
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

    def _save_json(self, text, output_file):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except json.JSONDecodeError:
            raw_file = output_file.with_suffix(".raw.txt")
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"JSON non valido salvato in {raw_file.name}")

    def process_folder(self, input_path: Path):
        if not input_path.exists():
            print(f"Percorso {input_path} non trovato.")
            return

        for img_path in input_path.rglob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            chart_class = img_path.parent.name
            
            prompt = PROMPT2CHARTCLASS.get(chart_class)
            if not prompt: continue

            # Costruisci il path di output relativo
            rel_path = img_path.relative_to(input_path)
            output_file = self.output_base_path / input_path.name / rel_path.with_suffix('.json')

            if output_file.exists():
                continue

            print(f"Processing: {img_path.name} con {self.client.__class__.__name__}")
            try:
                img_bytes = self._prepara_immagine(img_path)
                schema = SCHEMA2CHARTCLASS.get(chart_class)
                result = self.client.extract_data(prompt, img_bytes, schema)
                self._save_json(result, output_file)
            except Exception as e:
                print(f"Errore su {img_path.name}: {e}")

def ask_vllm(provider, model_name, path_to_dir_target):
    if provider.lower() == "gemini":
        client = GeminiClient(model_name=model_name)
    elif provider.lower() == "openai":
        client = OpenAIClient(model_name=model_name)
    else:
        print("Provider non presente nella lista.")
        return
    if not Path(path_to_dir_target).exists():
        print(f"Percorso {path_to_dir_target} non trovato.")
        return
    # Definizione percorso di output basato sul modello
    output_path = PREDICTIONS_DIR / model_name
    
    processor = ChartToTableProcessor(client=client, output_base_path=output_path)

    # Esecuzione sui dataset
    print(f"Avvio benchmark per {provider} ({model_name})...")
    processor.process_folder(Path(path_to_dir_target))
    print("Benchmark completato.")