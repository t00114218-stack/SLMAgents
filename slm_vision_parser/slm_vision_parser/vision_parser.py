import os
import sys
import yaml
from PIL import Image

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoProcessor = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_VISION_PARSER_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMVisionParser:
    """
    A local CPU-optimized Vision Parser agent powered by a local MIT-licensed Florence-2 model.
    Runs high-speed local OCR, layout parsing, and table parsing tasks offline.
    """
    def __init__(self, model_path=None, cache_dir=None):
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers and torch are required for Florence-2 parsing. Please install them:\n"
                "pip install torch transformers pillow"
            )

        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMVisionParser] Loading Florence-2 model from: {self.model_path}...")
        
        # Load model on CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("vision_parser", {})
        config_path = model_config.get("path", "../../models/florence-2-large")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
            
        if os.path.exists(config_path):
            return config_path
            
        repo_id = "microsoft/Florence-2-large"
        print(f"[SLMVisionParser] Model not found at configured path. Auto-downloading {repo_id}...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path
        )
        return config_path

    def parse_image(self, image_path: str, task: str = "<OCR>", system_prompt: str = None, user_input: str = None) -> str:
        """
        Executes a vision task on the specified image file.
        Tasks include:
            - '<OCR>' : Standard text extraction
            - '<OCR_WITH_REGION>' : Text extraction with coordinates
            - '<CAPTION>' : Basic captioning
            - '<DETAILED_CAPTION>' : Detailed description
            - '<MORE_DETAILED_CAPTION>' : Extremely detailed description
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        
        # Format prompt and extract features
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=image.size
        )
        
        # Return parsed string representation
        if task in parsed_answer:
            return str(parsed_answer[task])
        return generated_text
