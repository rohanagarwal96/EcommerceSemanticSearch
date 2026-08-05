"""Image/text embedding utilities wrapping OpenAI CLIP via transformers."""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ecomsearch.multimodal.config import CLIP_MODEL_NAME


class ClipEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def embed_images(self, image_paths: list) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        return self._normalize(self._as_tensor(outputs))

    def embed_text(self, texts: list) -> np.ndarray:
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
        return self._normalize(self._as_tensor(outputs))

    @staticmethod
    def _as_tensor(outputs):
        # As of transformers 5.14.1, CLIPModel.get_image_features()/get_text_features()
        # return a BaseModelOutputWithPooling (embedding in .pooler_output), not a bare
        # tensor. Fall back to the value itself in case that ever changes back.
        return outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs

    @staticmethod
    def _normalize(features) -> np.ndarray:
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
