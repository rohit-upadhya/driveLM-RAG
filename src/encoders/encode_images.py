import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from typing import Iterable, Union
from PIL import Image

from src.util.data_types import ModelType


class EncodeImage:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        model_type: ModelType = ModelType.CLIP,
    ) -> None:
        if model_type == ModelType.CLIP:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode(
        self,
        images: Iterable[Union[str, Image.Image]],
        batch_size: int = 16,
        normalize: bool = True,
    ) -> np.ndarray:
        pil_images: list[Image.Image] = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                pil_images.append(Image.open(img).convert("RGB"))

        all_feats = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            feats = self.model.get_image_features(pixel_values=pixel_values)  # (B, D)
            feats = feats.float()  # ensure float32
            if normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)

            all_feats.append(feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)
