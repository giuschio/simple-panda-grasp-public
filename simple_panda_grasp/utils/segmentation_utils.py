from contextlib import nullcontext
import gc

from PIL import Image
from transformers import Sam3Model, Sam3Processor

import numpy as np
import torch


class SamSeg:
    def __init__(self, device: str | None = None) -> None:
        self.preferred_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.preferred_device
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    @staticmethod
    def _clear_torch_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def park(self) -> None:
        self.model = self.model.to("cpu").eval()
        self.device = "cpu"
        self._clear_torch_cache()

    def use(self, device: str | None = None) -> None:
        self.device = device or self.preferred_device
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _union_masks_or_empty(masks: torch.Tensor | None, h: int, w: int) -> np.ndarray:
        if masks is None or masks.numel() == 0:
            return np.zeros((h, w), dtype=bool)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        return masks.bool().any(dim=0).detach().cpu().numpy()

    def __call__(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """
        Segment an RGB image with SAM3 from a text prompt.

        Args:
            image: NumPy array with shape (H, W, 3), RGB channel order,
                dtype `np.uint8`, and values in [0, 255].
            text_prompt: Text query describing the target object.

        Returns:
            Binary segmentation mask as a NumPy array of shape (H, W) and dtype bool.
            If no masks are returned, this is all False. If multiple masks are returned,
            this is their union.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"Expected image dtype np.uint8, got {image.dtype}")

        image_pil = Image.fromarray(image, mode="RGB")

        inputs = self.processor(images=[image_pil], text=[text_prompt], return_tensors="pt").to(self.device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        with torch.inference_mode(), autocast_context:
            outputs = self.model(**inputs)

        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        h, w = image.shape[0], image.shape[1]
        return self._union_masks_or_empty(result["masks"], h, w)
    
    def segment_image_data(self, image_data: dict, text_prompt: str):
        rgb_imgs = image_data["rgb_imgs"]
        segmentation_masks = [self(img, text_prompt) for img in rgb_imgs]
        image_data["segmentation_masks"] = segmentation_masks
        return image_data
