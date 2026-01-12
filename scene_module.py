# scene_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Optional
import requests
from io import BytesIO
from PIL import Image

# Lazy imports (torch/torchvision) only when needed
_TORCH_READY = False

SCENE_MAPPING = {
    "mountain": ["mountain", "mountain_path", "mountain_snowy", "cliff", "alp", "volcano"],
    "sea": ["ocean", "coast", "seashore", "promenade", "pier"],
    "beach": ["beach", "sandbar", "boardwalk", "lakeshore"],
    "forest": ["forest", "rainforest", "bamboo_forest", "jungle", "woods"],
    "city": ["street", "city", "downtown", "urban", "avenue", "plaza"],
    "desert": ["desert", "sand", "dune", "badlands"],
}

@dataclass
class SceneDetector:
    model: Optional[object] = None
    preprocess: Optional[object] = None
    classes: Optional[List[str]] = None
    device: str = "cpu"

    def _ensure_loaded(self):
        global _TORCH_READY
        if self.model is not None and self.preprocess is not None and self.classes is not None:
            return

        import torch
        from torchvision import models, transforms
        import torch.nn.functional as F

        _TORCH_READY = True

        # Load Places365 resnet18 weights
        model = models.resnet18(num_classes=365)

        checkpoint_url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=self.device)
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        classes_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
        classes = [line.strip() for line in requests.get(classes_url, timeout=20).text.splitlines()]

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = model
        self.preprocess = preprocess
        self.classes = classes

        # store F for later use
        self._F = F
        self._torch = torch

    def detect(self, image_url: str, topk: int = 5) -> List[str]:
        """Return user-friendly scene tags like ['beach', 'city']."""
        if not image_url:
            return []

        try:
            self._ensure_loaded()

            resp = requests.get(image_url, timeout=15)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0)

            with self._torch.no_grad():
                outputs = self.model(img_tensor)
                probs = self._F.softmax(outputs, dim=1)
                top_probs, top_idxs = probs.topk(topk)

            top_classes = [self.classes[i] for i in top_idxs[0].tolist()]

            detected: Set[str] = set()
            for cls in top_classes:
                for key, values in SCENE_MAPPING.items():
                    if any(v in cls for v in values):
                        detected.add(key)

            return sorted(detected)
        except Exception:
            # If anything fails, just return empty and do not break the app
            return []
