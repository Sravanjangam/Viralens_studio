import os
import json
import numpy as np
import torch
from PIL import Image
import open_clip


class TrendSimilarity:
    def __init__(self, bank_dir="cv_engine/viral_bank/", device=None):
        self.bank_dir = bank_dir
        os.makedirs(self.bank_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.to(self.device)
        self.model.eval()

        self.emb_path = os.path.join(self.bank_dir, "embeddings.npy")
        self.meta_path = os.path.join(self.bank_dir, "meta.json")

        self.embeddings = None
        self.meta = None

        self._load_bank()

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _load_bank(self):
        if os.path.exists(self.emb_path) and os.path.exists(self.meta_path):
            self.embeddings = np.load(self.emb_path)
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.embeddings = np.zeros((0, 512))
            self.meta = []
            self._save_bank()

    def _save_bank(self):
        np.save(self.emb_path, self.embeddings)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)

    # ------------------------------
    # Public functions
    # ------------------------------

    def _embed_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)[0]
        emb = emb / emb.norm()  # Normalize
        return emb.cpu().numpy()

    def add_to_bank(self, image_path: str, label="viral"):
        emb = self._embed_image(image_path)
        self.embeddings = np.vstack([self.embeddings, emb])
        self.meta.append({"path": image_path, "label": label})
        self._save_bank()

    def similarity_score(self, image_path: str):
        if len(self.embeddings) == 0:
            return 0.5  # neutral fallback

        query_emb = self._embed_image(image_path)

        sims = np.dot(self.embeddings, query_emb)
        best = float(np.max(sims))
        return best  # already between 0 and 1
