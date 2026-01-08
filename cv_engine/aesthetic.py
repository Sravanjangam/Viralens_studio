import torch
import clip
from PIL import Image
import requests
from io import BytesIO


class AestheticScorer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        prompts = [
            "a professional, high-quality, beautiful, aesthetic photo",
            "a low-quality, ugly, noisy, poorly composed photo"
        ]

        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.encode_text(tokens)
            self.text_features = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def load_image(self, src):
        if isinstance(src, str):
            if src.startswith("http"):
                resp = requests.get(src)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(src).convert("RGB")
        else:
            raise ValueError("Image must be URL or filepath")
        return img

    def score(self, src):
        img = self.load_image(src)
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feats = self.model.encode_image(img_input)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ self.text_features.T).cpu().numpy().flatten()

        pos, neg = sims[0], sims[1]
        raw = pos - neg  # [-1,1]
        score = (raw + 1) / 2
        return float(max(0.0, min(score, 1.0)))
