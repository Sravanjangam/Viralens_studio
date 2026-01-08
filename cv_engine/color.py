import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorAnalyzer:

    def extract_colors(self, image, k=3):
        """
        Main color extraction function expected by full_router.py
        Uses KMeans to get dominant colors.
        """
        return self.dominant_colors(image, k)

    def dominant_colors(self, image, k=3):
        """
        KMeans dominant color extractor
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.reshape((-1, 3))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(img)
        colors = kmeans.cluster_centers_

        # Normalize RGB values 0–255 → 0–1 for consistency
        normalized = (colors / 255.0).tolist()
        return normalized

    def brightness(self, image):
        """
        Average brightness normalized 0–1
        """
        return float(np.mean(image) / 255.0)

    def contrast(self, image):
        """
        Standard deviation of luminance normalized
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray) / 128.0)

    def analyze(self, image):
        """
        Unified analysis block (optional)
        """
        return {
            "brightness": self.brightness(image),
            "contrast": self.contrast(image),
            "dominant_colors": self.dominant_colors(image)
        }
