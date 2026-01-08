import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------- Rule of Thirds ----------
def rule_of_thirds_score(cx, cy, w, h):
    """
    cx, cy: center of main subject
    w, h: image width & height
    Returns score in [0, 1]
    """
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]

    # distance from nearest grid intersection
    dist_x = min(abs(cx - tx) for tx in thirds_x)
    dist_y = min(abs(cy - ty) for ty in thirds_y)
    dist = min(dist_x, dist_y)

    # Normalize: max distance we care about ~ one third of max dimension
    max_dist = max(w, h) / 3
    score = 1 - (dist / max_dist)
    return float(max(0.0, min(score, 1.0)))


# ---------- Symmetry ----------
def symmetry_score(image_bgr):
    """
    Horizontal symmetry based on SSIM between image and its mirror.
    Returns score in [-1, 1] but usually ~[0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype("float32")
    flipped = cv2.flip(gray, 1)

    # Since float, specify data_range explicitly
    score, _ = ssim(gray, flipped, full=True, data_range=255.0)
    return float(score)


# ---------- Clutter / Edge Density ----------
def clutter_score(image_bgr):
    """
    Edge density using Canny — higher = more clutter.
    Returns value in [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    if total_pixels == 0:
        return 0.0
    return float(edge_pixels / total_pixels)


# ---------- Brightness ----------
def brightness_score(image_bgr):
    """
    Mean pixel intensity scaled to [0,1]
    """
    return float(np.mean(image_bgr) / 255.0)


# ---------- Contrast ----------
def contrast_score(image_bgr):
    """
    Standard deviation of grayscale intensities, normalized.
    Roughly [0, 1] for typical images.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray) / 128.0)
