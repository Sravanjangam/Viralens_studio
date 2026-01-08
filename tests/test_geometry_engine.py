import cv2
import numpy as np
import requests

from cv_engine.geometry import (
    rule_of_thirds_score,
    symmetry_score,
    clutter_score,
    brightness_score,
    contrast_score,
)

IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def load_image_from_url(url: str):
    resp = requests.get(url)
    resp.raise_for_status()
    data = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def main():
    img = load_image_from_url(IMAGE_URL)
    if img is None:
        print("Failed to load image")
        return

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    thirds = rule_of_thirds_score(cx, cy, w, h)
    sym = symmetry_score(img)
    clut = clutter_score(img)
    bright = brightness_score(img)
    cont = contrast_score(img)

    print("Image size:", (w, h))
    print("Rule of Thirds Score:", thirds)
    print("Symmetry Score:", sym)
    print("Clutter Score:", clut)
    print("Brightness Score:", bright)
    print("Contrast Score:", cont)


if __name__ == "__main__":
    main()
