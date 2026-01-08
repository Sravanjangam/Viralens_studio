import cv2
import numpy as np
import requests

from cv_engine.color import dominant_colors, brightness, contrast

URL = "https://ultralytics.com/images/zidane.jpg"


def load(url):
    r = requests.get(url)
    data = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


img = load(URL)

dom = dominant_colors(img, k=3)
b = brightness(img)
c = contrast(img)

print("Dominant colors:", dom)
print("Brightness:", b)
print("Contrast:", c)
