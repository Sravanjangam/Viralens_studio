"""Reproduce the computer-vision latency numbers quoted in docs/ARCHITECTURE.md.

Run from the repository root:

    python scripts/bench_cv_pipeline.py

The benchmark uses synthetic images, so absolute timings will differ from real
photographs, but the resolution scaling is representative.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_engine.color import ColorAnalyzer
from cv_engine.geometry import clutter_score, symmetry_score


def synthetic_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def time_call(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - start) * 1000


def measure(image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    ssim_ms = time_call(symmetry_score, image)
    canny_ms = time_call(clutter_score, image)
    kmeans_ms = time_call(ColorAnalyzer().extract_colors, image)
    return {
        "resolution": f"{width}x{height}",
        "megapixels": round(width * height / 1_000_000, 2),
        "ssim_ms": round(ssim_ms, 1),
        "canny_ms": round(canny_ms, 1),
        "kmeans_ms": round(kmeans_ms, 1),
        "cv_total_ms": round(ssim_ms + canny_ms + kmeans_ms, 1),
    }


def main() -> None:
    four_k = synthetic_image(2160, 3840)
    downscaled = cv2.resize(four_k, (1024, 576), interpolation=cv2.INTER_AREA)

    rows = [
        measure(synthetic_image(720, 1280)),
        measure(synthetic_image(1080, 1920)),
        measure(four_k),
        measure(downscaled),
    ]

    headers = ["resolution", "megapixels", "ssim_ms", "canny_ms", "kmeans_ms", "cv_total_ms"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[key]) for key in headers) + " |")


if __name__ == "__main__":
    main()
