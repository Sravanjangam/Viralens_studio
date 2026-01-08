from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load image
img = io.imread("https://ultralytics.com/images/zidane.jpg")

# Convert to grayscale in float64 [0,1]
gray = color.rgb2gray(img)
flipped = np.fliplr(gray)

# Compute SSIM with explicit data_range
score, diff = ssim(gray, flipped, full=True, data_range=1.0)

print("scikit-image SSIM working!")
print("SSIM Score:", score)
