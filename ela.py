import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

def compute_ela(image_path, quality=90):
    """
    Perform Error Level Analysis (ELA) on the image.
    """
    # Load image
    original = Image.open(image_path).convert('RGB')

    # Save the image at specified JPEG quality
    temp_path = 'temp_ela.jpg'
    original.save(temp_path, 'JPEG', quality=quality)

    # Reload compressed image
    compressed = Image.open(temp_path)

    # Compute difference
    ela_image = ImageChops.difference(original, compressed)

    # Enhance to visualize differences better
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return np.array(ela_image)  # shape: (H, W, 3)

def compute_lga(image_path):
    """
    Perform Luminance Gradient Analysis (LGA) using Sobel filters.
    """
    # Load image and convert to Y channel
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients using Sobel filters
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize for visualization and compatibility
    normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)

# === Usage ===
image_path = 'Dataset/archive/Dataset/Train/Fake/fake_0.jpg'

# ela_output = compute_ela(image_path)
# lga_output = compute_lga(image_path)


def overlay_lga_on_image(image_path, alpha=0.6):
    """
    Perform Luminance Gradient Analysis (LGA) and overlay it on the original image.
    
    Parameters:
    - image_path: str, path to the input image.
    - alpha: float, blending factor for overlay (0 = only image, 1 = only LGA heatmap).
    
    Returns:
    - overlayed_image: np.ndarray, the image with LGA overlay.
    """
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize for consistency (optional)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients using Sobel filters
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the result to 0-255 and convert to uint8
    normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap (e.g., JET) to the normalized gradient
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Blend heatmap with original image
    overlayed_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlayed_image

# # Optional visualization
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(ela_output)
# plt.title('ELA Output')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(lga_output, cmap='gray')
# plt.title('LGA Output')
# plt.axis('off')
# plt.tight_layout()
# plt.show()

result = overlay_lga_on_image(image_path=image_path)
cv2.imwrite("output_with_lga_overlay.jpg", result)
cv2.imshow("LGA Overlay", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
