import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import yaml
from Model.model import EfficientViT  
import os
import cv2
import numpy as np

def overlay_lga_on_tensor_batch(batch_tensor, alpha=0.6):
    """
    Apply LGA overlay on a batch of tensor images and return a stacked tensor (B, 3, 224, 224).

    Parameters:
    - batch_tensor: torch.Tensor of shape (B, C, H, W) with values in [0, 1]
    - alpha: blending factor for overlay

    Returns:
    - torch.Tensor of shape (B, 3, 224, 224), dtype=torch.float32, values in [0, 1]
    """
    batch_size = batch_tensor.size(0)
    overlaid_images = []

    for i in range(batch_size):
        img_tensor = batch_tensor[i]

        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        img_np = cv2.resize(img_np, (224, 224))

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        overlaid = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

        overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

        overlaid_rgb = cv2.resize(overlaid_rgb, (224, 224))

        tensor_img = torch.from_numpy(overlaid_rgb).permute(2, 0, 1).float() / 255.0

        overlaid_images.append(tensor_img)

    # Stack into a single tensor: (B, 3, 224, 224)
    return torch.stack(overlaid_images)

st.set_page_config(page_title="DeepFake Detector", layout="centered")
st.title("🕵️‍♂️ DeepFake Detection")
st.markdown("Upload an image to detect if it's **Real** or **Fake**.")


@st.cache_resource
def load_model():
    with open("configs/arch.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = EfficientViT(config=config, channels=1280, selected_efficient_net=0)
    model.load_state_dict(torch.load("hybrid_model_v10.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    tensor = overlay_lga_on_tensor_batch(tensor)
    return tensor.to(device)


def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        label = "Real" if prob >= 0.5 else "Fake"
        confidence = prob if label == "Fake" else 1 - prob
        return label, confidence


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")
    image_tensor = preprocess_image(image)
    label, confidence = predict(image_tensor)

    if label == "Fake":
        st.error(f"This image is likely **Fake**.")
    else:
        st.success(f"This image is likely **Real**.")
