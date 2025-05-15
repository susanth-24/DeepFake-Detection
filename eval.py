from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
import numpy as np
from torch.optim import lr_scheduler
import os
import json
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim.lr_scheduler import LambdaLR
import collections
import math
import yaml
import argparse
import torch
from torchvision import transforms
from torch.utils.data import Subset
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from Model.model import EfficientViT
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

base_transform=transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
])


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

        # Convert tensor to numpy image in [0, 255] RGB
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Resize to 224x224 (if not already)
        img_np = cv2.resize(img_np, (224, 224))

        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Compute Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize and apply colormap
        normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Overlay heatmap on the original image
        overlaid = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

        # Convert back to RGB
        overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)

        # Resize to (224, 224) just in case
        overlaid_rgb = cv2.resize(overlaid_rgb, (224, 224))

        # Convert to tensor and normalize to [0, 1]
        tensor_img = torch.from_numpy(overlaid_rgb).permute(2, 0, 1).float() / 255.0

        overlaid_images.append(tensor_img)

    # Stack into a single tensor: (B, 3, 224, 224)
    return torch.stack(overlaid_images)

def get_class_indices(dataset, class_name, samples_required):
    class_idx = dataset.class_to_idx[class_name]
    indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
    return indices[:samples_required]

# Load full dataset
val_data_full = ImageFolder(root="Dataset/archive/Dataset/Validation", transform=base_transform)
test_data_full = ImageFolder(root="Dataset/archive/Dataset/Test", transform=base_transform)

# Get balanced indices
train_fake_indices = get_class_indices(val_data_full, "Fake", 5000)
train_real_indices = get_class_indices(val_data_full, "Real", 5000)
test_fake_indices = get_class_indices(test_data_full, "Fake", 5000)
test_real_indices = get_class_indices(test_data_full, "Real", 5000)

# Combine to form subsets
train_indices = train_fake_indices + train_real_indices
test_indices = test_fake_indices + test_real_indices

train_data = Subset(val_data_full, train_indices)
test_data = Subset(test_data_full, test_indices)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            images1 = overlay_lga_on_tensor_batch(images).to(device)  # returns tensor of shape [B, 3, 224, 224]
            outputs = model(images1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().squeeze(1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

if __name__ == "__main__":
    # Load config
    with open("configs/arch.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = EfficientViT(config=config, channels=1280, selected_efficient_net=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load weights
    model.load_state_dict(torch.load("hybrid_model_v10.pth", map_location=device))
    print("✅ Loaded model weights from hybrid_model_v10.pth")

    # Define test loader

    # Evaluate
    evaluate(model, train_loader, device)