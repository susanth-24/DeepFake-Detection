---

# 🕵️‍♂️ DeepFake Detection using EfficientDet + Vision Transformer + Hybrid Mathematical Models

This repository contains a **DeepFake detection pipeline** built using **EfficientDet (Backbone)**,  **Vision Transformer (ViT)**  and **Hybrid Mathematical Models**for global feature analysis. It uses the **CIFAKE** dataset, a synthetic dataset for real vs fake image classification.

---

## 🧠 Model Overview

- **Hybrid Fusion:** Innput+(LGA)
- **Feature Extractor:** EfficientDet-D0
- **Classifier:** Vision Transformer (ViT)
- **Loss Function:** Binary Cross Entropy with Logits

---

## 📦 Project Structure

```bash
DeepFakeV1/
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── app.py                  # Streamlit web app
├── model/
│   └── model.py            # Model definition (EfficientViT)
├── overlay.py              # LGA overlay preprocessing
├── configs/
│   └── arch.yaml           # Architecture configuration
├── hybrid_model_v10.pth    # Trained model weights
└── README.md               # Project documentation
```


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/susanth-24/DeepFakeV1.git
cd DeepFakeV1
````

### 2. Install requirements

```bash
pip install -r requirements.txt
```

Make sure your system has a working NVIDIA GPU driver and `torch` with CUDA installed.

---

## 🏋️ Training

Run the training script:

```bash
python train.py
```

This will:

* Train the model using CIFAKE dataset
* Save model weights to `hybrid_model_v10.pth`

---

## 📊 Evaluation

Run the evaluation script:

```bash
python eval.py
```

### ✅ Evaluation Results (on CIFAKE test set)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.9000 |
| Precision | 0.8788 |
| Recall    | 0.9280 |
| F1 Score  | 0.9027 |

---

## 🌐 Web App (Streamlit)

To launch a beautiful web interface for predictions:

```bash
streamlit run app.py
```

* Upload an image and it will classify it as **Real** or **Fake** using the trained model.

---

## 📁 Dataset

We use the **CIFAKE dataset**, a synthetic deepfake image dataset with real and fake categories.

You can download it from [here](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and structure it like:

```
dataset/
├── Train/
│   ├── real/
│   └── fake/
├── Test/
│   ├── real/
│   └── fake/
├── Validation/
│   ├── real/
│   └── fake/
```

Update your dataloader paths accordingly.

---

## 🧪 Model Summary

* Uses **EfficientNet-B0** for hybrid CNN features
* LGA overlays applied to emphasize local tampering cues
* Global ViT layers extract spatial dependencies
* Final classifier distinguishes between real and fake



---

## 📚 Course Information

This project was developed as part of the **CS512: Artificial Intelligence** coursework at **IIT Ropar**.

---

## 👨‍💻 Team Members

| Name                       | Roll Number     |
|----------------------------|-----------------|
| **Sanam Sai Susanth Reddy** | 2021CHB1053     |
| **Khushboo Gupta**          | 2021CSB1105     |
| **Nikhil Garg**             | 2021CSB1114     |
| **Abhinav Adarsh**          | 2021MEB1261     |

---

