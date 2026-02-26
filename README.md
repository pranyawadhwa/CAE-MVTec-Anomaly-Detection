# Anomaly Detection and Segmentation using Convolutional Autoencoder

This project demonstrates **unsupervised anomaly detection and segmentation** using a **Convolutional Autoencoder (CAE)** trained on the **MVTec Anomaly Detection dataset**.  
The model learns to reconstruct *normal* samples and identifies anomalies by analyzing reconstruction errors.

---

## Project Overview

### Objective
To detect and localize manufacturing defects (anomalies) such as scratches, dents, or misalignments in industrial objects by training only on **normal (non-defective)** images.

### Approach
- **Architecture:** Convolutional Autoencoder (Encoder–Decoder)
- **Training Data:** Only *normal* images per category
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metrics:** IoU, Dice Score, Pixel Accuracy
- **Hardware:** GPU-accelerated training (Colab / CUDA)

---

## Dataset

**Dataset:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Categories Used:**
- Bottle  
- Carpet  
- Metal Nut  
- Pill  
- Screw  

Each category contains:
- `train/good` → normal samples  
- `test/good` → normal test samples  
- `test/*defective*` → defective samples  
- `ground_truth` → binary masks for defects

---

## Model Architecture

**Encoder:**
- 4 Convolutional layers with ReLU + BatchNorm
- Downsampling via MaxPooling

**Decoder:**
- 4 Transposed Convolutions (upsampling)
- Sigmoid activation for output reconstruction

**Loss:**  
`MSELoss(input, reconstruction)`

---

## Training Details

| Parameter | Value |
|------------|--------|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Device | CUDA (GPU) |

---

## Results

| Category   | Mean IoU | Mean Dice | Accuracy |
|-------------|-----------|------------|-----------|
| Bottle      | 0.44      | 0.61       | 0.99 |
| Carpet      | 0.56      | 0.72       | 0.99 |
| Metal Nut   | 0.51      | 0.68       | 0.98 |
| Pill        | 0.65      | 0.79       | 0.99 |
| Screw       | 0.58      | 0.73       | 0.99 |

 **Overall performance:**  
- Excellent reconstruction quality  
- Realistic anomaly localization  
- High pixel-wise accuracy (~99%)  
- Stable and smooth training curves  

---

## Evaluation Metrics

- **IoU (Intersection over Union):** Measures overlap between predicted and true anomaly regions.
- **Dice Score:** Balances precision and recall for segmentation accuracy.
- **Pixel Accuracy:** Fraction of correctly predicted pixels.

---

## Installation & Usage

### Clone Repository
```bash
git clone https://github.com/pranyawadhwa/SwinUnet-MVTec-Defect-Detection.git
cd CAE-MVTec-Defect-Detection

