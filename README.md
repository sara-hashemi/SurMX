# SurvMx: Tumor-Centric Multimodal Survival Prediction

Official implementation of:

**A Multimodal Attention Transformer Framework for Tumor-Centric Survival Prediction**

SurvMx is a tumor-centric multimodal survival modeling framework that integrates 3D multiphase DCE-MRI and structured clinical metadata using a Multiple Instance Learning (MIL) architecture with multi-head attention and Cox-based optimization.

---

# Overview

Accurate survival prediction in breast cancer requires modeling tumor-level biological heterogeneity rather than relying solely on whole-organ representations.

SurvMx implements:

- 🧠 Tumor-centric volumetric extraction
- 🔄 Spatially aligned 3D augmentation
- 🔬 Deep (ViT) or radiomic feature encoding
- 🎯 Multi-head attention MIL aggregation
- 🧩 Multimodal fusion with structured clinical metadata
- 📊 Cox Proportional Hazards survival optimization
- ⚖ Separate models for Overall Survival (OS) and Recurrence-Free Survival (RFS)

Each patient is modeled as a **bag of tumor instances**, enabling biologically grounded risk aggregation.

---

# Repository Structure

```
postProcessing.py        # Tumor VOI extraction
augmentation.py          # 3D augmentation of cropped tumors
MIL.py                   # Tumor-centric MIL survival model
```

---

# ⚙️ Installation

## 1-Create Environment

```bash
conda create -n survmx python=3.9
conda activate survmx
```

## 2-Install Dependencies

```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install SimpleITK
pip install scikit-survival
```

(Optional: radiomics, timm, etc. depending on your embedding pipeline)

---

# 📂 Data Preparation

## Required Inputs

You need:

- 3D MRI/CT images (`.nii.gz`)
- Corresponding segmentation masks (`.nii.gz`)
- Clinical survival metadata (CSV)
- Tumor/global embeddings (deep or radiomic)

---

# 🚀 End-to-End Pipeline

---

## Step 1 — Extract Tumor VOIs

Run:

```bash
python postProcessing.py
```

### What It Does

- Loads full 3D scan
- Loads segmentation mask
- Identifies connected tumor components
- Computes 3D bounding boxes
- Crops tumor volumes
- Saves each tumor independently

### Output

```
cropped_voi_images/
├── patient001_tumor1_img.nii.gz
├── patient001_tumor1_mask.nii.gz
├── patient001_tumor2_img.nii.gz
├── patient001_tumor2_mask.nii.gz
...
```

Each tumor is saved separately for multi-instance modeling.

---

## Step 2 — Generate 3D Augmentations

Run:

```bash
python augmentation.py
```

### Augmentations Applied

**Geometric**
- 45°, 90°, 135°, 180°, 225°, 270°, 315° rotations

**Intensity**
- Brightness shift
- Contrast scaling
- Gaussian noise injection

All transformations are applied consistently to both image and mask to preserve spatial alignment.

### Output

```
augmented_cropped_voi_images/
├── patient001_tumor1_rot45_img.nii.gz
├── patient001_tumor1_rot45_mask.nii.gz
├── patient001_tumor1_bright_img.nii.gz
├── patient001_tumor1_bright_mask.nii.gz
...
```

---

## Step 3 — Feature Extraction

You may use:

- Pretrained Vision Transformer (ViT)
- Radiomic feature extraction
- Any volumetric encoder

The resulting CSV should contain:

```
features_merged_survival_only.csv
```

With columns including:

- tumor_id
- patient_id
- deep_tumor (embedding vector)
- deep_global (optional global embedding)
- days_to_death_or_last_followup
- death_event
- days_to_recurrence_or_last_followup
- recurrence_event

---

## Step 4 — Train MIL Survival Model

Run:

```bash
python MIL.py
```

---

# MIL Architecture Overview

Each patient is modeled as:

$begin:math:display$
Patient\^z \= \[Tumor\_1\^z\, Tumor\_2\^z\, \.\.\.\, Tumor\_k\^z\]
$end:math:display$

### Model Components

- Residual Tumor Encoder
- Multi-Head MIL Attention
- Global Embedding Encoder
- Cross-Attention Fusion
- Mixture-of-Experts (MoE)
- Cox Proportional Hazards Head

Separate models are trained for:

- Overall Survival (OS)
- Recurrence-Free Survival (RFS)

---

# Outputs

During training:

- Epoch-wise loss
- Validation C-index

Final evaluation:

- Death C-index
- Recurrence C-index
- Kaplan–Meier survival curves
- Trained model weights

---

# Reproducibility

- Patient-level independent train/test splits enforced
- Deterministic seed control supported
- Compatible with deep or radiomic features
- Modular design enables replacement of embedding backbone

---

# Example Results

| Model | OS C-index | RFS C-index |
|--------|------------|------------|
| SurvMx (Joint) | 0.835 | 0.752 |
| SurvMx (Single-task) | 0.858 | 0.641 |

(Results depend on dataset and preprocessing configuration.)

---

# Notes

- Designed for tumor-centric multimodal survival modeling
- Easily extendable to subtype classification
- Can be adapted to other cancer datasets

---

# Citation - TBA

---

# 📬 Contact

For questions regarding implementation or reproduction, please open an issue.

---
