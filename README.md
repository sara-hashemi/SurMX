# SurvMx: Tumor-Centric Multimodal Survival Prediction

Official implementation of:

**A Multimodal Attention Transformer Framework for Tumor-Centric Survival Prediction**

SurvMx is a tumor-centric multimodal survival modeling framework that integrates 3D multiphase DCE-MRI and structured clinical metadata using a Multiple Instance Learning (MIL) architecture with multi-head attention and Cox-based optimization.

---

# Overview

Accurate survival prediction in breast cancer requires modeling tumor-level biological heterogeneity rather than relying solely on whole-organ representations.

In SurMX, we propose a tumor-centric multimodal survival architecture that integrates instance-level MRI representations, global anatomical context, and structured clinical metadata within a unified Cox-based optimization framework. Tumor volumes are extracted from 3D segmentations and subjected to spatially consistent geometric rotations and intensity perturbations to enhance robustness and mitigate class imbalance. Each tumor volume is encoded using a deep feature extractor followed by a residual tumor encoder, producing instance-level embeddings. Survival prediction is formulated as a Multiple Instance Learning (MIL) problem in which each patient is represented as a variable-sized bag of tumor embeddings. A multi-head self-attention aggregation module learns adaptive instance weights to model intra-patient tumor heterogeneity. In parallel, a residual global encoder processes scan-level features to preserve anatomical context beyond localized lesions. Phase-specific embeddings are learned independently for each DCE-MRI phase to capture temporal contrast-enhancement dynamics. The aggregated tumor representation is fused with the global embedding through cross-attention and further integrated with learnable entity embeddings of categorical clinical variables. The resulting patient-level representation is refined through a Mixture-of-Experts module and optimized under a Cox Proportional Hazards objective in task-specific models for Overall Survival and Recurrence-Free Survival.

Each patient is modeled as a **bag of tumor instances**, enabling biologically grounded risk aggregation.

---

# Step 0 — Download and Prepare MAMA-MIA

## 0.1 Download Dataset

Download the MAMA-MIA dataset from the official source (insert link here).

After download, structure the dataset as follows:

```
MAMA-MIA/
└── raw/
    └── Dataset001_MAMA_MIA/
        ├── imagesTr/
        ├── labelsTr/
        ├── imagesTs/
        ├── labelsTs/
```

Each patient should contain **3 DCE-MRI phases** with some patients containing the last two DCE-MRI phases:

```
patient001_0000.nii.gz
patient001_0001.nii.gz
patient001_0002.nii.gz
```

Where:

- `_0000` = Pre-contrast
- `_0001–0004` = Post-contrast phases

---

# Step 1 — Install nnUNet v2

We use **nnUNet v2** for 3D tumor segmentation.

## 1.1 Create Environment

```bash
conda create -n nnunetv2 python=3.9
conda activate nnunetv2
```

## 1.2 Install nnUNet v2

```bash
pip install nnunetv2
```

Or from source:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

---

## 1.3 Set Environment Variables

```bash
export nnUNet_raw="/path/to/MAMA-MIA/raw"
export nnUNet_preprocessed="/path/to/MAMA-MIA/preprocessed"
export nnUNet_results="/path/to/MAMA-MIA/nnunet_results"
```

---

# Step 2 — Prepare 4-Phase Dataset for nnUNet

Ensure dataset follows nnUNet naming conventions:

```
Dataset001_MAMA_MIA/
├── imagesTr/
│   ├── patient001_0000.nii.gz
│   ├── patient001_0001.nii.gz
│   ├── patient001_0002.nii.gz
│   ├── patient001_0003.nii.gz
│   ...
├── labelsTr/
│   ├── patient001.nii.gz
```

Each patient must have 4 input channels.

Your `dataset.json` must specify:

```json
"channel_names": {
    "0": "DCE_phase0",
    "1": "DCE_phase1",
    "2": "DCE_phase2",
    "3": "DCE_phase3"
}
```

---

# Step 3 — Train nnUNet (3D Full Resolution)

## 3.1 Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

## 3.2 Train Model (Fold 0)

```bash
nnUNetv2_train 1 3d_fullres 0
```

To train all folds:

```bash
nnUNetv2_train 1 3d_fullres all
```

---

# Step 4 — Generate Tumor Segmentations

Run inference:

```bash
nnUNetv2_predict \
    -i /path/to/imagesTs \
    -o /path/to/nnUNet_predictions \
    -d 1 \
    -c 3d_fullres \
    -f 0 \
    -chk checkpoint_best.pth
```

Predicted masks will be saved as:

```
nnUNet_predictions/
├── patient001.nii.gz
├── patient002.nii.gz
...
```

---

# Step 5 — Tumor VOI Extraction

Update paths inside `postProcessing.py`:

```python
SOURCE_IMAGE_DIR = "/path/to/imagesTr"
SOURCE_MASK_DIR  = "/path/to/nnUNet_predictions"
```

Run:

```bash
python postProcessing.py
```

This:

- Identifies connected tumor components
- Computes 3D bounding boxes
- Extracts tumor volumes (VOIs)
- Saves each tumor independently

Output:

```
cropped_voi_images/
├── patient001_tumor1_img.nii.gz
├── patient001_tumor1_mask.nii.gz
...
```

---

# Step 6 — Tumor-Level Augmentation

```bash
python augmentation.py
```

Applies:

**Geometric Transformations**
- 45°, 90°, 135°, 180°, 225°, 270°, 315° rotations

**Intensity Transformations**
- Brightness shift
- Contrast scaling
- Gaussian noise injection

All transformations preserve spatial alignment.

---

# Step 7 — Feature Extraction

Use:

- Pretrained Vision Transformer (ViT), or
- Radiomic feature extraction

Generate:

```
features_merged_survival_only.csv
```

Required columns:

- tumor_id
- patient_id
- deep_tumor
- deep_global (optional)
- days_to_death_or_last_followup
- death_event
- days_to_recurrence_or_last_followup
- recurrence_event

---

# Step 8 — Train MIL Survival Model

```bash
python MIL.py
```

The model:

- Groups tumors into patient-level bags
- Applies multi-head attention aggregation
- Fuses tumor + global + clinical embeddings
- Optimizes Cox survival objective
- Trains separate models for OS and RFS

---

# Notes

- Ensure consistent DCE phase ordering.
- Patient-level splits must remain consistent.
- nnUNet handles resampling and normalization internally.
- GPU recommended for segmentation training.

---

# Outputs

- Death C-index
- Recurrence C-index
- Kaplan–Meier survival curves
- Saved model checkpoints

---

# Citation - TBA

---

# Contact
For questions regarding implementation or reproduction, please open an issue.

---
