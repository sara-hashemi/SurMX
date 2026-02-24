# SurvMX: Tumor-Centric Multimodal Survival Prediction

Official implementation of:

**A Multimodal Attention Transformer Framework for Tumor-Centric Survival Prediction**

SurvMX is a tumor-centric multimodal survival modeling framework that integrates 3D multiphase DCE-MRI and structured clinical metadata using a Multiple Instance Learning (MIL) architecture with multi-head attention and Cox Proportional Hazards optimization.

---

## 1. Overview

Accurate survival prediction in breast cancer requires modeling tumor-level biological heterogeneity rather than relying solely on global whole-breast representations.

SurvMX implements a fully tumor-centric pipeline:

1. 3D tumor segmentation via nnUNet v2  
2. Connected-component tumor extraction (VOIs)  
3. Spatially consistent 3D augmentation  
4. Tumor-level embedding extraction  
5. MIL-based patient-level aggregation  
6. Multimodal fusion with clinical metadata  
7. Cox-based survival optimization (OS and RFS trained separately)

Each patient is modeled as a bag of tumor instances, enabling biologically grounded risk aggregation.

---

## 2. End-to-End Pipeline

---

### STEP 0 — Download and Prepare MAMA-MIA

Download the MAMA-MIA dataset from the official source.

Expected directory structure:

```
MAMA-MIA/
└── raw/
    └── Dataset001_MAMA_MIA/
        ├── imagesTr/
        ├── labelsTr/
        ├── imagesTs/
        └── labelsTs/
```

Each patient must follow nnUNet naming convention:

```
patient001_0000.nii.gz
patient001_0001.nii.gz
patient001_0002.nii.gz
patient001_0003.nii.gz
```

Where:
- `_0000` = Pre-contrast
- `_0001–0003` = Post-contrast phases

Ensure consistent phase ordering across patients.

---

### STEP 1 — Install nnUNet v2

#### 1.1 Create Environment

```bash
conda create -n nnunetv2 python=3.9 -y
conda activate nnunetv2
```

#### 1.2 Install nnUNet

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

#### 1.3 Set Environment Variables

```bash
export nnUNet_raw="/path/to/MAMA-MIA/raw"
export nnUNet_preprocessed="/path/to/MAMA-MIA/preprocessed"
export nnUNet_results="/path/to/MAMA-MIA/nnunet_results"
```

---

### STEP 2 — Configure Dataset for 4-Phase Input

Inside `Dataset001_MAMA_MIA/dataset.json`:

```json
"channel_names": {
  "0": "DCE_phase0",
  "1": "DCE_phase1",
  "2": "DCE_phase2",
  "3": "DCE_phase3"
}
```

---

### STEP 3 — Train nnUNet (3D Full Resolution)

Preprocess:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Train (Fold 0):

```bash
nnUNetv2_train 1 3d_fullres 0
```

Train all folds (optional):

```bash
nnUNetv2_train 1 3d_fullres all
```

---

### STEP 4 — Generate Tumor Segmentations

```bash
nnUNetv2_predict \
    -i /path/to/imagesTs \
    -o /path/to/nnUNet_predictions \
    -d 1 \
    -c 3d_fullres \
    -f 0 \
    -chk checkpoint_best.pth
```

Predicted masks:

```
nnUNet_predictions/
├── patient001.nii.gz
├── patient002.nii.gz
```

---

### STEP 5 — Extract Tumor VOIs

Update paths inside `postProcessing.py`:

```python
SOURCE_IMAGE_DIR = "/path/to/imagesTr"
SOURCE_MASK_DIR  = "/path/to/nnUNet_predictions"
```

Run:

```bash
python postProcessing.py
```

Output:

```
cropped_voi_images/
├── patient001_tumor1_img.nii.gz
├── patient001_tumor1_mask.nii.gz
```

Each tumor is saved independently for MIL modeling.

---

### STEP 6 — Tumor-Level Augmentation

```bash
python augmentation.py
```

Augmentations applied:

Geometric:
- 45°, 90°, 135°, 180°, 225°, 270°, 315° rotations

Intensity:
- Brightness shift
- Contrast scaling
- Gaussian noise injection

All transformations preserve spatial alignment between image and mask.

---

### STEP 7 — Feature Extraction

Use your preferred encoder:

- Vision Transformer (ViT)
- 3D CNN
- Radiomics

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

### STEP 8 — Train Survival Models

Overall Survival:

```bash
python survmx_single_head_death.py
```

Recurrence-Free Survival:

```bash
python survmx_single_head_recurrence.py
```

---

## 3. MIL Architecture

Each patient is modeled as:

```
Patient^z = [Tumor_1^z, Tumor_2^z, ..., Tumor_k^z]
```

Model components:

- Residual Tumor Encoder  
- Multi-Head MIL Attention  
- Global Encoder  
- Cross-Attention Fusion  
- Mixture-of-Experts  
- Cox Proportional Hazards Head  

Separate models are trained for OS and RFS.

---

## 4. Outputs

During training:
- Epoch-wise loss
- Validation C-index

Final evaluation:
- Death C-index
- Recurrence C-index
- Kaplan–Meier survival curves
- Saved model checkpoints

---

## 5. Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib
pip install SimpleITK
pip install scikit-survival
```

---

## 6. Reproducibility

- Patient-level independent train/test splits
- Deterministic seed control
- Modular embedding backbone replacement
- Compatible with deep or radiomic features

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

TBA (MICCAI submission under review)

---

# Contact
For questions regarding implementation or reproduction, please open an issue.

---
