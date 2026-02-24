# SurvMX: Tumor-Centric Multimodal Survival Prediction

Official implementation of:

**A Multimodal Attention Transformer Framework for Tumor-Centric Survival Prediction**

SurvMX is a tumor-centric multimodal survival modeling framework that integrates 3D multiphase DCE-MRI and structured clinical metadata using a Multiple Instance Learning (MIL) architecture with multi-head attention and Cox Proportional Hazards optimization.

---

## 1. Overview

Accurate survival prediction in breast cancer requires modeling tumor-level biological heterogeneity rather than relying solely on global whole-breast representations. We propose a tumor-centric multimodal survival architecture that integrates instance-level MRI representations, global anatomical context, and structured clinical metadata within a unified Cox-based optimization framework. Tumor volumes are extracted from 3D segmentations and subjected to spatially consistent geometric rotations and intensity perturbations to enhance robustness and mitigate class imbalance. Each tumor volume is encoded using a deep feature extractor followed by a residual tumor encoder, producing instance-level embeddings. Survival prediction is formulated as a Multiple Instance Learning (MIL) problem in which each patient is represented as a variable-sized bag of tumor embeddings. A multi-head self-attention aggregation module learns adaptive instance weights to model intra-patient tumor heterogeneity. In parallel, a residual global encoder processes scan-level features to preserve anatomical context beyond localized lesions. Phase-specific embeddings are learned independently for each DCE-MRI phase to capture temporal contrast-enhancement dynamics. The aggregated tumor representation is fused with the global embedding through cross-attention and further integrated with learnable entity embeddings of categorical clinical variables. The resulting patient-level representation is refined through a Mixture-of-Experts module and optimized under a Cox Proportional Hazards objective in task-specific models for Overall Survival and Recurrence-Free Survival.

Each patient is modeled as a bag of tumor instances, enabling biologically grounded risk aggregation.

---

# 2. End-to-End Pipeline

---

## STEP 0 — Download and Prepare MAMA-MIA

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

## STEP 1 — Install nnUNet v2

### Create Environment

```bash
conda create -n nnunetv2 python=3.9 -y
conda activate nnunetv2
```

### Install nnUNet

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

### Set Environment Variables

```bash
export nnUNet_raw="/path/to/MAMA-MIA/raw"
export nnUNet_preprocessed="/path/to/MAMA-MIA/preprocessed"
export nnUNet_results="/path/to/MAMA-MIA/nnunet_results"
```

---

## STEP 2 — Configure Dataset for 4-Phase Input

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

## STEP 3 — Train nnUNet (3D Full Resolution)

Preprocess:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

Train (Fold 0):

```bash
nnUNetv2_train 1 3d_fullres 0
```

---

## STEP 4 — Generate Tumor Segmentations

```bash
nnUNetv2_predict \
    -i /path/to/imagesTs \
    -o /path/to/nnUNet_predictions \
    -d 1 \
    -c 3d_fullres \
    -f 0 \
    -chk checkpoint_best.pth
```

---

## STEP 5 — Extract Tumor VOIs (Size-Filtered)

```bash
python postProcessing.py \
    --image_dir /path/to/imagesTr \
    --mask_dir /path/to/nnUNet_predictions \
    --output_dir ./cropped_voi_images \
    --min_voxels 500
```

Connected components smaller than `min_voxels` are removed to suppress segmentation artifacts.

Output:

```
cropped_voi_images/
├── patient001_tumor1_img.nii.gz
├── patient001_tumor1_mask.nii.gz
```

---

## STEP 6 — Tumor-Level 3D Augmentation (MICCAI Version)

```bash
python augmentation.py \
    --cropped_dir ./cropped_voi_images \
    --out_dir ./augmented_cropped_voi_images \
    --seed 42
```

### Augmentations Applied

### Geometric (mask-safe)
- 45°, 90°, 135°, 180°, 225°, 270°, 315°
- Bilinear interpolation for images
- Nearest-neighbor for masks (prevents label corruption)

### Intensity
- Brightness shift
- Contrast scaling
- Gaussian noise

A reproducibility manifest is saved:

```
augmented_cropped_voi_images/augmentation_manifest.csv
```

All augmentations preserve spatial alignment between image and mask.

---

## STEP 7 — Deep Feature Extraction (ViT-Based)

```bash
python deepViT_FE.py \
    --orig_dir ./cropped_voi_images \
    --aug_dir ./augmented_cropped_voi_images \
    --out_csv features_deep_vit.csv \
    --model_name tiny_vit_21m_224 \
    --device cuda \
    --seed 42
```

### Feature Extraction Details

- Slice-based pretrained Vision Transformer (timm)
- Processes tumor-bearing slices only
- Extracts:
  - `deep_global_*` (global slice representation)
  - `deep_tumor_*` (tumor-masked representation)
- Averages embeddings across slices
- Automatically resumes if CSV already exists
- Deterministic seed control

Output:

```
features_deep_vit.csv
```

Columns:

```
id
deep_global_1 ... deep_global_K
deep_tumor_1 ... deep_tumor_K
```

---

## STEP 8 — Train Survival Models

### Overall Survival

```bash
python survmx_single_head_death.py
```

### Recurrence-Free Survival

```bash
python survmx_single_head_recurrence.py
```

---

# 3. MIL Architecture

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

# 4. Outputs

During training:
- Epoch-wise loss
- Validation C-index

Final evaluation:
- Death C-index
- Recurrence C-index
- Kaplan–Meier survival curves
- Saved model checkpoints

---

# 5. Requirements

```bash
pip install torch torchvision timm
pip install numpy pandas matplotlib nibabel scikit-image
pip install SimpleITK scikit-survival
```

GPU recommended for segmentation and feature extraction.

---

# 6. Reproducibility

- Deterministic seed control
- Mask-safe interpolation
- Augmentation manifest logging
- Feature extraction resume capability
- Patient-level independent splits
- Modular embedding backbone replacement

---

# 7. Citation

TBA (MICCAI 2026 submission)

---

# 8. Contact

For questions regarding implementation or reproduction, please open a GitHub issue.
