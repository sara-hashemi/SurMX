import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import label, center_of_mass
from skimage.measure import regionprops
from tqdm import tqdm

# --------------------------
# CONFIGURATION
# --------------------------
SOURCE_IMAGE_DIR = "/home/ec2-user/SageMaker/MAMA-MIA/raw/Dataset001_MAMA_MIA/imagesTr"
SOURCE_MASK_DIR = "/home/ec2-user/SageMaker/MAMA-MIA/raw/Dataset001_MAMA_MIA/labelsTr"
CROPPED_OUTDIR = "./2-cropped_voi_images"
SUMMARY_OUTFILE = "./summary_tumor_qc.csv"

MIN_TUMOR_SIZE = 500   # Minimum voxel count to keep
MAX_TUMORS = 10       # Maximum tumors to retain per image

os.makedirs(CROPPED_OUTDIR, exist_ok=True)

# --------------------------
# Pipeline
# --------------------------
summary = []

image_files = sorted([f for f in os.listdir(SOURCE_IMAGE_DIR) if f.endswith("_0000.nii.gz")])

for fname in tqdm(image_files, desc="Cropping tumors"):
    base_id = fname.replace("_0000.nii.gz", "")  # Used to find label

    # Check if this image was already processed (any tumor crop exists)
    existing_crops = [f for f in os.listdir(CROPPED_OUTDIR) if f.startswith(base_id) and "_tumor1_img" in f]
    if existing_crops:
        print(f"⏩ Skipping {fname} — crops already exist")
        continue
    
    img_path = os.path.join(SOURCE_IMAGE_DIR, fname)
    mask_path = os.path.join(SOURCE_MASK_DIR, f"{base_id}.nii.gz")

    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {fname} — mask not found at {mask_path}")
        continue

    img_nii = nib.load(img_path)
    mask_nii = nib.load(mask_path)
    img = img_nii.get_fdata()
    mask = mask_nii.get_fdata()
    affine = img_nii.affine

    # Connected components (3D)
    labeled_mask, num_raw = label(mask)
    props = regionprops(labeled_mask)
    tumor_sizes = [p.area for p in props]

    # Filter small tumors
    kept_vois = [p for p in props if p.area >= MIN_TUMOR_SIZE]
    kept_vois = sorted(kept_vois, key=lambda r: r.area, reverse=True)[:MAX_TUMORS]
    num_qc = len(kept_vois)

    # Cropping per tumor
    for i, p in enumerate(kept_vois):
        minr, minc, mins, maxr, maxc, maxs = *p.bbox[0:3], *p.bbox[3:6]
        crop_img = img[minr:maxr, minc:maxc, mins:maxs]
        crop_mask = (labeled_mask[minr:maxr, minc:maxc, mins:maxs] == p.label).astype(np.uint8)

        out_name_img = f"{base_id}_tumor{i+1}_img.nii.gz"
        out_name_mask = f"{base_id}_tumor{i+1}_mask.nii.gz"

        nib.save(nib.Nifti1Image(crop_img, affine), os.path.join(CROPPED_OUTDIR, out_name_img))
        nib.save(nib.Nifti1Image(crop_mask, affine), os.path.join(CROPPED_OUTDIR, out_name_mask))

    summary.append({
        "filename": fname,
        "tumors_before_qc": num_raw,
        "tumors_after_qc": num_qc,
        "tumor_sizes_before_qc": tumor_sizes,
        "tumor_sizes_after_qc": [p.area for p in kept_vois],
    })

# --------------------------
# Save Summary Table
# --------------------------
df_summary = pd.DataFrame(summary)
df_summary.to_csv(SUMMARY_OUTFILE, index=False)
print(f"✅ Tumor QC summary saved to {SUMMARY_OUTFILE}")
print(f"✅ Cropped VOIs saved to {CROPPED_OUTDIR}")