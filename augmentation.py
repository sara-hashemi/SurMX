import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

CROPPED_DIR = "./1-cropped_voi_images"
OUT_DIR = "./2-augmented_cropped_voi_images"
os.makedirs(OUT_DIR, exist_ok=True)

ROTATION_ANGLES = [45, 90, 135, 180, 225, 270, 315]
BRIGHTNESS_FACTOR = 1.1
CONTRAST_FACTOR = 1.1
NOISE_STD = 0.05

CLINICAL_CSV = "/home/ec2-user/SageMaker/MAMA-MIA/raw/Dataset001_MAMA_MIA/image_mapping_with_clinical.csv"
clinical_df = pd.read_csv(CLINICAL_CSV)
view_dict = dict(zip(clinical_df["nnunet_id"], clinical_df["view"]))

VIEW_TO_AXIS = {
    "axial": (1, 2),
    "sagittal": (0, 1),
    "coronal": (0, 2)
}

# -------------------------------
# FUNCTIONS
# -------------------------------
def load_nii(path):
    return nib.load(path).get_fdata(), nib.load(path).affine

def rotate_tensor(volume_np, angle_deg, axis):
    vol = torch.tensor(volume_np, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    theta = torch.eye(3)
    ax1, ax2 = axis
    theta[ax1, ax1] = cos_a
    theta[ax1, ax2] = -sin_a
    theta[ax2, ax1] = sin_a
    theta[ax2, ax2] = cos_a

    theta_4x4 = torch.eye(4)
    theta_4x4[:3, :3] = theta
    theta = theta_4x4[:3, :].unsqueeze(0).to(vol.dtype).to(DEVICE)

    grid = F.affine_grid(theta, vol.size(), align_corners=True)
    rotated = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return rotated.squeeze().cpu().numpy()

def adjust_brightness(vol, factor):
    return np.clip(vol * factor, 0, np.max(vol))

def adjust_contrast(vol, factor):
    mean = vol.mean()
    return np.clip((vol - mean) * factor + mean, 0, np.max(vol))

def add_noise(vol, std):
    noise = np.random.normal(0, std, vol.shape)
    return np.clip(vol + noise, 0, np.max(vol))

def save_pair(img_np, mask_np, base_name, suffix, affine):
    nib.save(nib.Nifti1Image(img_np, affine), os.path.join(OUT_DIR, f"{base_name}_{suffix}_img.nii.gz"))
    nib.save(nib.Nifti1Image(mask_np, affine), os.path.join(OUT_DIR, f"{base_name}_{suffix}_mask.nii.gz"))

# -------------------------------
# MAIN LOOP
# -------------------------------
files = sorted([f for f in os.listdir(CROPPED_DIR) if f.endswith("_img.nii.gz")])

for img_file in tqdm(files, desc="Augmenting Images + Masks"):
    base = img_file.replace("_img.nii.gz", "")
    img_path = os.path.join(CROPPED_DIR, f"{base}_img.nii.gz")
    mask_path = os.path.join(CROPPED_DIR, f"{base}_mask.nii.gz")

    try:
        view = view_dict.get(base, "axial")
        axis = VIEW_TO_AXIS.get(view.lower(), (1, 2))

        img_np, affine = load_nii(img_path)
        mask_np, _ = load_nii(mask_path)

        # --- Rotations ---
        for angle in ROTATION_ANGLES:
            rot_img = rotate_tensor(img_np, angle, axis)
            rot_mask = rotate_tensor(mask_np, angle, axis)
            save_pair(rot_img, rot_mask, base, f"rot{angle}", affine)

        # --- Brightness ---
        bright_img = adjust_brightness(img_np, BRIGHTNESS_FACTOR)
        save_pair(bright_img, mask_np, base, "bright", affine)

        # --- Contrast ---
        contrast_img = adjust_contrast(img_np, CONTRAST_FACTOR)
        save_pair(contrast_img, mask_np, base, "contrast", affine)

        # --- Noise ---
        noisy_img = add_noise(img_np, NOISE_STD)
        save_pair(noisy_img, mask_np, base, "noise", affine)

    except Exception as e:
        print(f"❌ Error processing {img_file}: {e}")