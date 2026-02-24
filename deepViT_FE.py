import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from timm import create_model
from skimage.transform import resize
import warnings

# =====================================
# CONFIGURATION
# =====================================
ORIG_DIR = "1-cropped_voi_images"
AUG_DIR = "2-augmented_cropped_voi_images"
OUT_CSV = "deep_features_cropped_vois_ALL.csv"
IMG_SIZE = 256
DEEP_SIZE = 64
FEAT_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 100

# =====================================
# SILENCE WARNINGS
# =====================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================
# MODEL SETUP
# =====================================
image_encoder = create_model("tiny_vit_21m_224", pretrained=True, features_only=True, out_indices=(3,)).to(DEVICE)
with torch.no_grad():
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    feat_ch = image_encoder(dummy)[-1].shape[1]
feat_proj = torch.nn.Conv2d(feat_ch, FEAT_DIM, 1).to(DEVICE)
image_encoder.eval(); feat_proj.eval()

# =====================================
# IMAGE PREPROCESSING
# =====================================
def normalize_and_window(slice2d, wmin=-1000, wmax=400):
    sl = np.clip(slice2d, wmin, wmax)
    return (sl - wmin) / float(wmax - wmin)

def preprocess_img2d(slice2d, size=IMG_SIZE):
    sl = normalize_and_window(slice2d)
    sl = resize(sl, (size, size), mode='reflect', anti_aliasing=True)
    img = np.stack([sl]*3, axis=0)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return ((img - mean) / std).astype(np.float32)

def preprocess_mask2d(mask2d, size=IMG_SIZE):
    mask = (mask2d > 0).astype(np.float32)
    return resize(mask, (size, size), order=0, preserve_range=True, anti_aliasing=False)[None, :, :]

def feat_to_vec(feat, mask=None):
    glob_vec = feat.mean(axis=(1, 2))
    if mask is not None:
        mask = mask[0]
        tumor_vec = np.sum(feat * mask, axis=(1, 2)) / (np.sum(mask) + 1e-8)
    else:
        tumor_vec = np.zeros_like(glob_vec)
    return glob_vec, tumor_vec

# =====================================
# COLLECT VOI IMAGE-MASK PAIRS
# =====================================
def collect_image_mask_pairs(folder):
    files = sorted(os.listdir(folder))
    img_files = [f for f in files if f.endswith("_img.nii.gz")]
    pairs = []
    for img in img_files:
        base = img.replace("_img.nii.gz", "")
        mask = f"{base}_mask.nii.gz"
        if mask in files:
            pairs.append((os.path.join(folder, img), os.path.join(folder, mask)))
    return pairs

pairs_orig = collect_image_mask_pairs(ORIG_DIR)
pairs_augm = collect_image_mask_pairs(AUG_DIR)
all_pairs = pairs_orig + pairs_augm
print(f" Total VOI samples (original + augmented): {len(all_pairs)}")

# =====================================
# RESUME FROM CSV
# =====================================
processed_ids = set()
if os.path.exists(OUT_CSV):
    df_existing = pd.read_csv(OUT_CSV)
    processed_ids = set(df_existing["id"].tolist())
    print(f" Resuming: {len(processed_ids)} already processed")
else:
    df_existing = pd.DataFrame()

remaining = [(img, msk) for (img, msk) in all_pairs if os.path.basename(img).replace("_img.nii.gz", "") not in processed_ids]
print(f" Remaining to process: {len(remaining)}")

# =====================================
# FEATURE EXTRACTION LOOP
# =====================================
results = []
start = time.time()

for idx, (img_path, mask_path) in enumerate(tqdm(remaining, desc="Extracting Deep Features")):
    sample_id = os.path.basename(img_path).replace("_img.nii.gz", "")
    try:
        img_3d = nib.load(img_path).get_fdata()
        msk_3d = nib.load(mask_path).get_fdata()

        global_vecs, tumor_vecs = [], []
        for z in range(img_3d.shape[2]):
            slice_img = img_3d[:, :, z]
            slice_mask = msk_3d[:, :, z]
            if np.sum(slice_mask) < 1e-4:
                continue

            img = preprocess_img2d(slice_img)
            msk = preprocess_mask2d(slice_mask, DEEP_SIZE)
            img_tensor = torch.from_numpy(img[None, ...]).to(DEVICE)

            with torch.no_grad():
                raw_feat = image_encoder(img_tensor)[-1]
                proj_feat = feat_proj(raw_feat)
                feat = F.interpolate(proj_feat, size=(DEEP_SIZE, DEEP_SIZE), mode='bilinear', align_corners=False)[0].cpu().numpy()

            glob_vec, tum_vec = feat_to_vec(feat, msk)
            global_vecs.append(glob_vec)
            tumor_vecs.append(tum_vec)

        if global_vecs:
            feat_row = {
                "id": sample_id,
                **{f"deep_global_{i+1}": v for i, v in enumerate(np.mean(global_vecs, axis=0))},
                **{f"deep_tumor_{i+1}": v for i, v in enumerate(np.mean(tumor_vecs, axis=0))}
            }
        else:
            feat_row = {
                "id": sample_id,
                **{f"deep_global_{i+1}": 0.0 for i in range(FEAT_DIM)},
                **{f"deep_tumor_{i+1}": 0.0 for i in range(FEAT_DIM)}
            }

        results.append(feat_row)

        if (idx + 1) % SAVE_EVERY == 0:
            print(" Saving checkpoint...")
            df_out = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
            df_out.to_csv(OUT_CSV, index=False)

    except Exception as e:
        print(f" Error processing {sample_id}: {e}")

# Final save
df_out = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
df_out.to_csv(OUT_CSV, index=False)

print(f" Feature extraction complete. {len(df_out)} samples saved.")
print(f" Total time: {((time.time() - start)/60):.2f} min")