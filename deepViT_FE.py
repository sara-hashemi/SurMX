#!/usr/bin/env python3
"""
SurvMx — Deep ViT Feature Extraction for Tumor VOIs 

This script extracts:
- deep_global_* : global (slice-level) embedding averaged over tumor-bearing slices
- deep_tumor_*  : tumor-masked embedding averaged over tumor-bearing slices

Inputs:
- Cropped tumor VOIs:  <id>_img.nii.gz and <id>_mask.nii.gz
- Augmented VOIs (optional): same naming convention

Outputs:
- CSV with one row per VOI id (original + augmented)
  Columns: id, deep_global_1..K, deep_tumor_1..K

Notes:
- Uses a pretrained **slice-based** ViT backbone from timm (features_only=True).
- Resumes automatically if --out_csv exists (skips already-processed ids).
- Mask resizing uses nearest-neighbor to avoid label corruption.
"""

import os
import re
import time
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from skimage.transform import resize

warnings.filterwarnings("ignore")


# ----------------------------
# Repro / device
# ----------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# I/O helpers
# ----------------------------
def collect_image_mask_pairs(folder: str) -> List[Tuple[str, str]]:
    """
    Returns list of (img_path, mask_path) for items in folder.
    """
    if folder is None or (not os.path.exists(folder)):
        return []
    pairs = []
    for f in os.listdir(folder):
        if not f.endswith("_img.nii.gz"):
            continue
        base = re.sub(r"_img\.nii\.gz$", "", f)
        img_path = os.path.join(folder, f)
        mask_path = os.path.join(folder, base + "_mask.nii.gz")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_img2d(slice_img: np.ndarray, out_size: int) -> np.ndarray:
    """
    Normalize + resize a 2D slice to (1, out_size, out_size).
    """
    x = slice_img.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x) + 1e-6
    x = x / std
    x = resize(x, (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return x[None, ...]  # (1,H,W)


def preprocess_mask2d(slice_mask: np.ndarray, out_size: int) -> np.ndarray:
    """
    Resize a 2D mask to (out_size, out_size), nearest-neighbor.
    """
    m = slice_mask.astype(np.float32)
    m = resize(m, (out_size, out_size), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
    m = (m > 0.5).astype(np.float32)
    return m


def feat_to_vec(feat_hwk: np.ndarray, mask_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    feat_hwk: (H,W,K) float
    mask_hw : (H,W) binary float
    Returns:
      global_vec: mean over all pixels
      tumor_vec : mean over masked pixels (fallback to global if empty)
    """
    H, W, K = feat_hwk.shape
    flat = feat_hwk.reshape(-1, K)
    global_vec = flat.mean(axis=0)

    m = mask_hw.reshape(-1) > 0.5
    if m.sum() == 0:
        tumor_vec = global_vec
    else:
        tumor_vec = flat[m].mean(axis=0)
    return global_vec, tumor_vec


# ----------------------------
# Model builder
# ----------------------------
class FeatProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    def forward(self, x):
        return self.proj(x)


def build_backbone(model_name: str, pretrained: bool, feat_dim: int, device: torch.device):
    """
    Returns:
      image_encoder: timm backbone returning list of feature maps
      feat_proj: 1x1 conv to feat_dim (K)
    """
    image_encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True)
    image_encoder.eval().to(device)

    # infer last stage channels
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        feats = image_encoder(dummy)
        last = feats[-1]
        in_ch = last.shape[1]

    feat_proj = FeatProjector(in_ch, feat_dim).eval().to(device)
    return image_encoder, feat_proj


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract deep tumor/global embeddings from cropped tumor VOIs.")
    ap.add_argument("--orig_dir", type=str, default="./cropped_voi_images",
                    help="Directory with original VOIs (*_img.nii.gz, *_mask.nii.gz).")
    ap.add_argument("--aug_dir", type=str, default="./augmented_cropped_voi_images",
                    help="Directory with augmented VOIs. Use empty string to disable.")
    ap.add_argument("--out_csv", type=str, default="./features_deep_vit.csv",
                    help="Output CSV path. Resumes if exists.")
    ap.add_argument("--model_name", type=str, default="tiny_vit_21m_224",
                    help="timm model name (slice-based ViT).")
    ap.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights.")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--deep_size", type=int, default=224, help="Input size to ViT.")
    ap.add_argument("--feat_dim", type=int, default=256, help="Projection feature dim (K).")
    ap.add_argument("--save_every", type=int, default=50, help="Checkpoint write frequency.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[ViT-FE] device={device} | model={args.model_name} | pretrained={args.pretrained}")

    # Collect pairs
    pairs_orig = collect_image_mask_pairs(args.orig_dir)
    pairs_augm = collect_image_mask_pairs(args.aug_dir) if (args.aug_dir and args.aug_dir.strip()) else []
    all_pairs = pairs_orig + pairs_augm
    print(f"[ViT-FE] Original VOIs: {len(pairs_orig)} | Augmented VOIs: {len(pairs_augm)} | Total: {len(all_pairs)}")
    if len(all_pairs) == 0:
        raise FileNotFoundError("No VOI pairs found. Check --orig_dir/--aug_dir.")

    # Resume support
    processed_ids = set()
    if os.path.exists(args.out_csv):
        df_existing = pd.read_csv(args.out_csv)
        if "id" in df_existing.columns:
            processed_ids = set(df_existing["id"].astype(str).tolist())
        else:
            df_existing = pd.DataFrame()
        print(f"[ViT-FE] Resuming: {len(processed_ids)} already processed")
    else:
        df_existing = pd.DataFrame()

    remaining = []
    for img_path, mask_path in all_pairs:
        sample_id = os.path.basename(img_path).replace("_img.nii.gz", "")
        if sample_id not in processed_ids:
            remaining.append((img_path, mask_path))
    print(f"[ViT-FE] Remaining: {len(remaining)}")

    # Build model
    image_encoder, feat_proj = build_backbone(args.model_name, args.pretrained, args.feat_dim, device)

    results = []
    start = time.time()

    for idx, (img_path, mask_path) in enumerate(tqdm(remaining, desc="Extracting Deep Features")):
        sample_id = os.path.basename(img_path).replace("_img.nii.gz", "")
        try:
            img_3d = nib.load(img_path).get_fdata()
            msk_3d = nib.load(mask_path).get_fdata()

            # Expect VOI shape either (H,W,D) or (D,H,W). Normalize to (H,W,D) for slicing.
            if img_3d.ndim != 3:
                raise ValueError(f"Expected 3D VOI, got shape {img_3d.shape}")
            if img_3d.shape != msk_3d.shape:
                raise ValueError(f"Image/mask mismatch: {img_3d.shape} vs {msk_3d.shape}")

            # If volume is (D,H,W), convert to (H,W,D)
            if img_3d.shape[0] < img_3d.shape[1] and img_3d.shape[0] < img_3d.shape[2]:
                img_3d = np.transpose(img_3d, (1, 2, 0))
                msk_3d = np.transpose(msk_3d, (1, 2, 0))

            global_vecs, tumor_vecs = [], []

            for z in range(img_3d.shape[2]):
                slice_img = img_3d[:, :, z]
                slice_mask = msk_3d[:, :, z]
                if np.sum(slice_mask) < 1e-4:
                    continue

                img = preprocess_img2d(slice_img, args.deep_size)          # (1,H,W)
                msk = preprocess_mask2d(slice_mask, args.deep_size)        # (H,W)

                # ViT expects 3-channel input; replicate grayscale
                img3 = np.repeat(img, 3, axis=0)                           # (3,H,W)
                img_tensor = torch.from_numpy(img3[None, ...]).to(device)  # (1,3,H,W)

                with torch.no_grad():
                    raw_feat = image_encoder(img_tensor)[-1]               # (1,C,h,w)
                    proj_feat = feat_proj(raw_feat)                        # (1,K,h,w)
                    feat = F.interpolate(
                        proj_feat,
                        size=(args.deep_size, args.deep_size),
                        mode="bilinear",
                        align_corners=False
                    )[0].permute(1, 2, 0).cpu().numpy()                    # (H,W,K)

                glob_vec, tum_vec = feat_to_vec(feat, msk)
                global_vecs.append(glob_vec)
                tumor_vecs.append(tum_vec)

            if len(global_vecs) > 0:
                g_mean = np.mean(global_vecs, axis=0)
                t_mean = np.mean(tumor_vecs, axis=0)
            else:
                g_mean = np.zeros(args.feat_dim, dtype=np.float32)
                t_mean = np.zeros(args.feat_dim, dtype=np.float32)

            feat_row = {"id": sample_id}
            feat_row.update({f"deep_global_{i+1}": float(v) for i, v in enumerate(g_mean)})
            feat_row.update({f"deep_tumor_{i+1}": float(v) for i, v in enumerate(t_mean)})
            results.append(feat_row)

            if (idx + 1) % args.save_every == 0:
                df_out = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
                df_out.to_csv(args.out_csv, index=False)

        except Exception as e:
            print(f"[ViT-FE] Error processing {sample_id}: {e}")

    df_out = pd.concat([df_existing, pd.DataFrame(results)], ignore_index=True)
    df_out.to_csv(args.out_csv, index=False)

    print(f"[ViT-FE] Done. Saved: {args.out_csv}")
    print(f"[ViT-FE] Total samples: {len(df_out)} | time: {((time.time() - start)/60):.2f} min")


if __name__ == "__main__":
    main()
