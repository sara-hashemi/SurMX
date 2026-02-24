#!/usr/bin/env python3
"""
SurvMx — Tumor-level 3D Augmentation

This script generates *paired* augmentations (image + mask) for cropped tumor VOIs
produced by `postprocessing.py`.

Key properties:
- Applies augmentation to image and mask (spatial alignment preserved)
- Uses **nearest-neighbor** interpolation for masks (prevents label corruption)
- Supports deterministic runs via --seed
- Writes a lightweight manifest (CSV) for reproducibility

Expected input naming (from postprocessing):
    <sample_id>_img.nii.gz
    <sample_id>_mask.nii.gz

Example output naming:
    <sample_id>_rot45_img.nii.gz / _mask.nii.gz
    <sample_id>_bright_img.nii.gz / _mask.nii.gz
"""

import os
import re
import json
import argparse
import random
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nii(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_nii(volume: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, header), out_path)


def parse_base_id(filename: str) -> str:
    # strips *_img.nii.gz or *_mask.nii.gz
    return re.sub(r"_(img|mask)\.nii\.gz$", "", filename)


def list_pairs(cropped_dir: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (base_id, img_path, mask_path)
    """
    items = []
    for f in os.listdir(cropped_dir):
        if not f.endswith("_img.nii.gz"):
            continue
        base = parse_base_id(f)
        img_path = os.path.join(cropped_dir, f)
        mask_path = os.path.join(cropped_dir, base + "_mask.nii.gz")
        if os.path.exists(mask_path):
            items.append((base, img_path, mask_path))
    items.sort(key=lambda x: x[0])
    return items


# ----------------------------
# Augmentations
# ----------------------------
def rotate_3d(volume_t: torch.Tensor, angle_deg: float, is_mask: bool) -> torch.Tensor:
    """
    Rotate a 3D tensor (D, H, W) slice-wise around the axial (H,W) plane.
    Uses:
      - bilinear for image
      - nearest for mask
    """
    angle = float(angle_deg) * np.pi / 180.0
    c, s = float(np.cos(angle)), float(np.sin(angle))

    # Affine for 2D rotation; applied to each slice
    theta = torch.tensor(
        [[c, -s, 0.0],
         [s,  c, 0.0]],
        dtype=torch.float32,
        device=volume_t.device
    )
    mode = "nearest" if is_mask else "bilinear"

    out_slices = []
    for z in range(volume_t.shape[0]):
        sl = volume_t[z][None, None, ...]  # (1,1,H,W)
        grid = F.affine_grid(theta[None, ...], sl.shape, align_corners=False)
        rot = F.grid_sample(sl, grid, mode=mode, padding_mode="zeros", align_corners=False)
        out_slices.append(rot[0, 0])

    out = torch.stack(out_slices, dim=0)
    if is_mask:
        # ensure binary after interpolation
        out = (out > 0.5).to(out.dtype)
    return out


def brightness_shift(img_t: torch.Tensor, shift: float) -> torch.Tensor:
    return img_t + shift


def contrast_scale(img_t: torch.Tensor, scale: float) -> torch.Tensor:
    mean = img_t.mean()
    return (img_t - mean) * scale + mean


def gaussian_noise(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    return img_t + torch.randn_like(img_t) * sigma


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="3D paired augmentation for tumor VOIs (image + mask).")
    ap.add_argument("--cropped_dir", type=str, default="./cropped_voi_images",
                    help="Directory containing *_img.nii.gz and *_mask.nii.gz.")
    ap.add_argument("--out_dir", type=str, default="./augmented_cropped_voi_images",
                    help="Output directory for augmented VOIs.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--rotations", type=str, default="45,90,135,180,225,270,315",
                    help="Comma-separated rotation angles in degrees.")
    ap.add_argument("--do_geometric", action="store_true", default=True, help="Enable rotations.")
    ap.add_argument("--do_intensity", action="store_true", default=True, help="Enable intensity augs.")
    ap.add_argument("--bright_shift", type=float, default=0.10, help="Brightness shift magnitude.")
    ap.add_argument("--contrast_scale", type=float, default=1.20, help="Contrast scale factor.")
    ap.add_argument("--noise_sigma", type=float, default=0.03, help="Gaussian noise sigma.")
    ap.add_argument("--save_manifest", action="store_true", default=True, help="Write augmentation_manifest.csv.")
    ap.add_argument("--manifest_path", type=str, default=None,
                    help="Optional manifest path. Default: <out_dir>/augmentation_manifest.csv")
    args = ap.parse_args()

    set_seed(args.seed)

    cropped_dir = args.cropped_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rotations = [float(x) for x in args.rotations.split(",") if x.strip()]

    pairs = list_pairs(cropped_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No pairs found in {cropped_dir}. Expected files like *_img.nii.gz and *_mask.nii.gz")

    manifest_rows: List[Dict] = []

    print(f"[Augment] Found {len(pairs)} VOIs in: {cropped_dir}")
    print(f"[Augment] Output dir: {out_dir}")
    print(f"[Augment] Geometric: {args.do_geometric} | Intensity: {args.do_intensity} | rotations={rotations}")

    for base, img_path, mask_path in pairs:
        img_np, affine, hdr = load_nii(img_path)
        msk_np, _, _ = load_nii(mask_path)

        img_t = torch.from_numpy(img_np.astype(np.float32))
        msk_t = torch.from_numpy(msk_np.astype(np.float32))

        # Note: input from postprocessing is typically (H,W,D). Normalize to (D,H,W) for slice-wise ops.
        # If the crop is saved as (H,W,D), convert:
        if img_t.ndim != 3:
            raise ValueError(f"Expected 3D volume for {img_path}, got shape {tuple(img_t.shape)}")
        if img_t.shape == msk_t.shape:
            pass
        else:
            raise ValueError(f"Image/mask shape mismatch for base={base}: {tuple(img_t.shape)} vs {tuple(msk_t.shape)}")

        # Heuristic: treat last axis as depth if it is smaller than spatial dims
        # Convert (H,W,D) -> (D,H,W)
        if img_t.shape[2] != img_t.shape[0] and img_t.shape[2] != img_t.shape[1]:
            img_t_dhw = img_t.permute(2, 0, 1).contiguous()
            msk_t_dhw = msk_t.permute(2, 0, 1).contiguous()
            needs_back = True
        else:
            # already (D,H,W)
            img_t_dhw = img_t
            msk_t_dhw = msk_t
            needs_back = False

        def write_pair(suffix: str, img_out_t: torch.Tensor, msk_out_t: torch.Tensor):
            if needs_back:
                img_out = img_out_t.permute(1, 2, 0).cpu().numpy()
                msk_out = msk_out_t.permute(1, 2, 0).cpu().numpy()
            else:
                img_out = img_out_t.cpu().numpy()
                msk_out = msk_out_t.cpu().numpy()

            out_img = os.path.join(out_dir, f"{base}_{suffix}_img.nii.gz")
            out_msk = os.path.join(out_dir, f"{base}_{suffix}_mask.nii.gz")
            save_nii(img_out, affine, hdr, out_img)
            save_nii(msk_out, affine, hdr, out_msk)

            manifest_rows.append({
                "original_id": base,
                "augmented_id": f"{base}_{suffix}",
                "augmentation": suffix,
                "img_path": out_img,
                "mask_path": out_msk,
                "seed": args.seed
            })

        # -----------------------
        # Geometric rotations
        # -----------------------
        if args.do_geometric:
            for ang in rotations:
                img_rot = rotate_3d(img_t_dhw, ang, is_mask=False)
                msk_rot = rotate_3d(msk_t_dhw, ang, is_mask=True)
                write_pair(f"rot{int(ang)}", img_rot, msk_rot)

        # -----------------------
        # Intensity-only (mask unchanged)
        # -----------------------
        if args.do_intensity:
            # Brightness
            write_pair("bright", brightness_shift(img_t_dhw, args.bright_shift), msk_t_dhw)
            # Contrast
            write_pair("contrast", contrast_scale(img_t_dhw, args.contrast_scale), msk_t_dhw)
            # Noise
            write_pair("noise", gaussian_noise(img_t_dhw, args.noise_sigma), msk_t_dhw)

    if args.save_manifest:
        manifest_path = args.manifest_path or os.path.join(out_dir, "augmentation_manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"[Augment] Saved manifest: {manifest_path}")

    print(f"[Augment] Done. Total augmented samples: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
