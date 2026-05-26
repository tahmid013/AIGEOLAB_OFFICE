"""Inference helpers: load checkpoint, run sliding-window prediction, vectorise heatmap to polygons.

Used by notebooks/03_app_colab.ipynb (the Gradio webapp). Importable so the app can
focus on UI and the inference logic is testable in isolation.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(ckpt_path: str | Path, device: str = "cuda"):
    """Load a trained checkpoint produced by `02_train_colab.ipynb` Cell 9.

    Returns (model, cfg_dict, meta_dict). meta has keys 'epoch', 'val_iou'.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    M = cfg["model"]
    model = getattr(smp, M["arch"])(
        encoder_name=M["encoder"],
        encoder_weights=None,                # init random, weights come from state_dict
        in_channels=M["in_channels"],
        classes=M["classes"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {"epoch": ckpt.get("epoch"), "val_iou": ckpt.get("val_iou")}
    return model, cfg, meta


# -----------------------------------------------------------------------------
# Inference (sliding-window)
# -----------------------------------------------------------------------------
_NORMALISE = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std =(0.229, 0.224, 0.225)),
                         ToTensorV2()])


def predict_heatmap(model, image_rgb_uint8: np.ndarray, patch: int, stride: int,
                    device: str = "cuda") -> np.ndarray:
    """Sliding-window inference. Returns float32 boundary-probability map [0,1], same H/W as input."""
    H, W = image_rgb_uint8.shape[:2]
    if H < patch or W < patch:
        raise ValueError(f"Image too small: {image_rgb_uint8.shape} for patch {patch}.")
    pred = np.zeros((H, W), dtype=np.float32)
    cnt  = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for top in range(0, H - patch + 1, stride):
            for left in range(0, W - patch + 1, stride):
                sub = image_rgb_uint8[top:top + patch, left:left + patch]
                x = _NORMALISE(image=sub)["image"].unsqueeze(0).to(device)
                p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                pred[top:top + patch, left:left + patch] += p
                cnt [top:top + patch, left:left + patch] += 1
    return pred / np.maximum(cnt, 1)


# -----------------------------------------------------------------------------
# Heatmap -> polygons (v1.5 vectorisation)
# -----------------------------------------------------------------------------
def extract_polygons(boundary_prob: np.ndarray, threshold: float = 0.4,
                     close_kernel: int = 3, min_area_px: int = 400,
                     approx_eps_frac: float = 0.004) -> List[np.ndarray]:
    """Convert a boundary heatmap into a list of polygon contours (each Nx2 int32).

    Steps: threshold -> dilate to close gaps -> invert (plot interiors become foreground)
    -> connected components -> outer contour per component -> Ramer-Douglas-Peucker simplify.
    Drops components smaller than `min_area_px` (noise).
    """
    bnd = (boundary_prob > threshold).astype(np.uint8)
    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        bnd = cv2.dilate(bnd, k)
    interior = (1 - bnd).astype(np.uint8)
    n_lbl, labels = cv2.connectedComponents(interior, connectivity=4)
    polygons: List[np.ndarray] = []
    for lbl in range(1, n_lbl):
        comp = (labels == lbl).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area_px:
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        eps = approx_eps_frac * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, eps, closed=True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append(approx)
    return polygons


def draw_polygons(image_rgb: np.ndarray, polygons: List[np.ndarray],
                  color: Tuple[int, int, int] = (255, 215, 0),
                  thickness: int = 3, fill_alpha: float = 0.0) -> np.ndarray:
    """Render polygons as bright outlines on a copy of `image_rgb` (uint8, H, W, 3)."""
    out = image_rgb.copy()
    if fill_alpha > 0:
        overlay = out.copy()
        for poly in polygons:
            cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
        cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, dst=out)
    for poly in polygons:
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def polygon_stats(polygons: List[np.ndarray], pixel_size_m: float = 0.1) -> dict:
    """Return a small dict summarising the detected plots — used by the webapp summary panel."""
    if not polygons:
        return {"n_plots": 0, "total_area_m2": 0.0, "mean_area_m2": 0.0,
                "median_area_m2": 0.0, "px_per_m": 1.0 / pixel_size_m}
    px_areas = [float(cv2.contourArea(p.astype(np.int32).reshape(-1, 1, 2))) for p in polygons]
    m2 = np.array(px_areas) * (pixel_size_m ** 2)
    return {
        "n_plots":        int(len(polygons)),
        "total_area_m2":  float(m2.sum()),
        "mean_area_m2":   float(m2.mean()),
        "median_area_m2": float(np.median(m2)),
        "px_per_m":       1.0 / pixel_size_m,
    }
