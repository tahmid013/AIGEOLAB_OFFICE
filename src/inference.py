"""Inference helpers: load checkpoint, run sliding-window prediction, vectorise heatmap to polygons,
regularise polygon shape, and apply a heuristic colour/texture-based land-use classifier.

Used by notebooks/03_app_colab.ipynb (the Gradio webapp). Importable so the app can
focus on UI and the inference logic is testable in isolation.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

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
    """Load a trained checkpoint produced by `02_train_colab.ipynb`.

    Returns (model, cfg_dict, meta_dict). meta has keys 'epoch', 'val_iou'.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    M = cfg["model"]
    model = getattr(smp, M["arch"])(
        encoder_name=M["encoder"],
        encoder_weights=None,
        in_channels=M["in_channels"],
        classes=M["classes"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {"epoch": ckpt.get("epoch"), "val_iou": ckpt.get("val_iou")}
    return model, cfg, meta


# -----------------------------------------------------------------------------
# Inference (sliding window)
# -----------------------------------------------------------------------------
_NORMALISE = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std =(0.229, 0.224, 0.225)),
                         ToTensorV2()])


def predict_heatmap(model, image_rgb_uint8: np.ndarray, patch: int, stride: int,
                    device: str = "cuda") -> np.ndarray:
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
# Heatmap -> polygons, with shape regularisation
# -----------------------------------------------------------------------------
def _snap_to_rect_if_good(contour: np.ndarray, fit_threshold: float = 0.86) -> np.ndarray:
    """If a min-area rectangle covers >= fit_threshold of the contour area, return the rect.
    Otherwise return the contour as-is.

    Cadastral plots are usually rectangular or trapezoidal; this snaps near-rectangular
    detections to clean 4-vertex polygons.
    """
    cnt_area = cv2.contourArea(contour)
    if cnt_area < 1:
        return contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)              # 4 corners as float32
    rect_area = cv2.contourArea(box.astype(np.float32))
    if rect_area <= 0:
        return contour
    iou = cnt_area / rect_area
    if iou >= fit_threshold:
        return box.astype(np.int32).reshape(-1, 2)
    return contour


def extract_polygons(boundary_prob: np.ndarray, threshold: float = 0.4,
                     close_kernel: int = 3, min_area_px: int = 400,
                     approx_eps_frac: float = 0.015,
                     regularise_to_rect: bool = True,
                     rect_iou_threshold: float = 0.86) -> List[np.ndarray]:
    """Convert a boundary heatmap into a list of polygon contours (each Nx2 int32).

    Pipeline: threshold -> close gaps -> invert (plot interiors become foreground)
    -> connected components -> outer contour -> R-D-P simplify
    -> (optional) snap to min-area rectangle when the fit is tight enough.
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
        if len(approx) < 3:
            continue
        if regularise_to_rect:
            approx = _snap_to_rect_if_good(approx, fit_threshold=rect_iou_threshold)
        polygons.append(approx)
    return polygons


# -----------------------------------------------------------------------------
# Heuristic land-use classifier (no training data yet — colour + texture rules)
# -----------------------------------------------------------------------------
CLASS_COLOURS = {
    "cultivated field":  (124, 191,  60),   # vivid green
    "vegetation/fallow": (160, 200, 100),   # pale green
    "pond / water":      ( 70, 145, 220),   # blue
    "built-up / house":  (240, 100,  80),   # red-orange
    "bare ground/path":  (220, 195, 130),   # tan
    "uncategorised":     (180, 180, 180),   # grey
}


def _polygon_stats_rgb(image_rgb: np.ndarray, polygon: np.ndarray) -> Dict[str, float]:
    """Mean RGB + HSV + texture (grayscale stdev) over the polygon interior."""
    H, W = image_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    if mask.sum() < 64:
        return {}
    pix = image_rgb[mask > 0]                            # (N, 3)
    r, g, b = pix[:, 0].mean(), pix[:, 1].mean(), pix[:, 2].mean()
    hsv = cv2.cvtColor(pix[None, ...], cv2.COLOR_RGB2HSV)[0]
    h, s, v = hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean()
    gray = (pix * np.array([0.299, 0.587, 0.114])).sum(axis=1)
    texture = float(gray.std())
    return {"r": r, "g": g, "b": b, "h": h, "s": s, "v": v, "texture": texture, "n_pix": int(mask.sum())}


def classify_plot_heuristic(image_rgb: np.ndarray, polygon: np.ndarray) -> Tuple[str, float]:
    """Rule-based land-use classification of one detected plot. Returns (class_name, confidence).

    NOT trained from labels — this is a colour/texture heuristic so the webapp can show
    *something* useful until a labelled v3 classifier exists.
    """
    f = _polygon_stats_rgb(image_rgb, polygon)
    if not f:
        return "uncategorised", 0.0
    r, g, b = f["r"], f["g"], f["b"]
    s, v, tex = f["s"], f["v"], f["texture"]

    # 1. Water: dark + bluish + low texture
    if v < 80 and b > r + 5 and tex < 20:
        return "pond / water", 0.75
    # 2. Built-up / house: bright, high texture, low saturation (roofs of mixed colour)
    if tex > 45 and v > 110:
        return "built-up / house", 0.65
    # 3. Cultivated field: clearly green dominant
    greenish = (g > r * 1.05) and (g > b * 1.10)
    if greenish:
        if tex > 28:
            return "cultivated field", 0.80
        return "vegetation/fallow", 0.65
    # 4. Bare / path: bright, low saturation, low texture
    if v > 160 and s < 50 and tex < 30:
        return "bare ground/path", 0.60
    return "uncategorised", 0.30


def classify_polygons(image_rgb: np.ndarray, polygons: List[np.ndarray]) -> List[Tuple[str, float]]:
    return [classify_plot_heuristic(image_rgb, p) for p in polygons]


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------
def draw_polygons(image_rgb: np.ndarray, polygons: List[np.ndarray],
                  color: Tuple[int, int, int] = (255, 215, 0),
                  thickness: int = 3, fill_alpha: float = 0.0) -> np.ndarray:
    """Draw all polygons with the same colour. Used for the 'no classification' view."""
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


def draw_polygons_classified(image_rgb: np.ndarray, polygons: List[np.ndarray],
                             classes: List[Tuple[str, float]],
                             thickness: int = 3, fill_alpha: float = 0.18,
                             label_classes: bool = True) -> np.ndarray:
    """Each polygon coloured by its predicted class. Optionally labels the centroid."""
    out = image_rgb.copy()
    overlay = out.copy()
    for poly, (cls, _) in zip(polygons, classes):
        col = CLASS_COLOURS.get(cls, CLASS_COLOURS["uncategorised"])
        cv2.fillPoly(overlay, [poly.astype(np.int32)], col)
    cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, dst=out)
    for poly, (cls, _) in zip(polygons, classes):
        col = CLASS_COLOURS.get(cls, CLASS_COLOURS["uncategorised"])
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=col, thickness=thickness)
    if label_classes:
        for poly, (cls, _) in zip(polygons, classes):
            M = cv2.moments(poly.astype(np.float32))
            if M["m00"] < 1: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            short = {"cultivated field": "field",
                     "vegetation/fallow": "veg",
                     "pond / water": "water",
                     "built-up / house": "house",
                     "bare ground/path": "bare",
                     "uncategorised": "?"}.get(cls, "?")
            cv2.putText(out, short, (cx - 14, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, short, (cx - 14, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def polygon_stats(polygons: List[np.ndarray], pixel_size_m: float = 0.1,
                  classes: List[Tuple[str, float]] | None = None) -> dict:
    """Summarise detected plots. If `classes` provided, also breaks down counts/area by class."""
    if not polygons:
        return {"n_plots": 0, "total_area_m2": 0.0, "mean_area_m2": 0.0,
                "median_area_m2": 0.0, "by_class": {}}
    px_areas = np.array([float(cv2.contourArea(p.astype(np.int32).reshape(-1, 1, 2))) for p in polygons])
    m2 = px_areas * (pixel_size_m ** 2)
    by_class: Dict[str, Dict[str, float]] = {}
    if classes is not None:
        for (cls, _), area in zip(classes, m2):
            d = by_class.setdefault(cls, {"count": 0, "total_m2": 0.0})
            d["count"] += 1
            d["total_m2"] += float(area)
    return {
        "n_plots":        int(len(polygons)),
        "total_area_m2":  float(m2.sum()),
        "mean_area_m2":   float(m2.mean()),
        "median_area_m2": float(np.median(m2)),
        "by_class":       by_class,
    }
