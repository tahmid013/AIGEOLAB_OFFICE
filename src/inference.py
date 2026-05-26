"""AIGEOLAB v2.0 inference helpers.

Multi-task model (boundary + interior + corner heatmap) with corner-guided
rect / trapezoid / quadrilateral fitting and a swappable land-use classifier
(CLIP zero-shot OR colour+texture heuristic).

Used by notebooks/03_app_colab.ipynb. Importable so the notebook stays UI-only.

Backward compatible with v1.x single-output (classes=1) checkpoints: in that
mode `predict_heatmaps` returns a dict whose 'interior' is the inverted
boundary prediction and 'corners' is None, and `extract_polygons` falls back
to the v1 inverted-boundary connected-component path.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# 1. Model loading
# =============================================================================
def load_model(ckpt_path: str | Path, device: str = "cuda"):
    """Load a checkpoint saved by `02_train_colab.ipynb`.

    Returns (model, cfg_dict, meta_dict). meta has keys 'epoch', 'val_iou',
    'classes' (number of output channels — 1 = v1.x boundary-only, 3 = v2.0).
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
    meta = {
        "epoch":   ckpt.get("epoch"),
        "val_iou": ckpt.get("val_iou"),
        "classes": int(M["classes"]),
    }
    return model, cfg, meta


# =============================================================================
# 2. Inference (sliding window over an arbitrary RGB crop)
# =============================================================================
_NORMALISE = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                                    std =(0.229, 0.224, 0.225)),
                         ToTensorV2()])


def predict_heatmaps(model, image_rgb_uint8: np.ndarray, patch: int, stride: int,
                     device: str = "cuda") -> Dict[str, np.ndarray]:
    """Sliding-window inference. Returns a dict with float32 [0,1] arrays.

    Keys are always present so callers don't need to branch:
      - 'boundary' : fence-line probability
      - 'interior' : plot-body probability  (= 1 - boundary on v1.x checkpoints)
      - 'corners'  : corner/junction probability  (zeros on v1.x checkpoints)
    """
    H, W = image_rgb_uint8.shape[:2]
    if H < patch or W < patch:
        raise ValueError(f"Image too small: {image_rgb_uint8.shape} for patch {patch}.")
    n_out = int(model.segmentation_head[0].out_channels)  # smp head conv
    pred = np.zeros((n_out, H, W), dtype=np.float32)
    cnt  = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for top in range(0, H - patch + 1, stride):
            for left in range(0, W - patch + 1, stride):
                sub = image_rgb_uint8[top:top + patch, left:left + patch]
                x = _NORMALISE(image=sub)["image"].unsqueeze(0).to(device)
                p = torch.sigmoid(model(x))[0].cpu().numpy()       # (C, h, w)
                pred[:, top:top + patch, left:left + patch] += p
                cnt[top:top + patch, left:left + patch] += 1
    pred = pred / np.maximum(cnt, 1)
    if n_out >= 3:
        return {"boundary": pred[0], "interior": pred[1], "corners": pred[2]}
    # v1.x compatibility — only boundary trained, derive a pseudo-interior.
    boundary = pred[0]
    return {
        "boundary": boundary,
        "interior": 1.0 - boundary,
        "corners":  np.zeros_like(boundary),
    }


# v1.x compatibility shim — old callers expect a single heatmap array.
def predict_heatmap(model, image_rgb_uint8: np.ndarray, patch: int, stride: int,
                    device: str = "cuda") -> np.ndarray:
    """Returns the boundary heatmap only. Prefer predict_heatmaps in new code."""
    return predict_heatmaps(model, image_rgb_uint8, patch, stride, device)["boundary"]


# =============================================================================
# 3. Polygon vectorisation: rect / trapezoid / quad fitting with corner refinement
# =============================================================================
def _polygon_area(poly: np.ndarray) -> float:
    return float(cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2)))


def _iou_with_contour(quad: np.ndarray, contour: np.ndarray, hw: Tuple[int, int]) -> float:
    """Rasterised IoU between a candidate 4-gon and the source contour."""
    H, W = hw
    mask_a = np.zeros((H, W), dtype=np.uint8)
    mask_b = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_a, [quad.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [contour.astype(np.int32)], 1)
    inter = int(((mask_a > 0) & (mask_b > 0)).sum())
    union = int(((mask_a > 0) | (mask_b > 0)).sum())
    return inter / max(1, union)


def _min_area_rect(contour: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(contour.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)            # (4, 2)


def _convex_hull_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    """Reduce the convex hull to its 4 most influential vertices by iteratively
    removing the vertex whose removal loses the least area. Returns a (4,2) array
    or None if the hull has < 4 vertices.
    """
    hull = cv2.convexHull(contour.astype(np.int32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 4:
        return None
    while len(hull) > 4:
        n = len(hull)
        full_area = _polygon_area(hull.astype(np.float32))
        best_i, best_loss = 0, float("inf")
        for i in range(n):
            trial = np.delete(hull, i, axis=0)
            loss = full_area - _polygon_area(trial.astype(np.float32))
            if loss < best_loss:
                best_loss, best_i = loss, i
        hull = np.delete(hull, best_i, axis=0)
    return hull.astype(np.float32)


def _is_trapezoid_like(quad: np.ndarray, parallel_cos_tol: float = 0.08) -> bool:
    """A 4-gon is trapezoid-like if at least one pair of opposite edges is
    near-parallel (|cos(theta) - 1| < tol after normalising direction)."""
    if quad is None or len(quad) != 4:
        return False
    e = []
    for i in range(4):
        v = quad[(i + 1) % 4] - quad[i]
        n = np.linalg.norm(v)
        if n < 1e-6: return False
        e.append(v / n)
    c_02 = abs(abs(float(np.dot(e[0], e[2]))) - 1.0)
    c_13 = abs(abs(float(np.dot(e[1], e[3]))) - 1.0)
    return c_02 < parallel_cos_tol or c_13 < parallel_cos_tol


def _refine_with_corners(quad: np.ndarray, corner_prob: np.ndarray,
                         band_px: int = 12, corner_threshold: float = 0.30) -> np.ndarray:
    """Snap each vertex of `quad` to the brightest corner-heatmap peak within
    `band_px`. Skips if no peak exceeds `corner_threshold`."""
    if corner_prob is None or corner_prob.size == 0 or corner_prob.max() < corner_threshold:
        return quad
    H, W = corner_prob.shape
    out = quad.copy()
    for i, (x, y) in enumerate(quad):
        x0 = int(max(0, x - band_px)); x1 = int(min(W, x + band_px + 1))
        y0 = int(max(0, y - band_px)); y1 = int(min(H, y + band_px + 1))
        if x1 <= x0 or y1 <= y0: continue
        roi = corner_prob[y0:y1, x0:x1]
        if roi.max() < corner_threshold: continue
        ay, ax = np.unravel_index(int(roi.argmax()), roi.shape)
        out[i] = np.array([x0 + ax, y0 + ay], dtype=np.float32)
    return out


def _fit_best_quad(contour: np.ndarray, hw: Tuple[int, int],
                   rect_iou_thr: float, trap_iou_thr: float, quad_iou_thr: float
                   ) -> Tuple[Optional[np.ndarray], str, float]:
    """Try rect → trapezoid → general quad. Return (quad, kind, iou).

    `kind` is one of {'rect', 'trapezoid', 'quad', 'none'}. `quad` is float32 (4,2).
    """
    rect = _min_area_rect(contour)
    rect_iou = _iou_with_contour(rect, contour, hw)
    if rect_iou >= rect_iou_thr:
        return rect, "rect", rect_iou

    hull_quad = _convex_hull_to_quad(contour)
    if hull_quad is not None:
        hq_iou = _iou_with_contour(hull_quad, contour, hw)
        if hq_iou >= trap_iou_thr and _is_trapezoid_like(hull_quad):
            return hull_quad, "trapezoid", hq_iou
        if hq_iou >= quad_iou_thr:
            return hull_quad, "quad", hq_iou
        # Best of {rect, hull_quad} as last resort
        if hq_iou > rect_iou:
            return hull_quad, "quad", hq_iou
    return rect, "rect", rect_iou


def extract_polygons(prob_or_heatmaps, threshold: float = 0.5,
                     close_kernel: int = 3, min_area_px: int = 400,
                     approx_eps_frac: float = 0.015,
                     regularise_to_rect: bool = True,
                     rect_iou_threshold: float = 0.86,
                     trapezoid_iou_threshold: float = 0.82,
                     quad_iou_threshold: float = 0.78,
                     corners_prob: Optional[np.ndarray] = None,
                     corner_threshold: float = 0.30,
                     corner_search_band_px: int = 12,
                     source: str = "interior",
                     ) -> List[np.ndarray]:
    """Convert model output into a list of polygon contours.

    Accepts EITHER:
      - dict from predict_heatmaps, with 'boundary' / 'interior' / 'corners' keys
      - a single np.ndarray (treated as a boundary or interior heatmap depending
        on `source`; default 'interior' assumes the array is interior-style)

    Pipeline (v2.0):
      1. Pick the foreground heatmap (interior preferred — survives broken edges).
      2. Threshold + light morphology.
      3. Connected components, drop tiny blobs.
      4. For each blob extract external contour.
      5. If regularise_to_rect: try rect → trapezoid → quad fits and pick the
         best one above its threshold. Refine vertices with the corner heatmap.
      6. Otherwise: return RDP-simplified general polygon.

    Returns a list of (N,2) int32 arrays, one polygon per detection.
    """
    # ---- Step 1: pick the source heatmap ---------------------------------
    if isinstance(prob_or_heatmaps, dict):
        if source == "interior":
            fg = prob_or_heatmaps.get("interior")
            if fg is None and "boundary" in prob_or_heatmaps:
                fg = 1.0 - prob_or_heatmaps["boundary"]
        else:
            b = prob_or_heatmaps.get("boundary")
            fg = 1.0 - b if b is not None else prob_or_heatmaps.get("interior")
        if corners_prob is None:
            corners_prob = prob_or_heatmaps.get("corners")
    else:
        # plain array
        if source == "interior":
            fg = prob_or_heatmaps
        else:
            fg = 1.0 - prob_or_heatmaps

    if fg is None:
        return []

    # ---- Step 2: binarise + close small gaps -----------------------------
    binmask = (fg > threshold).astype(np.uint8)
    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, k)

    # ---- Step 3: connected components ------------------------------------
    n_lbl, labels = cv2.connectedComponents(binmask, connectivity=4)
    hw = binmask.shape

    polygons: List[np.ndarray] = []
    for lbl in range(1, n_lbl):
        comp = (labels == lbl).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area_px:
            continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(cnt) < 3: continue

        if regularise_to_rect:
            quad, _kind, _iou = _fit_best_quad(
                cnt, hw,
                rect_iou_thr=rect_iou_threshold,
                trap_iou_thr=trapezoid_iou_threshold,
                quad_iou_thr=quad_iou_threshold,
            )
            if quad is None:
                eps = approx_eps_frac * cv2.arcLength(cnt.reshape(-1, 1, 2), True)
                quad = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float32)
            if corners_prob is not None and len(quad) == 4:
                quad = _refine_with_corners(quad, corners_prob,
                                            band_px=corner_search_band_px,
                                            corner_threshold=corner_threshold)
            polygons.append(quad.astype(np.int32))
        else:
            eps = approx_eps_frac * cv2.arcLength(cnt.reshape(-1, 1, 2), True)
            approx = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            if len(approx) >= 3:
                polygons.append(approx.astype(np.int32))
    return polygons


# =============================================================================
# 4. Land-use classifier — two backends, both return List[(class_name, conf)]
# =============================================================================
# v1.3 palette kept as a module constant so the webapp can render the legend
# without touching the config. Override-able by loading from cfg['classifier']['classes'].
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "cultivated field":  (124, 191,  60),
    "vegetation/fallow": (160, 200, 100),
    "pond / water":      ( 70, 145, 220),
    "built-up / house":  (240, 100,  80),
    "bare ground/path":  (220, 195, 130),
    "uncategorised":     (180, 180, 180),
}


# -------- 4a. heuristic backend (v1.3, lightly polished) ---------------------
def _polygon_features(image_rgb: np.ndarray, polygon: np.ndarray) -> Dict[str, float]:
    H, W = image_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    n_pix = int(mask.sum())
    if n_pix < 64:
        return {}
    pix = image_rgb[mask > 0].astype(np.float32)
    r, g, b = pix[:, 0].mean(), pix[:, 1].mean(), pix[:, 2].mean()
    hsv = cv2.cvtColor(pix[None, ...].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    h, s, v = hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean()
    gray = (pix * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=1)
    texture = float(gray.std())
    return {"r": float(r), "g": float(g), "b": float(b),
            "h": float(h), "s": float(s), "v": float(v),
            "texture": texture, "n_pix": n_pix}


def classify_plot_heuristic(image_rgb: np.ndarray, polygon: np.ndarray) -> Tuple[str, float]:
    f = _polygon_features(image_rgb, polygon)
    if not f: return "uncategorised", 0.0
    r, g, b, s, v, tex = f["r"], f["g"], f["b"], f["s"], f["v"], f["texture"]
    if v < 80 and b > r + 5 and tex < 20:
        return "pond / water", 0.75
    if tex > 45 and v > 110:
        return "built-up / house", 0.65
    if (g > r * 1.05) and (g > b * 1.10):
        return ("cultivated field", 0.80) if tex > 28 else ("vegetation/fallow", 0.65)
    if v > 160 and s < 50 and tex < 30:
        return "bare ground/path", 0.60
    return "uncategorised", 0.30


# -------- 4b. CLIP backend (zero-shot) ---------------------------------------
_CLIP_CACHE: Dict[str, object] = {}


def _load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
    """Lazy-load open_clip (preferred) or fall back to openai-clip if installed.
    Cached so repeated webapp clicks don't reload the model.
    """
    key = f"{model_name}|{pretrained}|{device}"
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        _CLIP_CACHE[key] = ("open_clip", model, preprocess, tokenizer, device)
        return _CLIP_CACHE[key]
    except ImportError:
        pass
    try:
        import clip as openai_clip
        model, preprocess = openai_clip.load(model_name, device=device, jit=False)
        _CLIP_CACHE[key] = ("openai_clip", model, preprocess, openai_clip.tokenize, device)
        return _CLIP_CACHE[key]
    except ImportError:
        raise RuntimeError(
            "CLIP backend requested but neither `open_clip` nor `clip` is installed. "
            "Install with `pip install open_clip_torch`."
        )


def _crop_polygon(image_rgb: np.ndarray, polygon: np.ndarray, pad_frac: float = 0.05,
                  min_px: int = 64) -> Optional[np.ndarray]:
    """Tight bounding-box crop around `polygon` with `pad_frac` margin. Returns
    None when the resulting crop would be smaller than `min_px` on either side
    or when the polygon falls entirely outside the image."""
    H, W = image_rgb.shape[:2]
    pts = polygon.reshape(-1, 2)
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    px = int(w * pad_frac); py = int(h * pad_frac)
    x0 = max(0, int(x0) - px); y0 = max(0, int(y0) - py)
    x1 = min(W, int(x1) + px); y1 = min(H, int(y1) + py)
    if (x1 - x0) < min_px or (y1 - y0) < min_px: return None
    return image_rgb[y0:y1, x0:x1].copy()


def classify_polygons_clip(image_rgb: np.ndarray, polygons: List[np.ndarray],
                           cfg_classifier: dict, device: str = "cuda"
                           ) -> List[Tuple[str, float]]:
    """Zero-shot CLIP classification over each polygon crop.

    Returns the same shape as `classify_polygons_heuristic`. Polygons too small
    to crop are returned as ('uncategorised', 0.0).
    """
    classes: List[dict] = cfg_classifier["classes"]
    # Build named text features. Drop classes with empty prompts (e.g. 'uncategorised').
    named_prompts = [(c["name"], c["clip_prompts"]) for c in classes if c.get("clip_prompts")]
    clip_cfg = cfg_classifier["clip"]

    backend, model, preprocess, tok, dev = _load_clip(
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        device=device,
    )
    with torch.no_grad():
        # Class text features: mean across each class's prompt set.
        text_feats: List[torch.Tensor] = []
        for _, prompts in named_prompts:
            tokens = tok(prompts).to(dev)        # open_clip + openai-clip tokenizers share this signature
            f = model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            text_feats.append(f.mean(dim=0, keepdim=True))
        text_mat = torch.cat(text_feats, dim=0)
        text_mat = text_mat / text_mat.norm(dim=-1, keepdim=True)

        # Per-polygon image features. Batch where possible.
        from PIL import Image
        results: List[Tuple[str, float]] = []
        pad = float(clip_cfg.get("crop_padding_frac", 0.05))
        min_px = int(clip_cfg.get("min_crop_px", 64))
        for poly in polygons:
            crop = _crop_polygon(image_rgb, poly, pad_frac=pad, min_px=min_px)
            if crop is None:
                results.append(("uncategorised", 0.0)); continue
            x = preprocess(Image.fromarray(crop)).unsqueeze(0).to(dev)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (f @ text_mat.T).squeeze(0) * 100.0
            probs = logits.softmax(dim=-1).cpu().numpy()
            i = int(probs.argmax())
            results.append((named_prompts[i][0], float(probs[i])))
        return results


# -------- 4c. unified entry point -------------------------------------------
def classify_polygons(image_rgb: np.ndarray, polygons: List[np.ndarray],
                      backend: str = "heuristic",
                      cfg_classifier: Optional[dict] = None,
                      device: str = "cuda") -> List[Tuple[str, float]]:
    """Dispatch to the requested backend. Falls back to heuristic if CLIP
    isn't installed so the webapp doesn't crash on a missing dep."""
    if not polygons:
        return []
    if backend == "clip":
        if cfg_classifier is None:
            raise ValueError("classify_polygons(backend='clip') requires cfg_classifier (from config.yaml).")
        try:
            return classify_polygons_clip(image_rgb, polygons, cfg_classifier, device=device)
        except RuntimeError as e:
            print(f"[classify_polygons] CLIP unavailable ({e}); falling back to heuristic.")
            backend = "heuristic"
    return [classify_plot_heuristic(image_rgb, p) for p in polygons]


# =============================================================================
# 5. Drawing helpers
# =============================================================================
def draw_polygons(image_rgb: np.ndarray, polygons: List[np.ndarray],
                  color: Tuple[int, int, int] = (255, 215, 0),
                  thickness: int = 3, fill_alpha: float = 0.0) -> np.ndarray:
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
                             label_classes: bool = True,
                             class_colours: Optional[Dict[str, Tuple[int, int, int]]] = None,
                             ) -> np.ndarray:
    if class_colours is None: class_colours = CLASS_COLOURS
    out = image_rgb.copy()
    overlay = out.copy()
    for poly, (cls, _) in zip(polygons, classes):
        col = class_colours.get(cls, class_colours.get("uncategorised", (180, 180, 180)))
        cv2.fillPoly(overlay, [poly.astype(np.int32)], col)
    cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, dst=out)
    for poly, (cls, _) in zip(polygons, classes):
        col = class_colours.get(cls, class_colours.get("uncategorised", (180, 180, 180)))
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=col, thickness=thickness)
    if label_classes:
        short_map = {"cultivated field": "field", "vegetation/fallow": "veg",
                     "pond / water": "water", "built-up / house": "house",
                     "bare ground/path": "bare", "uncategorised": "?"}
        for poly, (cls, _) in zip(polygons, classes):
            M = cv2.moments(poly.astype(np.float32))
            if M["m00"] < 1: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            short = short_map.get(cls, "?")
            cv2.putText(out, short, (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),     3, cv2.LINE_AA)
            cv2.putText(out, short, (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def polygon_stats(polygons: List[np.ndarray], pixel_size_m: float = 0.1,
                  classes: Optional[List[Tuple[str, float]]] = None) -> dict:
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
