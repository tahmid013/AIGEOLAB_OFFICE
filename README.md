# AIGEOLAB — Bangladesh Parcel Boundary Extraction (v2.0)

Cadastral plot detection on 10 cm RGB orthophotos over Bangladesh (DHAMRAI upazila, AOI5). v2.0 trains a **multi-task UNet++** that predicts boundary, plot-interior and corner heatmaps jointly, then fits each detected plot as a clean **rectangle / trapezoid / quadrilateral** and tags it with a land-use class (field / pond / built-up / bare ground / fallow). Class predictions come from either an OpenAI-CLIP zero-shot head or a colour+texture heuristic — selectable in the webapp — until the labelling team adds a class column to the shapefiles (v3).

## Quick start (Colab)

Run the notebooks in order. Each opens straight from GitHub:

| Step | Notebook | Open in Colab |
|---|---|---|
| 1 — stage tiles + labels into your Drive | `01_prep_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/01_prep_colab.ipynb) |
| 2 — train multi-task UNet++ (GPU) | `02_train_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/02_train_colab.ipynb) |
| 3 — webapp to preview predictions | `03_app_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/03_app_colab.ipynb) |

### Drive prerequisite

Once per Google account, add the public Bangladesh dataset to your Drive:

1. Open https://drive.google.com/drive/folders/13ktsk51AvQd_2LYw_E8mYZSlap9eCLBs
2. Right-click → **Organize → Add shortcut → My Drive**

The notebooks expect `/content/drive/MyDrive/Bangladesh/` after `drive.mount`.

### What the webapp does (step 3)

After training, `03_app_colab.ipynb` launches a Gradio UI with:

- A dropdown listing every staged tile (with the mouzas it covers).
- A **classifier-backend radio** (CLIP zero-shot vs colour heuristic) and a **rect/trapezoid snap** toggle.
- An "Analyze" button that runs the model on a 2048 × 2048 crop, fits each detected plot as a 4-vertex polygon, and colours it by guessed land-use.
- A summary panel: plots detected, total / mean / median area in m² and hectares, per-class breakdown table, and the raw plot-interior heatmap.

The last cell prints a `https://*.gradio.live` link — open it on any device, share with non-technical reviewers.

## v2.0 architecture

- **Backbone:** `segmentation_models_pytorch.UnetPlusPlus` with an `efficientnet-b3` encoder, ImageNet-initialised, AMP-enabled, AdamW + cosine LR.
- **Three output heads** (sigmoid), supervised by valid-coverage-masked BCE + Dice (head 0–1) and masked MSE + soft Dice (head 2):
  1. **Boundary** — 3-px fence-lines along every polygon edge.
  2. **Interior** — filled plot bodies, eroded inward so they don't overlap the boundary band. Source for connected components at inference time.
  3. **Corners** — gaussian peaks at every polygon vertex. Used to snap 4-gon vertices to true junctions.
- **Polygon vectoriser** (`src/inference.extract_polygons`): connected components of the interior head → external contour → try **oriented rectangle → trapezoid (best 4-gon with one pair of parallel sides) → general convex quadrilateral** in that order, keep whichever exceeds its IoU threshold against the contour. Falls back to RDP-simplified polygon if no 4-gon fits. Corner-heatmap peaks within `corner_search_band_px` of each vertex replace it when stronger.
- **Land-use classifier** — two backends, both unsupervised (no class labels in the .dbf yet):
  - `clip` — `open_clip` ViT-B-32 scores each polygon crop against per-class text prompts (e.g. "an aerial photograph of a small rural pond"). Default.
  - `heuristic` — RGB + HSV + grayscale-stdev rules from v1.3. Dependency-free fallback.

The training notebook saves a checkpoint with the full config embedded, so the webapp picks up architecture, head count and classifier config without re-editing.

### Backward compatibility

`src/inference.load_model` and `predict_heatmaps` still accept a v1.x single-output checkpoint (classes=1) — they'll fill `heatmaps['interior']` with `1 - boundary` so the rest of the v2.0 pipeline keeps working. You don't have to retrain to try the new polygon fitter on an old model, just keep an old `best.pt` in place.

## Local prep (Windows alternative)

If your home connection is faster than Colab's Drive bandwidth, run prep on Windows directly against the Drive Desktop mount and let it sync up:

```powershell
pip install pyyaml
python src/run_prep.py --env windows
```

Writes to `G:\My Drive\aigeolab_train\`. After Drive sync completes you can skip step 1 in Colab.

## Repo layout

```
.
├── config.yaml                  # all knobs (paths, model, training, polygons, classifier)
├── notebooks/
│   ├── 01_prep_colab.ipynb      # Drive mount + .rar extract + manifest
│   ├── 02_train_colab.ipynb     # rasterise 3 heads + patch + multi-task UNet++ + eval
│   └── 03_app_colab.ipynb       # Gradio webapp wrapping the trained model
├── src/
│   ├── run_prep.py              # cross-platform CLI version of prep
│   └── inference.py             # load_model + predict_heatmaps + extract_polygons
│                                #   + classify_polygons (CLIP + heuristic)
└── labeled_data/
    └── 01.Digitizing_Upload/    # ESRI shapefiles, polygon-only, keyed by mouza
```

## Dataset

- **Imagery:** 10 000 × 10 000 px 8-bit RGB GeoTIFF at 10 cm GSD, one `.rar` per 1 km tile, packaged under AOI folders. AOI5 covers the labelled DHAMRAI mouzas.
- **LiDAR:** LAS 1.2 / LAZ format 1, ~15.75 pts/m², same 1 km grid (LiDAR_Area5 pairs with AOI5). Not used in v2.0.
- **Labels:** ESRI Shapefile polygons in Bangladesh Transverse Mercator (custom — WGS84 + TM 90°E, FE 500 000). `.dbf` carries only a `FID` field — no class column. Class predictions are therefore unsupervised (CLIP zero-shot or heuristic) until v3.

## Configuration

Edit `config.yaml`. Common knobs:

| Key | Default | Notes |
|---|---|---|
| `prep.max_mouzas` | `6` | `null` = use all 16 |
| `dataset.patch_size` | `512` | Halve if you OOM |
| `dataset.boundary_thickness_px` | `3` | Raise to 5 if boundary IoU stalls |
| `dataset.interior_erode_px` | `3` | Smaller = more boundary/interior overlap (less crisp) |
| `dataset.corner_sigma_px` | `4` | Wider gaussian = more tolerant corner-refinement |
| `model.arch` | `UnetPlusPlus` | Any `smp.*` decoder |
| `model.encoder` | `efficientnet-b3` | Drop to `efficientnet-b0` or `resnet34` if OOM |
| `train.batch_size` | `12` | L4: try 20; A100: 32+ |
| `train.loss.boundary_weight` / `interior_weight` / `corners_weight` | `0.45 / 0.40 / 0.15` | Per-head weighting |
| `polygons.interior_threshold` | `0.5` | Lower = more detections |
| `polygons.rect_iou_threshold` | `0.86` | Higher = stricter rectangle fit |
| `polygons.trapezoid_iou_threshold` | `0.82` | Fallback below `rect_iou_threshold` |
| `polygons.quad_iou_threshold` | `0.78` | Fallback below `trapezoid_iou_threshold` |
| `classifier.default_backend` | `clip` | `"clip"` or `"heuristic"` |

## v1 → v2 → v3 roadmap

- **v1.0–v1.3 (shipped, archived):** single-output U-Net (ResNet-34) boundary heatmap; v1.2 added valid-coverage masking; v1.3 added the heuristic land-use classifier and rect-snap toggle.
- **v2.0 (this):** multi-task UNet++ (boundary + interior + corners) with corner-guided rectangle / trapezoid / quadrilateral fitting and a swappable CLIP zero-shot vs heuristic classifier in the webapp.
- **v2.1 (planned):** concatenate a LiDAR-derived nDSM as a 4th input channel (recovers building / pond elevation contrast).
- **v3 (planned):** trained per-polygon classifier head — requires the labelling team to add a class column to the shapefile `.dbf`.
