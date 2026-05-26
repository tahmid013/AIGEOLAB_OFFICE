# AIGEOLAB — Bangladesh Parcel Boundary Extraction

Cadastral plot-boundary segmentation on 10 cm RGB orthophotos over Bangladesh (DHAMRAI upazila, AOI5). v1 is a single-modality U-Net baseline; LiDAR fusion (v2) and per-parcel land-use classification — pond / house / field — (v3) come next.

## Quick start (Colab)

Run the notebooks in order. Each opens straight from GitHub:

| Step | Notebook | Open in Colab |
|---|---|---|
| 1 — stage tiles + labels into your Drive | `01_prep_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/01_prep_colab.ipynb) |
| 2 — train U-Net (GPU) | `02_train_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/02_train_colab.ipynb) |
| 3 — webapp to preview predictions | `03_app_colab.ipynb` | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tahmid013/AIGEOLAB_OFFICE/blob/main/notebooks/03_app_colab.ipynb) |

### Drive prerequisite

Once per Google account, add the public Bangladesh dataset to your Drive:

1. Open https://drive.google.com/drive/folders/13ktsk51AvQd_2LYw_E8mYZSlap9eCLBs
2. Right-click → **Organize → Add shortcut → My Drive**

The notebooks expect `/content/drive/MyDrive/Bangladesh/` after `drive.mount`.

### What the webapp does (step 3)

After training, `03_app_colab.ipynb` launches a small Gradio UI with:

- A dropdown listing every staged tile (with the mouzas it covers).
- An "Analyze" button that runs the model on a 2048 × 2048 crop and traces the detected plot boundaries in yellow on the aerial photo.
- A summary panel: number of plots detected, total / mean / median area in m² and hectares.
- Sliders to nudge crop position, boundary confidence threshold, and minimum plot area.

The last cell prints a `https://*.gradio.live` link — open it on any device, share with non-technical reviewers.

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
├── config.yaml                  # all knobs (paths, model, training)
├── notebooks/
│   ├── 01_prep_colab.ipynb      # Drive mount + .rar extract + manifest
│   ├── 02_train_colab.ipynb     # rasterise + patch + U-Net train + eval
│   └── 03_app_colab.ipynb       # Gradio webapp wrapping the trained model
├── src/
│   ├── run_prep.py              # cross-platform CLI version of prep
│   └── inference.py             # load_model + predict_heatmap + extract_polygons (used by the app)
└── labeled_data/
    └── 01.Digitizing_Upload/    # ESRI shapefiles, polygon-only, keyed by mouza
```

## Dataset

- **Imagery:** 10 000 × 10 000 px 8-bit RGB GeoTIFF at 10 cm GSD, one `.rar` per 1 km tile, packaged under AOI folders. AOI5 covers the labelled DHAMRAI mouzas.
- **LiDAR:** LAS 1.2 / LAZ format 1, ~15.75 pts/m², same 1 km grid (LiDAR_Area5 pairs with AOI5). Not used in v1.
- **Labels:** ESRI Shapefile polygons in Bangladesh Transverse Mercator (custom — WGS84 + TM 90°E, FE 500 000). `.dbf` carries only a `FID` field — no class column. v1 is therefore binary boundary segmentation only.

## Configuration

Edit `config.yaml`. Common knobs:

| Key | Default | Notes |
|---|---|---|
| `prep.max_mouzas` | `6` | `null` = use all 16 |
| `dataset.patch_size` | `512` | Halve if you OOM |
| `dataset.boundary_thickness_px` | `3` | Raise to 5 if val IoU stalls below 0.4 |
| `train.epochs` | `50` | Drop to 5-10 for fast iteration |
| `train.batch_size` | `16` | L4: try 32 |

## v1 → v2 → v3 roadmap

- **v1 (this repo):** RGB U-Net → boundary heatmap, vectorised to polygons in the webapp.
- **v1.2 (current):** loss + val IoU masked to annotated mouza coverage, so the metric tracks real performance instead of being dragged down by predictions outside the sparse GT.
- **v2:** concatenate a LiDAR-derived nDSM as a 4th channel; same U-Net.
- **v3:** per-polygon classifier head (pond / house / field) — needs a class column added to the labelling pass first.
