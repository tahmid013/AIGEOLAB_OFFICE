"""Cross-platform runner for the prep step — same logic as notebooks/01_prep_colab.ipynb.

Usage:
  python src/run_prep.py                # auto-detects platform: 'colab' on Linux, 'windows' on Windows
  python src/run_prep.py --env colab    # force one
"""
from __future__ import annotations
import sys, os, struct, glob, shutil, subprocess, csv, time, argparse
from pathlib import Path
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["colab", "windows"], default=None,
                    help="Override config.yaml's env. Auto-detected if omitted.")
    return ap.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parent.parent
    CFG = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))

    env = args.env or ("windows" if os.name == "nt" else "colab")
    P = CFG["paths"][env]
    print(f"env          : {env}")
    for k, v in P.items():
        print(f"  {k:18s} = {v}")
    print(f"AOI          : {CFG['aoi_imagery']}")
    print(f"batches      : {CFG['prep']['label_batches']}")
    print(f"max_mouzas   : {CFG['prep']['max_mouzas']}\n")

    def normalise_xy(stem):
        if "-" not in stem: return None
        a, b = stem.split("-", 1)
        try: x, y = int(a), int(b)
        except ValueError: return None
        if x >= 100_000:  x //= 1000
        if y >= 1_000_000: y //= 1000
        return (x, y)

    img_dir = Path(P["bangladesh_root"]) / CFG["aoi_imagery"]
    assert img_dir.is_dir(), f"AOI directory not found: {img_dir}"
    tile_index = {normalise_xy(p.stem): p.name for p in img_dir.glob("*.rar") if normalise_xy(p.stem)}
    print(f"Indexed {len(tile_index)} imagery tiles in {img_dir.name}")

    def read_shp_bbox(p):
        with open(p, "rb") as f: h = f.read(100)
        return struct.unpack("<4d", h[36:68])

    def tiles_for_bbox(xmin, ymin, xmax, ymax):
        return [(x, y)
                for x in range(int(xmin // 1000), int(xmax // 1000) + 1)
                for y in range(int(ymin // 1000), int(ymax // 1000) + 1)]

    labels_root = Path(P["labels_root"])
    mouzas = []
    for batch in CFG["prep"]["label_batches"]:
        bd = labels_root / batch
        if not bd.is_dir(): continue
        for shp in sorted(bd.glob("*.shp")):
            xmin, ymin, xmax, ymax = read_shp_bbox(shp)
            tiles = tiles_for_bbox(xmin, ymin, xmax, ymax)
            in_aoi = [t for t in tiles if t in tile_index]
            missing = [t for t in tiles if t not in tile_index]
            mouzas.append({"batch": batch, "shp": shp, "name": shp.stem,
                           "tiles_in_aoi": in_aoi, "tiles_missing": missing})

    if CFG["prep"]["drop_partial"]:
        before = len(mouzas)
        mouzas = [m for m in mouzas if m["tiles_in_aoi"] and not m["tiles_missing"]]
        print(f"Dropped {before - len(mouzas)} partial mouzas")
    if CFG["prep"]["max_mouzas"]:
        mouzas = mouzas[: CFG["prep"]["max_mouzas"]]
        print(f"Capped to first {len(mouzas)} mouzas")

    needed = sorted({t for m in mouzas for t in m["tiles_in_aoi"]})
    print(f"\nKeeping {len(mouzas)} mouzas covering {len(needed)} unique tiles")
    print(f"  ~{len(needed) * 0.3:.2f} GB of raw .tif will be staged\n")

    staging = Path(P["staging_root"])
    tiles_dir = staging / "tiles";  tiles_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = staging / "labels"; labels_dir.mkdir(parents=True, exist_ok=True)

    unrar = P["unrar_bin"]
    assert Path(unrar).exists() if os.path.isabs(unrar) else shutil.which(unrar), f"unrar binary not found: {unrar}"

    def force_cache(path):
        with open(path, "rb") as f:
            while f.read(16 * 1024 * 1024): pass

    print(f"Extracting {len(needed)} tiles to {tiles_dir} ...")
    extracted = skipped = 0
    failed = []
    t0 = time.time()
    for i, (x, y) in enumerate(needed, 1):
        rar_name = tile_index[(x, y)]
        archive = img_dir / rar_name
        stem = rar_name[:-4]
        tif_dst = tiles_dir / f"{stem}.tif"
        if tif_dst.exists() and tif_dst.stat().st_size > 0:
            skipped += 1
            print(f"  [{i}/{len(needed)}] {stem}  SKIP (present)"); continue
        tstart = time.time()
        try: force_cache(archive)
        except Exception as e:
            failed.append((rar_name, -1, f"cache: {e}"))
            print(f"  [{i}/{len(needed)}] {stem}  FAIL cache {e}"); continue
        cmd = [unrar, "e", "-y", "-o+", "-inul", str(archive), str(tiles_dir) + os.sep]
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - tstart
        if r.returncode != 0 or not tif_dst.exists():
            failed.append((rar_name, r.returncode, r.stderr[:200]))
            print(f"  [{i}/{len(needed)}] {stem}  FAIL rc={r.returncode}  {r.stderr[:80]}")
            continue
        extracted += 1
        size_mb = tif_dst.stat().st_size / 1e6
        print(f"  [{i}/{len(needed)}] {stem}  ok  {size_mb:.0f} MB  {dt:.1f}s")

    print(f"\nExtraction done in {time.time()-t0:.0f}s. extracted={extracted} skipped={skipped} failed={len(failed)}")
    for fn, rc, err in failed[:5]:
        print(f"  FAIL {fn}: rc={rc} {err}")

    SIDECARS = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd", ".sbn", ".sbx"]
    for m in mouzas:
        src_stem = m["shp"].with_suffix("")
        dst_base = f"{m['batch']}__{m['name']}"
        for ext in SIDECARS:
            src = src_stem.with_suffix(ext)
            if src.exists(): shutil.copy2(src, labels_dir / f"{dst_base}{ext}")
    print(f"\nCopied label sidecars for {len(mouzas)} mouzas")

    tile_to_mouzas = {}
    for m in mouzas:
        for t in m["tiles_in_aoi"]: tile_to_mouzas.setdefault(t, []).append(m)

    manifest_path = staging / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_x_km", "tile_y_km", "tile_tif_relpath", "mouza_count", "mouza_label_stems"])
        for (x, y), ms in sorted(tile_to_mouzas.items()):
            stem = tile_index[(x, y)][:-4]
            w.writerow([x, y, f"tiles/{stem}.tif", len(ms),
                        "|".join(f"{mm['batch']}__{mm['name']}" for mm in ms)])
    print(f"manifest -> {manifest_path}  ({len(tile_to_mouzas)} tiles)")

    tifs = sorted(tiles_dir.glob("*.tif"))
    shps = sorted(labels_dir.glob("*.shp"))
    gb = sum(p.stat().st_size for p in tifs) / 1e9
    print(f"\nSTAGING  n_tif={len(tifs)}  size={gb:.2f} GB  n_shp={len(shps)}  manifest={manifest_path}")


if __name__ == "__main__":
    main()
