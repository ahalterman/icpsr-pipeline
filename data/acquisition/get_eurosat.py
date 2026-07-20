"""
get_eurosat.py -- Download EuroSAT (RGB) and cache a small fixed sample.

Lab 4: CPU-only image classification on satellite land-cover patches.

Data source
-----------
EuroSAT: 27,000 labeled 64x64 Sentinel-2 RGB patches, 10 land-cover classes.
We download the RGB zip (~90 MB) from the official Zenodo deposit:

    https://zenodo.org/records/7711810  (EuroSAT_RGB.zip)

License: MIT. Imagery: Copernicus Sentinel data 2015-2018.
Citation: Helber, Bischke, Dengel, Borth (2019), "EuroSAT: A Novel Dataset
and Deep Learning Benchmark for Land Use and Land Cover Classification,"
IEEE JSTARS 12(7). https://github.com/phelber/EuroSAT

What this script does (re-runnable; same seed -> same sample)
-------------------------------------------------------------
1. Download EuroSAT_RGB.zip to /tmp (skipped if already present).
2. For each of the 6 classes used in the lab, sample 120 images with a
   seeded RNG and copy them to data/cached/eurosat_sample/<Class>/.
3. Write data/cached/eurosat_index.csv with a stratified 80/20
   train/test split (filepath, class, split), seed 42.
4. Embed all 720 images with a pretrained ViT on CPU and save
   data/cached/eurosat_embeddings.npz (fallback so the classifier-head
   exercise works even if students' model download fails).

Run from anywhere:  python data/acquisition/get_eurosat.py
"""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ZIP_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
ZIP_PATH = Path("/tmp/EuroSAT_RGB.zip")

# 6 of EuroSAT's 10 classes, chosen for the course's conflict/landscape
# narrative (crops, forests, built-up areas, rivers, grazing land).
CLASSES = ["AnnualCrop", "Forest", "Industrial", "Residential", "River", "Pasture"]
N_PER_CLASS = 120          # 6 * 120 = 720 images total (~2 MB of jpgs)
SEED = 42                  # controls BOTH the sample and the train/test split
TEST_FRAC = 0.20           # stratified 80/20 split within each class

# Embedding model for the precomputed-features fallback. Any small vision
# encoder works; we record the exact id in data/cached/README.md.
EMBED_MODEL = "google/vit-base-patch16-224"

REPO_ROOT = Path(__file__).resolve().parents[2]   # .../ICPSR_OSINT
CACHE_DIR = REPO_ROOT / "data" / "cached"
SAMPLE_DIR = CACHE_DIR / "eurosat_sample"
INDEX_CSV = CACHE_DIR / "eurosat_index.csv"
EMBED_NPZ = CACHE_DIR / "eurosat_embeddings.npz"


# ---------------------------------------------------------------------------
# Step 1: download the zip (only if not already in /tmp)
# ---------------------------------------------------------------------------
def download_zip() -> None:
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 50_000_000:
        print(f"Using existing {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")
        return
    print(f"Downloading {ZIP_URL} -> {ZIP_PATH} (~90 MB)...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("done.")


# ---------------------------------------------------------------------------
# Step 2 + 3: sample images, copy into the cache, write the index CSV
# ---------------------------------------------------------------------------
def build_sample() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    if SAMPLE_DIR.exists():
        shutil.rmtree(SAMPLE_DIR)   # re-runnable: rebuild from scratch

    rows = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        for cls in CLASSES:
            # Files inside the zip look like "EuroSAT_RGB/Forest/Forest_123.jpg"
            members = sorted(n for n in names
                             if f"/{cls}/" in n and n.lower().endswith(".jpg"))
            if len(members) < N_PER_CLASS:
                sys.exit(f"ERROR: only {len(members)} files found for {cls}")
            # Seeded sample of N_PER_CLASS images (sorted first => deterministic)
            chosen = rng.choice(len(members), size=N_PER_CLASS, replace=False)
            out_dir = SAMPLE_DIR / cls
            out_dir.mkdir(parents=True)
            for i in sorted(chosen):
                member = members[i]
                fname = Path(member).name
                with zf.open(member) as src, open(out_dir / fname, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                rows.append({
                    "filepath": f"data/cached/eurosat_sample/{cls}/{fname}",
                    "class": cls,
                })

    df = pd.DataFrame(rows)
    # Stratified 80/20 split: shuffle within each class with the seeded RNG,
    # mark the first 20% "test" and the rest "train".
    df["split"] = "train"
    for cls in CLASSES:
        idx = df.index[df["class"] == cls].to_numpy()
        shuffled = rng.permutation(idx)
        n_test = int(round(TEST_FRAC * len(idx)))
        df.loc[shuffled[:n_test], "split"] = "test"
    df = df.sort_values(["class", "filepath"]).reset_index(drop=True)
    df.to_csv(INDEX_CSV, index=False)
    print(f"Wrote {len(df)} rows to {INDEX_CSV}")
    print(df.groupby(["class", "split"]).size().unstack())
    return df


# ---------------------------------------------------------------------------
# Step 4: precompute ViT embeddings (CPU) as a fallback for the lab
# ---------------------------------------------------------------------------
def embed_images(df: pd.DataFrame) -> None:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    print(f"Loading {EMBED_MODEL} (CPU)...")
    processor = AutoImageProcessor.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)
    model.eval()

    feats = []
    batch_size = 32
    paths = [REPO_ROOT / p for p in df["filepath"]]
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt")  # resizes to 224
            out = model(**inputs)
            cls_tok = out.last_hidden_state[:, 0, :]   # CLS token
            feats.append(cls_tok.numpy().astype(np.float32))
            print(f"  embedded {start + len(batch)}/{len(paths)}", end="\r")
    embeddings = np.concatenate(feats, axis=0)
    print(f"\nEmbeddings: {embeddings.shape} ({embeddings.dtype})")

    # row_index[i] == i: row i of `embeddings` is row i of eurosat_index.csv
    np.savez_compressed(
        EMBED_NPZ,
        embeddings=embeddings,
        row_index=np.arange(len(df), dtype=np.int32),
        # str arrays (not object dtype) so the file loads with allow_pickle=False
        filepath=df["filepath"].to_numpy(dtype=str),
        label=df["class"].to_numpy(dtype=str),
        split=df["split"].to_numpy(dtype=str),
        model=np.array(EMBED_MODEL, dtype=str),
    )
    print(f"Wrote {EMBED_NPZ} ({EMBED_NPZ.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    download_zip()
    df = build_sample()
    embed_images(df)
