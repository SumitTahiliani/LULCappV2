# extract_rf_data.py

import numpy as np, json, pathlib, tqdm
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────
JSON_PATH = r"LULCmodelTraining/splits_unordered.json"
NPZ_PATH  = r"LULCmodelTraining/lulc_dataset_5band.npz"

# OUT_DIR   = pathlib.Path("rfData")      # ←  all outputs go here
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# OUT_X = OUT_DIR / "X.npy"
# OUT_Y = OUT_DIR / "y.npy"
NUM_CLASSES = 6

# DW → 6-class remap
MAP = np.array([0, 1, 2, 255, 3, 4, 5, 255, 255], dtype=np.uint8)
IGNORE = 255
RARE_CLASSES = {2, 4}  # Shrub & Bare
OVERSAMPLE_FACTOR = 3

def compute_indices(img):
    B11, B08, B04, B03, _ = [img[..., i] for i in range(5)]
    eps = 1e-6
    ndvi  = (B08 - B04) / (B08 + B04 + eps)
    mndwi = (B03 - B11) / (B03 + B11 + eps)
    nbi   = (B11 - B04) / (B11 + B04 + eps)
    return np.dstack([img, ndvi[..., None], mndwi[..., None], nbi[..., None]])

def collect_pixels(split="train",
                   max_pix_per_tile=5000,   # sample this many pixels per tile
                   max_per_class=50000):     # cap total per class
    data = np.load(NPZ_PATH, mmap_mode="r")
    ids = json.load(open(JSON_PATH))[split]

    class_buckets = {c: [] for c in range(NUM_CLASSES)}
    rare_samples = []

    for uid in tqdm.tqdm(ids, desc=f"Extracting {split}"):
        uid_str = f"{uid:06d}"
        img5 = data[f"img_{uid_str}"]
        msk6 = MAP[data[f"msk_{uid_str}"]]

        # pick valid pixels
        valid = (msk6 != IGNORE)
        if not np.any(valid):
            continue

        # random sub‑sample of valid indices
        coords = np.column_stack(np.where(valid))
        if len(coords) > max_pix_per_tile:
            coords = coords[np.random.choice(len(coords),
                                             max_pix_per_tile,
                                             replace=False)]

        H, W = msk6.shape
        img8 = compute_indices(img5)

        for (r, c) in coords:
            cls = int(msk6[r, c])
            if len(class_buckets[cls]) < max_per_class:
                feat = img8[r, c, :]
                class_buckets[cls].append(feat)

    # build X, y
    X_list, y_list = [], []
    for cls, feats in class_buckets.items():
        X_list.append(np.stack(feats))
        y_list.append(np.full(len(feats), cls, dtype=np.uint8))

    X_all = np.vstack(X_list).astype("float32")
    y_all = np.concatenate(y_list)

    for cls in [2, 4]:
        extras = X_all[y_all == cls]
        for _ in range(OVERSAMPLE_FACTOR - 1):
            X_all = np.vstack([X_all, extras])
            y_all = np.concatenate([y_all, np.full(len(extras), cls, dtype=np.uint8)])
            
    print("Final sample counts:", {c: len(v) for c, v in class_buckets.items()})
    return X_all, y_all


# X, y = collect_pixels(split="train")
# np.save(OUT_X, X)
# np.save(OUT_Y, y)
# print(f"Saved {X.shape[0]} samples to {OUT_X}, {OUT_Y}")
