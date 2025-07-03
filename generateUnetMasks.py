import pathlib, json, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from prepData import compute_indices  # You already use this
from math import ceil
import segmentation_models_pytorch as smp

TILE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(NUM_CLASSES):
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=8,
        classes=NUM_CLASSES
    )

model = build_model(NUM_CLASSES=6)
model.load_state_dict(torch.load("best_6cl_focalce_rarenpz_highlr.pt", map_location=DEVICE))
model.to(DEVICE).eval()

@torch.no_grad()
def generate_unet_prediction_mask(ds):
    ds = ds.isel(time=0)

    # --- Step 2: Get (H, W, 5) bands in correct order ---
    bands = ['B11', 'B08', 'B04', 'B03', 'B02']
    img = np.stack([ds[b].values for b in bands], axis=-1).astype(np.float32)

    # --- Step 3: Add indices (same as training) → (H, W, 8) ---
    img = compute_indices(img)

    # --- Step 4: Tile and pad ---
    H, W, C = img.shape
    pad_H = ceil(H / TILE_SIZE) * TILE_SIZE
    pad_W = ceil(W / TILE_SIZE) * TILE_SIZE
    img_padded = np.zeros((pad_H, pad_W, C), dtype=np.float32)
    img_padded[:H, :W, :] = img

    output_mask = np.zeros((pad_H, pad_W), dtype=np.uint8)

    # --- Step 5: Predict each tile ---
    for i in range(0, pad_H, TILE_SIZE):
        for j in range(0, pad_W, TILE_SIZE):
            tile = img_padded[i:i+TILE_SIZE, j:j+TILE_SIZE]
            tile = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)  # (1, 8, 256, 256)

            out = model(tile)  # (1, num_classes, 256, 256)
            pred = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
            output_mask[i:i+TILE_SIZE, j:j+TILE_SIZE] = pred

    # --- Step 6: Remove padding ---
    return output_mask[:H, :W]
