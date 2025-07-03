from prepData import compute_indices
import numpy as np
import joblib
from math import ceil

TILE_SIZE = 256

def tile_image(img, tile_size=TILE_SIZE):
    H, W, C = img.shape
    pad_H = (tile_size - H % tile_size) % tile_size
    pad_W = (tile_size - W % tile_size) % tile_size

    if pad_H > 0 or pad_W > 0:
        img = np.pad(img, ((0, pad_H), (0, pad_W), (0, 0)), mode='reflect')

    tiles = []
    positions = []
    for i in range(0, img.shape[0], tile_size):
        for j in range(0, img.shape[1], tile_size):
            tile = img[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            positions.append((i, j))
    return tiles, positions, img.shape[:2], (H, W)

def stitch_predictions(preds, positions, full_shape, original_shape):
    final = np.zeros(full_shape, dtype=np.uint8)
    for pred, (i, j) in zip(preds, positions):
        final[i:i+TILE_SIZE, j:j+TILE_SIZE] = pred
    return final[:original_shape[0], :original_shape[1]]

def predict_in_batches(model, tiles, batch_size=32):
    predictions = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i+batch_size]
        flat = np.vstack([tile.reshape(-1, 8) for tile in batch])
        preds = model.predict(flat)
        for j in range(len(batch)):
            start = j * TILE_SIZE * TILE_SIZE
            end = (j + 1) * TILE_SIZE * TILE_SIZE
            predictions.append(preds[start:end].reshape(TILE_SIZE, TILE_SIZE))
    return predictions

def generate_prediction_mask(ds, batch_size=8):
    ds = ds.isel(time=0)
    bands = ['B11', 'B08', 'B04', 'B03', 'B02']
    img = np.stack([ds[b].values for b in bands], axis=-1).astype(np.float32)

    img = compute_indices(img)  # shape: (H, W, 8)
    tiles, positions, padded_shape, orig_shape = tile_image(img)
    print("Tiling done")
    model = joblib.load("rf_model.pkl")
    predictions = predict_in_batches(model, tiles, batch_size=batch_size)
    return stitch_predictions(predictions, positions, padded_shape, orig_shape)
