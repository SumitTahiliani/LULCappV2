import xarray as xr
from predictionMask import generate_prediction_mask
import matplotlib.pyplot as plt
import numpy as np
from mapminer.miners import Sentinel2Miner
from shapely.geometry import Polygon
import requests
from saveMask import save_mask_as_geotiff
import tempfile
import os

miner = Sentinel2Miner()
def search_location(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 5, "polygon_geojson": 1}
    headers = {"User-Agent": "streamlit-gis-app/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Nominatim API error: {e}")
        return []

def get_s2_data(location, daterange):
    polygon_coords = location[0]['geojson']['coordinates'][0]
    lat = location[0]['lat']
    lon = location[0]['lon']
    if location[0]['geojson']['type'] != 'Polygon':
        ds = miner.fetch(lat, lon, radius = 21000,)
    else:
        polygon = Polygon(polygon_coords)
        ds = miner.fetch(polygon=polygon,)
    return ds

if __name__ == "__main__":
    location = search_location("Kanpur")
    ds = get_s2_data(miner, location)
    print(ds)
    mask = generate_prediction_mask(ds)
    # Simple assertions / sanity checks
    print("Mask shape:", mask.shape)
    print("Unique classes:", np.unique(mask, return_counts=True))

    # Plot the mask
    plt.imshow(mask, cmap="tab10")
    plt.title("Predicted Land Cover")
    plt.axis("off")
    plt.show()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tif_path = save_mask_as_geotiff(mask, ds.isel(time=0), tmp.name)

    print("GeoTIFF saved to:", tif_path)
    os.system(f"gdalinfo {tif_path}")