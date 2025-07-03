import tempfile
import os
from shapely.geometry import box, mapping
from predictionMask import generate_prediction_mask
from saveMask import save_mask_as_geotiff
from testing import get_s2_data  # ← your existing fetch function
from generateUnetMasks import generate_unet_prediction_mask

def get_or_generate_unet_raster(location, year):
    daterange = (f"{year}-01-01", f"{year}-12-31")
    ds = get_s2_data(location, daterange)

    mask = generate_unet_prediction_mask(ds)
    temp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    save_mask_as_geotiff(mask, ds.isel(time=0), temp.name)
    return temp.name
