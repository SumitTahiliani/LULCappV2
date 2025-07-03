import rasterio
from rasterio.transform import from_origin

def save_mask_as_geotiff(mask, ds, output_path):
    assert mask.ndim == 2, "Mask must be 2D"

    # Get georeferencing info
    res_x = abs(ds.x[1] - ds.x[0])
    res_y = abs(ds.y[1] - ds.y[0])
    transform = from_origin(ds.x[0], ds.y[0], res_x, res_y)
    crs = f"EPSG:{ds.spatial_ref.values}"

    # Write the raster
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    return output_path
