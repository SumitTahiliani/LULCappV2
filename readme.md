# LULCappV2

This project is a land use/land cover (LULC) classification app. It is an exact copy of the app in the `LandCoverExplorer-India` repository, with the key difference that this version uses a custom-trained model for land cover classification instead of pre-loaded rasters.

## Features
- Performs LULC classification using my own trained model (currently experimenting with different architectuers).
- Replicates the workflow and interface of the original UttarPradeshLULC app.
- Designed for flexibility and experimentation with custom models.

## Notes
- Model accuracy is a work in progress and may improve with further training and data.
- For usage instructions, refer to the original app's documentation in the `LandCoverExplorer-India` repository.

## Files
- `app.py`: Main application script.
- `best_6cl_focalce_rarenpz_highlr.pt`, `best_6cl_focalce_rarenpz.pt`: Custom-trained model weights.
- Other scripts: Data preparation, mask generation, and prediction utilities.

