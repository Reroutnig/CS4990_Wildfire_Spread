import rasterio
from rasterio.plot import show
tiff_img = rasterio.open('data/burn_usa_tifs/burn_2024_4.tif')
show(tiff_img)