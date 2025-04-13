import rasterio
from rasterio.plot import show
tiff_img = rasterio.open('data/2025-04-13-a6ee51/ndvi_tifs/MOD13C2.A2020001.061.2020328142931_NDVI_US.tif')
show(tiff_img)