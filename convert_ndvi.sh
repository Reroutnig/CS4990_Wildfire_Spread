#!/bin/bash

# Set the input directory
INPUT_DIR="data/2025-04-13-a6ee51"
OUTPUT_DIR="${INPUT_DIR}/ndvi_tifs"

# Make sure output folder exists
mkdir -p "$OUTPUT_DIR"

# Loop through all .hdf files
for hdf_file in "$INPUT_DIR"/*.hdf; do
  base=$(basename "$hdf_file" .hdf)
  ndvi_path="HDF4_EOS:EOS_GRID:\"$hdf_file\":MOD_Grid_monthly_CMG_VI:\"CMG 0.05 Deg Monthly NDVI\""
  gdal_translate "$ndvi_path" "${OUTPUT_DIR}/${base}_NDVI.tif"
done
