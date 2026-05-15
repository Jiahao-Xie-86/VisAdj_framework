#!/bin/bash
# Full script to generate OCTA500 benchmark dataset
# Split: 350 train, 50 val, 100 test
# Extract patches at 64x64, resize to 128x128
# Overlap: 16 pixels (25% of 64)

python3 dataset/create_octa500_benchmark_dataset.py \
    --raw_data_path dataset/raw/OCTA500 \
    --output_path dataset/processed/octa500_benchmark_128x128 \
    --patch_size 64 \
    --output_size 128 \
    --overlap 0 \
    --max_nodes 20 \
    --curvature-threshold 160.0 \
    --min-edge-length 10.0

