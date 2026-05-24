#!/bin/bash
# Full script to generate OCTA500 benchmark dataset
# Split: 350 train, 50 val, 100 test
# Extract patches at 256x256
# Overlap: 16 pixels 

python3 dataset/create_octa500_benchmark_dataset.py \
    --raw_data_path dataset/raw/OCTA500 \
    --output_path dataset/processed/octa500_benchmark \
    --patch_size 256 \
    --output_size 256 \
    --overlap 0 \
    --max_nodes 20 \
    --curvature-threshold 160.0 \
    --min-edge-length 10.0

