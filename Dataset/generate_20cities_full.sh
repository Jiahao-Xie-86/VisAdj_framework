#!/bin/bash

# Generate full 20cities benchmark dataset at 128×128 resolution
# 180 regions (144 train, 9 val, 27 test)
# Extracts overlapping 128×128 patches with 5% overlap
# Applies node simplification (curvature threshold: 160°)

echo "Generating full 20cities benchmark dataset at 128×128 resolution..."
echo "This will process 180 regions and extract ~46K patches (5% overlap)."
echo "Expected time: ~4-6 minutes"
echo ""

cd /usa/jiahaox/Image2matrix_baselines

python3 dataset/create_20cities_benchmark_dataset.py \
    --raw_data_path dataset/raw/20cities \
    --output_path dataset/processed/20cities_benchmark_128x128 \
    --patch_size 128 \
    --overlap 6 \
    --curvature_threshold 160.0

echo ""
echo "Dataset generation complete!"
echo "Location: dataset/processed/20cities_benchmark_128x128"

