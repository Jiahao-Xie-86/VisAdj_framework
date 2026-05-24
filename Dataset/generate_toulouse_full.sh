#!/bin/bash

# Generate full Toulouse benchmark dataset at 128×128 resolution
# Total samples: 111,034 (train: 80,357, val: 11,679, test: 18,998)

echo "Generating full Toulouse benchmark dataset at 128×128 resolution..."
echo "This will process 111,034 samples and may take some time."
echo ""

cd /usa/jiahaox/Image2matrix_baselines

python3 dataset/create_toulouse_benchmark_dataset.py \
    --output_path dataset/processed/toulouse_benchmark \
    --image_size 64 64

echo ""
echo "Dataset generation complete!"
echo "Location: dataset/processed/toulouse_benchmark"

