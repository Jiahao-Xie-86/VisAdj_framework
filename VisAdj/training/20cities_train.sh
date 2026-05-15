#!/bin/bash
# 20 U.S. Cities Training Script
# Adapted for 20cities benchmark dataset with 128x128 images
# Graph sizes: 2-28 nodes (mean: 6.7, median: 6)

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default paths (can be overridden)
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/../dataset/processed/20cities_benchmark_128x128}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/20cities_edgewareGT_RGB_mean_topology_hid256}"

echo "=========================================="
echo "20 U.S. Cities Training - 128x128 Images"
echo "=========================================="
echo "Dataset root: $DATASET_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: 128x128"
echo "Graph sizes: 2-20 nodes (mean: 6.7, median: 6)"
echo "Total samples: 40,873 (train: 33,151, val: 1,946, test: 5,776)"
echo "GPUs: 4"
echo "=========================================="

# Fix memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Image size parameters
# 20cities: 128x128 (same as Toulouse)
IMAGE_SIZE=128
HEATMAP_RESOLUTION=16  # Scaled from 64: 64 * (128/512) = 16
NEIGHBOR_RADIUS=128.0  # Scaled from 512.0: 512 * (128/512) = 128
MASK_POOL_RADIUS=4     # Scaled from 16: 16 * (128/512) = 4

# Run training with 20cities-adapted parameters
# Note: Node detector uses mask-based peaks, so mask threshold/pooling control detection density.
conda run -n sam_graph_split python "$SCRIPT_DIR/train.py" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --sam-version "vit_b" \
    --batch-size 144 \
    --num-epochs 200 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --gpus 4 \
    --strategy "ddp" \
    --num-workers 8 \
    --log-every-n-steps 10 \
    --val-check-interval 1.0 \
    --early-stopping-patience 20 \
    --node-loss-weight 1 \
    --edge-loss-weight 10 \
    --coverage-loss-weight 0 \
    --coverage-label-smoothing 0.1 \
    --heatmap-sigma 1.0 \
    --heatmap-resolution $HEATMAP_RESOLUTION \
    --image-size $IMAGE_SIZE \
    --use-lora \
    --lora-rank 8 \
    --edge-use-focal \
    --edge-focal-alpha 0.25 \
    --edge-focal-gamma 2.0 \
    --edge-pos-weight 1.0 \
    --adjacency-weight 2.0 \
    --pair-weight 5.0 \
    --k-neighbors 8 \
    --neighbor-radius $NEIGHBOR_RADIUS \
    --neighbor-sampler knn \
    --asns-entmax-alpha 1.5 \
    --max-nodes 20 \
    --relation-transformer-layers 3 \
    --relation-edge-dim 256 \
    --relation-hidden-dim 128 \
    --relation-num-heads 8 \
    --relation-dropout 0.1 \
    --rgb-feature-dim 32 \
    --rgb-sequence-model transformer \
    --rgb-seq-layers 2 \
    --rgb-seq-heads 4 \
    --rgb-neighborhood-aggregation mean \
    --rgb-neighborhood-radius 1.0 \
    --coordinate-noise-std 1.0 \
    --edge-model edge_aware_transformer \
    --no-edge-length-weighting \
    --mask-threshold 0.5 \
    --mask-pool-radius $MASK_POOL_RADIUS \
    --phase1-epochs 0 \
    --node-finetune-lr-scale 1 \
    --precision 32 \
    --enable-diagnostics
    # --use-hard-negative-mining \
    # --hard-negative-threshold 0.3 \
    # --max-hard-negatives-ratio 2.0 \
    # --no-detach-l-i  # Uncomment to allow gradients from edge/coverage losses to flow back to local descriptor head

echo ""
echo "=========================================="
echo "20cities training completed!"
echo "=========================================="
echo "Check outputs in: $OUTPUT_DIR"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo "  - Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  - Training log: $OUTPUT_DIR/training.log"
echo ""
echo "To plot losses, run:"
echo "  python $SCRIPT_DIR/plot_losses.py \\"
echo "    --train-csv $OUTPUT_DIR/logs/train_losses.csv \\"
echo "    --val-csv $OUTPUT_DIR/logs/val_losses.csv \\"
echo "    --output-dir $OUTPUT_DIR/plots"
echo "=========================================="

