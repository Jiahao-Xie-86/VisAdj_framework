#!/bin/bash
# SpaceNet Training Script
# Adapted for SpaceNet dataset with 400x400 images (vs original 512x512)
# Uses the filtered SpaceNet dataset with graphs N < 30

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default paths (can be overridden)
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/../dataset/processed/spacenet_benchmark_dataset_n_lt_30_resplit}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/spacenet_edgewareGT_RGB_mean_topology_hid256}"

echo "=========================================="
echo "SpaceNet Training - 400x400 Images"
echo "=========================================="
echo "Dataset root: $DATASET_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Image size: 400x400 (vs original 512x512)"
echo "GPUs: 1"
echo "=========================================="

# Fix memory fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Image size parameters
IMAGE_SIZE=400
HEATMAP_RESOLUTION=50  # Scaled from 64: 64 * (400/512) ≈ 50
NEIGHBOR_RADIUS=400.0  # Scaled from 512.0: 512 * (400/512) = 400
MASK_POOL_RADIUS=13    # Scaled from 16: 16 * (400/512) ≈ 12.5 → 13

# Run training with SpaceNet-adapted parameters
# Note: Node detector uses mask-based peaks, so mask threshold/pooling control detection density.
conda run -n sam_graph_split python "$SCRIPT_DIR/train.py" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --sam-version "vit_b" \
    --batch-size 72 \
    --num-epochs 200 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --gpus 4 \
    --strategy "ddp" \
    --num-workers 4 \
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
    --k-neighbors 20 \
    --neighbor-radius $NEIGHBOR_RADIUS \
    --neighbor-sampler knn \
    --asns-entmax-alpha 1.5 \
    --max-nodes 30 \
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
    --coordinate-noise-std 2.0 \
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
echo "SpaceNet training completed!"
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

