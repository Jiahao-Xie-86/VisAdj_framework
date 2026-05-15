#!/bin/bash
# Run inference and evaluation on a checkpoint

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Parse arguments
CHECKPOINT_DIR="$1"
SPLIT="${2:-test}"  # Default to 'test' if not provided
USE_GT_NODES="${3:-0}"  # Set to 1 to use ground-truth node coordinates during inference
# DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/dataset/processed/full_complete_benchmark_dataset}"
# DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/dataset/processed/benchmark_dataset_21-50}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/dataset/processed/toulouse_benchmark_128x128}"



if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> [split] [use_gt_nodes]"
    echo "Example: $0 outputs/full_training_ASNS_lr5e4_V4 test 1"
    echo "Set use_gt_nodes=1 to bypass node detection and evaluate edge module with GT nodes."
    exit 1
fi

# Handle both cases: with or without 'outputs/' prefix
if [[ "$CHECKPOINT_DIR" == outputs/* ]]; then
    # Already has outputs/ prefix
    FULL_CHECKPOINT_DIR="$CHECKPOINT_DIR"
else
    # Add outputs/ prefix
    FULL_CHECKPOINT_DIR="outputs/$CHECKPOINT_DIR"
fi

CHECKPOINT_PATH="$PROJECT_ROOT/sam_graph_split/$FULL_CHECKPOINT_DIR/checkpoints/best-epoch=epoch=005.ckpt"
OUTPUT_DIR="$PROJECT_ROOT/sam_graph_split/$FULL_CHECKPOINT_DIR/predictions_${SPLIT}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
if [ "$USE_GT_NODES" = "1" ]; then
    echo "Using ground-truth node coordinates for edge inference (edge-only evaluation mode)."
fi
echo "Running Inference and Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_ROOT"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run inference
echo ""
echo "Step 1: Running inference..."
cd "$PROJECT_ROOT"
INFER_EXTRA_ARGS=()
if [ "$USE_GT_NODES" = "1" ]; then
    INFER_EXTRA_ARGS+=(--use-gt-nodes)
fi
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" conda run -n sam_graph_split python "$SCRIPT_DIR/infer_model.py" \
    --checkpoint "$CHECKPOINT_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --batch-size 720 \
    --mask-threshold 0.5 \
    --mask-pool-radius 4 \
    --nms-radius 4 \
    --max-nodes 9 \
    --k-neighbors 4 \
    --neighbor-radius 64 \
    --edge-threshold 0.8 \
    --rgb-neighborhood-aggregation mean \
    --rgb-neighborhood-radius 1.0 \
    "${INFER_EXTRA_ARGS[@]}"

# Run evaluation
echo ""
echo "Step 2: Running evaluation..."
cd "$PROJECT_ROOT"
# Use env to ensure PYTHONPATH is passed to conda run
# Note: GED is disabled by default as it's very slow. Add --compute-ged to enable it.
env PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" conda run -n sam_graph_split python "$SCRIPT_DIR/../evaluation/evaluate_predictions.py" \
    --predictions-dir "$OUTPUT_DIR" \
    --ground-truth-dir "$DATASET_ROOT" \
    --split "$SPLIT"

echo ""
echo "=========================================="
echo "Inference and evaluation complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Evaluation results: $OUTPUT_DIR/evaluation_results.json"
echo "=========================================="

