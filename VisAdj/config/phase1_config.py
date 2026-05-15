"""
Configuration file for sam_graph_split Phase 1 (Node Detection)
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "processed" / "full_complete_benchmark_dataset"

# Model configuration
SAM_VERSION = 'vit_b'  # 'vit_b', 'vit_l', 'vit_h'
# Default checkpoint path (will be auto-detected if None)
SAM_CHECKPOINT = None  # Path to SAM checkpoint, None to use default from sam_checkpoint/ directory
FREEZE_ENCODER = True  # Freeze encoder by default (LoRA handles adaptation)
IMAGE_SIZE = 512

# Dual stream configuration
# Local: encoder resolution (32×32), Global: downsampled by ×4 (8×8)
LOCAL_FEATURE_DIM = 256
GLOBAL_FEATURE_DIM = 256

# Node detector configuration
NODE_DETECTOR_FEATURE_DIM = 256
HEATMAP_THRESHOLD = 0.3  # Lowered from 0.5 to detect more nodes initially
NMS_RADIUS = 10
MAX_NODES = 50

# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4  # Reduced from 5e-4 to prevent early collapse (model was collapsing in first epoch)
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# Loss configuration
# Equal weights: Since connectivity loss is disabled, node and edge losses have equal weight
NODE_LOSS_WEIGHT = 1.0  # Equal weight with edge loss
EDGE_LOSS_WEIGHT = 1.0  # Equal weight with node loss
COVERAGE_LOSS_WEIGHT = 0.3
BUDGET_LOSS_WEIGHT = 0.01  # Reduced from 0.1 since budget loss is now normalized
MASK_LOSS_WEIGHT = 0.0  # Optional, default 0
USE_MASK_LOSS = False
# Output configuration
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_model"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# Experiment tracking
USE_WANDB = False
WANDB_PROJECT = "sam_graph_split"
WANDB_NAME = "phase1_node_detection"

# Validation
VAL_INTERVAL = 1  # Validate every N epochs
SAVE_INTERVAL = 5  # Save checkpoint every N epochs

