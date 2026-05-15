# VisAdj: Learning Adjacency Matrices From Node-Link Images

A deep learning framework for converting node-link images to adjacency matrices using SAM (Segment Anything Model) as the visual backbone.

## Overview

This project implements a novel architecture that extracts graph structures from visual representations. It takes node-link diagram images as input and outputs corresponding adjacency matrices, enabling automated graph extraction from visualizations. The framework is particularly effective for road network extraction, vascular network analysis, and general graph structure recovery from images.

### Key Features

- **SAM2-based Visual Encoder**: Leverages frozen SAM2 encoder as a powerful visual backbone
- **Dual-Stream Architecture**: Combines local (geometry) and global (topology) feature streams
- **Attention-Sparse Neighbor Sampler (ASNS)**: Efficient neighbor sampling for edge prediction
- **Relation Transformer**: Set-based edge reasoning for accurate adjacency prediction
- **Multi-Dataset Support**: Works with various benchmark datasets including Synthetic graphs, US-Cities, OCTA500, Toulouse, and SpaceNet
- **Comprehensive Evaluation**: Supports node detection, edge prediction, and graph structure metrics

## Project Structure

```
.
├── VisAdj/                          # Main model implementation
│   ├── model/                       # Model components
│   │   ├── encoder.py               # SAM2 encoder wrapper
│   │   ├── dual_stream.py           # Dual-stream feature extraction
│   │   ├── node_detector.py         # Node detection module
│   │   ├── asns.py                  # Attention-Sparse Neighbor Sampler
│   │   ├── knn_neighbor_sampler.py # KNN-based neighbor sampler
│   │   ├── relation_transformer.py # Edge prediction transformer
│   │   ├── edge_aware_graph_transformer.py  # Edge-aware graph transformer
│   │   └── sam_graph_split.py      # Main model class
│   ├── dataset/                     # Dataset loaders
│   │   └── image2matrix_dataset.py
│   ├── training/                    # Training scripts
│   │   ├── train.py                 # Main training script
│   │   ├── full_train.sh            # Full training script
│   │   ├── 20cities_train.sh        # Dataset-specific scripts
│   │   ├── octa500_train.sh
│   │   ├── toulouse_train.sh
│   │   └── TRAINING_GUIDE.md        # Training guide
│   ├── inference/                   # Inference and evaluation scripts
│   │   ├── infer_model.py           # Inference script
│   │   └── run_inference_and_eval*.sh  # Evaluation scripts
│   ├── evaluation/                  # Evaluation metrics
│   │   ├── evaluate_predictions.py
│   │   ├── graph_isomorphism_evaluator.py
│   │   └── graph_structure_metrics.py
│   ├── losses/                      # Loss functions
│   │   └── combined_loss.py
│   ├── utils/                       # Utility functions
│   ├── config/                      # Configuration files
│   ├── sam2_checkpoints/           # SAM2 model checkpoints
│   └── sam_checkpoint/              # SAM checkpoints (legacy)
├── Dataset/                         # Dataset generation scripts
│   ├── README.md                    # Dataset generation guide
│   ├── raw/                         # Raw data directory
│   │   ├── synthetic_graphs_dataset/  # Synthetic graphs (provided)
│   │   ├── Toulouse/                # Toulouse dataset (download required)
│   │   ├── US-Cities/                # US-Cities dataset (download required)
│   │   └── OCTA500/                  # OCTA500 dataset (download required)
│   ├── create_benchmark_dataset.py  # Synthetic dataset generator
│   ├── create_20cities_benchmark_dataset.py
│   ├── create_octa500_benchmark_dataset.py
│   ├── create_toulouse_benchmark_dataset.py
│   └── generate_*.sh                 # Generation scripts
├── Baseline/                        # Baseline implementations
│   ├── 
│   ├── SAM-Road++/                  # SAM-Road++ baseline
│   ├── RNGDet++/                    # RNGDet++ baseline
│   ├── Any2Graph/                   # Any2Graph baseline
│   └── README.md                    # Baseline methods documentation
├── requirement.txt                  # Python dependencies
└── README.md                        # This file
```

> **Note**: The `Dataset/` and `Baseline/` folders contain their own README files with detailed documentation. Please refer to:
> - `Dataset/README.md` for dataset generation instructions and dataset-specific details
> - `Baseline/README.md` for baseline method implementations and comparisons

## Architecture

The model consists of several key components:

1. **SAM2 Encoder**: Frozen visual encoder that extracts rich visual features from input images
2. **Dual-Stream Extractor**: 
   - **Local stream**: High-resolution features at encoder grid resolution (32×32 for 512×512 input)
   - **Global stream**: Downsampled features for topology understanding (8×8)
3. **Node Detector**: Predicts node locations using heatmap-based detection with NMS
4. **Neighbor Sampler**: ASNS (Attention-Sparse Neighbor Sampler) or KNN-based sampling for efficient edge candidate selection
5. **Relation Transformer**: Processes node pairs to predict edge existence using transformer architecture
6. **Edge Predictor**: Final edge prediction module (Edge-Aware Graph Transformer or Pairwise MLP)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Conda (for environment management)
- Git

### Setup

1. **Create conda environment**：
```bash
conda create -n sam_graph_split python=3.10
conda activate sam_graph_split
```

2. **Install dependencies**:
```bash
pip install -r requirement.txt
```

3. **Download SAM and SAM2 checkpoints**:
   
   **SAM Checkpoints**:
   - Download SAM checkpoints from the [official repository](https://github.com/facebookresearch/segment-anything)
   - Place checkpoint files in `VisAdj/sam_checkpoint/`
   - Supported versions: `sam_vit_b.pth`, `sam_vit_l.pth`, `sam_vit_h.pth`
   - Download links:
     - [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
     - [SAM ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
     - [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
   
   **SAM2 Checkpoints**:
   - Download SAM2 checkpoints from the [official repository](https://github.com/facebookresearch/segment-anything-2)
   - Place checkpoint files in `VisAdj/sam2_checkpoints/`
   - Supported versions: `sam2_tiny`, `sam2_base_plus`, `sam2_large`
   - Checkpoint files should include both `.pt` and `.yaml` files
   - Download links:
     - [SAM2 Tiny](https://dl.fbaipublicfiles.com/segment_anything_2/095919/sam2.1_hiera_tiny.pt) and [config](https://github.com/facebookresearch/segment-anything-2/blob/main/sam2_hiera_t.yaml)
     - [SAM2 Base+](https://dl.fbaipublicfiles.com/segment_anything_2/095919/sam2.1_hiera_base_plus.pt) and [config](https://github.com/facebookresearch/segment-anything-2/blob/main/sam2_hiera_b+.yaml)
     - [SAM2 Large](https://dl.fbaipublicfiles.com/segment_anything_2/095919/sam2.1_hiera_large.pt) and [config](https://github.com/facebookresearch/segment-anything-2/blob/main/sam2_hiera_l.yaml)

## Quick Start

### 1. Prepare Dataset

The dataset should be organized as:
```
dataset/processed/<dataset_name>/
├── train/
│   ├── images/              # PNG images
│   ├── adjacency_matrices/  # NPY files (adjacency matrices)
│   ├── points/              # PKL files (node coordinates)
│   ├── masks/               # PKL files (optional, node/edge masks)
│   └── metadata.json        # Dataset metadata
├── val/                     # Same structure as train/
└── test/                    # Same structure as train/
```

**For detailed dataset generation instructions, please refer to `Dataset/README.md`**.


### 2. Training

#### Full Training

Run full training with default parameters:
```bash
cd VisAdj/training
bash full_train.sh
```

Or manually with custom parameters:
```bash
cd VisAdj/training
python train.py \
    --dataset-root /path/to/dataset/processed/full_complete_benchmark_dataset \
    --output-dir outputs/full_training \
    --sam-version sam2_tiny \
    --batch-size 96 \
    --num-epochs 200 \
    --gpus 4 \
    --strategy ddp \
    --learning-rate 1e-3 \
    --k-neighbors 20 \
    --max-nodes 20 \
    --node-loss-weight 1 \
    --edge-loss-weight 10
```

#### Dataset-Specific Training

The project includes dataset-specific training scripts:

- `full_train.sh`: General full training
- `20cities_train.sh`: Training for US-Cities dataset
- `octa500_train.sh`: Training for OCTA500 dataset
- `toulouse_train.sh`: Training for Toulouse dataset
- `spacenet_train.sh`: Training for SpaceNet dataset

**For detailed training instructions, see `VisAdj/training/TRAINING_GUIDE.md`**.

### 3. Inference and Evaluation

Run inference and evaluation on a trained checkpoint:

```bash
cd VisAdj/inference
bash run_inference_and_eval.sh <checkpoint_dir> [split]
```

Or use dataset-specific evaluation scripts:
```bash
bash run_inference_and_eval_toulouse.sh <checkpoint_dir> test
bash run_inference_and_eval_octa500.sh <checkpoint_dir> test
bash run_inference_and_eval_cityscale.sh <checkpoint_dir> test
```

## Evaluation Metrics

The evaluation module computes:

- **Node Metrics**: Precision, Recall, F1-score for node detection
- **Edge Metrics**: Precision, Recall, F1-score for edge prediction
- **Graph Structure Metrics**: 
  - Graph Isomorphsim Rate (GIR)
  - Graph Edit Distance (GED) 
  - Topological metrics (k-hop subgraph matching)

## Output Structure

After training, outputs are organized as:

```
VisAdj/outputs/<experiment_name>/
├── checkpoints/
│   ├── best-epoch=*.ckpt     # Best model checkpoint
│   └── last.ckpt              # Last checkpoint
├── logs/
│   ├── train_losses.csv       # Training loss history
│   ├── val_losses.csv         # Validation loss history
│   └── sam_graph_split/       # TensorBoard logs
├── predictions_test/          # Test set predictions
│   ├── adjacency_matrices/    # Predicted adjacency matrices
│   ├── points/                # Predicted node coordinates
│   └── evaluation_results.json
└── training.log               # Full training log
```


## Datasets

This project supports multiple datasets:

1. **Synthetic Dataset**: Various graphs from House of Graph (raw data provided in `Dataset/raw/synthetic_graphs_dataset/`)
2. **Toulouse Road Network**: Road network dataset from Toulouse, France
3. **US-Cities (20cities)**: Satellite imagery from 20 U.S. cities
4. **OCTA500**: Retinal vascular network dataset

**For detailed dataset information and download instructions, see `Dataset/README.md`**.

## Baselines

This project includes comparisons with several state-of-the-art baseline methods:

- **SAM-Road++**: Road graph extraction using SAM encoder
- **RNGDet++**: Road network graph detection by Transformer
- **Any2Graph**: General-purpose graph extraction framework
- **Sat2Graph**: Satellite image to graph conversion

**For baseline setup instructions, see `Baseline/README.md`**.


## Additional Documentation

For more detailed information, please refer to:

- **`Dataset/README.md`**: Comprehensive guide for dataset generation, including download instructions and processing details
- **`Baseline/README.md`**: Documentation for baseline methods setup and usage
