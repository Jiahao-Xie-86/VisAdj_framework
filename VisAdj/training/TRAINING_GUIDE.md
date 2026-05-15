# Training Guide

This guide explains how to run toy training (quick validation) and full training.

## Quick Start

### Step 1: Toy Training (Quick Validation)

Run a minimal training to verify everything works:

```bash
cd sam_graph_split/training
./toy_train.sh
```

Or manually:

```bash
python sam_graph_split/training/train.py \
    --dataset-root /path/to/dataset/processed/full_complete_benchmark_dataset \
    --output-dir sam_graph_split/outputs/toy_training \
    --sam-version vit_b \
    --batch-size 2 \
    --num-epochs 2 \
    --gpus 1 \
    --num-workers 2
```

**Toy Training Parameters:**
- **Batch size**: 2 (small for quick validation)
- **Epochs**: 2 (just to verify training loop works)
- **GPUs**: 1 (single GPU for quick test)
- **SAM version**: `vit_b` (smaller model, faster)
- **Workers**: 2 (reduced for quick startup)

**Expected Duration**: ~5-15 minutes depending on dataset size

**What to Check:**
1. ✅ Training starts without errors
2. ✅ Loss values are logged
3. ✅ Validation runs successfully
4. ✅ CSV files are created in `outputs/toy_training/logs/`
5. ✅ Checkpoints are saved in `outputs/toy_training/checkpoints/`

### Step 2: Full Training (Comprehensive)

After toy training succeeds, run the full training:

```bash
cd sam_graph_split/training
./full_train.sh
```

Or manually:

```bash
python sam_graph_split/training/train.py \
    --dataset-root /path/to/dataset/processed/full_complete_benchmark_dataset \
    --output-dir sam_graph_split/outputs/full_training \
    --sam-version sam2_base_plus \
    --batch-size 16 \
    --num-epochs 100 \
    --gpus 4 \
    --strategy ddp \
    --num-workers 8 \
    --early-stopping-patience 10
```

**Full Training Parameters:**
- **Batch size**: 16 (per GPU, total = 16 × 4 = 64)
- **Epochs**: 100 (with early stopping)
- **GPUs**: 4 (multi-GPU training)
- **SAM version**: `sam2_base_plus` (better performance)
- **Workers**: 8 (faster data loading)
- **Early stopping**: Patience of 10 epochs

**Expected Duration**: Several hours to days depending on dataset size

## Dataset Requirements

The dataset should be organized as:

```
dataset/processed/full_complete_benchmark_dataset/
├── train/
│   ├── images/              # PNG images
│   ├── adjacency_matrices/  # NPY files
│   ├── points/              # PKL files (node coordinates)
│   └── metadata.json
├── val/                     # Same structure
└── test/                    # Same structure
```

## Monitoring Training

### 1. Check Training Log

```bash
tail -f outputs/toy_training/training.log
# or
tail -f outputs/full_training/training.log
```

### 2. View TensorBoard

```bash
tensorboard --logdir outputs/toy_training/logs
# or
tensorboard --logdir outputs/full_training/logs
```

### 3. Plot Losses (After Training)

```bash
python sam_graph_split/training/plot_losses.py \
    --train-csv outputs/toy_training/logs/train_losses.csv \
    --val-csv outputs/toy_training/logs/val_losses.csv \
    --output-dir outputs/toy_training/plots
```

## Common Issues

### Issue: "Metadata file not found"
**Solution**: Check that `metadata.json` exists in `train/` and `val/` directories.

### Issue: "CUDA out of memory"
**Solution**: Reduce `--batch-size` (e.g., from 16 to 8 or 4).

### Issue: "No module named 'segment_anything'"
**Solution**: Install SAM or ensure the local SAM code is accessible.

### Issue: "Dataset is empty"
**Solution**: Verify dataset path and that `metadata.json` contains samples.

## Output Files

After training, you'll find:

```
outputs/toy_training/  (or full_training/)
├── checkpoints/
│   ├── best.ckpt          # Best model (lowest val loss)
│   └── last.ckpt          # Final model
├── logs/
│   ├── train_losses.csv   # Training losses per epoch
│   ├── val_losses.csv     # Validation losses per epoch
│   └── sam_graph_split/    # TensorBoard logs
├── plots/                  # Generated after running plot_losses.py
│   ├── total_loss.png
│   ├── individual_losses.png
│   └── ...
└── training.log           # Full training log
```

## Resuming Training

To resume from a checkpoint:

```bash
python sam_graph_split/training/train.py \
    --dataset-root /path/to/dataset \
    --output-dir outputs/full_training \
    # ... other args ...
    # Then manually load checkpoint in code or use PyTorch Lightning's resume_from_checkpoint
```

## Tips

1. **Start with toy training** to catch errors early
2. **Monitor GPU usage**: `nvidia-smi` to ensure GPUs are utilized
3. **Check CSV files** after each epoch to verify losses are decreasing
4. **Use early stopping** to avoid overfitting
5. **Save plots regularly** to visualize learning progress

