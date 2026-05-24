# Dataset Generation Guide

This directory contains scripts to generate benchmark datasets for the image-to-graph conversion task. The datasets are generated from raw data sources and processed into a standardized format suitable for training and evaluation.

## Overview

This project supports four types of datasets:

1. **Synthetic Dataset**: Various graphs from House of Graph (raw data provided)
2. **Toulouse Road Network**: Road network dataset from Toulouse, France
3. **US-Cities (20cities)**: Satellite imagery from 20 U.S. cities
4. **OCTA500**: Retinal vascular network dataset

## Dataset Structure

```
Dataset/
├── raw/                          # Raw data directory (download datasets here)
│   ├── synthetic_graphs_dataset/    # Synthetic dataset (already provided)
│   │   └── *.npy                    # Various graphs (.npy files)
│   ├── Toulouse/                    # Toulouse dataset (download required)
│   ├── US-Cities/                   # US-Cities dataset (download required)
│   └── OCTA500/                     # OCTA500 dataset (download required)
├── processed/                      # Processed datasets (generated)
├── create_benchmark_dataset.py     # Synthetic dataset generator
├── create_toulouse_benchmark_dataset.py
├── create_20cities_benchmark_dataset.py
├── create_octa500_benchmark_dataset.py
└── generate_*.sh                   # Generation scripts
```

## Raw Data Download

### 1. Synthetic Dataset (Already Provided)

The synthetic dataset is already available in `raw/synthetic_graphs_dataset/`. It contains:
- **Planar graphs**: Adjacency matrices (.npy files) with naming pattern `planar_*.npy`
- **Non-planar graphs**: Adjacency matrices (.npy files) with naming pattern `non_planar_*.npy`
- Graphs with 4-20 nodes
- Total: ~3,947 graph files

**No download required** - the data is already in the repository.

---

### 2. Toulouse Road Network Dataset

**Source**: [Toulouse Road Network Dataset](https://github.com/davide-belli/toulouse-road-network-dataset)

**Download Instructions**:
1. Clone or download the repository:
```bash
cd Dataset/raw
git clone https://github.com/davide-belli/toulouse-road-network-dataset.git Toulouse
```

2. Or download manually and extract to `Dataset/raw/Toulouse/`

```

---

### 3. US-Cities (20cities) Dataset

**Source**: [Google Drive](https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H)

**Download Instructions**:
1. Download from the Google Drive link above
2. Extract the downloaded files to `Dataset/raw/US-Cities/`

---

### 4. OCTA500 Dataset

**Source**: [IEEE DataPort - OCTA-500](https://ieee-dataport.org/open-access/octa-500)

**Download Instructions**:
1. Visit the IEEE DataPort page: https://ieee-dataport.org/open-access/octa-500
2. **Request Access**: Send an email to `chen2qiang@njust.edu.cn` with the subject:
   ```
   OCTA500: your_organization: your_name
   ```
3. After receiving the password, download the dataset
4. Extract to `Dataset/raw/OCTA500/`

```

---

## Dataset Generation

After downloading the raw data to the `raw/` folder with the correct folder names (`Toulouse/`, `US-Cities/`, `OCTA500/`), use the provided scripts to generate the processed benchmark datasets.

**Important**: Ensure the raw data folders match the expected names. The generation scripts may reference these paths, so verify the `--raw_data_path` parameter matches your folder structure.

### 1. Synthetic Dataset

Generate benchmark dataset from synthetic planar/non-planar graphs:

```bash
python create_benchmark_dataset.py \
    --raw_data_path raw/synthetic_graphs_dataset \
    --output_path processed/benchmark_dataset_synthetic \
    --image_size 512 512 \
    --num_visualizations 3
```

**Output**: Processed dataset with node-link visualizations and adjacency matrices

---

### 2. Toulouse Dataset

Generate Toulouse benchmark dataset:

```bash
bash generate_toulouse_full.sh
```


---

### 3. US-Cities (20cities) Dataset

Generate 20cities benchmark dataset:

```bash
bash generate_20cities_full.sh
```


---

### 4. OCTA500 Dataset

Generate OCTA500 benchmark dataset:

```bash
bash generate_octa500_full.sh
```


---

## Processed Dataset Format

All generated datasets follow a consistent structure:

```
processed/<dataset_name>/
├── train/
│   ├── images/              # PNG images
│   ├── adjacency_matrices/  # NPY files (adjacency matrices)
│   ├── points/              # PKL files (node coordinates)
│   ├── masks/               # PKL files (optional, node/edge masks)
│   └── metadata.json         # Dataset metadata
├── val/                     # Same structure as train/
└── test/                    # Same structure as train/
```



