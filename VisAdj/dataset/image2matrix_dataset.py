"""
Dataset loader for Image2Matrix task
"""

import json
import pickle
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
import random


class Image2MatrixDataset(Dataset):
    """
    Dataset loader for Image2Matrix benchmark dataset.
    
    Loads node-link diagram images with ground truth adjacency matrices.
    Compatible with the processed dataset structure used by SAM-Road.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        augment: bool = False,
        image_size: int = 512,
        heatmap_resolution: int = 32,
        heatmap_sigma: float = 1.5,
    ):
        """
        Args:
            dataset_path: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test')
            augment: Whether to apply data augmentation
            image_size: Target image size
            heatmap_resolution: Resolution of heatmap (default: 32, can be 64 for higher resolution)
            heatmap_sigma: Gaussian sigma for heatmap peaks (default: 1.5, reduced from 2.0 to reduce overlap)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.augment = augment
        self.image_size = image_size
        self.heatmap_resolution = heatmap_resolution
        self.heatmap_sigma = heatmap_sigma
        
        # Load metadata
        metadata_path = self.dataset_path / split / "metadata.json"
        if not metadata_path.exists():
            # Try alternative location
            metadata_path = self.dataset_path / "metadata" / f"{split}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Set up directories
        self.image_dir = self.dataset_path / split / "images"
        self.mask_dir = self.dataset_path / split / "masks"
        self.points_dir = self.dataset_path / split / "points"
        self.adjacency_dir = self.dataset_path / split / "adjacency_matrices"
        
        print(f"Loaded {len(self.metadata)} samples from {split} split")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Image directory: {self.image_dir}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample_metadata = self.metadata[idx]
        
        # Load image
        image_path = self.image_dir / sample_metadata['image_filename']
        rgb = cv2.imread(str(image_path))
        if rgb is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        # Initialize optional supervision
        node_mask = None
        edge_mask = None
        node_coords = None
        
        # Load masks (if available)
        
        if 'node_mask_filename' in sample_metadata:
            node_mask_path = self.mask_dir / "nodes" / sample_metadata['node_mask_filename']
            if node_mask_path.exists():
                node_mask = cv2.imread(str(node_mask_path), cv2.IMREAD_GRAYSCALE)
                if node_mask is not None:
                    node_mask = cv2.resize(node_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if 'edge_mask_filename' in sample_metadata:
            edge_mask_path = self.mask_dir / "edges" / sample_metadata['edge_mask_filename']
            if edge_mask_path.exists():
                edge_mask = cv2.imread(str(edge_mask_path), cv2.IMREAD_GRAYSCALE)
                if edge_mask is not None:
                    edge_mask = cv2.resize(edge_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Load adjacency matrix
        # Support both 'adjacency_filename' and 'adjacency_matrix_filename' for different dataset formats
        adj_filename = sample_metadata.get('adjacency_filename') or sample_metadata.get('adjacency_matrix_filename')
        if adj_filename is None:
            raise KeyError(f"Metadata must contain either 'adjacency_filename' or 'adjacency_matrix_filename'. Available keys: {list(sample_metadata.keys())}")
        adj_path = self.adjacency_dir / adj_filename
        adj_matrix = np.load(adj_path)
        
        # Load node coordinates and edge pairs if available
        node_coords = None
        gt_node_coords = None
        edge_pairs = None
        
        if 'points_filename' in sample_metadata:
            points_path = self.points_dir / sample_metadata['points_filename']
            if points_path.exists():
                with open(points_path, 'rb') as f:
                    points_data = pickle.load(f)
                    if isinstance(points_data, dict):
                        # Try multiple possible keys for node coordinates
                        if 'coords' in points_data:
                            node_coords = points_data['coords']
                            gt_node_coords = points_data['coords']
                        elif 'graph_points' in points_data:
                            node_coords = points_data['graph_points']
                            gt_node_coords = points_data['graph_points']
                        
                        # Extract edge pairs for pair-based edge loss (permutation-invariant)
                        if 'pairs' in points_data and 'valid' in points_data:
                            pairs = points_data['pairs']  # [N, 32, 2]
                            valid = points_data['valid']  # [N, 32]
                            
                            # Extract unique edge pairs
                            edge_pairs_set = set()
                            for i in range(pairs.shape[0]):
                                for j in range(pairs.shape[1]):
                                    if valid[i, j]:
                                        pair = tuple(sorted([int(pairs[i, j, 0]), int(pairs[i, j, 1])]))
                                        if pair[0] != pair[1]:  # No self-loops
                                            edge_pairs_set.add(pair)
                            
                            if edge_pairs_set:
                                edge_pairs = np.array(list(edge_pairs_set), dtype=np.int32)
                    elif isinstance(points_data, np.ndarray):
                        node_coords = points_data
                        gt_node_coords = points_data
        
        # Apply augmentation after all supervision is loaded so transforms stay aligned
        if self.augment:
            rgb, node_mask, edge_mask, node_coords = self._augment_sample(
                rgb,
                node_mask,
                edge_mask,
                node_coords
            )
            if node_coords is not None:
                gt_node_coords = node_coords.copy()
        
        # Compute reachability matrix (transitive closure)
        reachability_matrix = self._compute_reachability(adj_matrix)
        
        # Convert to PyTorch tensors
        # Make a copy to avoid negative stride issues
        rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        
        if node_mask is not None:
            node_mask_tensor = torch.from_numpy(node_mask).float() / 255.0  # [H, W]
        else:
            node_mask_tensor = torch.zeros(self.image_size, self.image_size)
        
        if edge_mask is not None:
            edge_mask_tensor = torch.from_numpy(edge_mask).float() / 255.0  # [H, W]
        else:
            edge_mask_tensor = torch.zeros(self.image_size, self.image_size)
        
        adj_matrix_tensor = torch.from_numpy(adj_matrix).float()
        reachability_tensor = torch.from_numpy(reachability_matrix).float()
        
        # Convert edge_pairs to tensor if available
        edge_pairs_tensor = None
        if edge_pairs is not None:
            edge_pairs_tensor = torch.from_numpy(edge_pairs).long()
        
        # Convert gt_node_coords to tensor if available
        gt_node_coords_tensor = None
        if gt_node_coords is not None:
            gt_node_coords_tensor = torch.from_numpy(gt_node_coords).float()
        
        return {
            'image': rgb_tensor,
            'node_mask': node_mask_tensor,
            'edge_mask': edge_mask_tensor,
            'adjacency_matrix': adj_matrix_tensor,
            'reachability_matrix': reachability_tensor,  # [N, N]
            'edge_pairs': edge_pairs_tensor,  # [E, 2] or None - for pair-based edge loss
            'gt_node_coords': gt_node_coords_tensor,  # [N, 2] or None - for node matching
            'image_filename': sample_metadata['image_filename'],
            'num_nodes': adj_matrix.shape[0],
            'num_edges': int(adj_matrix.sum() / 2),  # Undirected graph
        }
    
    def _compute_reachability(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Compute reachability matrix (transitive closure).
        
        Args:
            adj_matrix: Adjacency matrix [N, N]
        
        Returns:
            Reachability matrix [N, N]
        """
        # Use matrix exponentiation: R = (I + A)^k for large k
        # For efficiency, use iterative method
        N = adj_matrix.shape[0]
        reachability = adj_matrix.copy().astype(np.float32)
        
        # Iterative: R = R | (R @ A) until convergence
        for _ in range(N):  # At most N iterations
            prev_reachability = reachability.copy()
            # Ensure boolean operations work correctly
            new_reach = (reachability.astype(np.float32) @ adj_matrix.astype(np.float32)) > 0
            reachability = (reachability.astype(bool) | new_reach.astype(bool)).astype(np.float32)
            if np.array_equal(reachability, prev_reachability):
                break
        
        # Add self-loops (identity)
        reachability = (reachability.astype(bool) | np.eye(N, dtype=bool)).astype(np.float32)
        
        return reachability.astype(np.float32)
    
    def _augment_sample(
        self,
        rgb: np.ndarray,
        node_mask: Optional[np.ndarray],
        edge_mask: Optional[np.ndarray],
        node_coords: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply random augmentations to the image and keep supervision aligned."""
        # Work on copies to avoid modifying cached arrays
        if node_coords is not None:
            node_coords = node_coords.copy()
        if node_mask is not None:
            node_mask = node_mask.copy()
        if edge_mask is not None:
            edge_mask = edge_mask.copy()
        
        # Random contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            rgb = np.clip(alpha * rgb, 0, 255).astype(np.uint8)
        
        # Random brightness
        if random.random() < 0.5:
            beta = random.uniform(-20, 20)
            rgb = np.clip(rgb + beta, 0, 255).astype(np.uint8)
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, rgb.shape)
            rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if random.random() < 0.5:
            rgb = np.fliplr(rgb)
            if node_mask is not None:
                node_mask = np.fliplr(node_mask)
            if edge_mask is not None:
                edge_mask = np.fliplr(edge_mask)
            if node_coords is not None and len(node_coords) > 0:
                node_coords[:, 0] = (self.image_size - 1) - node_coords[:, 0]
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            rgb = np.rot90(rgb, k)
            if node_mask is not None:
                node_mask = np.rot90(node_mask, k)
            if edge_mask is not None:
                edge_mask = np.rot90(edge_mask, k)
            if node_coords is not None and len(node_coords) > 0:
                # Apply k counter-clockwise 90° rotations
                for _ in range(k):
                    x = node_coords[:, 0].copy()
                    y = node_coords[:, 1].copy()
                    node_coords[:, 0] = y
                    node_coords[:, 1] = (self.image_size - 1) - x
        
        return rgb.copy(), (
            node_mask.copy() if node_mask is not None else None
        ), (
            edge_mask.copy() if edge_mask is not None else None
        ), node_coords


def collate_fn(batch: list) -> Dict[str, Any]:
    """
    Custom collate function for variable-length node data.
    
    Args:
        batch: List of samples
    
    Returns:
        Batched data
    """
    # Images and masks can be batched normally
    images = torch.stack([item['image'] for item in batch])
    node_masks = torch.stack([item['node_mask'] for item in batch])
    edge_masks = torch.stack([item['edge_mask'] for item in batch])
    
    # Adjacency and reachability matrices have different sizes, keep as list
    adjacency_matrices = [item['adjacency_matrix'] for item in batch]
    reachability_matrices = [item['reachability_matrix'] for item in batch]
    
    # Edge pairs and GT node coordinates (for pair-based edge loss)
    # Keep as list since they have variable sizes
    edge_pairs_list = [item['edge_pairs'] for item in batch]  # List of [E_i, 2] or None
    gt_node_coords_list = [item['gt_node_coords'] for item in batch]  # List of [N_i, 2] or None
    
    # Metadata
    image_filenames = [item['image_filename'] for item in batch]
    num_nodes = [item['num_nodes'] for item in batch]
    num_edges = [item['num_edges'] for item in batch]
    
    return {
        'image': images,
        'node_mask': node_masks,
        'edge_mask': edge_masks,
        'adjacency_matrix': adjacency_matrices,
        'reachability_matrix': reachability_matrices,  # List of [N, N]
        'edge_pairs': edge_pairs_list,  # List of [E_i, 2] or None - for pair-based edge loss
        'gt_node_coords': gt_node_coords_list,  # List of [N_i, 2] or None - for node matching
        'image_filename': image_filenames,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
    }
