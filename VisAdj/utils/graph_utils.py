"""
Utility functions for graph operations
"""

import torch
import numpy as np
import networkx as nx
from typing import Tuple, Optional


def adjacency_to_graph(adj_matrix: np.ndarray) -> nx.Graph:
    """Convert adjacency matrix to NetworkX graph."""
    return nx.from_numpy_array(adj_matrix.astype(float))


def graph_to_adjacency(graph: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to adjacency matrix."""
    return nx.to_numpy_array(graph)


def extract_node_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    image_size: Tuple[int, int] = (512, 512)
) -> torch.Tensor:
    """
    Extract features at node coordinates using bilinear sampling.
    
    Args:
        features: Feature map [B, D, H, W]
        coords: Node coordinates [B, N, 2] in pixel space
        image_size: Original image size (H, W)
    
    Returns:
        node_features: Features at node locations [B, N, D]
    """
    B, D, H, W = features.shape
    _, N, _ = coords.shape
    
    # Normalize coordinates to [-1, 1]
    coords_norm = coords.clone()
    coords_norm[:, :, 0] = (coords[:, :, 0] / image_size[1]) * 2.0 - 1.0  # x
    coords_norm[:, :, 1] = (coords[:, :, 1] / image_size[0]) * 2.0 - 1.0  # y
    
    # Reshape for grid_sample
    coords_grid = coords_norm.unsqueeze(2)  # [B, N, 1, 2]
    
    # Sample features
    node_features = torch.nn.functional.grid_sample(
        features,
        coords_grid,
        mode='bilinear',
        align_corners=False
    )  # [B, D, N, 1]
    
    # Reshape to [B, N, D]
    node_features = node_features.squeeze(-1).permute(0, 2, 1)
    
    return node_features


def compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between coordinates.
    
    Args:
        coords: Coordinates [B, N, 2]
    
    Returns:
        distances: Pairwise distances [B, N, N]
    """
    return torch.cdist(coords, coords)


def pad_to_max_nodes(
    coords: torch.Tensor,
    features: torch.Tensor,
    max_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad node coordinates and features to maximum number of nodes.
    
    Args:
        coords: Node coordinates [B, N, 2]
        features: Node features [B, N, D]
        max_nodes: Maximum number of nodes
    
    Returns:
        padded_coords: Padded coordinates [B, max_nodes, 2]
        padded_features: Padded features [B, max_nodes, D]
        valid_mask: Valid mask [B, max_nodes]
    """
    B, N, _ = coords.shape
    device = coords.device
    
    if N >= max_nodes:
        return coords[:, :max_nodes], features[:, :max_nodes], torch.ones(B, max_nodes, dtype=torch.bool, device=device)
    
    # Pad
    pad_size = max_nodes - N
    padded_coords = torch.cat([
        coords,
        torch.zeros(B, pad_size, 2, device=device)
    ], dim=1)
    
    padded_features = torch.cat([
        features,
        torch.zeros(B, pad_size, features.shape[2], device=device)
    ], dim=1)
    
    valid_mask = torch.cat([
        torch.ones(B, N, dtype=torch.bool, device=device),
        torch.zeros(B, pad_size, dtype=torch.bool, device=device)
    ], dim=1)
    
    return padded_coords, padded_features, valid_mask

