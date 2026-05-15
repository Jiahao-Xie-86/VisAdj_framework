"""
KNN-based Neighbor Sampler (replacement for ASNS)

Simple distance-based neighbor selection using radius and top-k.
No learnable parameters, deterministic, stable gradients.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class KNNNeighborSampler(nn.Module):
    """
    KNN-based Neighbor Sampler (similar to SAM-road).
    
    Selects neighbors based on spatial distance:
    - Compute pairwise distances between nodes
    - For each node, select neighbors within radius
    - Take top-k nearest neighbors
    - No learnable parameters, fully differentiable
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        neighbor_radius: float = 256.0,  # In pixel space (32x32 grid)
    ):
        """
        Args:
            k_neighbors: Maximum number of neighbors to select per node
            neighbor_radius: Maximum distance (in pixels) for neighbor selection
        """
        super().__init__()
        
        self.k_neighbors = k_neighbors
        self.neighbor_radius = neighbor_radius
    
    def forward(
        self,
        l_i: torch.Tensor,  # Local descriptors (unused, kept for compatibility) [B, N, D]
        g_j: torch.Tensor,  # Global descriptors (unused, kept for compatibility) [B, N, D]
        node_coords: torch.Tensor,  # Node coordinates [B, N, 2] in Local grid space (32×32)
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate candidate edge mask using deterministic KNN.
        
        Returns:
            candidate_mask: Binary mask [B, N, N] (Top-K per row)
            attention_weights: Uniform weights over selected neighbors [B, N, N]
            attention_scores: Distance-based scores (higher = closer) [B, N, N]
        """
        B, N, _ = node_coords.shape
        device = node_coords.device
        
        # Compute pairwise distances [B, N, N]
        # node_coords: [B, N, 2]
        coords_i = node_coords.unsqueeze(2)  # [B, N, 1, 2]
        coords_j = node_coords.unsqueeze(1)  # [B, 1, N, 2]
        distances = torch.sqrt(((coords_i - coords_j) ** 2).sum(dim=-1) + 1e-8)  # [B, N, N]
        
        # Remove self-loops: set diagonal to large value
        eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]
        distances = distances.masked_fill(eye_mask, float('inf'))
        
        # Apply valid mask if provided
        if valid_mask is not None:
            invalid_mask = ~valid_mask  # [B, N]
            # Mask invalid query rows and key columns
            distances = distances.masked_fill(invalid_mask.unsqueeze(1).unsqueeze(2), float('inf'))
            distances = distances.masked_fill(invalid_mask.unsqueeze(1).unsqueeze(3), float('inf'))
        
        # For each node, select neighbors within radius
        # Then take top-k nearest
        within_radius = distances <= self.neighbor_radius  # [B, N, N]
        
        # For nodes with neighbors within radius, select top-k
        # For nodes with no neighbors, select top-k anyway (will be inf, but that's ok)
        # Set distances outside radius to inf so they're not selected
        distances_masked = torch.where(
            within_radius,
            distances,
            torch.full_like(distances, float('inf'))
        )
        
        # Get top-k nearest neighbors per node
        top_k_values, top_k_indices = distances_masked.topk(
            k=min(self.k_neighbors, N),
            dim=-1,
            largest=False  # Smallest distances (nearest neighbors)
        )  # [B, N, K]
        
        # Create binary candidate mask
        candidate_mask = torch.zeros(B, N, N, device=device, dtype=torch.float32)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        node_indices = torch.arange(N, device=device).unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        
        # Set top-k positions to 1
        candidate_mask[batch_indices, node_indices, top_k_indices] = 1.0
        
        # Attention weights: uniform distribution over selected neighbors
        neighbor_counts = candidate_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        attention_weights = candidate_mask / neighbor_counts
        
        # Attention scores: negative distances so nearer nodes have higher scores
        large_negative = torch.full_like(distances_masked, -1e9)
        attention_scores = torch.where(
            torch.isfinite(distances_masked),
            -distances_masked,
            large_negative
        )
        
        return candidate_mask, attention_weights, attention_scores
    
    def compute_coverage_loss(
        self,
        candidate_mask: torch.Tensor,
        target_adj: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        attention_scores: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Coverage loss: every GT edge must be selectable.
        
        For KNN, this is just a metric - it measures how many GT edges are
        within the KNN radius. Since KNN is fixed, this loss doesn't need gradients.
        It's kept for compatibility and monitoring.
        
        Args:
            candidate_mask: Candidate mask [B, N, N]
            target_adj: Target adjacency [B, N, N] or list
            attention_weights: Unused (kept for compatibility)
        
        Returns:
            Coverage loss (metric only, detached from graph)
        """
        # Handle list
        if isinstance(target_adj, list):
            max_n = max(adj.shape[0] for adj in target_adj)
            B = len(target_adj)
            target_tensor = torch.zeros(B, max_n, max_n, device=candidate_mask.device)
            for i, adj in enumerate(target_adj):
                n = adj.shape[0]
                if isinstance(adj, torch.Tensor):
                    target_tensor[i, :n, :n] = adj.to(candidate_mask.device).float()
                else:
                    import numpy as np
                    target_tensor[i, :n, :n] = torch.from_numpy(adj).float().to(candidate_mask.device)
            target_adj = target_tensor
        
        # Ensure same size
        min_n = min(candidate_mask.shape[1], target_adj.shape[1])
        candidate_mask = candidate_mask[:, :min_n, :min_n]
        target_adj = target_adj[:, :min_n, :min_n]
        
        # Compute coverage: GT edges that are not in candidate mask
        gt_edge_mask = target_adj > 0.5  # [B, N, N]
        if gt_edge_mask.sum() > 0:
            missing_edges = gt_edge_mask & (candidate_mask < 0.5)  # [B, N, N]
            # Coverage loss: metric only (detached)
            coverage_loss = missing_edges.float().mean().detach()
        else:
            coverage_loss = torch.tensor(0.0, device=candidate_mask.device, requires_grad=False)
        
        return coverage_loss
    
    def compute_budget_loss(
        self,
        candidate_mask: torch.Tensor,
        k_tgt: int = 8,
        attention_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Budget loss: encourage compact neighborhoods.
        
        For KNN, this is just a metric - KNN always selects exactly k_neighbors.
        Since KNN is fixed, this loss doesn't need gradients.
        It's kept for compatibility and monitoring.
        
        Args:
            candidate_mask: Candidate mask [B, N, N]
            k_tgt: Target number of neighbors per node
            attention_weights: Unused (kept for compatibility)
        
        Returns:
            Budget loss (metric only, detached from graph)
        """
        k_per_node = candidate_mask.sum(dim=-1)  # [B, N]
        
        # Hinge loss: only penalize when k_per_node > k_tgt
        k_target = torch.full_like(k_per_node, k_tgt, dtype=k_per_node.dtype)
        excess = k_per_node - k_target  # [B, N]
        hinge_loss_per_node = torch.nn.functional.relu(excess) ** 2  # [B, N]
        
        # Normalize and detach (metric only)
        budget_loss = (hinge_loss_per_node.mean() / (k_tgt ** 2 + 1e-8)).detach()
        
        return budget_loss

