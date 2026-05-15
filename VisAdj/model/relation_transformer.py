"""
Relation Micro-Transformer

Lightweight transformer for reasoning over candidate edges.
Uses features: [l_i || l_j || g_i || g_j || φ_ij || z*]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RelationMicroTransformer(nn.Module):
    """
    Relation Micro-Transformer for edge prediction.
    
    Processes candidate edges using set-based reasoning,
    enforcing junction structure consistency.
    Features: [l_i || l_j || g_i || g_j || φ_ij || z* || G'_i || G'_j]
    where:
    - l_i, l_j: Local node descriptors (enhanced with cross-attention)
    - g_i, g_j: Global node descriptors (enhanced with cross-attention)
    - φ_ij: Normalized direction (2D) and normalized distance (1D)
    - z*: Global topology embedding
    - G'_i, G'_j: Processed global features sampled at node coordinates
    """
    
    def __init__(
        self,
        node_feature_dim: int = 128,
        z_star_dim: int = 256,
        edge_feature_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        g_prime_dim: int = 256,  # Dimension of G_prime features
    ):
        """
        Args:
            node_feature_dim: Dimension of l_i and g_i
            z_star_dim: Dimension of z_star
            edge_feature_dim: Output edge feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            g_prime_dim: Dimension of G_prime (processed global features)
        """
        super().__init__()
        
        # Cross-attention for better node feature composition
        # This computes interactions between node features before concatenation
        self.node_cross_attn = nn.MultiheadAttention(
            embed_dim=node_feature_dim,
            num_heads=max(1, num_heads // 2),  # Use fewer heads for node features
            dropout=dropout,
            batch_first=True,
        )
        self.node_feature_proj = nn.Linear(node_feature_dim * 2, node_feature_dim)  # Project concatenated features
        
        # Edge feature composition: [l_i || l_j || g_i || g_j || φ_ij || z* || G'_i || G'_j]
        # l_i: node_feature_dim, l_j: node_feature_dim
        # g_i: node_feature_dim, g_j: node_feature_dim
        # φ_ij: 3 (normalized direction [2] + normalized distance [1])
        # z*: z_star_dim
        # G'_i: g_prime_dim, G'_j: g_prime_dim (sampled from G_prime at node coordinates)
        self.g_prime_dim = g_prime_dim
        edge_feature_raw_dim = node_feature_dim * 4 + 3 + z_star_dim + g_prime_dim * 2
        
        # MLP to project edge features
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(edge_feature_raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_feature_dim),
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=edge_feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        l_i: torch.Tensor,  # Local descriptors (detached) [B, N, D]
        g_i: torch.Tensor,  # Global descriptors [B, N, D]
        node_coords: torch.Tensor,  # Node coordinates [B, N, 2]
        z_star: torch.Tensor,  # Global embedding [B, D_z]
        candidate_mask: torch.Tensor,  # Candidate mask [B, N, N]
        G_prime: Optional[torch.Tensor] = None,  # Processed global features [B, C_G, H, W]
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict edge probabilities from edge features.
        
        Args:
            l_i: Local node descriptors (detached) [B, N, D]
            g_i: Global node descriptors [B, N, D]
            node_coords: Node coordinates [B, N, 2] (in 32×32 grid space)
            z_star: Global embedding [B, D_z]
            candidate_mask: Candidate edge mask [B, N, N]
            G_prime: Processed global features [B, C_G, H, W] (H=W=8 for global resolution)
            valid_mask: Valid node mask [B, N] (optional)
        
        Returns:
            edge_logits: Edge logits [B, N, N]
        """
        B, N, D_node = l_i.shape
        
        # Cross-attention between node features for better composition
        # Compute interactions between local and global descriptors
        # Query: l_i, Key/Value: g_i (or vice versa)
        l_i_enhanced, _ = self.node_cross_attn(l_i, g_i, g_i)  # [B, N, D_node]
        g_i_enhanced, _ = self.node_cross_attn(g_i, l_i, l_i)  # [B, N, D_node]
        
        # Concatenate and project to maintain dimension
        l_i_composed = self.node_feature_proj(torch.cat([l_i, l_i_enhanced], dim=-1))  # [B, N, D_node]
        g_i_composed = self.node_feature_proj(torch.cat([g_i, g_i_enhanced], dim=-1))  # [B, N, D_node]
        
        # Build edge features: [l_i || l_j || g_i || g_j || φ_ij || z* || G'_i || G'_j]
        # Expand for pairwise combinations
        l_i_expanded = l_i_composed.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        l_j_expanded = l_i_composed.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        g_i_expanded = g_i_composed.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        g_j_expanded = g_i_composed.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        
        # Sample G_prime at node coordinates
        # node_coords are in 32×32 grid space, need to scale to 8×8 (global resolution)
        if G_prime is not None:
            B_prime, C_G, H_global, W_global = G_prime.shape  # [B, C_G, 8, 8]
            # Scale coordinates from 32×32 to 8×8
            # Similar to node_detector: scale by (W_global-1)/(W_local-1)
            node_coords_global = node_coords.clone()
            W_local, H_local = 32.0, 32.0  # Local grid size
            if W_local > 1:
                node_coords_global[:, :, 0] = node_coords[:, :, 0] * (W_global - 1) / (W_local - 1)  # x
            else:
                node_coords_global[:, :, 0] = 0.0
            if H_local > 1:
                node_coords_global[:, :, 1] = node_coords[:, :, 1] * (H_global - 1) / (H_local - 1)  # y
            else:
                node_coords_global[:, :, 1] = 0.0
            
            # Normalize to [-1, 1] for grid_sample (matching node_detector pattern)
            node_coords_normalized = node_coords_global.clone()
            node_coords_normalized[:, :, 0] = (node_coords_global[:, :, 0] / (W_global - 1)) * 2.0 - 1.0  # x
            node_coords_normalized[:, :, 1] = (node_coords_global[:, :, 1] / (H_global - 1)) * 2.0 - 1.0  # y
            
            # grid_sample expects [B, N, 1, 2] format
            node_coords_grid = node_coords_normalized.unsqueeze(2)  # [B, N, 1, 2]
            
            # Sample G_prime at node coordinates
            # G_prime: [B, C_G, 8, 8], node_coords_grid: [B, N, 1, 2]
            G_prime_sampled = F.grid_sample(
                G_prime,
                node_coords_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B, C_G, N, 1]
            G_prime_sampled = G_prime_sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C_G]
            
            # Expand for pairwise combinations
            G_prime_i_expanded = G_prime_sampled.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, C_G]
            G_prime_j_expanded = G_prime_sampled.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, C_G]
        else:
            # If G_prime not provided, use zeros
            G_prime_i_expanded = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
            G_prime_j_expanded = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
        
        # Spatial features φ_ij (normalized relative position + distance)
        coords_i = node_coords.unsqueeze(2)  # [B, N, 1, 2]
        coords_j = node_coords.unsqueeze(1)  # [B, 1, N, 2]
        spatial = coords_j - coords_i  # [B, N, N, 2]
        
        # Compute distance (in grid cells, since coords are in 32×32 grid space)
        distance = torch.norm(spatial, dim=-1, keepdim=True)  # [B, N, N, 1]
        max_distance = 32.0 * (2.0 ** 0.5)  # Max distance in 32×32 grid: sqrt(32^2 + 32^2) ≈ 45.25
        distance_normalized = distance / (max_distance + 1e-8)  # [B, N, N, 1] - normalized to [0, 1]
        
        # Normalize direction (unit vector)
        spatial_norm = distance + 1e-8  # [B, N, N, 1]
        spatial_normalized = spatial / spatial_norm  # [B, N, N, 2] - normalized direction
        
        # Concatenate: [normalized_direction (2), normalized_distance (1)]
        spatial_features = torch.cat([spatial_normalized, distance_normalized], dim=-1)  # [B, N, N, 3]
        
        # z* (broadcast)
        z_star_expanded = z_star.unsqueeze(1).unsqueeze(2).expand(-1, N, N, -1)  # [B, N, N, D_z]
        
        # Concatenate edge features: [l_i || l_j || g_i || g_j || φ_ij || z* || G'_i || G'_j]
        edge_features_raw = torch.cat([
            l_i_expanded,
            l_j_expanded,
            g_i_expanded,
            g_j_expanded,
            spatial_features,  # [B, N, N, 3] - normalized direction + distance
            z_star_expanded,
            G_prime_i_expanded,  # [B, N, N, C_G] - sampled from G_prime at node i
            G_prime_j_expanded,  # [B, N, N, C_G] - sampled from G_prime at node j
        ], dim=-1)  # [B, N, N, D_raw]
        
        # Project through MLP
        edge_features = self.edge_feature_mlp(edge_features_raw)  # [B, N, N, D_edge]
        
        # Flatten for transformer: [B, N*N, D_edge]
        B, N, _, D_edge = edge_features.shape
        edge_features_flat = edge_features.reshape(B, N * N, D_edge)
        
        # Create padding mask for invalid edges
        candidate_mask_flat = candidate_mask.reshape(B, N * N)
        padding_mask = (candidate_mask_flat < 0.5)  # [B, N*N]
        
        # Handle empty sequences (all edges are invalid)
        for b in range(B):
            if padding_mask[b].all():
                # All edges invalid, make first edge valid
                padding_mask[b, 0] = False
                candidate_mask_flat[b, 0] = 1.0
        
        # Process with transformer (self-attention reasons over edges around each node)
        edge_features_processed = self.transformer(
            edge_features_flat,
            src_key_padding_mask=padding_mask
        )  # [B, N*N, D_edge]
        
        # Predict edge logits
        edge_logits_flat = self.edge_head(edge_features_processed).squeeze(-1)  # [B, N*N]
        
        # Reshape back to [B, N, N]
        edge_logits = edge_logits_flat.reshape(B, N, N)
        
        # Mask out non-candidate edges
        edge_logits = edge_logits * candidate_mask
        
        # Symmetrize: p_ij ← (p_ij + p_ji)/2
        edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0
        
        return edge_logits
