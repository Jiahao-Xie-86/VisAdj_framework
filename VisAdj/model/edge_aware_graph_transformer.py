"""
Edge-Aware Graph Transformer for SAM Graph Split

Adapted from the successful toy RGB classifier model.
Uses line graph attention and geometric positional encoding for edge detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class EdgeAwareGraphTransformer(nn.Module):
    """
    Edge-aware Graph Transformer with line graph structure and geometric positional encoding.
    
    Adapted for sam_graph_split framework:
    - Uses sam_graph_split's feature pipeline (l_i, g_i, images, etc.)
    - Supports RGB path sampling with coordinate conversion
    - Includes collinearity check in spatial features
    - Uses line graph attention for sparse, efficient computation
    
    Key features:
    1. Line graph attention: Edges only attend to edges sharing a common node (sparse attention)
    2. Geometric positional encoding: Uses actual node coordinates instead of indices
    3. Multi-modal features: RGB path features + spatial features + node features
    """
    
    def __init__(
        self,
        node_feature_dim: int = 128,
        z_star_dim: int = 256,  # Kept for interface compatibility, not used
        edge_feature_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        image_size: int = 512,
        heatmap_resolution: int = 32,
        use_rgb_path_features: bool = True,
        rgb_path_num_samples: int = 9,
        rgb_feature_dim: int = 32,
        rgb_sequence_model: str = 'transformer',  # 'mean', '1d_cnn', or 'transformer'
        rgb_seq_layers: int = 2,
        rgb_seq_heads: int = 4,
        rgb_neighborhood_aggregation: str = 'center',  # 'center', 'mean', or 'min_r_min_g_max_b'
        rgb_neighborhood_radius: float = 4.0,  # Radius in pixels for RGB neighborhood sampling (default: 4.0)
        g_prime_dim: int = 256,  # Dimension of G_prime features (topology features)
        topology_feature_dim: int = 0,  # Output dimension for topology features (0 = disabled)
        spatial_feature_dim: int = 4,  # Output dimension for spatial features (after projection)
    ):
        """
        Args:
            node_feature_dim: Dimension of l_i and g_i
            z_star_dim: Dimension of z_star (kept for interface compatibility, not used)
            edge_feature_dim: Output edge feature dimension
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_positional_encoding: Whether to use geometric positional encoding
            image_size: Image size in pixels (for coordinate normalization)
            heatmap_resolution: Heatmap resolution (for coordinate conversion)
            use_rgb_path_features: Whether to sample RGB along edge paths
            rgb_path_num_samples: Number of RGB samples along path
            rgb_feature_dim: Dimension after projecting RGB features
            rgb_sequence_model: How to process RGB sequence ('mean', '1d_cnn', 'transformer')
            rgb_seq_layers: Number of layers for RGB sequence transformer
            rgb_seq_heads: Number of heads for RGB sequence transformer
            rgb_neighborhood_aggregation: How to aggregate RGB from neighborhood ('center', 'mean', 'min_r_min_g_max_b')
            rgb_neighborhood_radius: Radius in pixels for RGB neighborhood sampling (default: 4.0)
            g_prime_dim: Dimension of G_prime features (topology features from processed global features)
            topology_feature_dim: Output dimension for topology features after projection (0 = disabled, default: 0)
            spatial_feature_dim: Output dimension for spatial features after projection (default: 4)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding
        self.image_size = image_size
        self.heatmap_resolution = heatmap_resolution
        self.use_rgb_path_features = use_rgb_path_features
        self.rgb_path_num_samples = rgb_path_num_samples
        self.rgb_sequence_model = rgb_sequence_model
        self.rgb_neighborhood_aggregation = rgb_neighborhood_aggregation
        self.rgb_feature_dim = rgb_feature_dim if use_rgb_path_features else 0
        self.rgb_neighborhood_radius = rgb_neighborhood_radius
        self.g_prime_dim = g_prime_dim
        self.topology_feature_dim = topology_feature_dim
        self.spatial_feature_dim = spatial_feature_dim
        
        # RGB path feature extraction
        if use_rgb_path_features:
            # Project raw RGB (3 channels) to feature space
            self.rgb_proj = nn.Sequential(
                nn.Linear(3, rgb_feature_dim),
                nn.LayerNorm(rgb_feature_dim),
                nn.GELU(),
            )
            
            # Sequence processing module for RGB samples along path
            if rgb_sequence_model == '1d_cnn':
                self.rgb_seq_conv1 = nn.Conv1d(
                    in_channels=rgb_feature_dim,
                    out_channels=rgb_feature_dim,
                    kernel_size=3,
                    padding=1,
                )
                self.rgb_seq_ln1 = nn.LayerNorm(rgb_feature_dim)
                self.rgb_seq_conv2 = nn.Conv1d(
                    in_channels=rgb_feature_dim,
                    out_channels=rgb_feature_dim,
                    kernel_size=3,
                    padding=1,
                )
                self.rgb_seq_ln2 = nn.LayerNorm(rgb_feature_dim)
                self.rgb_seq_dropout = nn.Dropout(dropout)
                self.rgb_seq_pool = nn.AdaptiveAvgPool1d(1)
            elif rgb_sequence_model == 'transformer':
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=rgb_feature_dim,
                    nhead=rgb_seq_heads,
                    dim_feedforward=rgb_feature_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.rgb_seq_processor = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=rgb_seq_layers,
                )
            elif rgb_sequence_model in {'mean', 'max'}:
                self.rgb_seq_processor = None
            else:
                raise ValueError(f"Unsupported rgb_sequence_model: {rgb_sequence_model}")
        else:
            self.rgb_proj = None
            self.rgb_seq_processor = None
        
        # Spatial feature dimension: distance (1) + direction (2) + collinearity (1) = 4
        # Will be projected to spatial_feature_dim (or used raw if spatial_feature_dim == 4)
        spatial_raw_dim = 4
        
        # Project spatial features (or use raw if dimension matches)
        if spatial_feature_dim == spatial_raw_dim:
            # No projection needed, use raw features
            self.spatial_proj = None
        else:
            # Project spatial features to different dimension
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_raw_dim, spatial_feature_dim),
                nn.LayerNorm(spatial_feature_dim),
                nn.GELU(),
            )
        
        # Topology features: G_prime_i and G_prime_j sampled from G_prime at node locations
        # Will be projected from 2*g_prime_dim to topology_feature_dim (if enabled)
        if topology_feature_dim > 0:
            topology_raw_dim = 2 * g_prime_dim  # G_prime_i and G_prime_j for each edge
            
            # Project topology features to lower dimension
            self.topology_proj = nn.Sequential(
                nn.Linear(topology_raw_dim, topology_feature_dim),
                nn.LayerNorm(topology_feature_dim),
                nn.GELU(),
            )
        else:
            self.topology_proj = None
        
        # Edge feature dimension: RGB features + spatial features + topology features (if enabled)
        edge_input_dim = self.rgb_feature_dim + spatial_feature_dim + (topology_feature_dim if topology_feature_dim > 0 else 0)
        
        # Project edge features to hidden dimension
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Geometric positional encoding for edges
        if use_positional_encoding:
            self.pos_encoding_dim = 4  # (x_i, y_i, x_j, y_j)
            self.pos_encoding_proj = nn.Sequential(
                nn.Linear(self.pos_encoding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.pos_encoding_proj = None
        
        # Transformer layers with line graph attention support
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Cache for base attention mask (only depends on N, not batch)
        self._cached_mask_N = None
        self._cached_mask = None
    
    def _create_line_graph_attention_mask_base(
        self,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create base attention mask for line graph structure (vectorized, cached).
        
        In a line graph, edge (i,j) only attends to edges that share a node:
        - Edges (i,k) for all k (edges sharing node i)
        - Edges (k,j) for all k (edges sharing node j)
        
        Returns:
            attention_mask: [N*N, N*N] - True means mask out (don't attend)
        """
        # Check cache
        if self._cached_mask_N == N and self._cached_mask is not None:
            if self._cached_mask.device == device:
                return self._cached_mask
            else:
                return self._cached_mask.to(device)
        
        # Vectorized mask creation
        edge_indices = torch.arange(N * N, device=device)  # [N*N]
        i_indices = edge_indices // N  # [N*N] - node i for each edge
        j_indices = edge_indices % N   # [N*N] - node j for each edge
        
        # Expand for all pairs
        i_i = i_indices.unsqueeze(1)  # [N*N, 1]
        j_j = j_indices.unsqueeze(1)  # [N*N, 1]
        i_i_prime = i_indices.unsqueeze(0)  # [1, N*N]
        j_j_prime = j_indices.unsqueeze(0)  # [1, N*N]
        
        # Two edges can attend if they share a node: (i==i') OR (j==j')
        share_node_i = (i_i == i_i_prime)  # [N*N, N*N]
        share_node_j = (j_j == j_j_prime)  # [N*N, N*N]
        can_attend = share_node_i | share_node_j  # [N*N, N*N]
        
        # attention_mask: True = mask out (don't attend), False = can attend
        attention_mask = ~can_attend
        
        # Cache the result
        self._cached_mask_N = N
        self._cached_mask = attention_mask
        
        return attention_mask
    
    def _create_line_graph_attention_mask(
        self,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask for line graph structure (batch agnostic).
        
        Returns:
            attention_mask: [1, N*N, N*N] - True means mask out (don't attend)
        """
        attention_mask = self._create_line_graph_attention_mask_base(N, device)
        return attention_mask.unsqueeze(0)
    
    def _sample_rgb_along_path(
        self,
        images: torch.Tensor,  # [B, 3, H, W] RGB images
        node_coords: torch.Tensor,  # [B, N, 2] in heatmap space OR image space
        num_samples: int,
        coords_in_image_space: bool = False,  # NEW: Flag to indicate coordinate space
    ) -> torch.Tensor:
        """
        Sample RGB colors at intermediate points along edge paths.
        
        Args:
            images: RGB images [B, 3, H_img, W_img]
            node_coords: Node coordinates [B, N, 2] in heatmap space (32×32) or image space (512×512)
            num_samples: Number of intermediate points to sample
            coords_in_image_space: If True, node_coords are already in image space (skip conversion)
        
        Returns:
            rgb_samples: [B, N, N, num_samples, 3] - RGB at each sample point
        """
        B, N, _ = node_coords.shape
        _, H_img, W_img = images.shape[1:]
        
        # Convert node_coords to pixel space if needed
        if coords_in_image_space:
            # Already in image space, use directly
            node_coords_pixel = node_coords
        else:
            # Convert from heatmap space (32×32) to pixel space (512×512)
            scale_factor = self.image_size / self.heatmap_resolution  # 512 / 32 = 16 or 512 / 64 = 8
            node_coords_pixel = node_coords * scale_factor  # [B, N, 2] in pixel space
        
        # Sample intermediate points along path (matching toy classifier: 20% → 80%)
        t = torch.linspace(0.2, 0.8, num_samples, device=node_coords.device)
        coords_i = node_coords_pixel.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 2]
        coords_j = node_coords_pixel.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, 2]
        t_expanded = t.view(1, 1, 1, -1, 1)  # [1, 1, 1, num_samples, 1]
        edge_coords_pixel = (1 - t_expanded) * coords_i + t_expanded * coords_j  # [B, N, N, num_samples, 2]
        
        # Normalize to [-1, 1] for grid_sample
        edge_coords_norm = edge_coords_pixel.clone()
        if W_img > 1:
            edge_coords_norm[:, :, :, :, 0] = (edge_coords_pixel[:, :, :, :, 0] / (W_img - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 0] = 0.0
        if H_img > 1:
            edge_coords_norm[:, :, :, :, 1] = (edge_coords_pixel[:, :, :, :, 1] / (H_img - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 1] = 0.0
        
        # Make sampling tolerant to offsets by averaging a small neighborhood around each point
        delta_px = self.rgb_neighborhood_radius  # radius in pixels (configurable, default: 4.0)
        if W_img > 1:
            offset_x = (delta_px / (W_img - 1)) * 2.0
        else:
            offset_x = 0.0
        if H_img > 1:
            offset_y = (delta_px / (H_img - 1)) * 2.0
        else:
            offset_y = 0.0
        offsets = edge_coords_norm.new_tensor([
            [0.0, 0.0],
            [offset_x, 0.0],
            [-offset_x, 0.0],
            [0.0, offset_y],
            [0.0, -offset_y],
            [offset_x, offset_y],
            [-offset_x, offset_y],
            [offset_x, -offset_y],
            [-offset_x, -offset_y],
        ])  # [K, 2]
        K = offsets.shape[0]
        edge_coords_neighborhood = edge_coords_norm.unsqueeze(-2) + offsets.view(1, 1, 1, 1, K, 2)
        edge_coords_neighborhood = edge_coords_neighborhood.clamp(min=-1.0, max=1.0)
        
        # Reshape for grid_sample: [B, N*N*num_samples*K, 1, 2]
        edge_coords_flat = edge_coords_neighborhood.reshape(B, N * N * num_samples * K, 1, 2)
        
        # Sample RGB colors
        rgb_features_flat = F.grid_sample(
            images,  # [B, 3, H_img, W_img]
            edge_coords_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B, 3, N*N*num_samples*K, 1]
        
        # Reshape: [B, N, N, num_samples, K, 3]
        rgb_samples_neighborhood = (
            rgb_features_flat.squeeze(-1)
            .permute(0, 2, 1)
            .reshape(B, N, N, num_samples, K, 3)
        )
        
        # Aggregate across neighborhood dimension based on method
        if self.rgb_neighborhood_aggregation == 'center':
            # Use only center point (first point, index 0)
            rgb_samples = rgb_samples_neighborhood[:, :, :, :, 0, :]  # [B, N, N, num_samples, 3]
        elif self.rgb_neighborhood_aggregation == 'mean':
            # Average across all neighborhood points (original behavior)
            rgb_samples = rgb_samples_neighborhood.mean(dim=-2)  # [B, N, N, num_samples, 3]
        elif self.rgb_neighborhood_aggregation == 'median':
            # Median across all neighborhood points for each channel (robust to outliers)
            rgb_samples = torch.stack([
                rgb_samples_neighborhood[:, :, :, :, :, 0].median(dim=-1)[0],  # median R
                rgb_samples_neighborhood[:, :, :, :, :, 1].median(dim=-1)[0],  # median G
                rgb_samples_neighborhood[:, :, :, :, :, 2].median(dim=-1)[0],  # median B
            ], dim=-1)  # [B, N, N, num_samples, 3]
        elif self.rgb_neighborhood_aggregation == 'min_r_min_g_max_b':
            # [min R, min G, max B] across neighborhood
            rgb_samples = torch.stack([
                rgb_samples_neighborhood[:, :, :, :, :, 0].min(dim=-1)[0],  # min R
                rgb_samples_neighborhood[:, :, :, :, :, 1].min(dim=-1)[0],  # min G
                rgb_samples_neighborhood[:, :, :, :, :, 2].max(dim=-1)[0],  # max B
            ], dim=-1)  # [B, N, N, num_samples, 3]
        else:
            raise ValueError(f"Unsupported rgb_neighborhood_aggregation: {self.rgb_neighborhood_aggregation}")
        
        return rgb_samples
    
    def _compute_spatial_features(
        self,
        node_coords: torch.Tensor,  # [B, N, 2] in heatmap space OR image space
        coords_in_image_space: bool = False,  # NEW: Flag to indicate coordinate space
    ) -> torch.Tensor:
        """
        Compute spatial/geometric features for all node pairs.
        
        Features:
        1. Euclidean distance (normalized)
        2. Direction vector (normalized)
        3. Collinearity score: Check if there's an intermediate node on the line
        
        Args:
            node_coords: Node coordinates [B, N, 2] in heatmap space or image space
            coords_in_image_space: If True, node_coords are already in image space (skip conversion)
        
        Returns:
            spatial_features: [B, N, N, 4] - distance, direction_x, direction_y, collinearity
        """
        B, N, _ = node_coords.shape
        device = node_coords.device
        
        # Convert to image space if needed (for collinearity check)
        if coords_in_image_space:
            # Already in image space, use directly
            node_coords_image = node_coords
        else:
            # Convert from heatmap space to image space
            scale_factor = self.image_size / self.heatmap_resolution
            node_coords_image = node_coords * scale_factor  # [B, N, 2] in image space
        
        # Expand coordinates for all pairs (using image space for collinearity)
        coords_i = node_coords_image.unsqueeze(2)  # [B, N, 1, 2] in image space
        coords_j = node_coords_image.unsqueeze(1)  # [B, 1, N, 2] in image space
        
        # 1. Euclidean distance (in image space, matching toy classifier)
        diff = coords_j - coords_i  # [B, N, N, 2]
        distance = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        # Normalize by image diagonal (512*sqrt(2) ≈ 724) - matching toy classifier
        max_distance = self.image_size * np.sqrt(2.0)
        distance_norm = distance / max_distance
        
        # 2. Direction vector (normalized)
        direction = diff / (distance + 1e-6)  # [B, N, N, 2]
        
        # 3. Collinearity check: Vectorized computation (using image space coordinates)
        # For each pair (i, j), check if any intermediate node k lies on the line segment
        # CRITICAL: Use image space coordinates for collinearity (matching toy classifier)
        p_i = coords_i.expand(B, N, N, 2)  # [B, N, N, 2] in image space
        p_j = coords_j.expand(B, N, N, 2)  # [B, N, N, 2] in image space
        p_k = node_coords_image.unsqueeze(1).unsqueeze(1).expand(B, N, N, N, 2)  # [B, N, N, N, 2] in image space
        
        # Vector from i to k and i to j
        v_ik = p_k - p_i.unsqueeze(-2)  # [B, N, N, N, 2]
        v_ij = (p_j - p_i).unsqueeze(-2)  # [B, N, N, 1, 2]
        
        # Project k onto line i->j
        dot_ik_ij = (v_ik * v_ij).sum(dim=-1)  # [B, N, N, N]
        dot_ij_ij = (v_ij * v_ij).sum(dim=-1)  # [B, N, N, 1]
        t_raw = dot_ik_ij / (dot_ij_ij + 1e-6)  # [B, N, N, N]
        
        # Clamp t to [0, 1] to stay on segment (but keep mask of original values)
        valid_t_mask = (t_raw >= 0.0) & (t_raw <= 1.0)  # [B, N, N, N]
        t = torch.clamp(t_raw, 0.0, 1.0)
        
        # Projected point on line
        p_proj = p_i.unsqueeze(-2) + t.unsqueeze(-1) * v_ij  # [B, N, N, N, 2]
        
        # Distance from k to projected point
        dist_to_line = torch.norm(p_k - p_proj, dim=-1)  # [B, N, N, N]
        
        # Normalize by segment length
        seg_length = distance.squeeze(-1)  # [B, N, N]
        dist_normalized = dist_to_line / (seg_length.unsqueeze(-1) + 1e-6)  # [B, N, N, N]
        
        # Create mask to exclude k=i and k=j
        k_indices = torch.arange(N, device=device).view(1, 1, 1, N)  # [1, 1, 1, N]
        i_indices = torch.arange(N, device=device).view(1, N, 1, 1)  # [1, N, 1, 1]
        j_indices = torch.arange(N, device=device).view(1, 1, N, 1)  # [1, 1, N, 1]
        
        # Mask: exclude k where k==i or k==j
        k_mask = (k_indices != i_indices) & (k_indices != j_indices)  # [B, N, N, N]
        
        # Apply mask: set invalid k to inf
        dist_normalized_masked = torch.where(
            (k_mask & valid_t_mask),
            dist_normalized,
            torch.ones_like(dist_normalized) * float('inf')
        )
        
        # Find minimum distance across k (excluding i and j)
        # Handle edge case where N < 3 (no intermediate nodes possible)
        # When N < 3, after excluding k==i and k==j, there are no valid k values
        if N < 3:
            # No intermediate nodes possible, set all to 1.0 (no collinearity)
            min_dist_to_line = torch.ones(B, N, N, device=dist_normalized.device, dtype=dist_normalized.dtype)
        else:
            min_dist_to_line, _ = dist_normalized_masked.min(dim=-1)  # [B, N, N]
        
        # Replace inf with 1.0 (no intermediate nodes found)
        collinearity_scores = torch.where(
            torch.isinf(min_dist_to_line),
            torch.ones_like(min_dist_to_line),
            min_dist_to_line
        ).unsqueeze(-1)  # [B, N, N, 1]
        
        # Set diagonal to 1.0 (self-connections)
        eye_mask = torch.eye(N, device=device).bool().unsqueeze(0).expand(B, N, N)  # [B, N, N]
        collinearity_scores = torch.where(
            eye_mask.unsqueeze(-1),
            torch.ones_like(collinearity_scores),
            collinearity_scores
        )
        
        # Concatenate all spatial features
        spatial_features = torch.cat([
            distance_norm,  # [B, N, N, 1]
            direction,  # [B, N, N, 2]
            collinearity_scores,  # [B, N, N, 1]
        ], dim=-1)  # [B, N, N, 4]
        
        return spatial_features
    
    def forward(
        self,
        l_i: torch.Tensor,  # Local descriptors [B, N, D]
        g_i: torch.Tensor,  # Global descriptors [B, N, D]
        node_coords: torch.Tensor,  # Node coordinates [B, N, 2] in heatmap space OR image space
        z_star: torch.Tensor,  # Global embedding [B, D_z] (kept for interface compatibility, not used)
        candidate_mask: torch.Tensor,  # Candidate edge mask [B, N, N]
        G_prime: Optional[torch.Tensor] = None,  # Processed global features [B, C_G, H, W] for topology features
        local_features: Optional[torch.Tensor] = None,  # Local features (not used in this model)
        images: Optional[torch.Tensor] = None,  # RGB images [B, 3, H, W]
        valid_mask: Optional[torch.Tensor] = None,  # [B, N]
        coords_in_image_space: bool = False,  # NEW: Flag to indicate if node_coords are in image space
        **kwargs,  # Accept additional parameters for future extensibility
    ) -> torch.Tensor:
        """
        Predict edge probabilities using edge-aware graph transformer.
        
        Args:
            l_i: Local node descriptors [B, N, D]
            g_i: Global node descriptors [B, N, D]
            node_coords: Node coordinates [B, N, 2] in heatmap space (32×32) or image space (512×512)
            z_star: Global embedding [B, D_z] (kept for interface compatibility, not used)
            candidate_mask: Candidate edge mask [B, N, N]
            G_prime: Processed global features [B, C_G, H, W] for topology feature sampling
            local_features: Local features (not used, kept for interface compatibility)
            images: RGB images [B, 3, H, W] for RGB path sampling
            valid_mask: Valid node mask [B, N] (optional)
            coords_in_image_space: If True, node_coords are already in image space (avoids double conversion)
        
        Returns:
            edge_logits: Edge logits [B, N, N]
        """
        B, N, _ = node_coords.shape
        device = node_coords.device
        
        # 1. RGB path features (if enabled) - matching toy model exactly
        rgb_path_features = None
        if self.use_rgb_path_features and images is not None:
            rgb_samples = self._sample_rgb_along_path(
                images, node_coords, self.rgb_path_num_samples,
                coords_in_image_space=coords_in_image_space  # Pass flag to avoid conversion
            )  # [B, N, N, num_samples, 3]
            
            # Project RGB samples to feature space
            rgb_features = self.rgb_proj(rgb_samples)  # [B, N, N, num_samples, rgb_feature_dim]
            
            # Process RGB sequence along path
            if self.rgb_sequence_model == '1d_cnn':
                B, N, num_samples, D = rgb_features.shape[0], rgb_features.shape[1], rgb_features.shape[3], rgb_features.shape[4]
                rgb_flat = rgb_features.reshape(B * N * N, D, num_samples)  # [B*N*N, D, num_samples]
                
                # Handle edge cases: skip CNN if batch is empty or sequence length is 0
                if rgb_flat.shape[0] == 0 or rgb_flat.shape[2] == 0:
                    rgb_path_features = torch.zeros(B, N, N, D, device=device, dtype=node_coords.dtype)
                else:
                    x = self.rgb_seq_conv1(rgb_flat)  # [B*N*N, D, num_samples]
                    x = x.permute(0, 2, 1)  # [B*N*N, num_samples, D] for LayerNorm
                    x = self.rgb_seq_ln1(x)
                    x = x.permute(0, 2, 1)  # [B*N*N, D, num_samples] back
                    x = F.gelu(x)
                    x = self.rgb_seq_dropout(x)
                    x = self.rgb_seq_conv2(x)  # [B*N*N, D, num_samples]
                    x = x.permute(0, 2, 1)  # [B*N*N, num_samples, D] for LayerNorm
                    x = self.rgb_seq_ln2(x)
                    x = x.permute(0, 2, 1)  # [B*N*N, D, num_samples] back
                    x = F.gelu(x)
                    rgb_processed = self.rgb_seq_pool(x)  # [B*N*N, D, 1]
                    rgb_path_features = rgb_processed.squeeze(-1).reshape(B, N, N, D)  # [B, N, N, D]
            elif self.rgb_sequence_model == 'transformer':
                B, N, num_samples, D = rgb_features.shape[0], rgb_features.shape[1], rgb_features.shape[3], rgb_features.shape[4]
                rgb_flat = rgb_features.reshape(B * N * N, num_samples, D)  # [B*N*N, num_samples, D]
                
                # Handle edge cases: skip transformer if batch is empty or sequence length is 0
                # CUDA transformers fail with "invalid configuration argument" for these cases
                if rgb_flat.shape[0] == 0 or rgb_flat.shape[1] == 0:
                    rgb_path_features = torch.zeros(B, N, N, D, device=device, dtype=node_coords.dtype)
                else:
                    rgb_transformed = self.rgb_seq_processor(rgb_flat)  # [B*N*N, num_samples, D]
                    rgb_path_features = rgb_transformed.mean(dim=1).reshape(B, N, N, D)  # [B, N, N, D]
            elif self.rgb_sequence_model == 'mean':
                rgb_path_features = rgb_features.mean(dim=3)  # [B, N, N, rgb_feature_dim]
            elif self.rgb_sequence_model == 'max':
                rgb_path_features = rgb_features.amax(dim=3)  # [B, N, N, rgb_feature_dim]
        else:
            # If RGB features disabled, use zeros (but this shouldn't happen in toy model)
            rgb_path_features = torch.zeros(B, N, N, self.rgb_feature_dim, device=device, dtype=node_coords.dtype)
        
        # 2. Spatial features (with collinearity check) - matching toy model exactly
        spatial_features_raw = self._compute_spatial_features(
            node_coords, 
            coords_in_image_space=coords_in_image_space
        )  # [B, N, N, 4]
        # Project spatial features (or use raw if no projection)
        if self.spatial_proj is not None:
            spatial_features = self.spatial_proj(spatial_features_raw)  # [B, N, N, spatial_feature_dim]
        else:
            spatial_features = spatial_features_raw  # [B, N, N, 4] - use raw features
        
        # 3. Topology features: Sample G_prime at node locations (if enabled)
        topology_features = None
        if self.topology_feature_dim > 0 and G_prime is not None:
            B_prime, C_G, H_global, W_global = G_prime.shape  # [B, C_G, 8, 8]
            # Convert coordinates to heatmap space first (if needed), then to global feature space
            if coords_in_image_space:
                # Convert from image space (512×512) to heatmap space (32×32)
                scale_to_heatmap = self.heatmap_resolution / self.image_size  # 32 / 512 = 0.0625
                node_coords_heatmap = node_coords * scale_to_heatmap  # [B, N, 2] in heatmap space
            else:
                # Already in heatmap space (32×32)
                node_coords_heatmap = node_coords
            
            # Scale coordinates from heatmap space (32×32) to global feature space (8×8)
            node_coords_global = node_coords_heatmap.clone()
            W_local, H_local = float(self.heatmap_resolution), float(self.heatmap_resolution)  # 32.0, 32.0
            if W_local > 1:
                node_coords_global[:, :, 0] = node_coords_heatmap[:, :, 0] * (W_global - 1) / (W_local - 1)  # x
            else:
                node_coords_global[:, :, 0] = 0.0
            if H_local > 1:
                node_coords_global[:, :, 1] = node_coords_heatmap[:, :, 1] * (H_global - 1) / (H_local - 1)  # y
            else:
                node_coords_global[:, :, 1] = 0.0
            
            # Normalize to [-1, 1] for grid_sample
            node_coords_normalized = node_coords_global.clone()
            node_coords_normalized[:, :, 0] = (node_coords_global[:, :, 0] / (W_global - 1)) * 2.0 - 1.0  # x
            node_coords_normalized[:, :, 1] = (node_coords_global[:, :, 1] / (H_global - 1)) * 2.0 - 1.0  # y
            
            # grid_sample expects [B, N, 1, 2] format
            node_coords_grid = node_coords_normalized.unsqueeze(2)  # [B, N, 1, 2]
            
            # Sample G_prime at node coordinates
            G_prime_sampled = F.grid_sample(
                G_prime,
                node_coords_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B, C_G, N, 1]
            G_prime_sampled = G_prime_sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C_G]
            
            # Expand for pairwise combinations: G_prime_i and G_prime_j
            G_prime_i_expanded = G_prime_sampled.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, C_G]
            G_prime_j_expanded = G_prime_sampled.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, C_G]
            
            # Concatenate topology features: [G_prime_i || G_prime_j]
            topology_features_raw = torch.cat([G_prime_i_expanded, G_prime_j_expanded], dim=-1)  # [B, N, N, 2*C_G]
            # Project topology features to lower dimension
            topology_features = self.topology_proj(topology_features_raw)  # [B, N, N, topology_feature_dim]
        elif self.topology_feature_dim > 0:
            # If G_prime not provided but topology is enabled, use zeros
            topology_features = torch.zeros(
                B, N, N, self.topology_feature_dim, 
                device=device, 
                dtype=node_coords.dtype
            )  # [B, N, N, topology_feature_dim]
        else:
            # Topology features disabled
            topology_features = None
        
        # 4. Combine edge features: RGB + spatial + topology (if enabled)
        if topology_features is not None:
            edge_features = torch.cat([
                rgb_path_features, 
                spatial_features, 
                topology_features
            ], dim=-1)  # [B, N, N, rgb_feature_dim + spatial_feature_dim + topology_feature_dim]
        else:
            edge_features = torch.cat([
                rgb_path_features, 
                spatial_features
            ], dim=-1)  # [B, N, N, rgb_feature_dim + spatial_feature_dim]
        
        # Track which edges are valid (for masking attention / gradients)
        edge_valid = None
        if valid_mask is not None or candidate_mask is not None:
            if valid_mask is not None:
                valid_i = valid_mask.unsqueeze(2)  # [B, N, 1]
                valid_j = valid_mask.unsqueeze(1)  # [B, 1, N]
                edge_valid = (valid_i & valid_j)
            else:
                edge_valid = torch.ones(B, N, N, device=device, dtype=torch.bool)
            if candidate_mask is not None:
                edge_valid = edge_valid & (candidate_mask > 0)
        
        # 5. Project edge features
        edge_embeds = self.edge_proj(edge_features)  # [B, N, N, hidden_dim]
        
        # 6. Add geometric positional encoding if enabled
        if self.use_positional_encoding and node_coords is not None:
            # Normalize coordinates to [0, 1] - matching toy model (uses image_size, not heatmap_resolution)
            # Convert from heatmap space to pixel space first (if needed), then normalize
            if coords_in_image_space:
                # Already in image space
                node_coords_pixel = node_coords
            else:
                # Convert from heatmap space to pixel space
                node_coords_pixel = node_coords * (self.image_size / self.heatmap_resolution)  # [B, N, 2] in pixel space
            coords_normalized = node_coords_pixel / self.image_size  # [B, N, 2] normalized to [0, 1]
            
            # Get coordinates for each edge (i,j): [coord_i, coord_j]
            coords_i = coords_normalized.unsqueeze(2)  # [B, N, 1, 2]
            coords_j = coords_normalized.unsqueeze(1)  # [B, 1, N, 2]
            
            # Concatenate: [x_i, y_i, x_j, y_j]
            pos_encoding = torch.cat([
                coords_i.expand(B, N, N, 2),  # [B, N, N, 2] - coordinates of node i
                coords_j.expand(B, N, N, 2),  # [B, N, N, 2] - coordinates of node j
            ], dim=-1)  # [B, N, N, 4]
            
            pos_embeds = self.pos_encoding_proj(pos_encoding)  # [B, N, N, hidden_dim]
            edge_embeds = edge_embeds + pos_embeds
        
        # 7. Reshape for transformer: [B, N*N, hidden_dim]
        edge_embeds_flat = edge_embeds.reshape(B, N * N, self.hidden_dim)
        if edge_valid is not None:
            edge_valid_flat = edge_valid.reshape(B, N * N)
            edge_embeds_flat = edge_embeds_flat.masked_fill(
                (~edge_valid_flat).unsqueeze(-1),
                0.0,
            )
        else:
            edge_valid_flat = None
        
        # 8. Create line graph attention mask (optimized, cached, vectorized)
        attention_mask = self._create_line_graph_attention_mask(
            N, device
        )  # [1, N*N, N*N]
        
        # 9. Create key padding mask for invalid edges (if needed)
        key_padding_mask = None  # Keep transformer stable even when all edges are masked
        
        # 10. Apply transformer with line graph attention mask
        x = edge_embeds_flat
        
        # Convert attention mask to format expected by MultiheadAttention
        if attention_mask.shape[0] == 1:
            attn_mask_base = attention_mask[0]  # [N*N, N*N]
            attn_mask_float = attn_mask_base.float().masked_fill(attn_mask_base, float('-inf'))  # [N*N, N*N]
        else:
            attn_mask_base = attention_mask[0]  # [N*N, N*N]
            attn_mask_float = attn_mask_base.float().masked_fill(attn_mask_base, float('-inf'))  # [N*N, N*N]
        
        for layer in self.layers:
            self_attn = layer.self_attn
            norm1 = layer.norm1
            norm2 = layer.norm2
            dropout1 = layer.dropout1
            dropout2 = layer.dropout2
            linear1 = layer.linear1
            linear2 = layer.linear2
            activation = layer.activation
            
            # Self-attention with line graph mask
            attn_output, _ = self_attn(
                x, x, x,
                attn_mask=attn_mask_float,  # [N*N, N*N]
                key_padding_mask=key_padding_mask,
            )
            
            # Residual connection and norm
            x = norm1(x + dropout1(attn_output))
            
            # Feedforward
            ff_output = linear2(dropout2(activation(linear1(x))))
            x = norm2(x + dropout2(ff_output))
        
        edge_transformed = x  # [B, N*N, hidden_dim]
        
        # 11. Reshape back: [B, N, N, hidden_dim]
        edge_transformed = edge_transformed.reshape(B, N, N, self.hidden_dim)
        
        # 12. Predict edge scores
        edge_logits = self.edge_head(edge_transformed).squeeze(-1)  # [B, N, N]
        
        # 13. Apply candidate / validity mask
        if candidate_mask is not None:
            edge_logits = edge_logits * candidate_mask
        if valid_mask is not None:
            pair_valid = (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)).float()
            edge_logits = edge_logits * pair_valid
        
        # 14. Ensure symmetry
        edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0
        
        # 15. Defensive clamp: replace NaN/Inf that can arise from fully-masked rows
        edge_logits = torch.nan_to_num(edge_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        return edge_logits

