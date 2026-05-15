import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualMLPBlock(nn.Module):
    """
    Small residual feed-forward block: LN → Linear → GELU → Dropout → Linear → Dropout + skip.
    """

    def __init__(self, dim: int, hidden_scale: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * hidden_scale
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class PairwiseEdgeMLP(nn.Module):
    """
    Simpler edge predictor that scores each candidate pair with a shared MLP.

    It reuses the same edge feature construction pipeline (spatial cues,
    optional local path features, endpoint descriptors) but replaces the
    GraphTransformer stack with a lightweight per-pair multilayer perceptron.
    """

    def __init__(
        self,
        node_feature_dim: int = 256,
        edge_feature_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_path_sampling: bool = True,
        path_num_samples: int = 4,
        local_feature_dim: int = 256,
        heatmap_resolution: int = 32,
        image_size: int = 512,
        use_rgb_path_features: bool = True,  # Sample RGB colors along path
        rgb_path_num_samples: int = 9,  # Number of intermediate points for RGB sampling
    ):
        super().__init__()

        self.local_feature_dim = local_feature_dim
        self.use_path_sampling = use_path_sampling
        self.path_num_samples = path_num_samples
        self.heatmap_resolution = heatmap_resolution
        self.image_size = image_size
        self.use_rgb_path_features = use_rgb_path_features
        self.rgb_path_num_samples = rgb_path_num_samples

        # Fuse local + global descriptors per node
        self.node_feature_proj = nn.Sequential(
            nn.Linear(node_feature_dim * 2, edge_feature_dim),
            nn.LayerNorm(edge_feature_dim),
            nn.GELU(),
        )

        # Raw edge feature dimension: spatial φ_ij plus optional path features
        edge_feature_raw_dim = 3  # Spatial features
        if use_path_sampling:
            edge_feature_raw_dim += 128  # Local path features (projected)
            self.local_path_proj = nn.Linear(local_feature_dim, 128)
        else:
            self.local_path_proj = None
        
        # RGB color features along path (for blue edge detection)
        if use_rgb_path_features:
            # Project RGB to feature space (learns color transformations)
            self.rgb_path_proj = nn.Sequential(
                nn.Linear(3, 8),  # Project RGB to small feature space
                nn.LayerNorm(8),
                nn.GELU(),
            )
            edge_feature_raw_dim += 8  # RGB path features (projected)
        else:
            self.rgb_path_proj = None

        self.edge_feature_proj = nn.Sequential(
            nn.Linear(edge_feature_raw_dim, edge_feature_dim),
            nn.LayerNorm(edge_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pairwise MLP (shared across all edges)
        pair_input_dim = edge_feature_dim * 3  # h_i || h_j || edge_features_ij
        self.edge_mlp_in = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_mlp_residual = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_scale=2, dropout=dropout),
            ResidualMLPBlock(hidden_dim, hidden_scale=2, dropout=dropout),
        ])
        self.edge_mlp_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        l_i: torch.Tensor,
        g_i: torch.Tensor,
        node_coords: torch.Tensor,
        z_star: torch.Tensor,
        candidate_mask: torch.Tensor,
        G_prime: Optional[torch.Tensor] = None,  # Not used, kept for interface compatibility
        local_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,  # RGB images for color sampling
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Accept additional parameters for future extensibility
    ) -> torch.Tensor:
        """
        Args mirror the previous GraphTransformer so the surrounding training
        code does not need to change.
        """
        del z_star  # Unused but kept for interface compatibility

        B, N, _ = l_i.shape

        # Node descriptor fusion
        node_features = torch.cat([l_i, g_i], dim=-1)
        node_features = self.node_feature_proj(node_features)  # [B, N, D_edge]

        # Spatial features φ_ij
        coords_i = node_coords.unsqueeze(2)  # [B, N, 1, 2]
        coords_j = node_coords.unsqueeze(1)  # [B, 1, N, 2]
        spatial = coords_j - coords_i
        distance = torch.norm(spatial, dim=-1, keepdim=True)
        max_distance = 32.0 * (2.0 ** 0.5)
        distance_normalized = distance / (max_distance + 1e-8)
        spatial_normalized = spatial / (distance + 1e-8)
        spatial_features = torch.cat([spatial_normalized, distance_normalized], dim=-1)

        edge_feature_list = [spatial_features]
        if self.use_path_sampling and local_features is not None:
            local_path = self._sample_local_path_features(
                local_features, node_coords, self.path_num_samples
            )
            local_path_reduced = self.local_path_proj(local_path)
            edge_feature_list.append(local_path_reduced)
        
        # Sample RGB colors along path (for blue edge detection)
        if self.use_rgb_path_features and images is not None:
            rgb_path = self._sample_rgb_path_features(
                images, node_coords, self.rgb_path_num_samples
            )
            if self.rgb_path_proj is not None:
                rgb_path = self.rgb_path_proj(rgb_path)  # [B, N, N, 8]
            edge_feature_list.append(rgb_path)

        edge_features = torch.cat(edge_feature_list, dim=-1) if len(edge_feature_list) > 1 else edge_feature_list[0]
        edge_features = self.edge_feature_proj(edge_features)

        # Pair features: concatenate endpoint embeddings + edge context
        h_i = node_features.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = node_features.unsqueeze(1).expand(-1, N, -1, -1)
        pair_input = torch.cat([h_i, h_j, edge_features], dim=-1)

        edge_hidden = self.edge_mlp_in(pair_input)
        for block in self.edge_mlp_residual:
            edge_hidden = block(edge_hidden)
        edge_logits = self.edge_mlp_out(edge_hidden).squeeze(-1)

        if candidate_mask is not None:
            if valid_mask is not None:
                pair_valid = (
                    valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
                ).float()
                candidate_mask = candidate_mask * pair_valid

            # Ensure symmetry matches what the hybrid loss expects
            candidate_mask = torch.maximum(
                candidate_mask, candidate_mask.transpose(-2, -1)
            )

            edge_logits = edge_logits * candidate_mask
            edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0
            edge_logits = edge_logits * candidate_mask
        else:
            edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0

        return edge_logits

    def _sample_local_path_features(
        self,
        features: torch.Tensor,
        node_coords: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Utility identical to the transformer version: sample intermediate points
        along each pair's straight line and average their features.
        """
        B, N, _ = node_coords.shape
        C, H, W = features.shape[1:]

        t = torch.linspace(0.2, 0.8, num_samples, device=node_coords.device)
        coords_i = node_coords.unsqueeze(2).unsqueeze(3)
        coords_j = node_coords.unsqueeze(1).unsqueeze(3)
        t_expanded = t.view(1, 1, 1, -1, 1)
        edge_coords = (1 - t_expanded) * coords_i + t_expanded * coords_j

        edge_coords_norm = edge_coords.clone()
        if W > 1:
            edge_coords_norm[:, :, :, :, 0] = (edge_coords[:, :, :, :, 0] / (W - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 0] = 0.0
        if H > 1:
            edge_coords_norm[:, :, :, :, 1] = (edge_coords[:, :, :, :, 1] / (H - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 1] = 0.0

        edge_coords_flat = edge_coords_norm.reshape(B, N * N * num_samples, 1, 2)
        edge_features_flat = F.grid_sample(
            features,
            edge_coords_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        edge_features = (
            edge_features_flat.squeeze(-1)
            .permute(0, 2, 1)
            .reshape(B, N, N, num_samples, C)
        )
        return edge_features.mean(dim=3)
    
    def _sample_rgb_path_features(
        self,
        images: torch.Tensor,  # [B, 3, H, W] RGB images
        node_coords: torch.Tensor,  # [B, N, 2] in heatmap space (32×32)
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample RGB colors at intermediate points along edge paths.
        This captures visual features like color (e.g., blue edges) that may
        not be preserved in high-level CNN features.
        
        Args:
            images: RGB images [B, 3, H_img, W_img] (e.g., 512×512)
            node_coords: Node coordinates [B, N, 2] in heatmap space (32×32)
            num_samples: Number of intermediate points to sample
        
        Returns:
            rgb_path: [B, N, N, 3] - RGB colors averaged along path
        """
        B, N, _ = node_coords.shape
        _, H_img, W_img = images.shape[1:]
        
        # Convert node_coords from heatmap space (32×32) to pixel space (512×512)
        scale_factor = self.image_size / self.heatmap_resolution  # 512 / 32 = 16
        node_coords_pixel = node_coords * scale_factor  # [B, N, 2] in pixel space
        
        # Sample intermediate points along path
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
        
        # Reshape for grid_sample: [B, N*N*num_samples, 1, 2]
        edge_coords_flat = edge_coords_norm.reshape(B, N * N * num_samples, 1, 2)
        
        # Sample RGB colors
        rgb_features_flat = F.grid_sample(
            images,  # [B, 3, H_img, W_img]
            edge_coords_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B, 3, N*N*num_samples, 1]
        
        # Reshape and average over sampled points
        rgb_features = (
            rgb_features_flat.squeeze(-1)
            .permute(0, 2, 1)
            .reshape(B, N, N, num_samples, 3)
        )
        rgb_path = rgb_features.mean(dim=3)  # [B, N, N, 3] - average RGB along path
        
        return rgb_path

