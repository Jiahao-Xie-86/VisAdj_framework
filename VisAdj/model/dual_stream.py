"""
Dual-Stream Feature Extraction

Extracts features at two resolutions:
- Local stream: Encoder grid resolution (≈1/16 of input, e.g., 32×32 for 512×512)
- Global stream: Downsampled Local by ×4 (e.g., 8×8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .layer_norm_2d import LayerNorm2d


class DualStreamExtractor(nn.Module):
    """
    Dual-stream feature extractor.
    
    Splits encoder features into:
    1. Local stream: Encoder grid resolution (32×32 for 512×512 input)
    2. Global stream: Downsampled Local by ×4 (8×8)
    """
    
    def __init__(
        self,
        encoder_feature_dim: int,
        local_feature_dim: int = 256,
        global_feature_dim: int = 256,
        local_resolution: int = 32,
        global_resolution: int = 8,
    ):
        """
        Args:
            encoder_feature_dim: Dimension of encoder output features
            local_feature_dim: Output dimension for local stream
            global_feature_dim: Output dimension for global stream
        """
        super().__init__()
        
        self.encoder_feature_dim = encoder_feature_dim
        self.local_resolution = local_resolution
        self.global_resolution = global_resolution
        
        # Local stream: Keep encoder grid resolution (32×32 for 512×512 input)
        self.local_stream = nn.Sequential(
            nn.Conv2d(encoder_feature_dim, local_feature_dim, 3, padding=1),
            LayerNorm2d(local_feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_feature_dim, local_feature_dim, 3, padding=1),
            LayerNorm2d(local_feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global stream: Downsample Local by ×4 (8×8)
        self.global_stream = nn.Sequential(
            nn.Conv2d(local_feature_dim, global_feature_dim, 3, padding=1),
            LayerNorm2d(global_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.global_resolution, self.global_resolution)),
            nn.Conv2d(global_feature_dim, global_feature_dim, 3, padding=1),
            LayerNorm2d(global_feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        encoder_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dual-stream features.
        
        Args:
            encoder_features: Encoder output [B, D, H, W] where H=W=32 for 512x512 input
        
        Returns:
            local_features: Local stream features [B, C_L, 32, 32]
            global_features: Global stream features [B, C_G, 8, 8]
        """
        # Local stream: Keep encoder grid resolution
        local_features = self.local_stream(encoder_features)
        if local_features.shape[-1] != self.local_resolution:
            local_features = F.interpolate(
                local_features,
                size=(self.local_resolution, self.local_resolution),
                mode='bilinear',
                align_corners=False,
            )
        # local_features: [B, C_L, 32, 32]
        
        # Global stream: Downsample Local by ×4
        global_features = self.global_stream(local_features)
        if global_features.shape[-1] != self.global_resolution:
            global_features = F.interpolate(
                global_features,
                size=(self.global_resolution, self.global_resolution),
                mode='bilinear',
                align_corners=False,
            )
        # global_features: [B, C_G, 8, 8]
        
        return local_features, global_features
    
    def get_local_resolution(self) -> int:
        """Get local stream resolution."""
        return self.local_resolution
    
    def get_global_resolution(self) -> int:
        """Get global stream resolution."""
        return self.global_resolution

