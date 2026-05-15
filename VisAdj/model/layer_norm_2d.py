"""
2D Layer Normalization for Convolutional Features

Normalizes features across spatial dimensions (H, W) for each channel.
Used for stabilizing convolutional feature maps.
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization for convolutional features.
    
    Normalizes across spatial dimensions (H, W) independently for each channel.
    This is more stable than BatchNorm2d for variable batch sizes and inference.
    
    Formula: (x - mean) / sqrt(var + eps) * weight + bias
    where mean and var are computed over spatial dimensions (H, W).
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Args:
            num_channels: Number of input channels
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Normalized tensor [B, C, H, W]
        """
        # Compute mean and variance across spatial dimensions (H, W)
        # Keep channel dimension for broadcasting
        u = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        s = (x - u).pow(2).mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Normalize
        x = (x - u) / torch.sqrt(s + self.eps)  # [B, C, H, W]
        
        # Apply learnable affine transformation
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # [B, C, H, W]
        
        return x

