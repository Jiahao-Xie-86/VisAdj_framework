"""
Global Topology Module

Learns global topology representation using transformer and learned topology tokens.
Returns G' (processed global features), Z' (processed tokens), and z_star (attention-pooled tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalTopologyModule(nn.Module):
    """
    Global topology module with learned topology tokens.
    
    Uses transformer layers to process global features and extract
    topology-aware embeddings. Returns G', Z', and z_star.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_tokens: int = 16,
        token_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Input feature dimension from global stream
            num_tokens: Number of learned topology tokens
            token_dim: Dimension of topology tokens
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Learned topology tokens
        self.topology_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        
        # Project global features to token dimension
        self.feature_proj = nn.Linear(feature_dim, token_dim)
        
        # Positional encoding for spatial features (8×8 = 64 positions)
        self.base_pos_encoding = nn.Parameter(torch.randn(1, 64, token_dim))
        self.register_buffer('pos_encoding', None, persistent=False)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection for G'
        self.output_proj = nn.Linear(token_dim, feature_dim)
        
        # Attention pooling for z_star
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.z_star_proj = nn.Linear(token_dim, token_dim)
    
    def forward(
        self,
        global_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process global features with topology tokens.
        
        Args:
            global_features: Global stream features [B, C_G, 8, 8]
        
        Returns:
            G_prime: Processed global features [B, C_G, 8, 8]
            Z_prime: Processed topology tokens [B, K, token_dim]
            z_star: Attention-pooled global embedding [B, token_dim]
        """
        B, C_G, H, W = global_features.shape  # [B, C_G, 8, 8]
        
        # Flatten spatial features
        global_flat = global_features.permute(0, 2, 3, 1).reshape(B, H * W, C_G)  # [B, 64, C_G]
        
        # Project to token dimension
        global_flat = self.feature_proj(global_flat)  # [B, H*W, token_dim]
        
        # Prepare positional encoding matching H*W
        if self.pos_encoding is None or self.pos_encoding.shape[1] != H * W:
            if H * W == self.base_pos_encoding.shape[1]:
                pos = self.base_pos_encoding
            else:
                # Interpolate base positional encoding to new resolution
                base_size = int(self.base_pos_encoding.shape[1] ** 0.5)
                base = self.base_pos_encoding.view(1, base_size, base_size, self.token_dim)
                pos = F.interpolate(
                    base.permute(0, 3, 1, 2),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1).reshape(1, H * W, self.token_dim)
            self.pos_encoding = pos
        global_flat = global_flat + self.pos_encoding[:, :H*W, :]
        
        # Expand topology tokens for batch
        topology_tokens = self.topology_tokens.expand(B, -1, -1)  # [B, K, token_dim]
        
        # Concatenate tokens and features
        x = torch.cat([topology_tokens, global_flat], dim=1)  # [B, K + 64, token_dim]
        
        # Process with transformer
        x = self.transformer(x)  # [B, K + 64, token_dim]
        
        # Extract topology tokens (first K)
        Z_prime = x[:, :self.num_tokens, :]  # [B, K, token_dim]
        
        # Extract processed global features (last 64)
        global_processed = x[:, self.num_tokens:, :]  # [B, 64, token_dim]
        
        # Project back to feature dimension and reshape to spatial
        global_processed_feat = self.output_proj(global_processed)  # [B, 64, C_G]
        G_prime = global_processed_feat.permute(0, 2, 1).reshape(B, C_G, H, W)  # [B, C_G, 8, 8]
        
        # Attention pooling for z_star: use a learnable query
        query = self.topology_tokens.mean(dim=1, keepdim=True).expand(B, 1, -1)  # [B, 1, token_dim]
        z_star, _ = self.attn_pool(query, Z_prime, Z_prime)  # [B, 1, token_dim]
        z_star = z_star.squeeze(1)  # [B, token_dim]
        z_star = self.z_star_proj(z_star)  # [B, token_dim]
        
        return G_prime, Z_prime, z_star
