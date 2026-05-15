"""
Graph Transformer for Edge Prediction

Node-based transformer that reasons over graph structure.
Uses edge features in attention mechanism for better edge prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EdgeAwareAttention(nn.Module):
    """
    Multi-head attention with edge features.
    
    Incorporates edge features into attention scores for graph-aware reasoning.
    """
    
    def __init__(
        self,
        embed_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge feature projection for attention bias
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,  # [B, N, D]
        key: torch.Tensor,  # [B, N, D]
        value: torch.Tensor,  # [B, N, D]
        edge_features: torch.Tensor,  # [B, N, N, D_edge]
        attention_mask: Optional[torch.Tensor] = None,  # [B, N, N] - True for valid edges
    ) -> torch.Tensor:
        """
        Compute edge-aware attention.
        
        Args:
            query: Query tensor [B, N, D]
            key: Key tensor [B, N, D]
            value: Value tensor [B, N, D]
            edge_features: Edge features [B, N, N, D_edge]
            attention_mask: Attention mask [B, N, N] - True for valid edges
        
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = query.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D_h]
        K = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D_h]
        V = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D_h]
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Add edge feature bias
        edge_bias = self.edge_proj(edge_features)  # [B, N, N, H]
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        scores = scores + edge_bias
        
        # Apply attention mask (if provided)
        if attention_mask is not None:
            # attention_mask: [B, N, N] - True for valid edges (can be float or bool)
            # Convert to boolean and then to [B, 1, N, N] for broadcasting
            mask = attention_mask.bool().unsqueeze(1)  # [B, 1, N, N]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, H, N, D_h]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        output = self.out_proj(output)
        
        return output


class GraphTransformerLayer(nn.Module):
    """
    Single layer of Graph Transformer.
    
    Processes node features with edge-aware attention.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Edge-aware attention
        self.attention = EdgeAwareAttention(
            embed_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, node_dim),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,  # [B, N, D]
        edge_features: torch.Tensor,  # [B, N, N, D_edge]
        attention_mask: Optional[torch.Tensor] = None,  # [B, N, N]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Node features [B, N, D]
            edge_features: Edge features [B, N, N, D_edge]
            attention_mask: Attention mask [B, N, N]
        
        Returns:
            Updated node features [B, N, D]
        """
        # Self-attention with edge features
        attn_output = self.attention(
            node_features,
            node_features,
            node_features,
            edge_features,
            attention_mask,
        )
        node_features = self.norm1(node_features + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(node_features)
        node_features = self.norm2(node_features + ffn_output)
        
        return node_features


class GraphTransformer(nn.Module):
    """
    Graph Transformer for edge prediction.
    
    Processes nodes with graph structure awareness.
    Uses edge features in attention mechanism.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 128,
        z_star_dim: int = 256,
        edge_feature_dim: int = 512,  # Increased from 256 to 512 for richer representations
        num_layers: int = 6,  # Increased from 4 to 6 for deeper reasoning
        num_heads: int = 16,  # Increased from 8 to 16 for more attention patterns
        hidden_dim: int = 256,  # Increased from 256 to 512 for richer representations
        dropout: float = 0.1,
        g_prime_dim: int = 256,
        use_edge_features_in_prediction: bool = True,  # New: include edge features in prediction
        use_path_sampling: bool = True,  # New: sample features along edge paths
        path_num_samples: int = 5,  # Number of points to sample along edge path
        local_feature_dim: int = 256,  # Dimension of local_features for path sampling
    ):
        """
        Args:
            node_feature_dim: Dimension of l_i and g_i
            z_star_dim: Dimension of z_star (kept for interface compatibility, but not used in edge features)
            edge_feature_dim: Output edge feature dimension
            num_layers: Number of graph transformer layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            g_prime_dim: Dimension of G_prime features
        """
        super().__init__()
        
        # Cross-attention for better node feature composition
        self.node_cross_attn = nn.MultiheadAttention(
            embed_dim=node_feature_dim,
            num_heads=max(1, num_heads // 2),
            dropout=dropout,
            batch_first=True,
        )
        self.node_feature_proj = nn.Linear(node_feature_dim * 2, node_feature_dim)
        
        # Node feature projection
        self.node_proj = nn.Linear(node_feature_dim, edge_feature_dim)
        
        # Edge feature composition: [φ_ij || G'_i || G'_j || (optional: G'_path || local_path)]
        # φ_ij: 3 (normalized direction [2] + normalized distance [1])
        # G'_i, G'_j: g_prime_dim each
        # G'_path: 128 (reduced from g_prime_dim, if use_path_sampling)
        # local_path: 128 (reduced from local_feature_dim, if use_path_sampling)
        # Note: z* removed to reduce redundancy (G'_i + G'_j already provide global context)
        edge_feature_raw_dim = 3 + g_prime_dim * 2
        if use_path_sampling:
            edge_feature_raw_dim += 128 + 128  # Add reduced-dimension path features (128 each)
        
        # Path feature dimension reduction (256 → 128)
        if use_path_sampling:
            self.g_prime_path_proj = nn.Linear(g_prime_dim, 128)  # Reduce G'_path from 256 → 128
            self.local_path_proj = nn.Linear(local_feature_dim, 128)  # Reduce local_path from 256 → 128
        else:
            self.g_prime_path_proj = None
            self.local_path_proj = None
        
        # Edge feature projection
        self.edge_feature_proj = nn.Sequential(
            nn.Linear(edge_feature_raw_dim, edge_feature_dim),
            nn.LayerNorm(edge_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                node_dim=edge_feature_dim,
                edge_dim=edge_feature_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Edge prediction head
        # Input: [h_i || h_j] or [h_i || h_j || edge_features_ij] if use_edge_features_in_prediction
        edge_pred_input_dim = edge_feature_dim * 2
        if use_edge_features_in_prediction:
            edge_pred_input_dim += edge_feature_dim  # Add edge features
        
        self.edge_head = nn.Sequential(
            nn.Linear(edge_pred_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Additional layer for better capacity
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.g_prime_dim = g_prime_dim
        self.use_edge_features_in_prediction = use_edge_features_in_prediction
        self.use_path_sampling = use_path_sampling
        self.path_num_samples = path_num_samples
        self.local_feature_dim = local_feature_dim
    
    def _sample_features_along_edge(
        self,
        features: torch.Tensor,  # [B, C, H, W] - local_features or G_prime
        node_coords: torch.Tensor,  # [B, N, 2] - node coordinates (will be expanded to all pairs)
        num_samples: int = 4,  # Number of intermediate points to sample along edge (endpoints excluded)
    ) -> torch.Tensor:
        """
        Sample features at multiple intermediate points along the edge path between all node pairs.
        
        Endpoints (t=0 and t=1) are excluded to avoid redundancy with endpoint features (G'_i, G'_j, l_i, l_j).
        Samples at t = [0.2, 0.4, 0.6, 0.8] for num_samples=4.
        
        Args:
            features: Image features [B, C, H, W]
            node_coords: Node coordinates [B, N, 2] (will be expanded to create all pairs)
            num_samples: Number of intermediate points to sample along edge (default: 4)
        
        Returns:
            edge_path_features: [B, N, N, C] - aggregated path features (mean pooled)
        """
        B, N, _ = node_coords.shape
        C, H, W = features.shape[1:]
        
        # Create interpolation points along edge
        # Exclude endpoints (t=0 and t=1) to avoid redundancy with G'_i, G'_j and l_i, l_j
        # Use intermediate points only: t = [0.2, 0.4, 0.6, 0.8] for num_samples=4
        # This makes path features complementary to endpoint features
        t = torch.linspace(0.2, 0.8, num_samples, device=node_coords.device)  # [num_samples]
        
        # Expand coordinates for all pairs (i, j)
        # coords_i: [B, N, 2] -> [B, N, 1, 1, 2] (expand along j dimension)
        coords_i_expanded = node_coords.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 2]
        # coords_j: [B, N, 2] -> [B, 1, N, 1, 2] (expand along i dimension)
        coords_j_expanded = node_coords.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, 2]
        # t: [num_samples] -> [1, 1, 1, num_samples, 1]
        t_expanded = t.view(1, 1, 1, -1, 1)  # [1, 1, 1, num_samples, 1]
        
        # Interpolate coordinates: coords(t) = (1-t) * coords_i + t * coords_j
        # Broadcasting: [B, N, 1, 1, 2] + [B, 1, N, 1, 2] + [1, 1, 1, num_samples, 1]
        # Result: [B, N, N, num_samples, 2]
        edge_coords = (1 - t_expanded) * coords_i_expanded + t_expanded * coords_j_expanded
        
        # Normalize to [-1, 1] for grid_sample
        edge_coords_norm = edge_coords.clone()
        if W > 1:
            edge_coords_norm[:, :, :, :, 0] = (edge_coords[:, :, :, :, 0] / (W - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 0] = 0.0
        if H > 1:
            edge_coords_norm[:, :, :, :, 1] = (edge_coords[:, :, :, :, 1] / (H - 1)) * 2.0 - 1.0
        else:
            edge_coords_norm[:, :, :, :, 1] = 0.0
        
        # Reshape for grid_sample: [B, N*N*num_samples, 1, 2]
        edge_coords_flat = edge_coords_norm.reshape(B, N * N * num_samples, 1, 2)
        
        # Sample features
        edge_features_flat = F.grid_sample(
            features,  # [B, C, H, W]
            edge_coords_flat,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, N*N*num_samples, 1]
        
        # Reshape back: [B, N, N, num_samples, C]
        edge_features = edge_features_flat.squeeze(-1).permute(0, 2, 1).reshape(B, N, N, num_samples, C)
        
        # Aggregate along path (mean pooling - captures average signal along path)
        edge_path_features = edge_features.mean(dim=3)  # [B, N, N, C]
        
        return edge_path_features
    
    def forward(
        self,
        l_i: torch.Tensor,  # Local descriptors (detached) [B, N, D]
        g_i: torch.Tensor,  # Global descriptors [B, N, D]
        node_coords: torch.Tensor,  # Node coordinates [B, N, 2]
        z_star: torch.Tensor,  # Global embedding [B, D_z]
        candidate_mask: torch.Tensor,  # Candidate mask [B, N, N]
        G_prime: Optional[torch.Tensor] = None,  # Processed global features [B, C_G, H, W]
        local_features: Optional[torch.Tensor] = None,  # Local features [B, C_L, H, W] for path sampling
        images: Optional[torch.Tensor] = None,  # Not used, kept for interface compatibility
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs,  # Accept additional parameters for future extensibility
    ) -> torch.Tensor:
        """
        Predict edge probabilities using graph transformer.
        
        Args:
            l_i: Local node descriptors (detached) [B, N, D]
            g_i: Global node descriptors [B, N, D]
            node_coords: Node coordinates [B, N, 2] (in 32×32 grid space)
            z_star: Global embedding [B, D_z] (kept for interface compatibility, but not used in edge features)
            candidate_mask: Candidate edge mask [B, N, N]
            G_prime: Processed global features [B, C_G, H, W] (H=W=8 for global resolution)
            valid_mask: Valid node mask [B, N] (optional)
        
        Returns:
            edge_logits: Edge logits [B, N, N]
        """
        B, N, D_node = l_i.shape
        
        # Cross-attention between node features for better composition
        l_i_enhanced, _ = self.node_cross_attn(l_i, g_i, g_i)  # [B, N, D_node]
        g_i_enhanced, _ = self.node_cross_attn(g_i, l_i, l_i)  # [B, N, D_node]
        
        # Concatenate and project to maintain dimension
        l_i_composed = self.node_feature_proj(torch.cat([l_i, l_i_enhanced], dim=-1))  # [B, N, D_node]
        g_i_composed = self.node_feature_proj(torch.cat([g_i, g_i_enhanced], dim=-1))  # [B, N, D_node]
        
        # Combine local and global node features
        node_features = (l_i_composed + g_i_composed) / 2.0  # [B, N, D_node]
        
        # Project to edge feature dimension
        node_features = self.node_proj(node_features)  # [B, N, D_edge]
        
        # Build edge features: [φ_ij || G'_i || G'_j || (optional: G'_path || local_path)]
        # Note: z* removed to reduce redundancy (G'_i + G'_j already provide global context)
        # Spatial features φ_ij
        coords_i = node_coords.unsqueeze(2)  # [B, N, 1, 2]
        coords_j = node_coords.unsqueeze(1)  # [B, 1, N, 2]
        spatial = coords_j - coords_i  # [B, N, N, 2]
        
        # Compute distance
        distance = torch.norm(spatial, dim=-1, keepdim=True)  # [B, N, N, 1]
        max_distance = 32.0 * (2.0 ** 0.5)  # Max distance in 32×32 grid
        distance_normalized = distance / (max_distance + 1e-8)  # [B, N, N, 1]
        
        # Normalize direction
        spatial_norm = distance + 1e-8
        spatial_normalized = spatial / spatial_norm  # [B, N, N, 2]
        
        # Concatenate spatial features
        spatial_features = torch.cat([spatial_normalized, distance_normalized], dim=-1)  # [B, N, N, 3]
        
        # Sample G_prime at node coordinates
        if G_prime is not None:
            B_prime, C_G, H_global, W_global = G_prime.shape
            # Scale coordinates from 32×32 to 8×8
            node_coords_global = node_coords.clone()
            W_local, H_local = 32.0, 32.0
            if W_local > 1:
                node_coords_global[:, :, 0] = node_coords[:, :, 0] * (W_global - 1) / (W_local - 1)
            else:
                node_coords_global[:, :, 0] = 0.0
            if H_local > 1:
                node_coords_global[:, :, 1] = node_coords[:, :, 1] * (H_global - 1) / (H_local - 1)
            else:
                node_coords_global[:, :, 1] = 0.0
            
            # Normalize to [-1, 1] for grid_sample
            node_coords_normalized = node_coords_global.clone()
            node_coords_normalized[:, :, 0] = (node_coords_global[:, :, 0] / (W_global - 1)) * 2.0 - 1.0
            node_coords_normalized[:, :, 1] = (node_coords_global[:, :, 1] / (H_global - 1)) * 2.0 - 1.0
            
            # grid_sample expects [B, N, 1, 2] format
            node_coords_grid = node_coords_normalized.unsqueeze(2)  # [B, N, 1, 2]
            
            # Sample G_prime
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
            
            # Sample features along edge paths if enabled
            if self.use_path_sampling:
                # Sample G_prime along edge paths
                G_prime_path = self._sample_features_along_edge(
                    G_prime,  # [B, C_G, 8, 8]
                    node_coords_global,  # [B, N, 2] in global space (8×8)
                    num_samples=self.path_num_samples
                )  # [B, N, N, C_G]
                
                # Sample local_features along path (higher resolution)
                if local_features is not None:
                    local_path = self._sample_features_along_edge(
                        local_features,  # [B, C_L, 32, 32]
                        node_coords,  # [B, N, 2] in local space (32×32)
                        num_samples=self.path_num_samples
                    )  # [B, N, N, C_L]
                else:
                    # If local_features not provided, use zeros
                    local_path = torch.zeros(B, N, N, self.local_feature_dim, device=l_i.device, dtype=l_i.dtype)
            else:
                # Path sampling disabled
                G_prime_path = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
                local_path = torch.zeros(B, N, N, self.local_feature_dim, device=l_i.device, dtype=l_i.dtype)
        else:
            G_prime_i_expanded = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
            G_prime_j_expanded = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
            if self.use_path_sampling:
                G_prime_path = torch.zeros(B, N, N, self.g_prime_dim, device=l_i.device, dtype=l_i.dtype)
                local_path = torch.zeros(B, N, N, self.local_feature_dim, device=l_i.device, dtype=l_i.dtype)
            else:
                G_prime_path = None
                local_path = None
        
        # Concatenate edge features: [φ_ij || G'_i || G'_j || (optional: G'_path || local_path)]
        # Note: z* removed to reduce redundancy (G'_i + G'_j already provide global context)
        edge_feature_list = [
            spatial_features,  # [B, N, N, 3]
            G_prime_i_expanded,  # [B, N, N, C_G] - endpoint
            G_prime_j_expanded,  # [B, N, N, C_G] - endpoint
        ]
        
        # Add path features if enabled (with dimension reduction)
        if self.use_path_sampling:
            # Reduce dimensions: 256 → 128
            # Project path features to reduced dimension (works for both real features and zero tensors)
            if G_prime_path is not None:
                G_prime_path_reduced = self.g_prime_path_proj(G_prime_path)  # [B, N, N, 128]
            else:
                G_prime_path_reduced = torch.zeros(B, N, N, 128, device=l_i.device, dtype=l_i.dtype)
            
            if local_path is not None:
                local_path_reduced = self.local_path_proj(local_path)  # [B, N, N, 128]
            else:
                local_path_reduced = torch.zeros(B, N, N, 128, device=l_i.device, dtype=l_i.dtype)
            
            edge_feature_list.append(G_prime_path_reduced)  # [B, N, N, 128] - reduced path features
            edge_feature_list.append(local_path_reduced)  # [B, N, N, 128] - reduced high-res path features
        
        edge_features = torch.cat(edge_feature_list, dim=-1)  # [B, N, N, D_raw]
        
        # Project edge features
        edge_features = self.edge_feature_proj(edge_features)  # [B, N, N, D_edge]
        
        # Process nodes with graph transformer layers
        for layer in self.layers:
            node_features = layer(
                node_features,  # [B, N, D_edge]
                edge_features,  # [B, N, N, D_edge]
                candidate_mask,  # [B, N, N] - attention mask
            )
        
        # Predict edges from node embeddings
        # For each pair (i, j): use [h_i || h_j] or [h_i || h_j || edge_features_ij]
        h_i = node_features.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D_edge]
        h_j = node_features.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D_edge]
        
        # Concatenate node embeddings
        edge_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2*D_edge]
        
        # Optionally include edge features in prediction for richer context
        if self.use_edge_features_in_prediction:
            edge_input = torch.cat([edge_input, edge_features], dim=-1)  # [B, N, N, 2*D_edge + D_edge]
        
        # Predict edge logits
        edge_logits = self.edge_head(edge_input).squeeze(-1)  # [B, N, N]
        
        # Mask out non-candidate edges
        edge_logits = edge_logits * candidate_mask
        
        # Symmetrize: p_ij ← (p_ij + p_ji)/2
        edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0
        
        return edge_logits

