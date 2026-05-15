"""
SAM Graph Split - Full Model

Integrates all components:
- SAM encoder
- Dual-stream extraction (Local: 32×32, Global: 8×8)
- Node detection (l_i from Local, g_i from Global)
- Global topology (G', Z', z_star)
- ASNS (attention-based candidate generation)
- Relation transformer (edge prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoder import SAM2Encoder
from .dual_stream import DualStreamExtractor
from .node_detector import NodeDetector
from .global_topology import GlobalTopologyModule
from .asns import AttentionSparseNeighborSampler
from .knn_neighbor_sampler import KNNNeighborSampler
from .pairwise_edge_mlp import PairwiseEdgeMLP
from .graph_transformer import GraphTransformer
from .edge_aware_graph_transformer import EdgeAwareGraphTransformer


class SAMGraphSplit(nn.Module):
    """
    Complete SAM Graph Split model.
    
    Architecture:
    1. Frozen SAM encoder
    2. Dual-stream feature extraction (Local: 32×32, Global: 8×8)
    3. Node detection from local stream (l_i from Local, g_i from Global)
    4. Global topology understanding (G', Z', z_star)
    5. ASNS for edge candidate generation
    6. Relation transformer for edge prediction
    """
    
    def __init__(
        self,
        sam_version: str = 'vit_b',
        sam_checkpoint: Optional[str] = None,
        sam_config: Optional[str] = None,
        freeze_encoder: bool = True,
        image_size: int = 512,
        local_feature_dim: int = 256,
        global_feature_dim: int = 256,
        node_feature_dim: int = 128,
        num_topology_tokens: int = 16,
        topology_token_dim: int = 256,
        k_neighbors: int = 20,
        neighbor_radius: float = 256.0,
        heatmap_resolution: int = 32,
        max_nodes: int = 50,
        relation_transformer_layers: int = 3,  # Default: 3 layers (matching successful toy model)
        relation_edge_dim: int = 256,  # Edge feature dimension (used by PairwiseEdgeMLP)
        relation_hidden_dim: int = 256,  # Hidden dimension (default: 256 for EdgeAwareGraphTransformer)
        relation_num_heads: int = 8,  # Number of attention heads (default: 8, matching successful toy model)
        relation_dropout: float = 0.1,
        rgb_feature_dim: int = 32,
        rgb_sequence_model: str = 'transformer',
        rgb_seq_layers: int = 2,
        rgb_seq_heads: int = 4,
        rgb_neighborhood_aggregation: str = 'center',  # 'center', 'mean', 'median', or 'min_r_min_g_max_b'
        rgb_neighborhood_radius: float = 4.0,  # Radius in pixels for RGB neighborhood sampling (default: 4.0)
        neighbor_sampler: str = 'asns',  # 'asns' (learnable) or 'knn' (deterministic)
        edge_model: str = 'edge_aware_transformer',  # 'mlp', 'graph_transformer', or 'edge_aware_transformer'
        # ASNS hyperparameters
        coverage_label_smoothing: float = 0.1,  # Label smoothing for coverage loss (default: 0.1 = 10% smoothing)
        asns_use_entmax: bool = True,  # Whether to use entmax (True) or sparsemax (False)
        asns_entmax_alpha: float = 1.5,  # Entmax alpha (1.0 = softmax, 1.5 = sparsemax, 2.0 = hardmax)
        # Node detection hyperparameters
        mask_threshold: float = 0.5,
        mask_pool_radius: int = 2,
    ):
        """
        Initialize SAM Graph Split model.
        
        Args:
            sam_version: SAM version ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint: Path to SAM checkpoint
            sam_config: Path to SAM config (required for SAM2 variants)
            freeze_encoder: Whether to freeze encoder
            image_size: Input image size
            local_feature_dim: Local feature dimension
            global_feature_dim: Global feature dimension
            node_feature_dim: Node feature dimension
            num_topology_tokens: Number of topology tokens
            topology_token_dim: Topology token dimension
            k_neighbors: Number of neighbors for the learned neighbor sampler (Top-K)
            neighbor_radius: Historical compatibility parameter (stored for CLI/inference overrides)
            heatmap_resolution: Resolution of the heatmap (32 or 64)
            max_nodes: Maximum number of nodes to detect per image (default: 50). Should be set based on dataset (e.g., 25 for max 20 nodes, 50 for larger datasets)
            relation_transformer_layers: Number of transformer layers
            relation_edge_dim: Edge feature dimension for relation transformer
            relation_hidden_dim: Hidden dimension for relation transformer
            relation_num_heads: Number of attention heads in relation transformer
            relation_dropout: Dropout rate for relation transformer
            mask_threshold: Probability threshold for selecting peaks from the predicted node mask.
            mask_pool_radius: Radius for local max pooling when extracting mask peaks (default: 16 → 33×33 window).
        """
        super().__init__()
        
        self.image_size = image_size
        self.heatmap_resolution = heatmap_resolution
        self.global_resolution = max(1, heatmap_resolution // 4)
        
        # SAM encoder
        self.encoder = SAM2Encoder(
            sam_version=sam_version,
            checkpoint_path=sam_checkpoint,
            config_path=sam_config,
            freeze=freeze_encoder,
            image_size=image_size
        )
        
        # Store LoRA parameters for later use
        self.use_lora = False
        self.lora_rank = None
        
        # Dual-stream extractor
        # Local: encoder resolution (32×32), Global: downsampled by ×4 (8×8)
        self.dual_stream = DualStreamExtractor(
            encoder_feature_dim=self.encoder.feature_dim,
            local_feature_dim=local_feature_dim,
            global_feature_dim=global_feature_dim,
            local_resolution=self.heatmap_resolution,
            global_resolution=self.global_resolution,
        )
        
        # Node detector
        # Returns l_i (from Local, detached) and g_i (from Global)
        self.node_detector = NodeDetector(
            local_feature_dim=local_feature_dim,
            global_feature_dim=global_feature_dim,
            node_feature_dim=node_feature_dim,
            mask_threshold=mask_threshold,
            mask_pool_radius=mask_pool_radius,
            max_nodes=max_nodes,
            heatmap_resolution=self.heatmap_resolution,
            global_resolution=self.global_resolution,
            image_size=self.image_size,
        )
        
        # Global topology module
        # Returns G' (processed global features), Z' (processed tokens), z_star (attention-pooled)
        self.global_topology = GlobalTopologyModule(
            feature_dim=global_feature_dim,
            num_tokens=num_topology_tokens,
            token_dim=topology_token_dim,
            num_layers=4,  # Increased from 2 to 4 for better topology reasoning
            num_heads=8,
        )
        
        sampler_choice = (neighbor_sampler or 'asns').lower()
        if sampler_choice not in {'asns', 'knn'}:
            raise ValueError(f"Unsupported neighbor_sampler='{neighbor_sampler}'. Expected 'asns' or 'knn'.")
        
        if sampler_choice == 'knn':
            self.asns = KNNNeighborSampler(
                k_neighbors=k_neighbors,
                neighbor_radius=neighbor_radius,
            )
        else:
            self.asns = AttentionSparseNeighborSampler(
                feature_dim=node_feature_dim,
                k_neighbors=k_neighbors,
                num_heads=max(1, relation_num_heads // 2),
                            use_entmax=asns_use_entmax,
                            entmax_alpha=asns_entmax_alpha,
                            coverage_label_smoothing=coverage_label_smoothing,
            )
        
        # Store neighbor_radius for backward compatibility with CLI/inference overrides
        self.asns.neighbor_radius = neighbor_radius
        
        # Edge prediction model selection
        if edge_model == 'mlp':
            # Pairwise edge MLP (default, lightweight)
            self.relation_transformer = PairwiseEdgeMLP(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=relation_edge_dim,
                hidden_dim=relation_hidden_dim,
                dropout=relation_dropout,
                use_path_sampling=True,  # Enable path sampling for richer edge features
                path_num_samples=4,  # Number of intermediate points to sample along edge path (excludes endpoints)
                local_feature_dim=local_feature_dim,  # Dimension of local_features for path sampling
                heatmap_resolution=heatmap_resolution,
                image_size=image_size,  # For RGB path sampling coordinate conversion
                use_rgb_path_features=True,  # Sample RGB colors along path (for blue edge detection)
                rgb_path_num_samples=9,  # Number of intermediate points for RGB sampling (more points for better color coverage)
            )
        elif edge_model == 'graph_transformer':
            # Node-based graph transformer
            self.relation_transformer = GraphTransformer(
                node_feature_dim=node_feature_dim,
                z_star_dim=topology_token_dim,
                edge_feature_dim=relation_edge_dim,
                num_layers=relation_transformer_layers,
                num_heads=relation_num_heads,
                hidden_dim=relation_hidden_dim,
                dropout=relation_dropout,
                g_prime_dim=global_feature_dim,
                use_edge_features_in_prediction=True,
                use_path_sampling=True,
                path_num_samples=4,
                local_feature_dim=local_feature_dim,
            )
        elif edge_model == 'edge_aware_transformer':
            # Edge-aware graph transformer (line graph attention, geometric positional encoding)
            # Matching toy model: rgb_feature_dim=32, hidden_dim=128 (default), layers=3 (default), heads=8 (default)
            # Note: edge_feature_dim parameter is kept for interface compatibility but not used
            # (toy model only uses RGB + spatial features, no node features)
            self.relation_transformer = EdgeAwareGraphTransformer(
                node_feature_dim=node_feature_dim,
                z_star_dim=topology_token_dim,
                edge_feature_dim=relation_edge_dim,  # Not used, kept for compatibility
                hidden_dim=relation_hidden_dim,  # Use parameter (default: 256)
                num_layers=relation_transformer_layers,  # Default: 3, matching toy model
                num_heads=relation_num_heads,  # Default: 8, matching toy model
                dropout=relation_dropout,
                use_positional_encoding=True,
                image_size=image_size,
                heatmap_resolution=heatmap_resolution,
                use_rgb_path_features=True,
                rgb_path_num_samples=9,
                rgb_feature_dim=rgb_feature_dim,
                rgb_sequence_model=rgb_sequence_model,
                rgb_seq_layers=rgb_seq_layers,
                rgb_seq_heads=rgb_seq_heads,
                rgb_neighborhood_aggregation=rgb_neighborhood_aggregation,
                rgb_neighborhood_radius=rgb_neighborhood_radius,
                g_prime_dim=global_feature_dim,  # Dimension of G_prime features (topology features)
                topology_feature_dim=0,  # Output dimension for topology features (0 = disabled)
                spatial_feature_dim=4,  # Output dimension for spatial features (after projection)
            )
        else:
            raise ValueError(f"Unsupported edge_model='{edge_model}'. Expected 'mlp', 'graph_transformer', or 'edge_aware_transformer'.")
    
    def forward(
        self,
        images: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            return_intermediates: Whether to return intermediate results
        
        Returns:
            Dictionary with predictions and optionally intermediates
        """
        B = images.shape[0]
        
        # 1. Encode image
        encoder_features = self.encoder(images)  # [B, D, 32, 32]
        
        # 2. Extract dual-stream features
        local_features, global_features = self.dual_stream(encoder_features)
        # local_features: [B, C_L, 32, 32]
        # global_features: [B, C_G, 8, 8]
        
        # 3. Detect nodes and sample descriptors
        node_coords, l_i, g_i, node_mask_logits = self.node_detector(
            local_features,
            global_features
        )
        node_coords_pixel = self.node_detector.latest_coords_pixel
        if node_coords_pixel is None:
            node_coords_pixel = torch.zeros_like(node_coords)
        # node_coords: [B, N, 2] in Local grid space (32×32)
        # l_i: [B, N, D]
        # g_i: [B, N, D]
        # heatmap: [B, 1, 32, 32]
        
        # 4. Global topology
        G_prime, Z_prime, z_star = self.global_topology(global_features)
        # G_prime: [B, C_G, 8, 8]
        # Z_prime: [B, K, token_dim]
        # z_star: [B, token_dim]
        
        # 5. Generate edge candidates with ASNS
        # Create valid mask (non-zero coordinates)
        valid_mask = (node_coords.sum(dim=-1) > 0)  # [B, N]
        
        candidate_mask, attention_weights, attention_scores = self.asns(
            l_i,  # Local descriptors
            g_i,  # Global descriptors
            node_coords,
            valid_mask
        )
        # candidate_mask: [B, N, N] (binary, Top-K̂ per row)
        # attention_weights: [B, N, N] (post-sparsemax)
        # attention_scores: [B, N, N] (pre-sparsemax, for coverage loss)
        
        # 6. Predict edges with relation transformer
        # Use keyword arguments for optional parameters to ensure flexibility and avoid parameter order issues
        # This allows each edge model to accept only the features it needs, making it easy to add new features in the future
        edge_logits = self.relation_transformer(
            l_i,  # Local descriptors [B, N, D]
            g_i,  # Global descriptors [B, N, D]
            node_coords,  # Node coordinates [B, N, 2] in heatmap space
            z_star,  # Global embedding [B, D_z]
            candidate_mask,  # Candidate edge mask [B, N, N]
            G_prime=G_prime,  # Processed global features [B, C_G, 8, 8] (optional)
            local_features=local_features,  # Local features [B, C_L, 32, 32] for path sampling (optional)
            images=images,  # RGB images [B, 3, 512, 512] for color path sampling (optional)
            valid_mask=valid_mask,  # Valid node mask [B, N] (optional)
        )
        # edge_logits: [B, N, N]
        
        # Prepare output
        output = {
            'edge_logits': edge_logits,
            'edge_probs': torch.sigmoid(edge_logits),
            'node_coords': node_coords,
            'node_coords_pixel': node_coords_pixel,
            'l_i': l_i,
            'g_i': g_i,
            'node_mask_logits': node_mask_logits,
            'candidate_mask': candidate_mask,
            'attention_scores': attention_scores,  # Always return for gradient flow in coverage/budget losses
            'z_star': z_star,
        }
        
        if return_intermediates:
            output.update({
                'encoder_features': encoder_features,
                'local_features': local_features,
                'global_features': global_features,
                'G_prime': G_prime,
                'Z_prime': Z_prime,
            })
        
        return output
