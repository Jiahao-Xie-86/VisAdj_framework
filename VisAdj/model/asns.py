"""
Attention-Sparse Neighbor Sampler (ASNS)

Selects edge candidates using attention over node descriptors.
q_i = Wq * l_i, k_j = Wk * g_j with geometric bias.
Uses entmax/sparsemax for sparse attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def entmax_bisect(inputs, alpha=1.5, dim=-1, n_iter=50):
    """
    Entmax activation (generalization of softmax and sparsemax).
    
    Args:
        inputs: Input tensor
        alpha: Entmax parameter (1.0 = softmax, 1.5 = sparsemax, 2.0 = hardmax)
        dim: Dimension to apply entmax
        n_iter: Number of bisection iterations
    
    Returns:
        Entmax probabilities
    """
    if alpha == 1.0:
        return F.softmax(inputs, dim=dim)
    elif alpha == 2.0:
        return sparsemax(inputs, dim=dim)
    
    # Bisection method for general alpha
    inputs_sorted, _ = torch.sort(inputs, descending=True, dim=dim)
    inputs_shifted = inputs_sorted - inputs_sorted[:, :1] if dim == -1 else inputs_sorted - inputs_sorted[:1]
    
    # Initialize tau bounds
    tau_low = torch.zeros_like(inputs_sorted[:, :1])
    tau_high = torch.ones_like(inputs_sorted[:, :1])
    
    for _ in range(n_iter):
        tau = (tau_low + tau_high) / 2.0
        # Compute p with numerical stability
        # p = max(0, (x - tau) / (alpha - 1))^(1/(alpha-1))
        diff = (inputs_shifted - tau) / (alpha - 1.0)
        diff_clamped = torch.clamp(diff, min=0.0, max=10.0)  # Clamp to prevent overflow
        # Use exp for numerical stability: x^y = exp(y * log(x))
        # But we need to handle the case where diff_clamped is 0
        p = torch.where(
            diff_clamped > 1e-10,
            torch.exp((1.0 / (alpha - 1.0)) * torch.log(diff_clamped)),
            torch.zeros_like(diff_clamped)
        )
        p_sum = p.sum(dim=dim, keepdim=True)
        tau_high = torch.where(p_sum > 1.0, tau, tau_high)
        tau_low = torch.where(p_sum <= 1.0, tau, tau_low)
    
    tau = (tau_low + tau_high) / 2.0
    # Final computation with same stability
    diff = (inputs_shifted - tau) / (alpha - 1.0)
    diff_clamped = torch.clamp(diff, min=0.0, max=10.0)
    p = torch.where(
        diff_clamped > 1e-10,
        torch.exp((1.0 / (alpha - 1.0)) * torch.log(diff_clamped)),
        torch.zeros_like(diff_clamped)
    )
    
    # Reorder back to original order
    _, indices = torch.sort(inputs, descending=True, dim=dim)
    p_reordered = torch.zeros_like(inputs)
    # Ensure p has the same dtype as p_reordered
    p = p.to(dtype=p_reordered.dtype)
    if dim == -1:
        p_reordered.scatter_(dim, indices, p)
    else:
        p_reordered.scatter_(dim, indices, p)
    
    return p_reordered


def sparsemax(inputs, dim=-1):
    """
    Sparsemax activation (alpha=2.0 entmax).
    
    Args:
        inputs: Input tensor
        dim: Dimension to apply sparsemax
    
    Returns:
        Sparsemax probabilities
    """
    # Sort inputs descending
    inputs_sorted, indices = torch.sort(inputs, descending=True, dim=dim)
    dim_size = inputs.shape[dim]
    
    # Compute cumulative sums
    cumsum = torch.cumsum(inputs_sorted, dim=dim)
    
    # Find threshold
    k = torch.arange(1, dim_size + 1, device=inputs.device, dtype=inputs.dtype)
    if dim == -1:
        k = k.view(1, -1)
    else:
        k = k.view(-1, 1)
    
    threshold = (cumsum - 1.0) / k
    threshold_prev = F.pad(threshold[:, :-1], (1, 0) if dim == -1 else (0, 1), value=float('-inf'))
    
    # Find support
    support = (inputs_sorted > threshold_prev).float()
    k_support = support.sum(dim=dim, keepdim=True).float()
    tau = (cumsum.gather(dim, (k_support.long() - 1).clamp(min=0)) - 1.0) / k_support.clamp(min=1.0)
    
    # Compute sparsemax
    output = torch.clamp(inputs_sorted - tau, min=0.0)
    
    # Reorder back
    output_reordered = torch.zeros_like(inputs)
    output_reordered.scatter_(dim, indices, output)
    
    return output_reordered


class AttentionSparseNeighborSampler(nn.Module):
    """
    Attention-Sparse Neighbor Sampler.
    
    Uses attention mechanism to select plausible neighbors for each node:
    - q_i = Wq * l_i_detached
    - k_j = Wk * g_j
    - Scores with geometric bias
    - Entmax/sparsemax for sparse weights
    - Top-K̂ straight-through binary mask
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        k_neighbors: int = 12,
        num_heads: int = 4,
        use_entmax: bool = True,
        entmax_alpha: float = 1.5,
        geom_bias_dim: int = 4,
        coverage_focal_gamma: float = 1.0,  # Focal weight gamma for hard examples (reduced from 1.5 to 1.0: less aggressive, allow overall improvement)
        coverage_margin: float = 1.0,  # Margin for ranking loss (GT scores should be higher than non-GT by this margin)
        coverage_margin_weight: float = 0.0,  # Weight for margin loss component (set to 0.0 to disable margin loss)
        use_pairwise_ranking: bool = False,  # Whether to use pairwise ranking (stronger but more expensive) instead of max-based
        coverage_label_smoothing: float = 0.1,  # Label smoothing for coverage loss (0.0 = no smoothing, 0.1 = 10% smoothing, default: 0.1)
    ):
        """
        Args:
            feature_dim: Node feature dimension
            k_neighbors: Number of neighbors to select per node (K̂)
            num_heads: Number of attention heads
            use_entmax: Whether to use entmax (True) or sparsemax (False)
            entmax_alpha: Entmax alpha parameter (1.0 = softmax, 1.5 = sparsemax, 2.0 = hardmax)
                         Set to 1.0 to use softmax activation
            geom_bias_dim: Dimension of geometric bias features
            coverage_focal_gamma: Gamma parameter for focal-style weighting of hard examples (default: 2.0)
            coverage_margin: Margin threshold for ranking loss (default: 1.0)
            coverage_margin_weight: Weight for margin loss component (default: 0.5)
        """
        super().__init__()
        
        self.k_neighbors = k_neighbors
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.use_entmax = use_entmax
        self.entmax_alpha = entmax_alpha
        self.coverage_focal_gamma = coverage_focal_gamma
        self.coverage_margin = coverage_margin
        self.coverage_margin_weight = coverage_margin_weight
        self.use_pairwise_ranking = use_pairwise_ranking
        self.coverage_label_smoothing = coverage_label_smoothing
        
        # Query projection (for l_i)
        self.Wq = nn.Linear(feature_dim, feature_dim)
        
        # Key projection (for g_j)
        self.Wk = nn.Linear(feature_dim, feature_dim)
        
        # Geometric bias MLP: (Δx, Δy, d, θ) → bias
        self.geom_bias_mlp = nn.Sequential(
            nn.Linear(geom_bias_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
        )
        
        # Temperature (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0 / math.sqrt(feature_dim)))
    
    def forward(
        self,
        l_i: torch.Tensor,  # Local descriptors (detached) [B, N, D]
        g_j: torch.Tensor,  # Global descriptors [B, N, D]
        node_coords: torch.Tensor,  # Node coordinates [B, N, 2]
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate edge mask using attention.
        
        Args:
            l_i: Local node descriptors (detached) [B, N, D]
            g_j: Global node descriptors [B, N, D]
            node_coords: Node coordinates [B, N, 2] in Local grid space
            valid_mask: Valid node mask [B, N] (optional)
        
        Returns:
            candidate_mask: Candidate edge mask [B, N, N] (binary, Top-K̂ per row)
            attention_scores: Attention scores [B, N, N]
        """
        B, N, D = l_i.shape
        
        # Project to queries and keys
        # CRITICAL: Even though l_i is detached, we still compute q_i for forward pass
        # But gradients will only flow through k_j (from g_j) to Wk
        q_i = self.Wq(l_i)  # [B, N, D] - no gradients through l_i (detached)
        k_j = self.Wk(g_j)  # [B, N, D] - SHOULD have gradients through g_j
        
        # Reshape for multi-head attention
        q_i = q_i.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, N, D/H]
        k_j = k_j.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, N, D/H]
        
        # Compute attention scores: (q_i · k_j) / sqrt(d)
        # Gradients flow through k_j even if q_i is detached
        scores = torch.matmul(q_i, k_j.transpose(-2, -1)) / (self.temperature * math.sqrt(D // self.num_heads))  # [B, H, N, N]
        
        # Add geometric bias
        geom_bias = self._compute_geometric_bias(node_coords)  # [B, N, N, H]
        geom_bias = geom_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        scores = scores + geom_bias
        
        # Remove self-loops: mask diagonal before activation
        eye_mask = torch.eye(N, device=scores.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        # Use very low but finite score instead of -inf to preserve gradient flow
        very_low_score = torch.tensor(-1e4, device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(eye_mask, very_low_score)
        
        # Apply valid mask robustly: avoid rows/cols becoming fully -inf
        if valid_mask is not None:
            invalid_mask = ~valid_mask  # [B, N]
            
            # Mask invalid query rows
            scores = scores.masked_fill(invalid_mask.unsqueeze(1).unsqueeze(3), very_low_score)
            # Mask invalid key columns
            scores = scores.masked_fill(invalid_mask.unsqueeze(1).unsqueeze(2), very_low_score)
            
            # Robust handling: if a row has NO valid positions (all columns invalid),
            # ensure it doesn't break entmax by having at least one finite value
            # Check if row_max is <= very_low_score (meaning all positions are invalid)
            row_max = scores.max(dim=-1, keepdim=True)[0]  # [B, H, N, 1]
            fully_invalid_rows = (row_max <= very_low_score + 1e-3).squeeze(-1)  # [B, H, N]
            if fully_invalid_rows.any():
                # For fully invalid rows, set diagonal to a small finite value so entmax doesn't break
                # This handles edge cases where a node has no valid neighbors
                uniform_low = torch.tensor(-10.0, device=scores.device, dtype=scores.dtype)
                # Set diagonal for fully invalid rows (self-loop, but at least finite)
                eye_indices = torch.arange(N, device=scores.device)
                for b in range(B):
                    for h in range(scores.shape[1]):
                        invalid_row_indices = torch.where(fully_invalid_rows[b, h])[0]
                        if len(invalid_row_indices) > 0:
                            scores[b, h, invalid_row_indices, invalid_row_indices] = uniform_low
        
        # Reduce across heads using max (not average) to avoid diluting gradients
        # Do this BEFORE stabilization so we stabilize the final scores
        scores = scores.max(dim=1)[0]  # [B, N, N] - max over heads
        
        # Stabilize scores: subtract row-wise max and clamp to prevent overflow/underflow
        # This helps with numerical stability in entmax/sparsemax
        # CRITICAL: Only subtract max if it's positive, otherwise scores become too negative
        row_max = scores.max(dim=-1, keepdim=True)[0]  # [B, N, 1]
        # Subtract max only if max > very_low_score (i.e., row has valid positions)
        # For rows with only invalid positions, keep scores as-is (they're already -10.0 on diagonal)
        valid_row_mask = (row_max > very_low_score + 1e-3)  # [B, N, 1]
        scores = torch.where(
            valid_row_mask,
            scores - row_max,  # Subtract max for valid rows (standard stabilization)
            scores  # Keep as-is for invalid rows (already have -10.0 on diagonal)
        )
        scores = scores.clamp(min=-50.0, max=50.0)  # Clamp to reasonable range
        
        # CRITICAL FIX: Ensure at least one score per row is positive for entmax to work
        # After stabilization, max should be 0, but if all scores are negative, entmax produces zeros
        # Add a small positive constant to the max position to ensure entmax produces non-zero weights
        row_max_after = scores.max(dim=-1, keepdim=True)[0]  # [B, N, 1]
        # For rows where max is <= 0, add a small positive value to ensure entmax works
        needs_boost = (row_max_after <= 0.0)  # [B, N, 1]
        if needs_boost.any():
            # Find the max position for each row and add a small positive value
            row_max_indices = scores.argmax(dim=-1, keepdim=True)  # [B, N, 1]
            boost_value = torch.tensor(1e-3, device=scores.device, dtype=scores.dtype)
            # Use non-in-place scatter to preserve gradients
            max_values = scores.gather(-1, row_max_indices) + boost_value  # [B, N, 1]
            scores = scores.scatter(-1, row_max_indices, max_values)  # Non-in-place
        
        # Apply activation function: softmax, sparsemax, or entmax
        if self.use_entmax:
            if self.entmax_alpha == 1.0:
                # Use softmax (alpha=1.0 in entmax is equivalent to softmax)
                attention_weights = F.softmax(scores, dim=-1)  # [B, N, N]
            else:
                # Use entmax with specified alpha
                attention_weights = entmax_bisect(scores, alpha=self.entmax_alpha, dim=-1)  # [B, N, N]
        else:
            # Use sparsemax (alpha=2.0 entmax)
            attention_weights = sparsemax(scores, dim=-1)  # [B, N, N]
        
        # Top-K̂ straight-through: select top-k neighbors per node
        top_k_values, top_k_indices = attention_weights.topk(
            k=min(self.k_neighbors, N),
            dim=-1
        )  # [B, N, K̂]
        
        # Create binary candidate mask with straight-through estimator
        # Forward: use hard Top-K mask; Backward: gradients flow through full soft attention weights
        candidate_mask = torch.zeros(B, N, N, device=l_i.device, dtype=torch.float32)
        batch_indices = torch.arange(B, device=l_i.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        node_indices = torch.arange(N, device=l_i.device).unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        
        # Create hard binary mask for top-k positions
        candidate_mask_hard = torch.zeros_like(candidate_mask)
        candidate_mask_hard[batch_indices, node_indices, top_k_indices] = 1.0
        
        # Straight-through estimator: hard mask in forward, soft weights in backward
        # Gradients flow through ALL attention_weights, enabling learning
        candidate_mask = candidate_mask_hard + (attention_weights - attention_weights.detach())
        
        # Do NOT symmetrize here - keep row-wise supervision
        # Symmetry will be applied downstream if needed
        
        # Return: candidate_mask, attention_weights (post-sparsemax), attention_scores (pre-sparsemax)
        # attention_scores is used for coverage loss to provide strong gradients
        return candidate_mask, attention_weights, scores
    
    def _compute_geometric_bias(
        self,
        node_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geometric bias from node coordinates.
        
        Args:
            node_coords: Node coordinates [B, N, 2] (x, y)
        
        Returns:
            Geometric bias [B, N, N, H]
        """
        B, N, _ = node_coords.shape
        
        # Compute pairwise differences
        coords_i = node_coords.unsqueeze(2)  # [B, N, 1, 2]
        coords_j = node_coords.unsqueeze(1)  # [B, 1, N, 2]
        
        delta = coords_j - coords_i  # [B, N, N, 2]
        delta_x = delta[:, :, :, 0]  # [B, N, N]
        delta_y = delta[:, :, :, 1]  # [B, N, N]
        
        # Distance
        d = torch.sqrt(delta_x ** 2 + delta_y ** 2 + 1e-8)  # [B, N, N]
        
        # Angle
        theta = torch.atan2(delta_y, delta_x)  # [B, N, N]
        
        # Normalize features
        delta_x_norm = delta_x / (d + 1e-8)
        delta_y_norm = delta_y / (d + 1e-8)
        d_norm = d / (d.max() + 1e-8)
        theta_norm = theta / (math.pi + 1e-8)
        
        # Concatenate geometric features
        geom_features = torch.stack([delta_x_norm, delta_y_norm, d_norm, theta_norm], dim=-1)  # [B, N, N, 4]
        
        # Compute bias through MLP
        geom_bias = self.geom_bias_mlp(geom_features)  # [B, N, N, H]
        
        return geom_bias
    
    def compute_coverage_loss(
        self,
        candidate_mask: torch.Tensor,
        target_adj: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Coverage loss: treat attention as a "neighbor distribution" and
        use cross-entropy to fit the ground-truth neighbor distribution.

        For each node i:
          - P_{i·} : attention distribution (entmax / sparsemax output, each row sums to 1)
          - T_{i·} : target distribution from GT adjacency (with label smoothing)
                     (smoothed uniform over GT neighbors, smoothed uniform elsewhere)
          - loss_i = - sum_{j} T_{ij} * log P_{ij}  (cross-entropy)

        Final loss = first average over nodes per sample, then average over batch.
        This ensures nodes with higher degrees do not overly dominate the loss.

        IMPORTANT: Uses full attention_weights (not top-k cut-off) to allow learning
        of all neighbors, not just those in the top-k candidate set.

        Args:
            candidate_mask: Candidate mask [B, N, N] (not used, kept for compatibility)
            target_adj: Target adjacency [B, N, N] or list
            attention_scores: Pre-sparsemax attention scores [B, N, N] (optional, for fallback)
            attention_weights: Post-sparsemax attention weights [B, N, N] (preferred)
                              Uses FULL weights, not top-k cut-off
            valid_mask: Valid node mask [B, N] - True for valid nodes (optional)
        
        Returns:
            Coverage loss (scalar tensor)
        """
        device = candidate_mask.device
        eps = 1e-8
        label_smoothing = self.coverage_label_smoothing

        # 1. Handle the case where target_adj is a list (pad to a common size)
        if isinstance(target_adj, list):
            max_n = max(adj.shape[0] for adj in target_adj if adj is not None)
            B = len(target_adj)
            target_tensor = torch.zeros(B, max_n, max_n, device=device, dtype=torch.float32)
            for i, adj in enumerate(target_adj):
                if adj is None:
                    continue
                n = adj.shape[0]
                if isinstance(adj, torch.Tensor):
                    target_tensor[i, :n, :n] = adj.to(device).float()
                else:
                    import numpy as np
                    target_tensor[i, :n, :n] = torch.from_numpy(adj).float().to(device)
            target_adj = target_tensor
        else:
            target_adj = target_adj.to(device).float()

        B_c, N_cand, _ = candidate_mask.shape
        B_t, N_gt, _ = target_adj.shape
        assert B_c == B_t, "Batch size mismatch between candidate_mask and target_adj"
        # Use B_t as the batch size (works for both list and tensor cases)
        B = B_t

        # 2. Align sizes (in case candidate and GT use different N)
        min_n = min(N_cand, N_gt)
        target_adj = target_adj[:, :min_n, :min_n]

        # 3. Get attention distribution P
        #    CRITICAL: Use FULL attention_weights (not top-k cut-off) to allow learning
        #    of all neighbors during training. Top-k is only used for candidate_mask
        #    (which masks the relation transformer), but loss uses full distribution.
        if attention_weights is not None:
            P = attention_weights[:, :min_n, :min_n].to(device)
        elif attention_scores is not None:
            # Fallback: convert scores to a probability distribution
            P = F.softmax(attention_scores[:, :min_n, :min_n], dim=-1)
        else:
            # No attention information available: return zero loss but keep graph connection
            return candidate_mask.sum() * 0.0

        # 4. Apply valid_mask if provided
        #    We only supervise edges between valid nodes and renormalize P along rows.
        if valid_mask is not None:
            valid = valid_mask[:, :min_n].bool()  # [B, N]
            # Only keep GT edges between valid nodes
            valid_2d = valid.unsqueeze(1) & valid.unsqueeze(2)  # [B, N, N]
            target_adj = target_adj * valid_2d.float()

            # Mask out probabilities involving invalid nodes and renormalize per row
            P = P * valid_2d.float()
            row_sum = P.sum(dim=-1, keepdim=True).clamp_min(eps)
            P = P / row_sum
        else:
            valid = None

        # Optionally remove self-loops from GT (if GT does not contain i→i edges)
        eye = torch.eye(min_n, device=device).bool().unsqueeze(0)  # [1, N, N]
        target_adj = target_adj.masked_fill(eye, 0.0)

        # 5. Build GT mask indicating which (i, j) are GT neighbors
        gt_mask = target_adj > 0.5  # [B, N, N] bool

        # 6. Apply label smoothing to target distribution
        #    For each node i:
        #      - GT neighbors: (1 - smoothing) / |N(i)| + smoothing / (N_valid - 1)
        #      - Non-GT neighbors: smoothing / (N_valid - 1)
        #    This prevents overconfidence and improves generalization
        gt_counts = gt_mask.sum(dim=-1, keepdim=True)  # [B, N, 1] - number of GT neighbors per node
        
        # Compute number of valid positions per row (accounting for valid_mask if provided)
        if valid is not None:
            # Per-row valid positions: number of valid nodes minus self (exclude self-loop)
            # For each sample, count valid nodes (same for all nodes in the sample)
            num_valid_per_sample = valid.sum(dim=-1) - 1  # [B] - number of valid nodes per sample (minus self)
            num_valid_per_sample = num_valid_per_sample.clamp_min(1.0)  # At least 1 to avoid division by zero
            # Expand to [B, N, 1] - same value for all nodes in each sample
            num_valid_positions_per_row = num_valid_per_sample.unsqueeze(-1).unsqueeze(-1).expand(B, min_n, 1)  # [B, N, 1]
        else:
            # Global: min_n - 1 (exclude self-loop), but ensure at least 1
            num_valid_positions_global = max(1, min_n - 1)  # Scalar, at least 1
            num_valid_positions_per_row = torch.full((B, min_n, 1), num_valid_positions_global, 
                                                     device=device, dtype=torch.float32)  # [B, N, 1]
        
        # Build smoothed target distribution T
        if label_smoothing > 0.0:
            # Smoothed probability for GT neighbors
            gt_prob = (1.0 - label_smoothing) / gt_counts.clamp_min(1.0) + label_smoothing / num_valid_positions_per_row  # [B, N, 1]
            # Smoothed probability for non-GT neighbors
            non_gt_prob = label_smoothing / num_valid_positions_per_row  # [B, N, 1]
            
            # Expand to [B, N, N, 1] for broadcasting with gt_mask
            # gt_prob: [B, N, 1] -> [B, N, 1, 1] -> expand to [B, N, N, 1]
            gt_prob_expanded = gt_prob.unsqueeze(2).expand(B, min_n, min_n, 1)  # [B, N, N, 1]
            non_gt_prob_expanded = non_gt_prob.unsqueeze(2).expand(B, min_n, min_n, 1)  # [B, N, N, 1]
            
            # Create target distribution
            T = torch.where(
                gt_mask.unsqueeze(-1),  # [B, N, N, 1]
                gt_prob_expanded,  # [B, N, N, 1]
                non_gt_prob_expanded  # [B, N, N, 1]
            ).squeeze(-1)  # [B, N, N]
            
            # Ensure row-wise normalization (each row sums to 1)
            # For rows with GT neighbors: |N(i)| * gt_prob + (N-1-|N(i)|) * non_gt_prob should ≈ 1
            # But due to smoothing, we renormalize to ensure exact sum = 1
            row_sum_T = T.sum(dim=-1, keepdim=True)  # [B, N, 1]
            T = T / row_sum_T.clamp_min(eps)
        else:
            # No smoothing: hard targets (uniform over GT neighbors, zero elsewhere)
            T = gt_mask.float() / gt_counts.clamp_min(1.0)  # [B, N, N]

        # 7. Compute cross-entropy loss: -sum_j T_{ij} * log P_{ij}
        logP = torch.log(P.clamp_min(eps))  # [B, N, N]
        per_edge_loss = -T * logP  # [B, N, N] - cross-entropy per edge

        # 8. Per-node loss: sum over all neighbors (not just GT)
        #    This is the full cross-entropy for each node's neighbor distribution
        loss_per_node = per_edge_loss.sum(dim=-1)  # [B, N] - loss per node

        # 9. Average over nodes per sample (first average)
        #    This ensures nodes with higher degrees don't dominate
        #    Each sample contributes equally regardless of node degrees
        row_mask = gt_counts.squeeze(-1) > 0  # [B, N] - nodes with at least one GT neighbor
        if valid is not None:
            # Also require node i itself to be valid
            row_mask = row_mask & valid

        # For each sample, average over valid nodes
        loss_per_sample = []  # List of per-sample losses
        for b in range(B):
            sample_node_mask = row_mask[b]  # [N]
            if sample_node_mask.any():
                # Average over valid nodes in this sample
                sample_loss = loss_per_node[b][sample_node_mask].mean()
                loss_per_sample.append(sample_loss)
        
        # 10. Average over batch (second average)
        #     This gives equal weight to each sample
        if len(loss_per_sample) > 0:
            loss = torch.stack(loss_per_sample).mean()
        else:
            # No valid samples – return 0 but preserve gradient graph
            loss = P.sum() * 0.0

        # 11. Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            # In case of numerical issues, fall back to a zero-like loss tied to P
            loss = P.mean() * 0.0

        return loss
    
    # REMOVED: compute_budget_loss method - budget loss is no longer used
    # Budget loss was removed to simplify training and focus on coverage loss
