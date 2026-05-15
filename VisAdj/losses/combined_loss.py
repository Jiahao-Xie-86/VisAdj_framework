"""
Loss Functions

Node heatmap loss, edge prediction loss, coverage loss, budget loss,
and optional edge mask loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict


class NodeMaskLoss(nn.Module):
    """Loss for dense node mask prediction.
    
    Addresses class imbalance (17% positive, 83% negative pixels) by combining:
    - Weighted BCE Loss: Balances positive/negative pixel contributions
    - Focal Loss: Focuses on hard examples, down-weights easy negatives
    - MSE Loss: Provides spatial accuracy regularization
    
    This combination prevents the model from collapsing to uniform predictions.
    """
    
    def __init__(
        self,
        mse_weight: float = 0.30,  # Increased from 0.25 - MSE provides stronger gradient signal
        dice_weight: float = 0.25,  # Increased from 0.15 - Dice loss helps with imbalanced segmentation
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        weighted_bce_weight: float = 0.20,  # Reduced to make room for stronger sparsity/background penalties
        focal_weight: float = 0.0,  # Disabled - focal loss down-weights uniform predictions (counterproductive)
        sparsity_weight: float = 0.10,  # REDUCED: Prevent amplification of gradient issues
        mean_matching_weight: float = 0.15,  # INCREASED: Stronger supervision for mean matching
        background_weight: float = 0.40,  # New: penalize spurious background activations
        use_dynamic_pos_weight: bool = True,
        fixed_pos_weight: float = 4.8,
        pos_weight_scale: float = 0.5,
        target_mean: float = 0.22,  # Target heatmap mean (sparse)
    ):
        """
        Args:
            mse_weight: Weight for MSE loss component
            focal_alpha: Focal loss balance factor (typically 0.25)
            focal_gamma: Focal loss focusing parameter (typically 2.0)
            weighted_bce_weight: Weight for weighted BCE in final combination
            focal_weight: Weight for focal loss in final combination
            use_dynamic_pos_weight: If True, compute pos_weight per batch; else use fixed_pos_weight
            fixed_pos_weight: Fixed positive class weight (used if use_dynamic_pos_weight=False)
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weighted_bce_weight = weighted_bce_weight
        self.focal_weight = focal_weight
        self.sparsity_weight = sparsity_weight
        self.mean_matching_weight = mean_matching_weight
        self.background_weight = background_weight
        self.use_dynamic_pos_weight = use_dynamic_pos_weight
        self.fixed_pos_weight = fixed_pos_weight
        self.pos_weight_scale = pos_weight_scale
        self.target_mean = target_mean
        
        # Standard BCE (will be used with pos_weight)
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self._last_pos_weight = None
    
    def _compute_focal_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss for binary classification.
        
        Formula: FL = -α(1-p_t)^γ log(p_t)
        where p_t = p if y=1, else (1-p)
        
        Args:
            pred_logits: Predicted logits [B, 1, H, W]
            target: Target probabilities [B, 1, H, W]
        
        Returns:
            Focal loss value (scalar)
        """
        # Compute BCE for each pixel
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        
        # Compute probabilities
        pred_probs = torch.sigmoid(pred_logits)
        
        # Compute p_t: probability of true class
        # For positive pixels (target > 0.5): p_t = pred_probs
        # For negative pixels (target <= 0.5): p_t = 1 - pred_probs
        p_t = torch.where(target > 0.5, pred_probs, 1.0 - pred_probs)
        
        # Focal weight: α(1-p_t)^γ
        focal_weight = self.focal_alpha * (1.0 - p_t) ** self.focal_gamma
        
        # Focal loss
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()
    
    def _compute_dice_loss(
        self,
        pred_probs: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute Dice Loss for binary segmentation.
        
        Dice Loss = 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
        
        Args:
            pred_probs: Predicted probabilities [B, 1, H, W]
            target: Target probabilities [B, 1, H, W]
            smooth: Smoothing factor to avoid division by zero
        
        Returns:
            Dice loss value (scalar)
        """
        # Flatten tensors
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        
        # Dice loss (1 - dice coefficient)
        dice_loss = 1.0 - dice
        
        return dice_loss
    
    def forward(
        self,
        pred_mask_logits: torch.Tensor,  # [B, 1, H, W]
        target_mask: torch.Tensor,  # [B, 1, H, W] or [B, H, W]
    ) -> torch.Tensor:
        """
        Compute node heatmap loss using Weighted BCE + Focal Loss + MSE.
        
        Args:
            pred_mask_logits: Predicted mask [B, 1, H, W] (logits)
            target_mask: Target mask [B, 1, H, W] or [B, H, W] (probabilities)
        
        Returns:
            Loss value (Weighted BCE + Focal + MSE)
        """
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        
        # Ensure same size
        if pred_mask_logits.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask,
                size=pred_mask_logits.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Compute positive class weight for weighted BCE
        if self.use_dynamic_pos_weight:
            # Compute per batch: pos_weight = neg_count / pos_count
            pos_mask = target_mask > 0.5
            pos_count = pos_mask.sum().float() + 1e-8  # Avoid division by zero
            neg_count = (~pos_mask).sum().float() + 1e-8
            pos_weight = neg_count / pos_count
            # Clamp to reasonable range [1.0, 20.0] - increased to handle higher imbalance
            pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0)
        else:
            pos_weight = torch.tensor([self.fixed_pos_weight], device=pred_mask_logits.device)
        pos_weight = pos_weight * self.pos_weight_scale
        self._last_pos_weight = pos_weight.clone().detach() if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight)
        
        # Weighted BCE loss (on logits)
        # Use reduction='none' then apply pos_weight manually for better control
        bce_per_pixel = self.bce_loss_fn(pred_mask_logits, target_mask)  # [B, 1, H, W]
        
        # Apply positive weight: multiply positive pixels by pos_weight
        # Ensure pos_weight is a scalar for broadcasting
        if pos_weight.dim() > 0:
            pos_weight = pos_weight.item()
        # Scale loss contribution smoothly using target intensity as a soft weight
        soft_pos_weight = 1.0 + (pos_weight - 1.0) * target_mask
        weighted_bce_per_pixel = bce_per_pixel * soft_pos_weight
        weighted_bce_loss = weighted_bce_per_pixel.mean()
        
        # Focal loss (on logits)
        focal_loss = self._compute_focal_loss(pred_mask_logits, target_mask)
        
        # MSE loss (on probabilities) - provides spatial accuracy
        pred_probs = torch.sigmoid(pred_mask_logits)
        mse_loss = F.mse_loss(pred_probs, target_mask)
        
        # Dice loss (on probabilities) - helps with imbalanced segmentation
        dice_loss = self._compute_dice_loss(pred_probs, target_mask)
        
        # CRITICAL FIX: Mean-matching loss - explicitly match target mean
        # FIXED: Use per-pixel version to avoid gradient dilution (1/N scaling)
        # CRITICAL: Only apply to non-background pixels to avoid conflict with background_loss
        # Background pixels should be 0, not target_mean, so mean_matching should only apply to foreground
        background_mask = (target_mask < 0.1).float()
        foreground_mask = 1.0 - background_mask  # Non-background pixels
        pred_mean = pred_probs.mean()
        target_mean = target_mask.mean()
        # Per-pixel MSE: each pixel gets gradient proportional to (pred - target_mean)
        # But only for foreground pixels (background pixels handled by background_loss)
        mean_matching_loss = F.mse_loss(
            pred_probs * foreground_mask,
            torch.full_like(pred_probs, target_mean) * foreground_mask
        )
        
        # CRITICAL FIX: Sparsity regularization - penalize predictions above target mean
        # FIXED: Removed exponential penalty to prevent gradient explosion and vanishing
        # Use per-pixel sparsity for stronger gradient signals (no mean dilution)
        # target_mean is already computed above, reuse it
        # FIXED: Remove redundant sparsity_mask - F.relu already handles the > target_mean condition
        sparsity_penalty_per_pixel = F.relu(pred_probs - target_mean)  # Per-pixel penalty (already non-negative)
        # Use quadratic only (removed cubic to prevent gradient instability)
        sparsity_loss = (sparsity_penalty_per_pixel ** 2).mean()

        # Background suppression: penalize predictions that exceed the target in background regions
        # CRITICAL: This works with mean_matching_loss now - no conflict!
        # mean_matching_loss only applies to foreground, background_loss only applies to background
        background_probs = pred_probs * background_mask
        background_loss = (background_probs ** 2).mean()
        
        # Combined loss: Weighted BCE + Focal + MSE + Dice + Sparsity + Mean Matching
        total_loss = (
            self.weighted_bce_weight * weighted_bce_loss +
            self.focal_weight * focal_loss +
            self.mse_weight * mse_loss +
            self.dice_weight * dice_loss +
            self.sparsity_weight * sparsity_loss +
            self.mean_matching_weight * mean_matching_loss +
            self.background_weight * background_loss
        )
        
        # Store individual components for monitoring (detached for logging only)
        # These will be used to verify gradient contributions
        self._last_components = {
            'weighted_bce': weighted_bce_loss.detach(),
            'focal': focal_loss.detach(),
            'mse': mse_loss.detach(),
            'dice': dice_loss.detach(),
            'sparsity': sparsity_loss.detach(),
            'mean_matching': mean_matching_loss.detach(),
            'background': background_loss.detach(),
        }
        
        return total_loss


def match_nodes_hungarian(
    pred_coords: torch.Tensor,  # [B, N_pred, 2] in image space
    gt_coords: torch.Tensor,    # [B, N_gt, 2] in image space
    valid_mask: torch.Tensor,   # [B, N_pred] - True for valid predicted nodes
    max_distance: float = 100.0,  # Maximum distance for matching (pixels)
    gt_valid_mask: Optional[torch.Tensor] = None  # [B, N_gt] - True for valid GT nodes (to filter padded nodes)
) -> torch.Tensor:
    """
    Match predicted nodes to GT nodes using Hungarian algorithm (optimal matching).
    
    OPTIMIZED: Batched distance computation on GPU, then Hungarian on CPU.
    This provides optimal matches while minimizing GPU-CPU transfer overhead.
    
    Args:
        pred_coords: Predicted node coordinates [B, N_pred, 2] in image space
        gt_coords: GT node coordinates [B, N_gt, 2] in image space
        valid_mask: [B, N_pred] - True for valid predicted nodes
        max_distance: Maximum distance for matching (pixels)
    
    Returns:
        permutation: [B, N_pred] where permutation[b, i] = j means
                     predicted node i matches GT node j
                     -1 if no match found
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise ImportError("scipy is required for Hungarian matching. Install with: pip install scipy")
    
    B, N_pred, _ = pred_coords.shape
    B_gt, N_gt, _ = gt_coords.shape
    
    device = pred_coords.device
    permutation = torch.full((B, N_pred), -1, dtype=torch.long, device=device)
    
    # Early return if no GT nodes (nothing to match)
    if N_gt == 0:
        return permutation
    
    # OPTIMIZATION: Compute all distance matrices on GPU in one batch
    # This is much faster than computing them one-by-one
    dists = torch.cdist(pred_coords, gt_coords)  # [B, N_pred, N_gt] - batched on GPU!
    
    # Mask invalid distances
    valid_mask_expanded = valid_mask.unsqueeze(-1)  # [B, N_pred, 1]
    dists = dists.masked_fill(~valid_mask_expanded, float('inf'))
    
    # CRITICAL FIX: Filter out padded GT nodes (if gt_valid_mask is provided)
    # This matches the behavior of greedy matching which only considers valid GT nodes
    if gt_valid_mask is not None:
        # gt_valid_mask: [B, N_gt], dists: [B, N_pred, N_gt]
        # Expand to [B, 1, N_gt] to broadcast correctly with dists
        gt_valid_mask_expanded = gt_valid_mask.unsqueeze(1)  # [B, 1, N_gt]
        dists = dists.masked_fill(~gt_valid_mask_expanded, float('inf'))
    
    # Mask distances exceeding max_distance
    dists = dists.masked_fill(dists > max_distance, float('inf'))
    
    # Move to CPU for Hungarian algorithm (scipy is CPU-only)
    # Only transfer the distance matrices, not the full coordinate tensors
    dists_cpu = dists.cpu().numpy()  # [B, N_pred, N_gt]
    
    # Process each batch sample (Hungarian algorithm is per-sample)
    for b in range(B):
        cost_matrix = dists_cpu[b]  # [N_pred, N_gt]
        
        # Get valid predicted nodes for this batch
        valid_pred = valid_mask[b].cpu().numpy()  # [N_pred]
        valid_indices = np.where(valid_pred)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Extract cost matrix for valid predicted nodes only
        cost_matrix_valid = cost_matrix[valid_indices]  # [N_valid, N_gt]
        
        # CRITICAL FIX: Filter out padded GT nodes (if gt_valid_mask is provided)
        # This matches greedy matching behavior - only consider valid GT nodes
        gt_valid_indices = None
        if gt_valid_mask is not None:
            gt_valid_indices = np.where(gt_valid_mask[b].cpu().numpy())[0]
            if len(gt_valid_indices) == 0:
                continue  # No valid GT nodes for this sample
            cost_matrix_valid = cost_matrix_valid[:, gt_valid_indices]  # [N_valid, N_gt_valid]
        else:
            gt_valid_indices = np.arange(N_gt)  # Use all GT nodes if no mask provided
        
        # CRITICAL FIX: Handle edge cases before calling Hungarian algorithm
        # Check if cost matrix is valid (not empty, not all inf, no NaN)
        if cost_matrix_valid.shape[0] == 0 or cost_matrix_valid.shape[1] == 0:
            continue
        
        # Check if all values are inf (no valid matches)
        # This happens when all distances exceed max_distance
        # CRITICAL: Only check if ALL distances are inf (not just some)
        # If there are ANY finite distances, Hungarian algorithm can still find matches
        if np.isinf(cost_matrix_valid).all():
            continue
        
        # Check for NaN values
        if np.isnan(cost_matrix_valid).any():
            # Replace NaN with inf (treat as invalid)
            cost_matrix_valid = np.where(np.isnan(cost_matrix_valid), np.inf, cost_matrix_valid)
        
        # Hungarian algorithm: find optimal assignment
        # This minimizes total distance across all matches
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix_valid)
        except ValueError as e:
            # Handle infeasible cost matrix (e.g., all inf values after filtering)
            # This can happen if all distances exceed max_distance
            if "infeasible" in str(e).lower():
                # Skip this batch sample - no valid matches
                continue
            else:
                raise
        
        # Map back to original indices and filter by max_distance
        for local_idx, gt_idx_local in zip(row_indices, col_indices):
            pred_idx = valid_indices[local_idx]
            # Map back to original GT index (accounting for filtered GT nodes)
            if gt_valid_mask is not None:
                gt_idx = gt_valid_indices[gt_idx_local]
            else:
                gt_idx = gt_idx_local
            dist = cost_matrix_valid[local_idx, gt_idx_local]
            
            if dist < max_distance and dist < float('inf'):
                permutation[b, pred_idx] = gt_idx
    
    return permutation


def match_nodes_greedy(
    pred_coords: torch.Tensor,  # [B, N_pred, 2] in image space
    gt_coords: torch.Tensor,    # [B, N_gt, 2] in image space
    valid_mask: torch.Tensor,   # [B, N_pred] - True for valid predicted nodes
    max_distance: float = 100.0  # Maximum distance for matching (pixels) - INCREASED from 50.0
) -> torch.Tensor:
    """
    Match predicted nodes to GT nodes using greedy spatial matching.
    
    DEPRECATED: Use match_nodes_hungarian for optimal matching.
    Kept for backward compatibility.
    
    Args:
        pred_coords: Predicted node coordinates [B, N_pred, 2] in image space
        gt_coords: GT node coordinates [B, N_gt, 2] in image space
        valid_mask: [B, N_pred] - True for valid predicted nodes
        max_distance: Maximum distance for matching (pixels)
    
    Returns:
        permutation: [B, N_pred] where permutation[b, i] = j means
                     predicted node i matches GT node j
                     -1 if no match found
    """
    B, N_pred, _ = pred_coords.shape
    B_gt, N_gt, _ = gt_coords.shape
    
    device = pred_coords.device
    permutation = torch.full((B, N_pred), -1, dtype=torch.long, device=device)
    
    for b in range(B):
        # Get valid predicted nodes
        valid_pred = valid_mask[b]  # [N_pred]
        valid_indices = torch.where(valid_pred)[0]
        
        if len(valid_indices) == 0:
            continue
        
        pred_coords_b = pred_coords[b][valid_indices]  # [N_valid, 2]
        gt_coords_b = gt_coords[b]  # [N_gt, 2]
        
        # Compute distance matrix
        # pred_coords_b: [N_valid, 2], gt_coords_b: [N_gt, 2]
        dists = torch.cdist(pred_coords_b, gt_coords_b)  # [N_valid, N_gt]
        
        # Greedy matching: for each predicted node, find closest unmatched GT node
        matched_gt = set()
        
        # Sort by distance to prioritize closer matches
        for _ in range(min(len(valid_indices), N_gt)):
            best_pred_idx = None
            best_gt_idx = None
            best_dist = float('inf')
            
            for local_idx, pred_idx in enumerate(valid_indices):
                if permutation[b, pred_idx] >= 0:  # Already matched
                    continue
                
                for gt_idx in range(N_gt):
                    if gt_idx in matched_gt:  # Already matched
                        continue
                    
                    dist = dists[local_idx, gt_idx].item()
                    if dist < best_dist and dist < max_distance:
                        best_dist = dist
                        best_pred_idx = pred_idx
                        best_gt_idx = gt_idx
            
            if best_pred_idx is not None:
                permutation[b, best_pred_idx] = best_gt_idx
                matched_gt.add(best_gt_idx)
            else:
                break  # No more valid matches
    
    return permutation


class EdgePredictionLoss(nn.Module):
    """Loss for edge prediction (legacy - element-wise adjacency comparison)."""
    
    def __init__(self, use_focal: bool = False):
        super().__init__()
        if use_focal:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_adj: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute edge prediction loss (legacy method - has permutation invariance issue).
        
        Args:
            pred_logits: Predicted edge logits [B, N, N]
            target_adj: Target adjacency matrix [B, N, N] or list of arrays
            candidate_mask: Candidate edge mask [B, N, N] (optional)
        
        Returns:
            loss: Scalar loss value
        """
        # Handle list of target adjacencies
        if isinstance(target_adj, list):
            max_n = max(adj.shape[0] for adj in target_adj)
            B = len(target_adj)
            target_tensor = torch.zeros(B, max_n, max_n, device=pred_logits.device)
            for i, adj in enumerate(target_adj):
                n = adj.shape[0]
                if isinstance(adj, np.ndarray):
                    adj = torch.from_numpy(adj).to(pred_logits.device)
                elif isinstance(adj, torch.Tensor):
                    adj = adj.to(pred_logits.device)
                target_tensor[i, :n, :n] = adj.float()
            target_adj = target_tensor
        
        # Ensure same size
        B, N_pred, _ = pred_logits.shape
        B_target, N_target, _ = target_adj.shape
        
        if N_pred != N_target:
            min_n = min(N_pred, N_target)
            pred_logits = pred_logits[:, :min_n, :min_n]
            target_adj = target_adj[:, :min_n, :min_n]
        
        # Apply candidate mask if provided
        if candidate_mask is not None:
            if candidate_mask.shape != pred_logits.shape:
                min_n = min(pred_logits.shape[1], candidate_mask.shape[1])
                pred_logits = pred_logits[:, :min_n, :min_n]
                target_adj = target_adj[:, :min_n, :min_n]
                candidate_mask = candidate_mask[:, :min_n, :min_n]
            
            # Only compute loss on candidate edges
            mask = candidate_mask > 0.5
            if mask.sum() > 0:
                pred_masked = pred_logits[mask]
                target_masked = target_adj[mask]
                loss = self.loss_fn(pred_masked, target_masked)
            else:
                loss = torch.tensor(0.0, device=pred_logits.device)
        else:
            # Compute loss on all edges
            loss = self.loss_fn(pred_logits, target_adj)
        
        return loss


class PairBasedEdgeLoss(nn.Module):
    """
    Pair-based edge loss (permutation-invariant) with Focal Loss.
    
    Uses edge pairs from points files instead of element-wise adjacency comparison.
    This avoids the permutation invariance issue by comparing edge sets rather than
    matrix positions.
    
    Focal Loss addresses class imbalance by focusing on hard examples:
    FL = -α(1-p_t)^γ log(p_t)
    where p_t = p if y=1, else (1-p)
    """
    
    def __init__(
        self,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,  # Positive class weight for handling class imbalance
    ):
        """
        Args:
            use_focal: Whether to use focal loss (True) or standard BCE (False)
            focal_alpha: Balance factor for focal loss (typically 0.25)
            focal_gamma: Focusing parameter for focal loss (typically 2.0)
            pos_weight: Positive class weight for handling class imbalance (matching toy classifier)
        """
        super().__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' for focal loss
    
    def _compute_focal_loss(
        self,
        pred_logits: torch.Tensor,  # [N]
        targets: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """
        Compute Focal Loss for binary classification.
        
        Formula: FL = -α(1-p_t)^γ log(p_t)
        where p_t = p if y=1, else (1-p)
        
        Args:
            pred_logits: Predicted logits [N]
            targets: Target labels (0 or 1) [N]
        
        Returns:
            Focal loss value (scalar)
        """
        # Handle empty tensors (can happen when no valid pairs remain)
        if pred_logits.numel() == 0:
            return torch.tensor(0.0, device=targets.device if targets is not None else pred_logits.device)
        
        # Compute BCE for each sample
        bce = self.bce_loss(pred_logits, targets)  # [N]
        
        # Compute probabilities
        pred_probs = torch.sigmoid(pred_logits)  # [N]
        
        # Compute p_t: probability of true class
        # For positive samples (target=1): p_t = pred_probs
        # For negative samples (target=0): p_t = 1 - pred_probs
        p_t = torch.where(targets > 0.5, pred_probs, 1.0 - pred_probs)  # [N]
        
        # Focal weight: α(1-p_t)^γ
        focal_weight = self.focal_alpha * (1.0 - p_t) ** self.focal_gamma  # [N]
        
        # Add positive weighting to handle class imbalance (matching toy classifier)
        pos_mask = targets > 0.5
        focal_weight = focal_weight.clone()
        focal_weight[pos_mask] = focal_weight[pos_mask] * self.pos_weight
        
        # Focal loss
        focal_loss = focal_weight * bce  # [N]
        
        return focal_loss.mean()
    
    def forward(
        self,
        pred_edge_logits: torch.Tensor,  # [B, N, N]
        gt_edge_pairs: List[Optional[torch.Tensor]],  # List of [E_i, 2] or None
        node_permutation: torch.Tensor,  # [B, N] - maps pred node i to GT node j
        valid_mask: torch.Tensor,  # [B, N] - valid predicted nodes
        pred_node_coords_image: Optional[torch.Tensor] = None,  # [B, N, 2] - for fallback matching
        gt_node_coords: Optional[torch.Tensor] = None,  # [B, N_gt, 2] - for fallback matching
    ) -> torch.Tensor:
        """
        Compute pair-based edge loss.
        
        Args:
            pred_edge_logits: [B, N, N] predicted edge logits
            gt_edge_pairs: List of [E_i, 2] GT edge pairs (GT node indices) or None
            node_permutation: [B, N] where permutation[b, i] = j means
                             predicted node i matches GT node j (-1 if no match)
            valid_mask: [B, N] - True for valid predicted nodes
        
        Returns:
            loss: Scalar loss value
        """
        pred_edge_logits = torch.nan_to_num(pred_edge_logits, nan=0.0, posinf=0.0, neginf=0.0)
        edge_probs = torch.sigmoid(pred_edge_logits)
        B = pred_edge_logits.shape[0]
        
        batch_losses = []
        for b in range(B):
            # Get valid predicted nodes
            valid_pred = valid_mask[b]  # [N]
            valid_indices = torch.where(valid_pred)[0]
            n_valid = len(valid_indices)
            
            if n_valid == 0:
                # No valid nodes, use small constant loss
                batch_losses.append(torch.tensor(0.0, device=pred_edge_logits.device))
                continue
            
            # Get GT edge pairs for this batch
            gt_pairs = gt_edge_pairs[b]
            
            if gt_pairs is None or len(gt_pairs) == 0:
                # No GT edges, penalize all predicted edges (encourage sparsity)
                batch_losses.append(edge_probs[b].mean())
                continue
            
            # Convert GT pairs to set for fast lookup
            gt_pair_set = set()
            for pair in gt_pairs.cpu().numpy():
                pair_tuple = tuple(sorted([int(pair[0]), int(pair[1])]))
                gt_pair_set.add(pair_tuple)
            
            # Extract predicted edge pairs and remap to GT node indices
            # CRITICAL FIX: Also consider edges between matched and unmatched nodes
            # This ensures we get more supervision signal even if matching is imperfect
            
            pred_probs_list = []
            targets_list = []
            
            # Strategy 1: Matched node pairs (preferred - most accurate)
            matched_pairs = []
            for i_idx, i in enumerate(valid_indices):
                for j_idx, j in enumerate(valid_indices):
                    if j_idx <= i_idx:  # Only consider upper triangle (undirected)
                        continue
                    
                    # Get corresponding GT node indices
                    gt_i = node_permutation[b, i].item()
                    gt_j = node_permutation[b, j].item()
                    
                    if gt_i < 0 or gt_j < 0:  # One or both nodes not matched
                        continue
                    
                    # CRITICAL FIX: Use only upper triangle (i,j) to match toy classifier
                    pred_prob = edge_probs[b, i, j]  # Only upper triangle
                    
                    # Remap to GT node indices
                    pair_gt = tuple(sorted([gt_i, gt_j]))
                    
                    # Check if this edge exists in GT
                    is_gt_edge = pair_gt in gt_pair_set
                    
                    matched_pairs.append((pred_prob, is_gt_edge))
            
            # Strategy 2: If we have very few matched pairs, use lenient matching
            # Re-match all unmatched nodes to their closest GT nodes (no distance limit)
            # Apply lenient matching if we have fewer matched pairs than 20% of GT edges
            min_required_pairs = max(5, len(gt_pair_set) // 5)  # At least 20% of GT edges
            if len(matched_pairs) < min_required_pairs and pred_node_coords_image is not None and gt_node_coords is not None:
                # Re-match unmatched nodes with lenient criteria (always match to closest)
                for i_idx, i in enumerate(valid_indices):
                    gt_i = node_permutation[b, i].item()
                    
                    # If not matched, find closest GT node (no distance limit)
                    if gt_i < 0:
                        pred_coord = pred_node_coords_image[b, i]
                        if pred_coord.sum() > 0:  # Valid coordinate
                            dists_to_gt = torch.cdist(
                                pred_coord.unsqueeze(0),
                                gt_node_coords[b]
                            ).squeeze(0)  # [N_gt]
                            
                            closest_gt_idx = dists_to_gt.argmin().item()
                            # Use this match (lenient: allow any distance)
                            node_permutation[b, i] = closest_gt_idx
                
                # Re-extract matched pairs with updated matching
                matched_pairs = []
                for i_idx, i in enumerate(valid_indices):
                    for j_idx, j in enumerate(valid_indices):
                        if j_idx <= i_idx:
                            continue
                        
                        gt_i = node_permutation[b, i].item()
                        gt_j = node_permutation[b, j].item()
                        
                    if gt_i >= 0 and gt_j >= 0:
                        # CRITICAL FIX: Use only upper triangle (i,j) to match toy classifier
                        pred_prob = edge_probs[b, i, j]  # Only upper triangle
                        
                        pair_gt = tuple(sorted([gt_i, gt_j]))
                        is_gt_edge = pair_gt in gt_pair_set
                        
                        matched_pairs.append((pred_prob, is_gt_edge))
            
            # Use matched pairs
            for pred_prob, is_gt_edge in matched_pairs:
                pred_probs_list.append(pred_prob)
                targets_list.append(1.0 if is_gt_edge else 0.0)
            
            if len(pred_probs_list) == 0:
                # No pairs at all, use small constant loss
                batch_losses.append(torch.tensor(0.0, device=pred_edge_logits.device))
                continue
            
            # Convert to tensors
            pred_probs_tensor = torch.stack(pred_probs_list)
            targets_tensor = torch.tensor(targets_list, device=pred_edge_logits.device, dtype=torch.float32)
            
            # CRITICAL FIX: Get edge logits directly for matched pairs (better gradient flow)
            # Re-extract logits for the matched pairs we're using
            pred_logits_list = []
            targets_list_for_logits = []
            
            for i_idx, i in enumerate(valid_indices):
                for j_idx, j in enumerate(valid_indices):
                    if j_idx <= i_idx:
                        continue
                    gt_i = node_permutation[b, i].item()
                    gt_j = node_permutation[b, j].item()
                    
                    # Only use pairs where both nodes are matched (for accurate supervision)
                    if gt_i >= 0 and gt_j >= 0:
                        # CRITICAL FIX: Use only upper triangle (i,j) logits to match toy classifier
                        # This avoids redundant processing and provides cleaner gradients
                        pred_logit = pred_edge_logits[b, i, j]  # Only upper triangle
                        
                        # Check if this edge exists in GT
                        pair_gt = tuple(sorted([gt_i, gt_j]))
                        is_gt_edge = pair_gt in gt_pair_set
                        
                        pred_logits_list.append(pred_logit)
                        targets_list_for_logits.append(1.0 if is_gt_edge else 0.0)
            
            # Use logits if available, otherwise fallback to probabilities
            if len(pred_logits_list) > 0:
                pred_logits_tensor = torch.stack(pred_logits_list)
                targets_tensor = torch.tensor(targets_list_for_logits, device=pred_edge_logits.device, dtype=torch.float32)
                
                if self.use_focal:
                    # Compute focal loss
                    loss = self._compute_focal_loss(pred_logits_tensor, targets_tensor)
                else:
                    # Standard BCE loss
                    loss = self.bce_loss(pred_logits_tensor, targets_tensor).mean()
            else:
                # Fallback: use probabilities (less accurate but works)
                pred_logits_tensor = torch.logit(pred_probs_tensor.clamp(min=1e-7, max=1-1e-7))
                if self.use_focal:
                    loss = self._compute_focal_loss(pred_logits_tensor, targets_tensor)
                else:
                    loss = self.bce_loss(pred_logits_tensor, targets_tensor).mean()
            
            batch_losses.append(loss)
        
        if len(batch_losses) == 0:
            return torch.tensor(0.0, device=pred_edge_logits.device)
        
        return torch.stack(batch_losses).mean()


def normalize_adj(P: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix to row/col stochastic.
    
    Args:
        P: Edge probabilities or logits [B, N, N]
    
    Returns:
        Normalized adjacency [B, N, N]
    """
    # Check for NaN in input and replace with 0
    if torch.isnan(P).any():
        P = torch.where(torch.isnan(P), torch.zeros_like(P), P)
    
    # Convert logits to probabilities if needed
    if P.min() < 0 or P.max() > 1:
        P = torch.sigmoid(P)
        # Check again after sigmoid (shouldn't happen, but safety)
        if torch.isnan(P).any():
            P = torch.where(torch.isnan(P), torch.zeros_like(P), P)
    
    # Row normalization with proper handling of all-zero rows
    row_sum = P.sum(dim=-1, keepdim=True)
    # Use larger epsilon and ensure we never divide by exactly 0
    row_sum_safe = torch.clamp(row_sum, min=1e-7)
    P_row = P / row_sum_safe
    
    # Check for NaN after row normalization (shouldn't happen, but verify)
    if torch.isnan(P_row).any():
        # If a row is all zeros, P_row will be 0/eps = 0, which is fine
        # But if we get NaN, something else is wrong - use uniform distribution as fallback
        P_row = torch.where(torch.isnan(P_row), torch.ones_like(P_row) / P.shape[-1], P_row)
    
    # Column normalization with proper handling of all-zero columns
    col_sum = P.sum(dim=-2, keepdim=True)
    col_sum_safe = torch.clamp(col_sum, min=1e-7)
    P_col = P / col_sum_safe
    
    # Check for NaN after column normalization
    if torch.isnan(P_col).any():
        P_col = torch.where(torch.isnan(P_col), torch.ones_like(P_col) / P.shape[-2], P_col)
    
    # Average of row and column normalized
    P_norm = (P_row + P_col) / 2.0
    
    # Final NaN check
    if torch.isnan(P_norm).any():
        # Ultimate fallback: uniform distribution
        P_norm = torch.where(torch.isnan(P_norm), torch.ones_like(P_norm) / P.shape[-1], P_norm)
    
    return P_norm


def soft_reachability(P: torch.Tensor, Kpow: int = 4) -> torch.Tensor:
    """
    Compute soft reachability matrix.
    
    R = sigmoid(Ã + Ã² + ... + Ã^Kpow)
    
    Args:
        P: Edge probabilities or logits [B, N, N]
        Kpow: Maximum power for reachability
    
    Returns:
        Reachability matrix [B, N, N]
    """
    # Normalize adjacency
    A_tilde = normalize_adj(P)  # [B, N, N]
    
    # Check for NaN in normalized adjacency before starting
    if torch.isnan(A_tilde).any():
        # If input is invalid, return zeros
        return torch.zeros_like(A_tilde)
    
    # Compute powers with NaN checking at each step
    R_sum = A_tilde.clone()
    A_power = A_tilde.clone()
    
    for k in range(2, Kpow + 1):
        A_power = torch.bmm(A_power, A_tilde)  # [B, N, N]
        
        # Check for NaN after matrix multiplication
        if torch.isnan(A_power).any():
            # If NaN appears, stop accumulation and use what we have so far
            break
        
        # Clamp to prevent overflow (but don't clamp if already NaN)
        A_power = torch.clamp(A_power, min=-10.0, max=10.0)
        R_sum = R_sum + A_power
        
        # Check R_sum for NaN
        if torch.isnan(R_sum).any():
            # If accumulation produces NaN, use previous valid value
            R_sum = R_sum - A_power
            break
    
    # Clamp before sigmoid to prevent overflow
    R_sum = torch.clamp(R_sum, min=-10.0, max=10.0)
    
    # Check for NaN before sigmoid
    if torch.isnan(R_sum).any():
        # If still NaN, return zeros
        return torch.zeros_like(A_tilde)
    
    # Apply sigmoid
    R = torch.sigmoid(R_sum)
    
    # Final NaN check (sigmoid should never produce NaN, but safety first)
    if torch.isnan(R).any():
        R = torch.where(torch.isnan(R), torch.zeros_like(R), R)
    
    return R


class SoftReachabilityConnectivityLoss(nn.Module):
    """Soft reachability connectivity loss with optional BCE supervision."""

    def __init__(
        self,
        Kpow: int = 4,
        use_mse: bool = False,
        temperature: float = 1.0,
        eps: float = 1e-4,
        pos_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.Kpow = Kpow
        self.use_mse = use_mse
        self.temperature = temperature
        self.eps = eps
        self.pos_weight = pos_weight
        if use_mse:
            self.loss_fn = nn.MSELoss()

    def _pad_to_tensor(self, matrices, device):
        if isinstance(matrices, torch.Tensor):
            return matrices.to(device)
        max_n = max(mat.shape[0] for mat in matrices)
        B = len(matrices)
        tensor = torch.zeros(B, max_n, max_n, device=device)
        for i, mat in enumerate(matrices):
            m = mat
            if isinstance(mat, np.ndarray):
                m = torch.from_numpy(mat)
            tensor[i, :m.shape[0], :m.shape[1]] = m.to(device)
        return tensor

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_adj: torch.Tensor,
        target_reachability: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = pred_logits.device

        target_adj = self._pad_to_tensor(target_adj, device)

        min_n = min(pred_logits.shape[1], target_adj.shape[1])
        pred_logits = pred_logits[:, :min_n, :min_n]
        target_adj = target_adj[:, :min_n, :min_n]

        if target_reachability is not None:
            target_reachability = self._pad_to_tensor(target_reachability, device)
            target_reachability = target_reachability[:, :min_n, :min_n]

        if candidate_mask is not None:
            candidate_mask = candidate_mask[:, :min_n, :min_n].to(device)

        if self.use_mse:
            R_pred = soft_reachability(pred_logits, Kpow=self.Kpow)
            R_target = (
                target_reachability.float()
                if target_reachability is not None
                else soft_reachability(target_adj.float(), Kpow=self.Kpow)
            )
            loss = self.loss_fn(R_pred, R_target)
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️  WARNING: connectivity_loss is NaN/Inf in SoftReachabilityConnectivityLoss")
                loss = torch.tensor(0.0, device=device)
            return loss

        # BCE path with strengthened supervision
        prob = torch.sigmoid(pred_logits)
        if candidate_mask is not None:
            prob = prob * candidate_mask

        row_sum = prob.sum(dim=-1, keepdim=True)
        prob_normalized = prob / (row_sum + self.eps)

        reachability_prob = prob_normalized.clone()
        current = prob_normalized
        for _ in range(2, self.Kpow + 1):
            current = torch.bmm(current, prob_normalized)
            reachability_prob = reachability_prob + current

        reachability_prob = torch.clamp(reachability_prob / self.Kpow, min=self.eps, max=1.0 - self.eps)

        if target_reachability is not None:
            target = target_reachability.float()
        else:
            target = soft_reachability(target_adj.float(), Kpow=self.Kpow)

        # Ensure diagonal supervision
        diag = torch.eye(reachability_prob.shape[-1], device=device).unsqueeze(0)
        target = torch.where(diag.bool(), torch.ones_like(target), target)

        weight = torch.ones_like(target)
        if candidate_mask is not None:
            cm = candidate_mask.clone()
            cm = torch.where(diag.bool(), torch.ones_like(cm), cm)
            weight = weight * cm
        if self.pos_weight is not None and self.pos_weight != 1.0:
            weight = torch.where(target > 0.5, weight * self.pos_weight, weight)

        loss = F.binary_cross_entropy(
            reachability_prob,
            target,
            weight=weight,
            reduction="sum",
        )
        denom = weight.sum().clamp_min(self.eps)
        loss = loss / denom

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  WARNING: connectivity_loss is NaN/Inf in SoftReachabilityConnectivityLoss (BCE path)")
            loss = torch.tensor(0.0, device=device)

        return loss


class AdjacencyBasedEdgeLoss(nn.Module):
    """
    Dense adjacency-based edge loss.
    
    Uses adjacency matrix directly for all matched node pairs.
    Provides explicit positive and negative supervision.
    """
    
    def __init__(
        self,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
        use_hard_negative_mining: bool = True,  # Hard negative mining for class imbalance
        hard_negative_threshold: float = 0.3,  # Threshold for hard negatives (probability > threshold)
        max_hard_negatives_ratio: float = 2.0,  # Max hard negatives = max_hard_negatives_ratio * num_positives
        use_edge_length_weighting: bool = False,  # Weight loss by edge length (longer edges = higher weight) - DISABLED by default
        edge_length_weight_power: float = 1.5,  # Power for edge length weighting (1.0 = linear, >1.0 = emphasize long edges)
    ):
        super().__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_threshold = hard_negative_threshold
        self.max_hard_negatives_ratio = max_hard_negatives_ratio
        self.use_edge_length_weighting = use_edge_length_weighting
        self.edge_length_weight_power = edge_length_weight_power
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def _compute_focal_loss(self, pred_logits, targets, sample_weights=None):
        """
        Compute Focal Loss with positive weighting.
        
        Formula: FL = α(1-p_t)^γ * BCE
        where p_t = p if y=1, else (1-p)
        
        Positive weighting: Multiply positive sample weights by pos_weight
        to handle class imbalance (81% negatives, 19% positives).
        
        Args:
            pred_logits: Predicted logits [N]
            targets: Target labels [N]
            sample_weights: Optional sample weights [N] (e.g., edge length weights)
        """
        bce = self.bce_loss(pred_logits, targets)
        pred_probs = torch.sigmoid(pred_logits)
        p_t = torch.where(targets > 0.5, pred_probs, 1.0 - pred_probs)
        focal_weight = self.focal_alpha * (1.0 - p_t) ** self.focal_gamma
        
        # Add positive weighting to handle class imbalance
        # Multiply positive sample weights by pos_weight
        pos_mask = targets > 0.5
        focal_weight = focal_weight.clone()  # Avoid in-place modification
        focal_weight[pos_mask] = focal_weight[pos_mask] * self.pos_weight
        
        # FIX: Apply sample weights (e.g., edge length weights) only to positive edges
        # This reduces compound weighting effects and improves stability
        # For positives: focal_weight * pos_weight * edge_length_weight
        # For negatives: focal_weight (no edge length weighting)
        if sample_weights is not None:
            # Apply edge length weights only to positive samples to reduce compound effects
            pos_edge_length_weights = torch.where(
                pos_mask,
                sample_weights,  # Use edge length weights for positives
                torch.ones_like(sample_weights)  # No weighting for negatives
            )
            focal_weight = focal_weight * pos_edge_length_weights
        
        return (focal_weight * bce).mean()
    
    def forward(
        self,
        pred_edge_logits: torch.Tensor,  # [B, N_pred, N_pred]
        target_adj: torch.Tensor,  # [B, N_gt, N_gt] or List
        node_permutation: torch.Tensor,  # [B, N_pred] - maps pred node i to GT node j
        valid_mask: torch.Tensor,  # [B, N_pred] - valid predicted nodes
        pred_node_coords_image: Optional[torch.Tensor] = None,  # [B, N_pred, 2] - node coordinates in image space
    ) -> torch.Tensor:
        """
        Compute adjacency-based edge loss.
        
        Args:
            pred_edge_logits: [B, N_pred, N_pred] predicted edge logits
            target_adj: [B, N_gt, N_gt] or List of [N_gt, N_gt] ground truth adjacency
            node_permutation: [B, N_pred] where permutation[b, i] = j means
                             predicted node i matches GT node j (-1 if no match)
            valid_mask: [B, N_pred] - True for valid predicted nodes
        
        Returns:
            loss: Scalar loss value
        """
        pred_edge_logits = torch.nan_to_num(pred_edge_logits, nan=0.0, posinf=0.0, neginf=0.0)
        B, N_pred, _ = pred_edge_logits.shape
        
        # Handle list format
        if isinstance(target_adj, list):
            # Convert to tensor (pad to max size)
            max_n_gt = max(adj.shape[0] for adj in target_adj if adj is not None)
            target_adj_tensor = torch.zeros(B, max_n_gt, max_n_gt, 
                                           device=pred_edge_logits.device, dtype=pred_edge_logits.dtype)
            for b, adj in enumerate(target_adj):
                if adj is not None:
                    n_gt = adj.shape[0]
                    if isinstance(adj, np.ndarray):
                        target_adj_tensor[b, :n_gt, :n_gt] = torch.from_numpy(adj).float().to(pred_edge_logits.device)
                    else:
                        target_adj_tensor[b, :n_gt, :n_gt] = adj.float()
            target_adj = target_adj_tensor
        else:
            # Already tensor
            max_n_gt = target_adj.shape[1]
        
        # Create supervision matrix for predicted nodes
        edge_targets = torch.zeros_like(pred_edge_logits)  # [B, N_pred, N_pred]
        supervision_mask = torch.zeros_like(pred_edge_logits, dtype=torch.bool)  # [B, N_pred, N_pred]
        
        for b in range(B):
            # Get valid predicted nodes
            valid_pred = valid_mask[b]  # [N_pred]
            valid_indices = torch.where(valid_pred)[0]
            
            # Build supervision for all matched pairs
            for i_idx, i in enumerate(valid_indices):
                gt_i = node_permutation[b, i].item()
                if gt_i < 0:  # Unmatched node
                    continue
                
                for j_idx, j in enumerate(valid_indices):
                    if j_idx <= i_idx:  # Only upper triangle (symmetric)
                        continue
                    
                    gt_j = node_permutation[b, j].item()
                    if gt_j < 0:  # Unmatched node
                        continue
                    
                    # Get adjacency value (upper triangle only, matching toy classifier)
                    if gt_i < max_n_gt and gt_j < max_n_gt:
                        adj_value = target_adj[b, gt_i, gt_j].item()
                        # CRITICAL FIX: Only supervise upper triangle (i,j) to match toy classifier
                        # This avoids redundant supervision and provides cleaner gradients
                        edge_targets[b, i, j] = adj_value
                        supervision_mask[b, i, j] = True
        
        # Compute loss only on supervised pairs
        if supervision_mask.sum() == 0:
            # No valid supervision, return small constant
            return torch.tensor(0.0, device=pred_edge_logits.device)
        
        # Compute edge length weights if enabled and coordinates are provided
        # CRITICAL FIX: Extract in the exact same order as boolean indexing (row-major: batch, row, col)
        edge_length_weights = None
        if self.use_edge_length_weighting and pred_node_coords_image is not None:
            # Extract edge lengths in the exact same order as pred_logits_supervised (boolean indexing)
            # Boolean indexing extracts in row-major order: (b,0,0), (b,0,1), ..., (b,0,N), (b,1,0), ...
            num_supervised = supervision_mask.sum().item()
            if num_supervised > 0:
                edge_lengths_list = []
                
                # Extract in the exact same order as boolean indexing (upper triangle only)
                # CRITICAL FIX: Only extract upper triangle to match supervision_mask (which now only has upper triangle)
                for b in range(B):
                    for i in range(pred_edge_logits.shape[1]):  # N_pred
                        for j in range(i + 1, pred_edge_logits.shape[2]):  # Only upper triangle (j > i)
                            if supervision_mask[b, i, j]:
                                # Compute edge length for this supervised pair
                                # Note: supervision_mask only includes valid matched nodes, so both should be valid
                                coord_i = pred_node_coords_image[b, i]  # [2]
                                coord_j = pred_node_coords_image[b, j]  # [2]
                                edge_length = torch.norm(coord_i - coord_j).item()
                                edge_lengths_list.append(edge_length)
                
                # Convert to tensor and compute weights
                if len(edge_lengths_list) > 0 and len(edge_lengths_list) == num_supervised:
                    edge_lengths_tensor = torch.tensor(
                        edge_lengths_list, 
                        device=pred_edge_logits.device, 
                        dtype=pred_edge_logits.dtype
                    )
                    # FIX: Clip first, then normalize to preserve loss scale
                    edge_length_mean = edge_lengths_tensor.mean()
                    if edge_length_mean > 1e-6:
                        edge_length_normalized = edge_lengths_tensor / edge_length_mean
                        # Apply power weighting (reduced from 1.5 to 1.2 for stability)
                        power = min(self.edge_length_weight_power, 1.2)  # Cap at 1.2 for stability
                        edge_length_weights = edge_length_normalized ** power
                        # Clip first to prevent extreme values (max 3.0x mean weight)
                        edge_length_weights = torch.clamp(edge_length_weights, min=0.1, max=3.0)
                        # Then normalize to have mean=1.0 to prevent loss scale changes
                        weight_mean = edge_length_weights.mean()
                        if weight_mean > 1e-6:
                            edge_length_weights = edge_length_weights / weight_mean
                    else:
                        edge_length_weights = torch.ones_like(edge_lengths_tensor)
                else:
                    # Length mismatch - disable edge length weighting for this batch
                    edge_length_weights = None
            else:
                edge_length_weights = None
        
        # Extract supervised pairs
        pred_logits_supervised = pred_edge_logits[supervision_mask]  # [M]
        targets_supervised = edge_targets[supervision_mask]  # [M]
        
        # Hard negative mining: Focus on hard negatives (high probability but should be 0)
        # This balances the loss by reducing easy negative contributions
        # Track indices for edge length weight mapping
        combined_indices = None
        if self.use_hard_negative_mining and len(pred_logits_supervised) > 0:
            # Separate positives and negatives
            pos_mask = targets_supervised > 0.5
            neg_mask = ~pos_mask
            
            pos_logits = pred_logits_supervised[pos_mask]
            neg_logits = pred_logits_supervised[neg_mask]
            pos_targets = targets_supervised[pos_mask]
            neg_targets = targets_supervised[neg_mask]
            
            # Get indices for mapping edge length weights
            pos_indices = torch.where(pos_mask)[0]  # Indices in pred_logits_supervised
            neg_indices = torch.where(neg_mask)[0]  # Indices in pred_logits_supervised
            
            # Keep all positives (rare, important)
            # For negatives: only keep hard negatives (high probability but should be 0)
            if len(neg_logits) > 0:
                neg_probs = torch.sigmoid(neg_logits)
                
                # Hard negatives: probability > threshold (model thinks it's an edge but it's not)
                hard_neg_mask = neg_probs > self.hard_negative_threshold
                n_pos = len(pos_logits)
                
                if hard_neg_mask.sum() > 0:
                    # Use hard negatives above threshold
                    hard_neg_logits = neg_logits[hard_neg_mask]
                    hard_neg_targets = neg_targets[hard_neg_mask]
                    hard_neg_indices_in_neg = torch.where(hard_neg_mask)[0]  # Indices within neg_logits
                    hard_neg_indices = neg_indices[hard_neg_indices_in_neg]  # Map back to pred_logits_supervised
                    n_hard_neg = len(hard_neg_logits)
                    
                    # Ensure at least as many hard negatives as positives for better balance
                    # If we have fewer hard negatives than positives, add more from top-K hardest
                    if n_hard_neg < n_pos:
                        # Get remaining negatives (not yet selected)
                        remaining_neg_mask = ~hard_neg_mask
                        remaining_neg_logits = neg_logits[remaining_neg_mask]
                        remaining_neg_probs = neg_probs[remaining_neg_mask]
                        remaining_neg_targets = neg_targets[remaining_neg_mask]
                        remaining_neg_indices_in_neg = torch.where(remaining_neg_mask)[0]
                        remaining_neg_indices = neg_indices[remaining_neg_indices_in_neg]
                        
                        if len(remaining_neg_logits) > 0:
                            # Calculate how many more we need
                            remaining_needed = n_pos - n_hard_neg
                            # Get top-K hardest from remaining (even if below threshold)
                            k = min(remaining_needed, len(remaining_neg_logits))
                            _, additional_indices_in_remaining = torch.topk(remaining_neg_probs, k=k)
                            additional_hard_neg_logits = remaining_neg_logits[additional_indices_in_remaining]
                            additional_hard_neg_targets = remaining_neg_targets[additional_indices_in_remaining]
                            additional_hard_neg_indices = remaining_neg_indices[additional_indices_in_remaining]
                            
                            # Combine threshold-based and top-K hard negatives
                            hard_neg_logits = torch.cat([hard_neg_logits, additional_hard_neg_logits])
                            hard_neg_targets = torch.cat([hard_neg_targets, additional_hard_neg_targets])
                            hard_neg_indices = torch.cat([hard_neg_indices, additional_hard_neg_indices])
                else:
                    # If no hard negatives above threshold, use top-K hardest
                    # Keep at least num_positives negatives (or max_hard_negatives_ratio * num_positives)
                    max_hard_negatives = max(n_pos, int(n_pos * self.max_hard_negatives_ratio))
                    
                    # Sort by probability (descending) - highest probability = hardest
                    _, topk_indices_in_neg = torch.topk(neg_probs, k=min(max_hard_negatives, len(neg_probs)))
                    hard_neg_logits = neg_logits[topk_indices_in_neg]
                    hard_neg_targets = neg_targets[topk_indices_in_neg]
                    hard_neg_indices = neg_indices[topk_indices_in_neg]  # Map back to pred_logits_supervised
                
                # Combine positives and hard negatives
                if len(pos_logits) > 0:
                    combined_logits = torch.cat([pos_logits, hard_neg_logits])
                    combined_targets = torch.cat([pos_targets, hard_neg_targets])
                    combined_indices = torch.cat([pos_indices, hard_neg_indices])
                else:
                    # No positives, use hard negatives only
                    combined_logits = hard_neg_logits
                    combined_targets = hard_neg_targets
                    combined_indices = hard_neg_indices
            else:
                # No negatives, use positives only
                combined_logits = pos_logits
                combined_targets = pos_targets
                combined_indices = pos_indices
        else:
            # No hard negative mining, use all supervised pairs
            combined_logits = pred_logits_supervised
            combined_targets = targets_supervised
            # All indices are used (0 to len(pred_logits_supervised)-1)
            if len(pred_logits_supervised) > 0:
                combined_indices = torch.arange(len(pred_logits_supervised), device=pred_logits_supervised.device)
        
        # Compute loss on filtered pairs
        if len(combined_logits) == 0:
            return torch.tensor(0.0, device=pred_edge_logits.device)
        
        # Map edge length weights to combined_logits if available
        # FIX: Use safer indexing approach to handle out-of-bounds indices
        combined_edge_length_weights = None
        if edge_length_weights is not None and len(edge_length_weights) > 0:
            if combined_indices is not None and len(combined_indices) > 0:
                # Filter valid indices (within bounds)
                valid_mask_idx = (combined_indices >= 0) & (combined_indices < len(edge_length_weights))
                
                if valid_mask_idx.all():
                    # All indices are valid, use direct indexing
                    combined_edge_length_weights = edge_length_weights[combined_indices]
                elif valid_mask_idx.any():
                    # Some indices are invalid, use fallback for invalid ones
                    valid_indices = combined_indices[valid_mask_idx]
                    valid_weights = edge_length_weights[valid_indices]
                    mean_weight = valid_weights.mean() if len(valid_weights) > 0 else edge_length_weights.mean()
                    
                    # Initialize with mean weight, then fill valid positions
                    combined_edge_length_weights = torch.ones_like(combined_logits) * mean_weight
                    combined_edge_length_weights[valid_mask_idx] = valid_weights
                else:
                    # No valid indices, use mean weight
                    combined_edge_length_weights = torch.ones_like(combined_logits) * edge_length_weights.mean()
                
                # Verify length matches (safety check)
                if len(combined_edge_length_weights) != len(combined_logits):
                    # Fallback: use mean weight if mismatch
                    combined_edge_length_weights = torch.ones_like(combined_logits) * edge_length_weights.mean()
            else:
                # No indices tracked, use all weights (shouldn't happen, but safety check)
                if len(edge_length_weights) == len(combined_logits):
                    combined_edge_length_weights = edge_length_weights
                else:
                    # Mismatch, use mean weight
                    combined_edge_length_weights = torch.ones_like(combined_logits) * edge_length_weights.mean()
        
        # Compute loss with edge length weighting
        if self.use_focal:
            loss = self._compute_focal_loss(
                combined_logits, 
                combined_targets,
                sample_weights=combined_edge_length_weights
            )
        else:
            # Standard BCE with positive weighting
            # For BCE, we apply edge length weights manually
            bce = F.binary_cross_entropy_with_logits(
                combined_logits,
                combined_targets,
                pos_weight=torch.tensor(self.pos_weight, device=pred_edge_logits.device),
                reduction='none'  # Get per-sample loss
            )
            
            # FIX: Apply edge length weights only to positives (for consistency with focal loss)
            # This reduces compound weighting effects and improves stability
            if combined_edge_length_weights is not None:
                pos_mask = combined_targets > 0.5
                pos_edge_length_weights = torch.where(
                    pos_mask,
                    combined_edge_length_weights,  # Use edge length weights for positives
                    torch.ones_like(combined_edge_length_weights)  # No weighting for negatives
                )
                bce = bce * pos_edge_length_weights
            
            loss = bce.mean()
        
        return loss


class HybridEdgeLoss(nn.Module):
    """
    Hybrid edge loss combining adjacency matrix and point pairs.
    
    Uses both supervision signals with equal weights (default: 1.0 each).
    """
    
    def __init__(
        self,
        adjacency_weight: float = 1.0,
        pair_weight: float = 1.0,
        adjacency_use_focal: bool = True,
        pair_use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
        use_hard_negative_mining: bool = True,  # Hard negative mining for class imbalance
        hard_negative_threshold: float = 0.3,  # Threshold for hard negatives
        max_hard_negatives_ratio: float = 2.0,  # Max hard negatives ratio
        use_edge_length_weighting: bool = False,  # Edge length weighting (longer edges = higher weight) - DISABLED by default
        edge_length_weight_power: float = 1.5,  # Power for edge length weighting
    ):
        super().__init__()
        self.adjacency_weight = adjacency_weight
        self.pair_weight = pair_weight
        
        # Initialize both loss functions
        self.adjacency_loss = AdjacencyBasedEdgeLoss(
            use_focal=adjacency_use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,  # Pass positive weight
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_threshold=hard_negative_threshold,
            max_hard_negatives_ratio=max_hard_negatives_ratio,
            use_edge_length_weighting=use_edge_length_weighting,
            edge_length_weight_power=edge_length_weight_power,
        )
        
        self.pair_loss = PairBasedEdgeLoss(
            use_focal=pair_use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,  # Pass positive weight to match toy classifier
        )
    
    def forward(
        self,
        pred_edge_logits: torch.Tensor,  # [B, N_pred, N_pred]
        target_adj: torch.Tensor,  # [B, N_gt, N_gt] or List
        gt_edge_pairs: List[Optional[torch.Tensor]],  # List of [E_i, 2] or None
        node_permutation: torch.Tensor,  # [B, N_pred] - maps pred node i to GT node j
        valid_mask: torch.Tensor,  # [B, N_pred] - valid predicted nodes
        pred_node_coords_image: Optional[torch.Tensor] = None,
        gt_node_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid edge loss.
        
        Returns:
            Dictionary with:
            - 'loss': Total combined loss
            - 'adjacency_loss': Adjacency-based loss component
            - 'pair_loss': Pair-based loss component
        """
        # Compute both losses
        adjacency_loss = self.adjacency_loss(
            pred_edge_logits,
            target_adj,
            node_permutation,
            valid_mask,
            pred_node_coords_image=pred_node_coords_image,
        )
        
        pair_loss = self.pair_loss(
            pred_edge_logits,
            gt_edge_pairs,
            node_permutation,
            valid_mask,
            pred_node_coords_image=pred_node_coords_image,
            gt_node_coords=gt_node_coords,
        )
        
        # Combine weighted components
        total_loss = self.adjacency_weight * adjacency_loss + self.pair_weight * pair_loss
        
        return {
            'loss': total_loss,
            'adjacency_loss': adjacency_loss,
            'pair_loss': pair_loss,
        }


class EdgeMaskLoss(nn.Module):
    """Optional edge mask loss for QA."""
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred_heatmap: torch.Tensor,  # Can use edge predictions or heatmap
        target_mask: torch.Tensor,  # [B, H, W]
    ) -> torch.Tensor:
        """
        Compute edge mask loss (optional, for QA only).
        
        Args:
            pred_heatmap: Predicted heatmap or edge visualization [B, 1, H, W]
            target_mask: Target edge mask [B, H, W]
        
        Returns:
            Mask loss
        """
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        
        # Ensure same size
        if pred_heatmap.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask,
                size=pred_heatmap.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Dice loss + BCE
        pred_sigmoid = torch.sigmoid(pred_heatmap)
        dice_loss = 1.0 - (2.0 * (pred_sigmoid * target_mask).sum() + 1e-8) / (
            pred_sigmoid.sum() + target_mask.sum() + 1e-8
        )
        bce_loss = self.bce_loss(pred_heatmap, target_mask)
        
        return dice_loss + bce_loss


def _permute_edge_logits(
    pred_edge_logits: torch.Tensor,  # [B, N_pred, N_pred]
    node_permutation: torch.Tensor,   # [B, N_pred] - maps pred node i to GT node j (-1 if no match)
    valid_mask: torch.Tensor,        # [B, N_pred] - valid predicted nodes
    gt_node_coords: torch.Tensor,    # [B, N_gt, 2] - GT node coordinates
) -> torch.Tensor:
    """
    Permute predicted edge logits to match GT node order.
    
    This fixes the permutation invariance issue for connectivity loss.
    
    CRITICAL: This function must preserve gradients to allow backpropagation.
    Uses scatter_add to preserve computation graph.
    
    Args:
        pred_edge_logits: [B, N_pred, N_pred] predicted edge logits
        node_permutation: [B, N_pred] where permutation[b, i] = j means predicted node i matches GT node j
        valid_mask: [B, N_pred] - True for valid predicted nodes
        gt_node_coords: [B, N_gt, 2] - GT node coordinates
    
    Returns:
        permuted_logits: [B, N_gt, N_gt] edge logits reordered to match GT node order
    """
    B, N_pred, _ = pred_edge_logits.shape
    B_gt, N_gt, _ = gt_node_coords.shape
    device = pred_edge_logits.device
    dtype = pred_edge_logits.dtype
    
    # Initialize output with zeros (unmatched edges will be 0)
    # Don't set requires_grad here - scatter_add will create a new tensor that inherits requires_grad from source
    permuted_logits = torch.zeros(B, N_gt, N_gt, device=device, dtype=dtype)
    
    for b in range(B):
        # Build mapping using tensor operations to preserve gradients
        # Create masks for valid matches
        valid_pred_mask = valid_mask[b]  # [N_pred]
        matched_mask = (node_permutation[b] >= 0)  # [N_pred] - True if matched to GT
        valid_matched = valid_pred_mask & matched_mask  # [N_pred]
        
        if not valid_matched.any():
            # No matches, keep zeros for this batch
            continue
        
        # Get matched GT indices (as tensor, not .item())
        matched_gt_indices = node_permutation[b][valid_matched]  # [M] where M = num matched
        matched_pred_indices = torch.where(valid_matched)[0]  # [M] - predicted node indices
        
        M = matched_pred_indices.shape[0]
        
        if M == 0:
            continue
        
        # Build all pairs of matched nodes
        # pred_i_idx, pred_j_idx: indices in matched_pred_indices
        # gt_i, gt_j: corresponding GT indices
        pred_i_indices = matched_pred_indices.unsqueeze(1).expand(M, M)  # [M, M]
        pred_j_indices = matched_pred_indices.unsqueeze(0).expand(M, M)  # [M, M]
        gt_i_indices = matched_gt_indices.unsqueeze(1).expand(M, M)  # [M, M]
        gt_j_indices = matched_gt_indices.unsqueeze(0).expand(M, M)  # [M, M]
        
        # Get edge logits for all pairs
        logits_ij = pred_edge_logits[b][pred_i_indices, pred_j_indices]  # [M, M]
        logits_ji = pred_edge_logits[b][pred_j_indices, pred_i_indices]  # [M, M]
        logits_sym = (logits_ij + logits_ji) / 2.0  # [M, M]
        
        # Flatten for scatter
        logits_flat = logits_sym.reshape(-1)  # [M*M]
        gt_i_flat = gt_i_indices.reshape(-1)  # [M*M]
        gt_j_flat = gt_j_indices.reshape(-1)  # [M*M]
        
        # Use scatter_add to preserve gradients
        # scatter_add creates a new tensor that inherits requires_grad from source values
        flat_target = gt_i_flat * N_gt + gt_j_flat  # [M*M]
        permuted_logits_flat = permuted_logits[b].view(-1).clone()  # [N_gt*N_gt] - clone to avoid in-place on leaf
        permuted_logits_flat = permuted_logits_flat.scatter_add(0, flat_target, logits_flat)
        permuted_logits = permuted_logits.clone()  # Clone entire tensor to avoid in-place on leaf
        permuted_logits[b] = permuted_logits_flat.view(N_gt, N_gt)
    
    return permuted_logits


def permute_adjacency_to_predicted_space(
    target_adj: torch.Tensor,  # [B, N_gt, N_gt] or List of [N_gt, N_gt] - GT adjacency
    node_permutation: torch.Tensor,  # [B, N_pred] - maps pred node i to GT node j (-1 if no match)
    valid_mask: torch.Tensor,  # [B, N_pred] - valid predicted nodes
    max_n_pred: int,  # Maximum number of predicted nodes
) -> torch.Tensor:
    """
    Permute GT adjacency matrix from GT node space to predicted node space.
    
    This fixes the node alignment issue for coverage loss.
    
    Args:
        target_adj: [B, N_gt, N_gt] or List of [N_gt, N_gt] GT adjacency matrix
        node_permutation: [B, N_pred] where permutation[b, i] = j means predicted node i matches GT node j
        valid_mask: [B, N_pred] - True for valid predicted nodes
        max_n_pred: Maximum number of predicted nodes
    
    Returns:
        permuted_adj: [B, N_pred, N_pred] adjacency matrix in predicted node space
    """
    # Handle list format
    if isinstance(target_adj, list):
        B = len(target_adj)
        max_n_gt = max(adj.shape[0] for adj in target_adj if adj is not None)
        target_adj_tensor = torch.zeros(B, max_n_gt, max_n_gt, 
                                       device=node_permutation.device, dtype=torch.float32)
        for b, adj in enumerate(target_adj):
            if adj is not None:
                n_gt = adj.shape[0]
                if isinstance(adj, np.ndarray):
                    target_adj_tensor[b, :n_gt, :n_gt] = torch.from_numpy(adj).float().to(node_permutation.device)
                else:
                    target_adj_tensor[b, :n_gt, :n_gt] = adj.float().to(node_permutation.device)
        target_adj = target_adj_tensor
    else:
        B = target_adj.shape[0]
    
    device = node_permutation.device
    dtype = target_adj.dtype if isinstance(target_adj, torch.Tensor) else torch.float32
    
    # Initialize output with zeros (unmatched edges will be 0)
    permuted_adj = torch.zeros(B, max_n_pred, max_n_pred, device=device, dtype=dtype)
    
    for b in range(B):
        # Get valid predicted nodes that are matched
        valid_pred_mask = valid_mask[b]  # [N_pred]
        matched_mask = (node_permutation[b] >= 0)  # [N_pred] - True if matched to GT
        valid_matched = valid_pred_mask & matched_mask  # [N_pred]
        
        if not valid_matched.any():
            # No matches, keep zeros for this batch
            continue
        
        # Get matched GT indices and predicted indices
        matched_gt_indices = node_permutation[b][valid_matched]  # [M] where M = num matched
        matched_pred_indices = torch.where(valid_matched)[0]  # [M] - predicted node indices
        
        M = matched_pred_indices.shape[0]
        if M == 0:
            continue
        
        # Build all pairs of matched nodes
        # pred_i, pred_j: predicted node indices
        # gt_i, gt_j: corresponding GT indices
        pred_i_indices = matched_pred_indices.unsqueeze(1).expand(M, M)  # [M, M]
        pred_j_indices = matched_pred_indices.unsqueeze(0).expand(M, M)  # [M, M]
        gt_i_indices = matched_gt_indices.unsqueeze(1).expand(M, M)  # [M, M]
        gt_j_indices = matched_gt_indices.unsqueeze(0).expand(M, M)  # [M, M]
        
        # Get GT adjacency values for matched pairs
        # target_adj[b, gt_i, gt_j] gives the GT edge value
        gt_i_flat = gt_i_indices.reshape(-1)  # [M*M]
        gt_j_flat = gt_j_indices.reshape(-1)  # [M*M]
        
        # Extract GT adjacency values
        adj_values = target_adj[b][gt_i_flat, gt_j_flat]  # [M*M]
        
        # Map to predicted node space
        pred_i_flat = pred_i_indices.reshape(-1)  # [M*M]
        pred_j_flat = pred_j_indices.reshape(-1)  # [M*M]
        
        # Use scatter_add to handle potential duplicates (though there shouldn't be any)
        flat_target = pred_i_flat * max_n_pred + pred_j_flat  # [M*M]
        permuted_adj_flat = permuted_adj[b].view(-1).clone()
        permuted_adj_flat = permuted_adj_flat.scatter_add(0, flat_target, adj_values)
        permuted_adj[b] = permuted_adj_flat.view(max_n_pred, max_n_pred)
        
        # Make symmetric (undirected graphs)
        permuted_adj[b] = (permuted_adj[b] + permuted_adj[b].T) / 2.0
    
    return permuted_adj


class CombinedLoss(nn.Module):
    """Combined loss function."""
    
    def __init__(
        self,
        node_weight: float = 1.0,
        edge_weight: float = 2.0,
        coverage_weight: float = 1.0,
        mask_weight: float = 0.0,  # Optional, default 0
        use_mask_loss: bool = False,
        edge_use_focal: bool = True,
        edge_focal_alpha: float = 0.25,
        edge_focal_gamma: float = 2.0,
        edge_pos_weight: float = 1.0,
        use_hard_negative_mining: bool = True,  # Hard negative mining for class imbalance
        hard_negative_threshold: float = 0.3,  # Threshold for hard negatives (probability > threshold)
        max_hard_negatives_ratio: float = 2.0,  # Max hard negatives = ratio * num_positives
        use_edge_length_weighting: bool = False,  # Edge length weighting (longer edges = higher weight) - DISABLED by default
        edge_length_weight_power: float = 1.5,  # Power for edge length weighting
        adjacency_weight: float = 2.0,  # Weight for adjacency-based edge loss component
        pair_weight: float = 1.0,  # Weight for pair-based edge loss component
    ):
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.coverage_weight = coverage_weight
        self.mask_weight = mask_weight
        self.use_mask_loss = use_mask_loss
        
        # Node loss: Weighted BCE + MSE + Dice + Sparsity + Mean Matching
        # Default weights: 35% Weighted BCE, 0% Focal Loss, 30% MSE, 25% Dice, 25% Sparsity, 15% Mean Matching
        # Weighted BCE addresses class imbalance
        # MSE provides stronger gradient signal (30%)
        # Dice loss helps with imbalanced segmentation (25%)
        # Sparsity regularization: Penalizes high mean predictions (25%) - STRONGER FIX
        # Mean matching: Explicitly matches target mean ~0.22 (15%) - STRONGER FIX
        # Focal loss disabled because it down-weights uniform predictions (counterproductive)
        self.node_loss_fn = NodeMaskLoss(
            mse_weight=0.30,  # Restored to provide stronger gradient signal
            dice_weight=0.25,  # Restored to help with imbalanced segmentation
            focal_alpha=0.25,
            focal_gamma=2.0,
            weighted_bce_weight=0.20,  # Core loss for class imbalance
            focal_weight=0.0,  # Disabled - focal loss doesn't help with uniform predictions
            sparsity_weight=0.10,  # Moderate sparsity regularization
            mean_matching_weight=0.15,  # CRITICAL FIX: Reduced from 0.50 to prevent gradient vanishing
            background_weight=0.20,  # CRITICAL FIX: Reduced from 0.40 to prevent conflict with mean matching
            use_dynamic_pos_weight=True,  # Compute pos_weight per batch
            fixed_pos_weight=4.8,  # Fallback if not using dynamic
            target_mean=0.22,  # Target heatmap mean (sparse)
        )
        # Use hybrid edge loss (adjacency + pairs) with configurable weights
        # Fallback to legacy loss if pair data is not available
        self.edge_loss_fn = EdgePredictionLoss(use_focal=False)  # Legacy (fallback)
        self.hybrid_edge_loss_fn = HybridEdgeLoss(
            adjacency_weight=adjacency_weight,  # Weight for adjacency-based loss
            pair_weight=pair_weight,  # Weight for pair-based loss
            adjacency_use_focal=edge_use_focal,
            pair_use_focal=edge_use_focal,
            focal_alpha=edge_focal_alpha,  # 0.5 (increased for class imbalance)
            focal_gamma=edge_focal_gamma,
            pos_weight=edge_pos_weight,  # 3.0 (positive weighting for class imbalance)
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_threshold=hard_negative_threshold,
            max_hard_negatives_ratio=max_hard_negatives_ratio,
            use_edge_length_weighting=use_edge_length_weighting,
            edge_length_weight_power=edge_length_weight_power,
        )  # Hybrid loss with equal weights
        self.mask_loss_fn = EdgeMaskLoss() if use_mask_loss else None
    
    def forward(
        self,
        pred_node_mask_logits: torch.Tensor,
        pred_edge_logits: torch.Tensor,
        target_node_mask: torch.Tensor,
        target_adj: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        coverage_loss: Optional[torch.Tensor] = None,
        target_edge_mask: Optional[torch.Tensor] = None,
        # New parameters for pair-based edge loss
        gt_edge_pairs: Optional[List[Optional[torch.Tensor]]] = None,  # List of [E_i, 2] or None
        gt_node_coords: Optional[torch.Tensor] = None,  # [B, N_gt, 2] or None
        pred_node_coords: Optional[torch.Tensor] = None,  # [B, N_pred, 2] or None
        valid_mask: Optional[torch.Tensor] = None,  # [B, N_pred] or None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred_node_mask_logits: Predicted node mask [B, 1, H, W]
            pred_edge_logits: Predicted edge logits [B, N, N]
            target_node_mask: Target node mask [B, H, W] or [B, 1, H, W]
            target_adj: Target adjacency matrix [B, N, N] or list
            candidate_mask: Candidate edge mask [B, N, N]
            coverage_loss: Coverage loss from ASNS (optional)
            target_edge_mask: Target edge mask [B, H, W] (optional)
        
        Returns:
            Dictionary with individual losses and total loss
        """
        # Node heatmap loss
        node_loss = self.node_loss_fn(pred_node_mask_logits, target_node_mask)
        
        # Edge prediction loss
        # Use pair-based loss if data is available (permutation-invariant)
        # Otherwise fallback to legacy adjacency-based loss
        use_pair_based = (
            gt_edge_pairs is not None and
            gt_node_coords is not None and
            pred_node_coords is not None and
            valid_mask is not None and
            all(pairs is not None for pairs in gt_edge_pairs)  # All batches have pair data
        )
        
        # Initialize loss components (will be set if using hybrid loss)
        adjacency_loss_component = None
        pair_loss_component = None
        
        if use_pair_based:
            # Match nodes by spatial distance
            node_permutation = match_nodes_greedy(
                pred_node_coords,  # [B, N_pred, 2] in image space
                gt_node_coords,    # [B, N_gt, 2] in image space
                valid_mask,        # [B, N_pred]
                max_distance=100.0  # pixels - INCREASED to improve matching
            )
            
            # Compute hybrid edge loss (adjacency + pairs) with equal weights
            edge_loss_dict = self.hybrid_edge_loss_fn(
                pred_edge_logits,
                target_adj,  # Adjacency matrix for dense supervision
                gt_edge_pairs,  # Point pairs for robust supervision
                node_permutation,
                valid_mask,
                pred_node_coords_image=pred_node_coords,  # For fallback matching
                gt_node_coords=gt_node_coords,  # For fallback matching
            )
            edge_loss = edge_loss_dict['loss']
            adjacency_loss_component = edge_loss_dict['adjacency_loss']
            pair_loss_component = edge_loss_dict['pair_loss']
            
        else:
            # Fallback to legacy adjacency-based loss (has permutation invariance issue)
            edge_loss = self.edge_loss_fn(pred_edge_logits, target_adj, candidate_mask)
        
        # Coverage loss (from ASNS)
        coverage_loss_val = coverage_loss if coverage_loss is not None else torch.tensor(0.0, device=pred_edge_logits.device)
        
        # Optional edge mask loss
        mask_loss_val = torch.tensor(0.0, device=pred_edge_logits.device)
        if self.use_mask_loss and target_edge_mask is not None:
            mask_loss_val = self.mask_loss_fn(pred_node_mask_logits, target_edge_mask)
        
        # Check all losses for NaN/Inf - if found, print warning and use fallback
        # We should NOT have NaN at this point if root causes are fixed
        if torch.isnan(node_loss) or torch.isinf(node_loss):
            print(f"⚠️  WARNING: node_loss is NaN/Inf: {node_loss.item()}")
            node_loss = torch.tensor(0.0, device=node_loss.device)
        if torch.isnan(edge_loss) or torch.isinf(edge_loss):
            print(f"⚠️  WARNING: edge_loss is NaN/Inf: {edge_loss.item()}")
            edge_loss = torch.tensor(0.0, device=edge_loss.device)
        if torch.isnan(coverage_loss_val) or torch.isinf(coverage_loss_val):
            print(f"⚠️  WARNING: coverage_loss is NaN/Inf: {coverage_loss_val.item()}")
            coverage_loss_val = torch.tensor(0.0, device=coverage_loss_val.device)
        
        # Total loss
        total_loss = (
            self.node_weight * node_loss +
            self.edge_weight * edge_loss +
            self.coverage_weight * coverage_loss_val +
            self.mask_weight * mask_loss_val
        )
        
        # Final NaN check on total loss - use small epsilon instead of 0
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1e-8, device=total_loss.device)
        
        result = {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'coverage_loss': coverage_loss_val,
            'mask_loss': mask_loss_val,
        }
        
        # Add hybrid loss components if available
        if use_pair_based and adjacency_loss_component is not None and pair_loss_component is not None:
            result['edge_adjacency_loss'] = adjacency_loss_component
            result['edge_pair_loss'] = pair_loss_component
        
        return result







