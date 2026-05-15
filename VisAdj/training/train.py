"""
SAM Graph Split - Full Training Script

PyTorch Lightning training with comprehensive logging and monitoring.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

# Add project root to path to use VisAdj as a package
# This avoids conflicts with other 'dataset' modules in sibling directories
image2matrix_root = Path(__file__).parent.parent.parent
if str(image2matrix_root) not in sys.path:
    sys.path.insert(0, str(image2matrix_root))

# Use absolute imports with the VisAdj package prefix to avoid conflicts
from VisAdj.model.sam_graph_split import SAMGraphSplit
from VisAdj.dataset.image2matrix_dataset import Image2MatrixDataset, collate_fn
from VisAdj.losses.combined_loss import CombinedLoss
from VisAdj.training.csv_logger import CSVLogger
from VisAdj.training.tqdm_logger import FileTQDMProgressBar
from VisAdj.training.tee_output import TeeOutput


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class PhaseAwareEarlyStopping(EarlyStopping):
    """Early stopping that activates only after a given epoch (phase boundary)."""

    def __init__(self, phase_start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.phase_start_epoch = phase_start_epoch
        self._phase_active = False

    def on_validation_end(self, trainer, pl_module):
        # Skip early stopping checks before the specified phase starts
        if trainer.current_epoch < self.phase_start_epoch:
            return
        if not self._phase_active:
            self._reset_state(trainer)
            self._phase_active = True
        return super().on_validation_end(trainer, pl_module)

    def _reset_state(self, trainer):
        # Reset best_score and wait_count when phase 2 starts so patience
        # only considers metrics from the new phase.
        device = trainer.lightning_module.device if trainer is not None else torch.device('cpu')
        if self.mode == 'min':
            best_score = torch.tensor(float('inf'), device=device)
        else:
            best_score = torch.tensor(float('-inf'), device=device)
        self.best_score = best_score
        self.wait_count = 0
        self.stopped_epoch = 0


class PhaseAwareModelCheckpoint(ModelCheckpoint):
    """Model checkpoint that ignores best-metric tracking until phase 2."""

    def __init__(self, phase_start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.phase_start_epoch = phase_start_epoch
        self._phase_active = False

    def on_validation_end(self, trainer, pl_module):
        # Always allow "last" checkpoint to be saved for resuming
        if trainer.current_epoch < self.phase_start_epoch:
            if self.save_last:
                # Pass empty monitor_candidates dict for last checkpoint saving
                self._save_last_checkpoint(trainer, monitor_candidates={})
            return
        if not self._phase_active:
            self._reset_state(trainer)
            self._phase_active = True
        return super().on_validation_end(trainer, pl_module)

    def _reset_state(self, trainer):
        device = trainer.lightning_module.device if trainer is not None else torch.device('cpu')
        if self.mode == 'min':
            best_score = torch.tensor(float('inf'), device=device)
        else:
            best_score = torch.tensor(float('-inf'), device=device)
        self.best_model_score = best_score
        self.best_model_path = ""
        self.kth_best_model_path = ""
        self.best_k_models = {}


class SAMGraphSplitLightning(pl.LightningModule):
    """PyTorch Lightning module for SAM Graph Split training."""
    
    def __init__(
        self,
        # Model parameters
        sam_version: str = 'vit_b',
        sam_checkpoint: Optional[str] = None,
        sam_config: Optional[str] = None,
        image_size: int = 512,
        heatmap_resolution: int = 32,
        heatmap_sigma: float = 1.5,
        max_nodes: int = 50,
        k_neighbors: int = 20,
        neighbor_radius: float = 256.0,
        neighbor_sampler: str = 'asns',
        relation_transformer_layers: int = 2,
        relation_edge_dim: int = 256,
        relation_hidden_dim: int = 256,
        relation_num_heads: int = 4,
        relation_dropout: float = 0.1,
        rgb_feature_dim: int = 32,
        rgb_sequence_model: str = 'transformer',
        rgb_seq_layers: int = 2,
        rgb_seq_heads: int = 4,
        rgb_neighborhood_aggregation: str = 'center',  # 'center', 'mean', 'median', or 'min_r_min_g_max_b'
        rgb_neighborhood_radius: float = 4.0,  # Radius in pixels for RGB neighborhood sampling (default: 4.0)
        edge_model: str = 'edge_aware_transformer',  # 'mlp', 'graph_transformer', or 'edge_aware_transformer'
        use_lora: bool = False,
        lora_rank: int = 8,
        # ASNS activation parameters
        asns_use_entmax: bool = True,  # Whether to use entmax (True) or sparsemax (False)
        asns_entmax_alpha: float = 1.5,  # Entmax alpha (1.0 = softmax, 1.5 = sparsemax, 2.0 = hardmax)
        # Loss parameters
        node_loss_weight: float = 2.0,
        edge_loss_weight: float = 5.0,
        coverage_loss_weight: float = 0.1,
        coverage_label_smoothing: float = 0.1,  # Label smoothing for coverage loss (default: 0.1 = 10% smoothing)
        mask_loss_weight: float = 0.0,
        use_mask_loss: bool = False,
        edge_use_focal: bool = True,
        edge_focal_alpha: float = 0.5,
        edge_focal_gamma: float = 2.0,
        edge_pos_weight: float = 3.0,
        use_hard_negative_mining: bool = True,
        hard_negative_threshold: float = 0.3,
        max_hard_negatives_ratio: float = 2.0,
        use_edge_length_weighting: bool = False,
        edge_length_weight_power: float = 1.5,
        adjacency_weight: float = 2.0,  # Weight for adjacency-based edge loss component
        pair_weight: float = 1.0,  # Weight for pair-based edge loss component
        # Node detection hyperparameters
        mask_threshold: float = 0.5,
        mask_pool_radius: int = 2,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 5.0,
        enable_diagnostics: bool = False,
        detach_l_i: bool = True,  # Whether to detach l_i before passing to ASNS and relation transformer
        phase1_epochs: int = 20,
        node_finetune_lr_scale: float = 0.2,
        teacher_forcing_epochs: int = 30,
        coordinate_noise_std: float = 2.0,  # Standard deviation of coordinate noise in pixels (for training robustness). Updated to match window=9 accuracy (mean offset: 1.52px, std: 0.73px)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SAMGraphSplit(
            sam_version=sam_version,
            sam_checkpoint=sam_checkpoint,
            sam_config=sam_config,
            freeze_encoder=True,
            image_size=image_size,
            heatmap_resolution=heatmap_resolution,
            max_nodes=max_nodes,
            k_neighbors=k_neighbors,
            neighbor_radius=neighbor_radius,
            relation_transformer_layers=relation_transformer_layers,
            relation_edge_dim=relation_edge_dim,
            relation_hidden_dim=relation_hidden_dim,
            relation_num_heads=relation_num_heads,
            relation_dropout=relation_dropout,
            rgb_feature_dim=rgb_feature_dim,
            rgb_sequence_model=rgb_sequence_model,
            rgb_seq_layers=rgb_seq_layers,
            rgb_seq_heads=rgb_seq_heads,
            rgb_neighborhood_aggregation=rgb_neighborhood_aggregation,
            rgb_neighborhood_radius=rgb_neighborhood_radius,
            edge_model=edge_model,
            neighbor_sampler=neighbor_sampler,
            # ASNS hyperparameters
            coverage_label_smoothing=coverage_label_smoothing,
            asns_use_entmax=asns_use_entmax,
            asns_entmax_alpha=asns_entmax_alpha,
            # Node detection hyperparameters
            mask_threshold=mask_threshold,
            mask_pool_radius=mask_pool_radius,
        )
        
        # Apply LoRA if requested
        if use_lora:
            logger.info(f"Applying LoRA with rank={lora_rank}")
            self.model.encoder.apply_lora(lora_rank=lora_rank)
        
        # Initialize loss function
        self.loss_fn = CombinedLoss(
            node_weight=node_loss_weight,
            edge_weight=edge_loss_weight,
            coverage_weight=coverage_loss_weight,
            mask_weight=mask_loss_weight,
            use_mask_loss=use_mask_loss,
            edge_use_focal=edge_use_focal,
            edge_focal_alpha=edge_focal_alpha,
            edge_focal_gamma=edge_focal_gamma,
            edge_pos_weight=edge_pos_weight,
            use_hard_negative_mining=use_hard_negative_mining,
            hard_negative_threshold=hard_negative_threshold,
            max_hard_negatives_ratio=max_hard_negatives_ratio,
            use_edge_length_weighting=use_edge_length_weighting,
            edge_length_weight_power=edge_length_weight_power,
            adjacency_weight=adjacency_weight,
            pair_weight=pair_weight,
        )
        
        # Store parameters
        self.heatmap_sigma = heatmap_sigma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.enable_diagnostics = enable_diagnostics
        self.detach_l_i = detach_l_i
        self.phase1_epochs = phase1_epochs
        self.node_finetune_lr_scale = node_finetune_lr_scale
        self.teacher_forcing_epochs = teacher_forcing_epochs
        self.neighbor_sampler = neighbor_sampler.lower() if neighbor_sampler else 'asns'
        self._base_param_lrs = {}
        self._current_phase = None
        self._edge_weight_full = edge_loss_weight
        self._coverage_weight_full = coverage_loss_weight
        self._grad_modules = {
            'mask_head': getattr(self.model.node_detector, 'mask_head', None),
            'local_descriptor_head': getattr(self.model.node_detector, 'local_descriptor_head', None),
            'global_descriptor_head': getattr(self.model.node_detector, 'global_descriptor_head', None),
            'relation_transformer': getattr(self.model, 'relation_transformer', None),
            'asns': getattr(self.model, 'asns', None),  # Add ASNS to track NaN gradients
        }

    def _log_diag(self, message: str):
        """Log diagnostic information only when enabled (rank 0)."""
        if not self.enable_diagnostics:
            return
        trainer = getattr(self, 'trainer', None)
        rank = getattr(trainer, 'global_rank', 0) if trainer is not None else 0
        if rank == 0:
            logger.info(message)

    def _teacher_forcing_probability(self) -> float:
        """Linearly decay teacher forcing probability from 1 to 0."""
        horizon = int(getattr(self.hparams, "teacher_forcing_epochs", 0))
        if horizon <= 0:
            return 0.0

        edge_epoch = max(0, int(self.current_epoch) - int(getattr(self.hparams, "phase1_epochs", 0)))
        return max(0.0, 1.0 - float(edge_epoch) / float(horizon))

    def _log_gradients(self):
        """Log gradient norms for key sub-modules (every optimizer step)."""
        if not self.enable_diagnostics:
            return
        grad_info = {}
        for name, module in self._grad_modules.items():
            if module is None:
                continue
            grads = []
            nan_grads = []
            num_params = 0
            num_with_grad = 0
            num_nan = 0
            for param in module.parameters():
                num_params += 1
                if param.grad is not None:
                    num_with_grad += 1
                    grad_norm = param.grad.detach().norm(2)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        num_nan += 1
                        nan_grads.append(grad_norm.item())
                    else:
                        grads.append(grad_norm)
            
            if grads:
                # Only compute mean over non-NaN gradients
                grad_norm = torch.stack(grads).mean()
                if num_nan > 0:
                    grad_info[name] = f"{grad_norm.item():.6f} ({num_nan} NaN grads)"
                    logger.warning(f"[GRAD DIAG] {name}: {num_nan}/{num_with_grad} params have NaN/Inf gradients")
                else:
                    grad_info[name] = f"{grad_norm.item():.6f}"
                self.log(
                    f'grads/{name}',
                    grad_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )
            elif num_nan > 0:
                grad_info[name] = f"ALL_NAN ({num_nan}/{num_with_grad} params have NaN grads)"
                logger.warning(f"[GRAD DIAG] {name}: all {num_with_grad} params have NaN/Inf gradients")
            else:
                grad_info[name] = f"NO_GRAD ({num_with_grad}/{num_params} params have grad)"
        
        # Print to stdout so we can see it in the log file
        if grad_info:
            self._log_diag(f"[GRADIENTS @ step {self.global_step}] {grad_info}")
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(images, return_intermediates=True)
    
    def _permute_node_features_to_gt_order(
        self,
        node_coords: torch.Tensor,  # [B, N_pred, 2]
        l_i: torch.Tensor,  # [B, N_pred, D]
        g_i: torch.Tensor,  # [B, N_pred, D]
        node_permutation: torch.Tensor,  # [B, N_pred] - maps pred node i to GT node j
        valid_mask: torch.Tensor,  # [B, N_pred]
        gt_node_coords: torch.Tensor,  # [B, N_gt, 2] or list
        local_features: Optional[torch.Tensor] = None,  # [B, C_L, H, W] - for extracting unmatched GT node features
        global_features: Optional[torch.Tensor] = None,  # [B, C_G, H_g, W_g] - for extracting unmatched GT node features
    ) -> tuple:
        """
        Permute node features to GT node order.
        
        For unmatched GT nodes, extracts features from local_features and global_features
        at GT node coordinates, ensuring all GT nodes have useful features (not zeros).
        
        Args:
            node_coords: Predicted node coordinates [B, N_pred, 2] in heatmap space
            l_i: Local descriptors [B, N_pred, D]
            g_i: Global descriptors [B, N_pred, D]
            node_permutation: [B, N_pred] where permutation[b, i] = j means pred node i matches GT node j
            valid_mask: [B, N_pred] - valid predicted nodes
            gt_node_coords: GT node coordinates [B, N_gt, 2] or list in image space
            local_features: [B, C_L, H, W] - local features for extracting unmatched GT node features
            global_features: [B, C_G, H_g, W_g] - global features for extracting unmatched GT node features
            
        Returns:
            permuted_node_coords: [B, N_gt, 2] in GT order (heatmap space)
            permuted_l_i: [B, N_gt, D] in GT order
            permuted_g_i: [B, N_gt, D] in GT order
            permuted_valid_mask: [B, N_gt] in GT order (True for all GT nodes, matched or unmatched)
        """
        B, N_pred, D = l_i.shape
        
        # Get N_gt
        if isinstance(gt_node_coords, list):
            N_gt = max(len(coords) if coords is not None else 0 for coords in gt_node_coords)
        else:
            N_gt = gt_node_coords.shape[1]
        
        device = node_coords.device
        dtype = node_coords.dtype
        
        # CRITICAL FIX: Use vectorized indexing to preserve gradients!
        # Python for loops with .item() break the computational graph.
        # We need to use advanced indexing to maintain gradient flow.
        
        # Initialize with zeros (will be filled with matched nodes and unmatched GT nodes)
        permuted_node_coords = torch.zeros(B, N_gt, 2, device=device, dtype=dtype)
        permuted_l_i = torch.zeros(B, N_gt, D, device=device, dtype=l_i.dtype)
        permuted_g_i = torch.zeros(B, N_gt, D, device=device, dtype=g_i.dtype)
        permuted_valid_mask = torch.zeros(B, N_gt, dtype=torch.bool, device=device)
        
        # Get feature extraction parameters
        heatmap_resolution = self.hparams.heatmap_resolution  # e.g., 32
        image_size = self.hparams.image_size  # e.g., 512
        scale_factor = heatmap_resolution / image_size  # 32 / 512 = 0.0625
        
        # Prepare GT coordinates (convert to tensor if list)
        if isinstance(gt_node_coords, list):
            gt_coords_tensor = torch.zeros(B, N_gt, 2, device=device, dtype=torch.float32)
            gt_coords_valid_mask = torch.zeros(B, N_gt, dtype=torch.bool, device=device)
            for b in range(B):
                coords_b = gt_node_coords[b] if b < len(gt_node_coords) else None
                if coords_b is not None and len(coords_b) > 0:
                    if isinstance(coords_b, torch.Tensor):
                        coords_b = coords_b.to(device)
                    else:
                        coords_b = torch.tensor(coords_b, device=device, dtype=torch.float32)
                    n_gt_b = min(len(coords_b), N_gt)
                    gt_coords_tensor[b, :n_gt_b] = coords_b[:n_gt_b]
                    gt_coords_valid_mask[b, :n_gt_b] = True
        else:
            gt_coords_tensor = gt_node_coords.to(device=device, dtype=torch.float32)  # [B, N_gt, 2] in image space
            gt_coords_valid_mask = (gt_coords_tensor.sum(dim=-1) > 0)  # [B, N_gt]

        gt_coords_tensor_heatmap = gt_coords_tensor * scale_factor  # convert once to heatmap space
        
        # OPTIMIZATION: Process descriptor heads ONCE for entire batch (not per sample)
        # This reduces descriptor head calls from B to 1, providing massive speedup
        # 
        # NO PADDING ISSUES: 
        # - local_features and global_features are already batched tensors [B, C, H, W] from dual_stream
        # - All samples in batch have identical spatial dimensions (32x32 for local, 8x8 for global)
        # - Descriptor heads (Conv2d with padding=1) preserve spatial dimensions, output [B, D, H, W]
        # - We slice per-sample inside the loop: local_processed[b:b+1] extracts one sample
        # - grid_sample operations are still per-sample, handling variable unmatched GT nodes correctly
        local_processed = None
        global_processed = None
        if local_features is not None and global_features is not None:
            local_processed = self.model.node_detector.local_descriptor_head(local_features)  # [B, D, H, W]
            global_processed = self.model.node_detector.global_descriptor_head(global_features)  # [B, D, H_g, W_g]
            H_local, W_local = local_processed.shape[2], local_processed.shape[3]  # e.g., 32, 32
            H_global, W_global = global_processed.shape[2], global_processed.shape[3]  # e.g., 8, 8
        
        # Vectorized permutation for gradient flow
        for b in range(B):
            valid_pred = valid_mask[b]  # [N_pred] bool
            matched_gt_indices = node_permutation[b]  # [N_pred] long
            
            # Filter: only process valid predicted nodes
            valid_pred_indices = torch.where(valid_pred)[0]  # [num_valid]
                
            # Track which GT nodes are matched
            matched_gt_set = set()
            
            if len(valid_pred_indices) > 0:
                gt_indices_for_valid = matched_gt_indices[valid_pred_indices]  # [num_valid]
                valid_gt_mask = (gt_indices_for_valid >= 0) & (gt_indices_for_valid < N_gt)
            
                if valid_gt_mask.any():
                    final_pred_indices = valid_pred_indices[valid_gt_mask]  # [num_matched]
                    final_gt_indices = gt_indices_for_valid[valid_gt_mask]  # [num_matched]
            
                    permuted_node_coords[b, final_gt_indices] = gt_coords_tensor_heatmap[b, final_gt_indices]
                    permuted_l_i[b, final_gt_indices] = l_i[b, final_pred_indices]
                    permuted_g_i[b, final_gt_indices] = g_i[b, final_pred_indices]
                    permuted_valid_mask[b, final_gt_indices] = True
                    matched_gt_set = set(final_gt_indices.cpu().tolist())
            
            # Extract features for unmatched GT nodes
            if local_processed is not None and global_processed is not None:
                # Get valid GT nodes for this batch
                valid_gt_indices = torch.where(gt_coords_valid_mask[b])[0]  # [num_valid_gt]
                
                # Find unmatched GT nodes
                unmatched_gt_indices = [idx for idx in valid_gt_indices.cpu().tolist() if idx not in matched_gt_set]
                
                if len(unmatched_gt_indices) > 0:
                    unmatched_gt_indices_tensor = torch.tensor(unmatched_gt_indices, device=device, dtype=torch.long)  # [num_unmatched]
                    unmatched_gt_coords_image = gt_coords_tensor[b, unmatched_gt_indices_tensor]  # [num_unmatched, 2] in image space
                    unmatched_gt_coords_local = gt_coords_tensor_heatmap[b, unmatched_gt_indices_tensor].clone()
                    
                    # Use pre-processed features (already processed through descriptor heads for entire batch)
                    local_processed_b = local_processed[b:b+1]  # [1, D, H, W] - slice from batch
                    global_processed_b = global_processed[b:b+1]  # [1, D, H_g, W_g] - slice from batch
                    
                    # Convert GT coordinates from image space to heatmap space (for local_processed)
                    # Normalize coordinates to [-1, 1] for grid_sample
                    coords_local_norm = unmatched_gt_coords_local.clone()
                    if W_local > 1:
                        coords_local_norm[:, 0] = (coords_local_norm[:, 0] / (W_local - 1)) * 2.0 - 1.0
                    else:
                        coords_local_norm[:, 0] = 0.0
                    if H_local > 1:
                        coords_local_norm[:, 1] = (coords_local_norm[:, 1] / (H_local - 1)) * 2.0 - 1.0
                    else:
                        coords_local_norm[:, 1] = 0.0
                    
                    # Sample processed local features for unmatched GT nodes
                    coords_grid_local = coords_local_norm.unsqueeze(0).unsqueeze(2)  # [1, num_unmatched, 1, 2]
                    l_i_unmatched = F.grid_sample(
                        local_processed_b,  # [1, D, H, W] - already processed through descriptor head
                        coords_grid_local,
                        mode='bilinear',
                        align_corners=False
                    )  # [1, D, num_unmatched, 1]
                    l_i_unmatched = l_i_unmatched.squeeze(-1).permute(0, 2, 1)  # [1, num_unmatched, D]
                    
                    # Convert GT coordinates from image space to global space (for global_processed)
                    # First convert to heatmap space, then to global space (same as node_detector)
                    # Convert image space ->heatmap space ->global space (matching node_detector logic)
                    unmatched_gt_coords_global = unmatched_gt_coords_local.clone()  # Start from heatmap space coords
                    if W_local > 1:
                        unmatched_gt_coords_global[:, 0] = unmatched_gt_coords_local[:, 0] * (W_global - 1) / (W_local - 1)
                    else:
                        unmatched_gt_coords_global[:, 0] = 0.0
                    if H_local > 1:
                        unmatched_gt_coords_global[:, 1] = unmatched_gt_coords_local[:, 1] * (H_global - 1) / (H_local - 1)
                    else:
                        unmatched_gt_coords_global[:, 1] = 0.0
                    
                    # Normalize coordinates to [-1, 1] for grid_sample
                    coords_global_norm = unmatched_gt_coords_global.clone()
                    if W_global > 1:
                        coords_global_norm[:, 0] = (coords_global_norm[:, 0] / (W_global - 1)) * 2.0 - 1.0
                    else:
                        coords_global_norm[:, 0] = 0.0
                    if H_global > 1:
                        coords_global_norm[:, 1] = (coords_global_norm[:, 1] / (H_global - 1)) * 2.0 - 1.0
                    else:
                        coords_global_norm[:, 1] = 0.0
                    
                    # Sample processed global features for unmatched GT nodes
                    coords_grid_global = coords_global_norm.unsqueeze(0).unsqueeze(2)  # [1, num_unmatched, 1, 2]
                    g_i_unmatched = F.grid_sample(
                        global_processed_b,  # [1, D, H_g, W_g] - already processed through descriptor head
                        coords_grid_global,
                        mode='bilinear',
                        align_corners=False
                    )  # [1, D, num_unmatched, 1]
                    g_i_unmatched = g_i_unmatched.squeeze(-1).permute(0, 2, 1)  # [1, num_unmatched, D]
                    
                    # Place unmatched GT node features
                    permuted_node_coords[b, unmatched_gt_indices_tensor] = unmatched_gt_coords_local
                    permuted_l_i[b, unmatched_gt_indices_tensor] = l_i_unmatched[0]  # Remove batch dimension
                    permuted_g_i[b, unmatched_gt_indices_tensor] = g_i_unmatched[0]  # Remove batch dimension
                    permuted_valid_mask[b, unmatched_gt_indices_tensor] = True
        
        return permuted_node_coords, permuted_l_i, permuted_g_i, permuted_valid_mask
    
    def _prepare_gt_node_coords(
        self,
        gt_node_coords: Optional[Any],
        device: torch.device,
    ) -> Optional[Any]:
        """
        Move GT node coords to device while preserving list structure.
        """
        if gt_node_coords is None:
            return None
        
        if isinstance(gt_node_coords, list):
            gt_device = []
            for coords in gt_node_coords:
                if coords is None:
                    gt_device.append(None)
                else:
                    gt_device.append(coords.to(device=device, dtype=torch.float32))
            return gt_device
        
        # Tensor case
        return gt_node_coords.to(device=device, dtype=torch.float32)
    
    def _run_backbone(self, images: torch.Tensor, detailed_profile: bool = False) -> Dict[str, torch.Tensor]:
        """
        Run encoder, dual streams, node detector, and global topology once.
        
        PERFORMANCE NOTE: The main bottleneck is in node_detector.forward() which has a 
        sequential loop `for b in range(batch_size)` calling _extract_nodes() for each sample.
        _extract_nodes() contains CPU-GPU synchronization points (torch.nonzero, NMS) that 
        are executed batch_size times sequentially. With batch_size=128, this causes ~20s overhead.
        
        To fix: Vectorize node detection to process all batch samples in parallel on GPU,
        or use a smaller batch size (8-16) to reduce the number of sequential operations.
        """
        if detailed_profile:
            t_enc_start = time.time()
            encoder_features = self.model.encoder(images)
            t_enc = time.time() - t_enc_start
            
            t_dual_start = time.time()
            local_features, global_features = self.model.dual_stream(encoder_features)
            t_dual = time.time() - t_dual_start
            
            t_node_start = time.time()
            node_coords, l_i_detached, g_i, node_mask_logits = self.model.node_detector(
                local_features,
                global_features
            )
            node_coords_pixel = self.model.node_detector.latest_coords_pixel
            if node_coords_pixel is None:
                node_coords_pixel = torch.zeros_like(node_coords)
            t_node = time.time() - t_node_start
            
            t_topo_start = time.time()
            G_prime, _, z_star = self.model.global_topology(global_features)
            t_topo = time.time() - t_topo_start
            
            total = t_enc + t_dual + t_node + t_topo
            self._log_diag(f"[BACKBONE DETAIL] Encoder={t_enc:.2f}s ({100*t_enc/total:.1f}%), "
                       f"DualStream={t_dual:.2f}s ({100*t_dual/total:.1f}%), "
                       f"NodeDetector={t_node:.2f}s ({100*t_node/total:.1f}%), "
                       f"GlobalTopo={t_topo:.2f}s ({100*t_topo/total:.1f}%), "
                       f"Total={total:.2f}s")
            
            return {
                'node_coords': node_coords,
                'node_coords_pixel': node_coords_pixel,
                'l_i': l_i_detached,
                'g_i': g_i,
                'node_mask_logits': node_mask_logits,
                'G_prime': G_prime,
                'z_star': z_star,
                'local_features': local_features,  # [B, C_L, 32, 32] for path sampling
                'global_features': global_features,  # [B, C_G, 8, 8] for extracting unmatched GT node features
            }
        else:
            encoder_features = self.model.encoder(images)
            local_features, global_features = self.model.dual_stream(encoder_features)
            node_coords, l_i_detached, g_i, node_mask_logits = self.model.node_detector(
                local_features,
                global_features
            )
            node_coords_pixel = self.model.node_detector.latest_coords_pixel
            if node_coords_pixel is None:
                node_coords_pixel = torch.zeros_like(node_coords)
            G_prime, _, z_star = self.model.global_topology(global_features)
        
        return {
            'node_coords': node_coords,
            'node_coords_pixel': node_coords_pixel,
            'l_i': l_i_detached,
            'g_i': g_i,
            'node_mask_logits': node_mask_logits,
            'G_prime': G_prime,
            'z_star': z_star,
            'local_features': local_features,  # [B, C_L, 32, 32] for path sampling
            'global_features': global_features,  # [B, C_G, 8, 8] for extracting unmatched GT node features
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        images = batch['image']  # [B, 3, H, W]
        target_node_mask = batch['node_mask']  # [B, H, W]
        target_adj = batch['adjacency_matrix']  # List of [N_i, N_i] tensors
        gt_edge_pairs = batch.get('edge_pairs', None)  # List of [E_i, 2] or None
        gt_node_coords = batch.get('gt_node_coords', None)  # [B, N_gt, 2] or None
        target_edge_mask = batch.get('edge_mask', None)  # [B, H, W] or None
        gt_node_coords_device = self._prepare_gt_node_coords(gt_node_coords, images.device)
        
        # Forward pass - run backbone once to extract features before matching
        t0 = time.time()
        # Enable detailed profiling on first batch of first epoch to identify bottleneck within backbone
        detailed_profile = (batch_idx == 0 and self.training and self.current_epoch == 0)
        backbone_outputs = self._run_backbone(images, detailed_profile=detailed_profile)
        t_backbone = time.time() - t0
        pred_node_mask_logits = backbone_outputs['node_mask_logits']  # [B, 1, H, W]
        node_coords = backbone_outputs['node_coords']  # [B, N, 2] - detected nodes
        l_i = backbone_outputs['l_i']  # [B, N, D] - local descriptors (detached)
        g_i = backbone_outputs['g_i']  # [B, N, D] - global descriptors
        G_prime = backbone_outputs['G_prime']  # [B, C_G, 8, 8] - processed global features
        z_star = backbone_outputs['z_star']  # [B, D_z] - global embedding
        local_features = backbone_outputs['local_features']  # [B, C_L, 32, 32] - local features for path sampling
        global_features = backbone_outputs['global_features']  # [B, C_G, 8, 8] - global features for extracting unmatched GT node features
        
        # CRITICAL: Ensure pred_node_mask_logits maintains gradient connection
        # Check if logits require grad (should be True for training)
        if self.training and batch_idx == 0 and self.current_epoch < 3:
            self._log_diag(f"[GRAD CHECK] pred_node_mask_logits.requires_grad={pred_node_mask_logits.requires_grad}")
            if not pred_node_mask_logits.requires_grad:
                logger.warning("[GRAD CHECK] WARNING: pred_node_mask_logits does NOT require grad! This will break training!")
        
        # Get batch size
        B = images.shape[0]
        
        # DIAGNOSTIC: Log node mask statistics to diagnose detection issues
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_node_mask_logits)
            mask_mean = pred_probs.mean().item()
            mask_max = pred_probs.max().item()
            mask_min = pred_probs.min().item()
            mask_std = pred_probs.std().item()
            if batch_idx % 10 == 0:
                self.log('train/node_mask_mean', mask_mean, on_step=True, on_epoch=False, sync_dist=True)
                self.log('train/node_mask_max', mask_max, on_step=True, on_epoch=False, sync_dist=True)
                self.log('train/node_mask_min', mask_min, on_step=True, on_epoch=False, sync_dist=True)
                self.log('train/node_mask_std', mask_std, on_step=True, on_epoch=False, sync_dist=True)
        
        # Create valid mask (non-zero coordinates)
        # CRITICAL: Check if coordinates are non-zero AND not the dummy center node
        # The dummy node is at [H//2, W//2] = [16, 16] for 32x32 heatmap
        valid_mask = (node_coords.sum(dim=-1) > 0)  # [B, N]
        
        # Additional check: filter out dummy center nodes if they exist
        # Dummy nodes are at [16, 16] in 32x32 grid space
        # Use a tighter tolerance (0.1 instead of 0.5) to only filter exact center matches
        # This prevents filtering real nodes that happen to be near the center
        center_coord = torch.tensor([16.0, 16.0], device=node_coords.device)
        is_center = torch.all(torch.abs(node_coords - center_coord.unsqueeze(0).unsqueeze(0)) < 0.1, dim=-1)  # [B, N]
        valid_mask = valid_mask & (~is_center)  # Exclude center dummy nodes
        
        # DIAGNOSTIC: Log number of detected nodes (before and after filtering dummy nodes)
        with torch.no_grad():
            num_detected_nodes_raw = (node_coords.sum(dim=-1) > 0).sum(dim=1).float().mean().item()
            num_detected_nodes = valid_mask.sum(dim=1).float().mean().item()
            num_center_filtered = is_center.sum().item()
            if batch_idx % 10 == 0:
                self.log('train/num_detected_nodes_raw', num_detected_nodes_raw, on_step=True, on_epoch=False, sync_dist=True)
                self.log('train/num_detected_nodes', num_detected_nodes, on_step=True, on_epoch=False, sync_dist=True)
                if num_center_filtered > 0:
                    self._log_diag(f"[Dummy Node Filtering] Filtered {num_center_filtered} center nodes at [16, 16]")
        
        # Match nodes for permutation invariance
        # CRITICAL: We need to match nodes first, then permute node features to GT order
        # Then edge detection will use matched nodes
        node_permutation = None
        target_adj_permuted = None
        
        # Handle list format from collate_fn (variable-sized data)
        # OPTIMIZATION: Pad GT coordinates to enable batched matching (eliminates 128 sequential function calls)
        t1 = time.time()
        if gt_node_coords is not None and isinstance(gt_node_coords, list):
            from VisAdj.losses.combined_loss import match_nodes_hungarian, permute_adjacency_to_predicted_space
            
            N_pred = node_coords.shape[1]
            device = node_coords.device
            
            # Find max N_gt in batch for padding
            max_N_gt = max(len(coords) if coords is not None else 0 for coords in gt_node_coords)
            
            # Pad GT coordinates to [B, max_N_gt, 2] for batched matching
            gt_coords_padded = torch.zeros(B, max_N_gt, 2, device=device, dtype=torch.float32)
            gt_coords_valid_mask = torch.zeros(B, max_N_gt, dtype=torch.bool, device=device)
            
            for b in range(B):
                coords_b = gt_node_coords_device[b] if gt_node_coords_device is not None else None
                if coords_b is not None and coords_b.shape[0] > 0:
                    n_gt_b = coords_b.shape[0]
                    gt_coords_padded[b, :n_gt_b] = coords_b
                    gt_coords_valid_mask[b, :n_gt_b] = True
            
            # CRITICAL FIX: Scale GT coordinates from image space to heatmap space
            # Predicted coords are in heatmap space (32x32), GT coords are in image space (512x512)
            scale_factor = self.hparams.heatmap_resolution / self.hparams.image_size  # 32 / 512 = 0.0625
            gt_coords_padded_scaled = gt_coords_padded * scale_factor  # Scale to heatmap space
            
            # CRITICAL FIX: Set padded (invalid) GT coordinates to far location to prevent matching
            # This ensures match_nodes_hungarian won't match predicted nodes to padded zeros
            # Use a location far outside the heatmap space (32x32) so it exceeds max_distance=15.0
            far_location = torch.tensor([1000.0, 1000.0], device=device, dtype=torch.float32)
            gt_coords_padded_scaled = torch.where(
                gt_coords_valid_mask.unsqueeze(-1),  # [B, max_N_gt, 1]
                gt_coords_padded_scaled,  # Keep valid coordinates
                far_location.unsqueeze(0).unsqueeze(0).expand(B, max_N_gt, -1)  # Set padded to far location
            )
            
            # OPTIMIZATION: Use Hungarian algorithm for optimal node matching
            # Batched distance computation on GPU, then optimal assignment on CPU
            # This provides optimal matches (minimizes total distance) which is critical for training quality
            node_permutation = match_nodes_hungarian(
                pred_coords=node_coords,  # [B, N_pred, 2] in heatmap space
                gt_coords=gt_coords_padded_scaled,  # [B, max_N_gt, 2] in heatmap space (padded positions set to far location)
                valid_mask=valid_mask,  # [B, N_pred]
                max_distance=15.0,  # Increased from 5.0: in 32x32 heatmap space, allow up to ~15 pixels (equivalent to ~240 pixels in 512x512 image space)
                gt_valid_mask=gt_coords_valid_mask  # [B, max_N_gt] - Filter out padded GT nodes (matches greedy behavior)
            )  # [B, N_pred]
            
            # DIAGNOSTIC: Log matching statistics (only for first batch, first sample in training)
            if batch_idx == 0 and self.training:
                num_matched = (node_permutation[0] >= 0).sum().item()
                num_valid_pred = valid_mask[0].sum().item()
                num_gt = gt_coords_valid_mask[0].sum().item()
                self._log_diag(f"[Node Matching] Batch {batch_idx}, Sample 0: "
                          f"Valid Pred={num_valid_pred}, GT={num_gt}, Matched={num_matched}")
            
            # Permute adjacency for each sample (still need per-sample due to variable sizes)
            target_adj_permuted = []
            for b in range(B):
                if gt_node_coords_device[b] is not None:
                    target_adj_b = target_adj[b].unsqueeze(0).to(device)  # [1, N_gt, N_gt]
                    perm_b = node_permutation[b:b+1]  # [1, N_pred]
                    valid_mask_b = valid_mask[b:b+1] if valid_mask is not None else None  # [1, N_pred]
                    adj_perm_b = permute_adjacency_to_predicted_space(
                        target_adj=target_adj_b,
                        node_permutation=perm_b,
                        valid_mask=valid_mask_b,
                        max_n_pred=N_pred
                    )  # [1, N_pred, N_pred]
                    target_adj_permuted.append(adj_perm_b[0])
                else:
                    # No GT coords, use identity permutation
                    target_adj_permuted.append(target_adj[b].to(device))
        t_matching = time.time() - t1 if gt_node_coords is not None and isinstance(gt_node_coords, list) else 0.0
        
        # Store original valid_mask for node count error computation
        # (before it gets replaced with permuted mask)
        valid_mask_original = valid_mask.clone()  # [B, N_pred] - original predicted nodes

        teacher_forcing_prob = self._teacher_forcing_probability()
        use_teacher_forcing = (
            node_permutation is not None and
            gt_node_coords is not None and
            torch.rand((), device=images.device).item() < teacher_forcing_prob
        )
        self.log(
            'train/teacher_forcing_prob',
            torch.tensor(teacher_forcing_prob, device=images.device),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            'train/use_teacher_forcing',
            torch.tensor(float(use_teacher_forcing), device=images.device),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        
        # Scheduled teacher forcing: only condition edge reasoning on GT-order nodes
        # for a decaying fraction of training batches.
        t2 = time.time()
        use_matched_nodes = False
        if use_teacher_forcing:
            node_coords, l_i, g_i, valid_mask = self._permute_node_features_to_gt_order(
                node_coords=node_coords,
                l_i=l_i,
                g_i=g_i,
                node_permutation=node_permutation,
                valid_mask=valid_mask,
                gt_node_coords=gt_node_coords,
                local_features=local_features,  # For extracting unmatched GT node features
                global_features=global_features,  # For extracting unmatched GT node features
            )
            use_matched_nodes = True
            # target_adj_permuted was already computed during matching (lines 368-383)
            # After permuting nodes to GT order, we use target_adj directly (already in GT order)
            # Do NOT overwrite target_adj_permuted - it's only used when NOT using matched nodes
            
            # DIAGNOSTIC: Log permuted valid_mask statistics (only for first batch in training, less frequently)
            if batch_idx == 0 and self.training and self.current_epoch == 0:
                num_valid_after_permute = valid_mask.sum(dim=1).float().mean().item()
                self._log_diag(f"[After Permutation] Avg valid nodes per sample: {num_valid_after_permute:.2f}")
        else:
            use_matched_nodes = False
        t_permutation = time.time() - t2

        scale_to_image = self.hparams.image_size / self.hparams.heatmap_resolution
        node_coords_image_for_loss = node_coords * scale_to_image
        
        # Run ASNS and graph transformer once (after optional permutation)
        t3 = time.time()
        # CRITICAL FIX: Detach g_i when used in ASNS to prevent NaN gradient propagation
        # Even with coverage_loss_weight=0, ASNS creates a computation graph through g_i
        # ASNS operations (entmax/sparsemax) can produce NaN, which propagates back to global_descriptor_head
        # Since coverage loss is disabled, we don't need gradients through g_i for ASNS
        g_i_for_asns = g_i.detach() if self.loss_fn.coverage_weight == 0 else g_i
        # HYPERPARAMETER: Conditionally detach l_i based on detach_l_i flag
        # If detach_l_i=True: gradients from edge/coverage losses won't flow back to local descriptor head
        # If detach_l_i=False: gradients can flow back, allowing edge/coverage losses to influence local features
        l_i_for_asns = l_i.detach() if self.detach_l_i else l_i
        candidate_mask, attention_weights, attention_scores = self.model.asns(
            l_i_for_asns,  # Local descriptors (detached if detach_l_i=True)
            g_i_for_asns,  # Global descriptors (detached if coverage_loss_weight=0)
            node_coords,
            valid_mask
        )
        # CRITICAL FIX: Detach candidate_mask when coverage loss is disabled
        # The straight-through estimator in ASNS allows gradients to flow back to ASNS parameters
        # If ASNS has NaN gradients (from entmax/sparsemax), they propagate through candidate_mask
        # Since coverage loss is disabled, we don't need gradients through ASNS
        if self.loss_fn.coverage_weight == 0:
            candidate_mask = candidate_mask.detach()
        t_asns = time.time() - t3
        
        t4 = time.time()
        # CRITICAL FIX: Detach g_i when used in relation_transformer if edge_loss_weight=0
        # This prevents NaN gradient propagation when edge loss is disabled
        g_i_for_transformer = g_i.detach() if self.loss_fn.edge_weight == 0 else g_i
        # HYPERPARAMETER: Use the same l_i (detached or not) as used in ASNS for consistency
        l_i_for_transformer = l_i_for_asns
        
        # Add coordinate noise during training to make model robust to prediction errors
        node_coords_for_rgb = node_coords.clone()
        coords_in_image_space = False
        if self.training and self.hparams.coordinate_noise_std > 0:
            # Convert to image space for noise addition
            scale_to_image = self.hparams.image_size / self.hparams.heatmap_resolution
            node_coords_image = node_coords * scale_to_image  # [B, N, 2] in image space
            
            # Add Gaussian noise matching observed offset distribution
            noise = torch.randn_like(node_coords_image) * self.hparams.coordinate_noise_std
            noisy_coords_image = node_coords_image + noise
            
            # Clamp to image bounds
            noisy_coords_image = torch.clamp(noisy_coords_image, 0, self.hparams.image_size - 1)
            
            # Use noisy coordinates in image space for RGB sampling
            node_coords_for_rgb = noisy_coords_image
            coords_in_image_space = True
        
        pred_edge_logits = self.model.relation_transformer(
            l_i_for_transformer,  # Local descriptors (detached if detach_l_i=True)
            g_i_for_transformer,  # Global descriptors (detached if edge_loss_weight=0)
            node_coords_for_rgb,  # Use noisy coordinates for RGB sampling during training
            z_star,
            candidate_mask,
            G_prime=G_prime,  # Processed global features [B, C_G, 8, 8] (optional)
            local_features=local_features,  # Local features [B, C_L, 32, 32] for path sampling (optional)
            images=images,  # RGB images [B, 3, 512, 512] for color path sampling (optional)
            valid_mask=valid_mask,  # Valid node mask [B, N] (optional)
            coords_in_image_space=coords_in_image_space,  # Pass flag to indicate coordinate space
        )
        t_transformer = time.time() - t4
        
        # ============================================================
        # Loss Computation in Order: Node ->Edge ->Coverage ->Budget
        # ============================================================
        
        # Step 1: Compute Node Loss
        target_node_mask = target_node_mask.to(
            device=pred_node_mask_logits.device, dtype=pred_node_mask_logits.dtype
        )
        
        if self.training and batch_idx == 0 and self.current_epoch < 3:
            self._log_diag(
                "[GRAD CHECK] Before loss: pred_node_mask_logits.requires_grad="
                f"{pred_node_mask_logits.requires_grad}, is_leaf={pred_node_mask_logits.is_leaf}, "
                f"grad_fn={pred_node_mask_logits.grad_fn}"
            )
            if torch.isnan(pred_node_mask_logits).any() or torch.isinf(pred_node_mask_logits).any():
                logger.warning(
                    "[NAN CHECK] pred_node_mask_logits has NaN/Inf! "
                    f"nan={torch.isnan(pred_node_mask_logits).sum()}, inf={torch.isinf(pred_node_mask_logits).sum()}"
                )
            if torch.isnan(target_node_mask).any() or torch.isinf(target_node_mask).any():
                logger.warning("[NAN CHECK] target_node_mask has NaN/Inf!")
        
        node_loss = self.loss_fn.node_loss_fn(pred_node_mask_logits, target_node_mask)
        
        # CRITICAL: Verify node_loss has gradient connection after computation
        if self.training and batch_idx == 0 and self.current_epoch < 3:
            self._log_diag(f"[GRAD CHECK] After loss: node_loss.requires_grad={node_loss.requires_grad}, "
                       f"is_leaf={node_loss.is_leaf}, grad_fn={node_loss.grad_fn}")
        
        # DIAGNOSTIC: Check if node loss is NaN
        if self.training and batch_idx == 0 and self.current_epoch < 3:
            if torch.isnan(node_loss) or torch.isinf(node_loss):
                logger.warning(f"[NAN CHECK] node_loss is NaN/Inf! value={node_loss.item()}")
            else:
                self._log_diag(f"[NAN CHECK] node_loss value: {node_loss.item():.6f}")
        
        # DIAGNOSTIC: Log node loss components if available (training step only)
        if self.training and hasattr(self.loss_fn.node_loss_fn, '_last_components'):
            components = self.loss_fn.node_loss_fn._last_components
            if batch_idx == 0 and self.current_epoch < 3:
                self._log_diag(f"[NAN CHECK] Node loss components: {[(k, v.item()) for k, v in components.items()]}")
            if batch_idx % 10 == 0:
                for name, value in components.items():
                    self.log(f'train/node_loss_{name}', value.item(), on_step=True, on_epoch=False, sync_dist=True)
        
        # Step 2: Compute Edge Loss (with hybrid supervision)
        # CRITICAL FIX: Skip edge loss computation if weight=0 to prevent NaN gradient propagation
        # Even with weight=0, computing the loss creates a computation graph through pred_edge_logits
        # If the loss computation produces NaN, gradients propagate back to relation_transformer
        edge_loss = torch.tensor(0.0, device=pred_edge_logits.device, requires_grad=False)
        adjacency_loss_component = None
        pair_loss_component = None
        
        if self.loss_fn.edge_weight > 0:
            # Check if we have all data needed for hybrid edge loss (adjacency + pair-based)
            has_pair_data = (
                gt_edge_pairs is not None and
                gt_node_coords is not None and
                node_coords is not None and
                valid_mask is not None and
                all(pairs is not None for pairs in gt_edge_pairs)
            )
            
            if use_matched_nodes:
                N_gt = node_coords.shape[1]
                identity_permutation = torch.full((B, N_gt), -1, dtype=torch.long, device=node_coords.device)
                for b in range(B):
                    all_gt_positions = torch.where(valid_mask[b])[0]
                    if len(all_gt_positions) > 0:
                        identity_permutation[b, all_gt_positions] = all_gt_positions
                
                if has_pair_data:
                    edge_loss_dict = self.loss_fn.hybrid_edge_loss_fn(
                        pred_edge_logits,
                        target_adj,
                        gt_edge_pairs,
                        identity_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                        gt_node_coords=gt_node_coords_device,
                    )
                    edge_loss = edge_loss_dict['loss']
                    adjacency_loss_component = edge_loss_dict['adjacency_loss']
                    pair_loss_component = edge_loss_dict['pair_loss']
                else:
                    edge_loss = self.loss_fn.hybrid_edge_loss_fn.adjacency_loss(
                        pred_edge_logits,
                        target_adj,
                        identity_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                    )
            else:
                if has_pair_data:
                    edge_loss_dict = self.loss_fn.hybrid_edge_loss_fn(
                        pred_edge_logits,
                        target_adj,
                        gt_edge_pairs,
                        node_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                        gt_node_coords=gt_node_coords_device,
                    )
                    edge_loss = edge_loss_dict['loss']
                    adjacency_loss_component = edge_loss_dict['adjacency_loss']
                    pair_loss_component = edge_loss_dict['pair_loss']
                else:
                    if node_permutation is not None:
                        adj_for_edge = target_adj_permuted if target_adj_permuted is not None else target_adj
                        edge_loss = self.loss_fn.hybrid_edge_loss_fn.adjacency_loss(
                            pred_edge_logits,
                            adj_for_edge,
                            node_permutation,
                            valid_mask,
                            pred_node_coords_image=node_coords_image_for_loss,
                        )
                    else:
                        adj_for_edge = target_adj_permuted if target_adj_permuted is not None else target_adj
                        edge_loss = self.loss_fn.edge_loss_fn(
                            pred_edge_logits,
                            adj_for_edge,
                            candidate_mask,
                        )
        
        # Step 3: Compute ASNS Coverage Loss
        # CRITICAL FIX: Skip coverage loss computation if weight=0 to prevent NaN gradient propagation
        # Even with weight=0, computing the loss creates a computation graph through attention_weights
        # If the loss computation produces NaN, gradients propagate back to global_descriptor_head
        coverage_loss = torch.tensor(0.0, device=pred_edge_logits.device, requires_grad=False)
        if self.loss_fn.coverage_weight > 0:
            if candidate_mask is not None and (attention_weights is not None or attention_scores is not None):
                if use_matched_nodes:
                    adj_for_coverage = target_adj
                else:
                    adj_for_coverage = target_adj_permuted if target_adj_permuted is not None else target_adj
                
                coverage_loss = self.model.asns.compute_coverage_loss(
                    candidate_mask=candidate_mask,
                    target_adj=adj_for_coverage,
                    attention_weights=attention_weights,  # Preferred: normalized distribution from entmax/sparsemax
                    attention_scores=attention_scores,  # Fallback: pre-sparsemax scores
                    valid_mask=valid_mask,
                )
        
        # Step 4: Compute optional mask loss
        mask_loss = torch.tensor(0.0, device=pred_edge_logits.device)
        if self.loss_fn.use_mask_loss and target_edge_mask is not None:
            mask_loss = self.loss_fn.mask_loss_fn(pred_node_mask_logits, target_edge_mask)
        
        # Step 5: Combine all losses with weights
        t5 = time.time()
        total_loss = (
            self.loss_fn.node_weight * node_loss +
            self.loss_fn.edge_weight * edge_loss +
            self.loss_fn.coverage_weight * coverage_loss +
            self.loss_fn.mask_weight * mask_loss
        )
        t_loss = time.time() - t5
        
        # DIAGNOSTIC: Check if loss requires grad (should be True during training)
        if self.training and batch_idx == 0 and self.current_epoch < 3:
            self._log_diag(f"[LOSS CHECK] total_loss.requires_grad={total_loss.requires_grad}, "
                       f"total_loss={total_loss.item():.6f}, "
                       f"node_loss.requires_grad={node_loss.requires_grad}, "
                       f"edge_loss.requires_grad={edge_loss.requires_grad}")
            
            # Check if key parameters require grad
            mask_head = getattr(self.model.node_detector, 'mask_head', None)
            mask_requires_grad = any(p.requires_grad for p in mask_head.parameters()) if mask_head is not None else False
            transformer_requires_grad = any(p.requires_grad for p in self.model.relation_transformer.parameters())
            self._log_diag(f"[PARAM CHECK] mask_head.requires_grad={mask_requires_grad}, "
                       f"relation_transformer.requires_grad={transformer_requires_grad}")
        
        # PERFORMANCE PROFILING: Log timing breakdown (only for first batch, first epoch)
        if batch_idx == 0 and self.training and self.current_epoch == 0:
            total_time = t_backbone + t_matching + t_permutation + t_asns + t_transformer + t_loss
            self._log_diag(f"[PERF] Timing breakdown (batch {batch_idx}): "
                      f"Backbone={t_backbone:.2f}s ({100*t_backbone/total_time:.1f}%), "
                      f"Matching={t_matching:.2f}s ({100*t_matching/total_time:.1f}%), "
                      f"Permutation={t_permutation:.2f}s ({100*t_permutation/total_time:.1f}%), "
                      f"ASNS={t_asns:.2f}s ({100*t_asns/total_time:.1f}%), "
                      f"Transformer={t_transformer:.2f}s ({100*t_transformer/total_time:.1f}%), "
                      f"Loss={t_loss:.2f}s ({100*t_loss/total_time:.1f}%), "
                      f"Total={total_time:.2f}s")
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'edge_adjacency_loss': adjacency_loss_component,
            'edge_pair_loss': pair_loss_component,
            'coverage_loss': coverage_loss,
            'mask_loss': mask_loss,
        }
        
        # Log losses
        self.log('train/loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/node_loss', loss_dict['node_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/edge_loss', loss_dict['edge_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        # Log edge loss components if available (hybrid loss)
        if 'edge_adjacency_loss' in loss_dict and loss_dict['edge_adjacency_loss'] is not None:
            self.log('train/edge_adjacency_loss', loss_dict['edge_adjacency_loss'], on_step=False, on_epoch=True, sync_dist=True)
        if 'edge_pair_loss' in loss_dict and loss_dict['edge_pair_loss'] is not None:
            self.log('train/edge_pair_loss', loss_dict['edge_pair_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        self.log('train/coverage_loss', loss_dict['coverage_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        if loss_dict['mask_loss'] > 0:
            self.log('train/mask_loss', loss_dict['mask_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        return loss_dict['total_loss']
    
    def on_before_optimizer_step(self, optimizer):
        """Hook called before each optimizer step to log gradients and handle NaN."""
        self._log_gradients()
        
        # CRITICAL FIX: Zero out NaN gradients to prevent optimizer failures
        # NaN gradients can cause the optimizer to skip updates or fail silently
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"[NAN GRAD FIX] Zeroing NaN/Inf gradients for param with shape {param.shape}")
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CRITICAL FIX: Apply explicit gradient clipping to ensure it's used
        # PyTorch Lightning may not always apply gradient clipping automatically
        # Use the value from trainer if available, otherwise use the module's value
        clip_val = getattr(self.trainer, 'gradient_clip_val', None) if self.trainer else None
        if clip_val is None:
            clip_val = self.gradient_clip_val
        
        if clip_val is not None and clip_val > 0:
            # Get all parameters from all parameter groups
            parameters = []
            for param_group in optimizer.param_groups:
                parameters.extend([p for p in param_group['params'] if p.grad is not None])
            
            if parameters:
                # Compute gradient norm before clipping (for logging)
                total_norm_before = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
                
                # Apply norm-based gradient clipping
                total_norm_after = torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip_val)
                
                # Log the gradient norm before and after clipping (for debugging)
                if self.trainer and self.global_step % self.trainer.log_every_n_steps == 0:
                    self._log_diag(f"[GRAD CLIP] Applied gradient clipping: max_norm={clip_val:.2f}, "
                                   f"norm_before={total_norm_before.item():.4f}, norm_after={total_norm_after.item():.4f}")
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Extract batch data
        images = batch['image']
        target_node_mask = batch['node_mask']
        target_adj = batch['adjacency_matrix']
        gt_edge_pairs = batch.get('edge_pairs', None)
        gt_node_coords = batch.get('gt_node_coords', None)
        target_edge_mask = batch.get('edge_mask', None)
        gt_node_coords_device = self._prepare_gt_node_coords(gt_node_coords, images.device)
        
        # Forward pass - run backbone once to extract features before matching
        backbone_outputs = self._run_backbone(images)
        pred_node_mask_logits = backbone_outputs['node_mask_logits']
        node_coords = backbone_outputs['node_coords']  # [B, N, 2] - detected nodes
        l_i = backbone_outputs['l_i']  # [B, N, D] - local descriptors
        g_i = backbone_outputs['g_i']  # [B, N, D] - global descriptors
        G_prime = backbone_outputs['G_prime']  # [B, C_G, 8, 8] - processed global features
        z_star = backbone_outputs['z_star']  # [B, D_z] - global embedding
        local_features = backbone_outputs['local_features']  # [B, C_L, 32, 32] - local features for path sampling
        global_features = backbone_outputs['global_features']  # [B, C_G, 8, 8] - global features for extracting unmatched GT node features
        
        # Get batch size
        B = images.shape[0]
        
        # Create valid mask (non-zero coordinates)
        # CRITICAL: Check if coordinates are non-zero AND not the dummy center node
        # The dummy node is at [H//2, W//2] = [16, 16] for 32x32 heatmap
        valid_mask = (node_coords.sum(dim=-1) > 0)  # [B, N]
        
        # Additional check: filter out dummy center nodes if they exist
        # Dummy nodes are at [16, 16] in 32x32 grid space
        # Use a tighter tolerance (0.1 instead of 0.5) to only filter exact center matches
        # This prevents filtering real nodes that happen to be near the center
        center_coord = torch.tensor([16.0, 16.0], device=node_coords.device)
        is_center = torch.all(torch.abs(node_coords - center_coord.unsqueeze(0).unsqueeze(0)) < 0.1, dim=-1)  # [B, N]
        valid_mask = valid_mask & (~is_center)  # Exclude center dummy nodes
        
        # Match nodes for permutation invariance
        # CRITICAL: We need to match nodes first, then permute node features to GT order
        # Then edge detection will use matched nodes
        node_permutation = None
        target_adj_permuted = None
        
        # Handle list format from collate_fn (variable-sized data)
        # OPTIMIZATION: Pad GT coordinates to enable batched matching (eliminates 128 sequential function calls)
        if gt_node_coords is not None and isinstance(gt_node_coords, list):
            from VisAdj.losses.combined_loss import match_nodes_hungarian, permute_adjacency_to_predicted_space
            
            N_pred = node_coords.shape[1]
            device = node_coords.device
            
            # Find max N_gt in batch for padding
            max_N_gt = max(len(coords) if coords is not None else 0 for coords in gt_node_coords)
            
            # Pad GT coordinates to [B, max_N_gt, 2] for batched matching
            gt_coords_padded = torch.zeros(B, max_N_gt, 2, device=device, dtype=torch.float32)
            gt_coords_valid_mask = torch.zeros(B, max_N_gt, dtype=torch.bool, device=device)
            
            for b in range(B):
                coords_b = gt_node_coords_device[b] if gt_node_coords_device is not None else None
                if coords_b is not None and coords_b.shape[0] > 0:
                    n_gt_b = coords_b.shape[0]
                    gt_coords_padded[b, :n_gt_b] = coords_b
                    gt_coords_valid_mask[b, :n_gt_b] = True
            
            # CRITICAL FIX: Scale GT coordinates from image space to heatmap space
            # Predicted coords are in heatmap space (32x32), GT coords are in image space (512x512)
            scale_factor = self.hparams.heatmap_resolution / self.hparams.image_size  # 32 / 512 = 0.0625
            gt_coords_padded_scaled = gt_coords_padded * scale_factor  # Scale to heatmap space
            
            # CRITICAL FIX: Set padded (invalid) GT coordinates to far location to prevent matching
            # This ensures match_nodes_hungarian won't match predicted nodes to padded zeros
            # Use a location far outside the heatmap space (32x32) so it exceeds max_distance=15.0
            far_location = torch.tensor([1000.0, 1000.0], device=device, dtype=torch.float32)
            gt_coords_padded_scaled = torch.where(
                gt_coords_valid_mask.unsqueeze(-1),  # [B, max_N_gt, 1]
                gt_coords_padded_scaled,  # Keep valid coordinates
                far_location.unsqueeze(0).unsqueeze(0).expand(B, max_N_gt, -1)  # Set padded to far location
            )
            
            # OPTIMIZATION: Use Hungarian algorithm for optimal node matching
            # Batched distance computation on GPU, then optimal assignment on CPU
            # This provides optimal matches (minimizes total distance) which is critical for training quality
            node_permutation = match_nodes_hungarian(
                pred_coords=node_coords,  # [B, N_pred, 2] in heatmap space
                gt_coords=gt_coords_padded_scaled,  # [B, max_N_gt, 2] in heatmap space (padded positions set to far location)
                valid_mask=valid_mask,  # [B, N_pred]
                max_distance=15.0,  # Increased from 5.0: in 32x32 heatmap space, allow up to ~15 pixels (equivalent to ~240 pixels in 512x512 image space)
                gt_valid_mask=gt_coords_valid_mask  # [B, max_N_gt] - Filter out padded GT nodes (matches greedy behavior)
            )  # [B, N_pred]
            
            # Permute adjacency for each sample (still need per-sample due to variable sizes)
            target_adj_permuted = []
            for b in range(B):
                if gt_node_coords_device[b] is not None:
                    target_adj_b = target_adj[b].unsqueeze(0).to(device)  # [1, N_gt, N_gt]
                    perm_b = node_permutation[b:b+1]  # [1, N_pred]
                    valid_mask_b = valid_mask[b:b+1] if valid_mask is not None else None  # [1, N_pred]
                    adj_perm_b = permute_adjacency_to_predicted_space(
                        target_adj=target_adj_b,
                        node_permutation=perm_b,
                        valid_mask=valid_mask_b,
                        max_n_pred=N_pred
                    )  # [1, N_pred, N_pred]
                    target_adj_permuted.append(adj_perm_b[0])
                else:
                    # No GT coords, use identity permutation
                    target_adj_permuted.append(target_adj[b].to(device))
        
        # Store original valid_mask for node count error computation
        # (before it gets replaced with permuted mask)
        valid_mask_original = valid_mask.clone()  # [B, N_pred] - original predicted nodes
        
        # OPTIMIZATION: Skip expensive edge/coverage operations during Phase 1 validation
        # when edge and coverage losses are disabled (weight=0)
        skip_edge_ops = (self.loss_fn.edge_weight == 0 and self.loss_fn.coverage_weight == 0)
        
        # Validation follows the test-time setting: edge reasoning uses predicted
        # nodes only. Matching is still computed above for supervision/evaluation.
        use_teacher_forcing = False
        use_matched_nodes = False
        if use_teacher_forcing and node_permutation is not None and gt_node_coords is not None:
            # OPTIMIZATION: Skip expensive unmatched GT node feature extraction if edge/coverage are disabled
            if skip_edge_ops:
                # Only permute matched nodes (skip unmatched GT node feature extraction)
                # This is much faster since we don't need to process descriptor heads
                # CRITICAL FIX: Use GT coordinates (not predicted) for matched nodes
                B, N_pred, D = l_i.shape
                if isinstance(gt_node_coords, list):
                    N_gt = max(len(coords) if coords is not None else 0 for coords in gt_node_coords)
                else:
                    N_gt = gt_node_coords.shape[1]
                
                permuted_node_coords = torch.zeros(B, N_gt, 2, device=node_coords.device, dtype=node_coords.dtype)
                permuted_l_i = torch.zeros(B, N_gt, D, device=l_i.device, dtype=l_i.dtype)
                permuted_g_i = torch.zeros(B, N_gt, D, device=g_i.device, dtype=g_i.dtype)
                permuted_valid_mask = torch.zeros(B, N_gt, dtype=torch.bool, device=valid_mask.device)
                
                # Convert GT coordinates from image space to heatmap space
                scale_factor = self.hparams.heatmap_resolution / self.hparams.image_size  # 32 / 512 = 0.0625
                if isinstance(gt_node_coords, list):
                    gt_coords_tensor = torch.zeros(B, N_gt, 2, device=node_coords.device, dtype=torch.float32)
                    for b in range(B):
                        coords_b = gt_node_coords_device[b] if gt_node_coords_device is not None else None
                        if coords_b is not None and len(coords_b) > 0:
                            n_gt_b = min(len(coords_b), N_gt)
                            gt_coords_tensor[b, :n_gt_b] = coords_b[:n_gt_b]
                else:
                    gt_coords_tensor = gt_node_coords_device.to(device=node_coords.device, dtype=torch.float32)
                gt_coords_tensor_heatmap = gt_coords_tensor * scale_factor  # Convert to heatmap space
                
                for b in range(B):
                    valid_pred = valid_mask[b]
                    matched_gt_indices = node_permutation[b]
                    valid_pred_indices = torch.where(valid_pred)[0]
                    if len(valid_pred_indices) > 0:
                        gt_indices_for_valid = matched_gt_indices[valid_pred_indices]
                        valid_gt_mask = (gt_indices_for_valid >= 0) & (gt_indices_for_valid < N_gt)
                        if valid_gt_mask.any():
                            final_pred_indices = valid_pred_indices[valid_gt_mask]
                            final_gt_indices = gt_indices_for_valid[valid_gt_mask]
                            # CRITICAL FIX: Use GT coordinates (not predicted) for matched nodes
                            permuted_node_coords[b, final_gt_indices] = gt_coords_tensor_heatmap[b, final_gt_indices]
                            permuted_l_i[b, final_gt_indices] = l_i[b, final_pred_indices]
                            permuted_g_i[b, final_gt_indices] = g_i[b, final_pred_indices]
                            permuted_valid_mask[b, final_gt_indices] = True
                
                node_coords = permuted_node_coords
                l_i = permuted_l_i
                g_i = permuted_g_i
                valid_mask = permuted_valid_mask
            else:
                # Full permutation including unmatched GT node feature extraction
                node_coords, l_i, g_i, valid_mask = self._permute_node_features_to_gt_order(
                    node_coords=node_coords,
                    l_i=l_i,
                    g_i=g_i,
                    node_permutation=node_permutation,
                    valid_mask=valid_mask,
                    gt_node_coords=gt_node_coords,
                    local_features=local_features,  # For extracting unmatched GT node features
                    global_features=global_features,  # For extracting unmatched GT node features
                )
            use_matched_nodes = True
            # target_adj_permuted was already computed during matching (lines 728-743)
            # After permuting nodes to GT order, we use target_adj directly (already in GT order)
            # Do NOT overwrite target_adj_permuted - it's only used when NOT using matched nodes
        else:
            use_matched_nodes = False
        
        scale_to_image = self.hparams.image_size / self.hparams.heatmap_resolution
        node_coords_image_for_loss = node_coords * scale_to_image
        
        # Run ASNS and graph transformer once (after optional permutation)
        # OPTIMIZATION: Skip expensive ASNS and transformer operations if edge/coverage losses are disabled
        g_i_for_transformer = g_i
        l_i_for_transformer = l_i
        if skip_edge_ops:
            # Create dummy outputs to avoid errors downstream
            B, N = node_coords.shape[0], node_coords.shape[1]
            candidate_mask = torch.ones(B, N, N, dtype=torch.bool, device=node_coords.device)
            attention_weights = None
            attention_scores = None
            pred_edge_logits = torch.zeros(B, N, N, device=node_coords.device)
        else:
            # CRITICAL FIX: Detach g_i when used in ASNS/transformer if corresponding loss weights are 0
            # This prevents NaN gradient propagation during validation
            g_i_for_asns = g_i.detach() if self.loss_fn.coverage_weight == 0 else g_i
            # HYPERPARAMETER: Conditionally detach l_i based on detach_l_i flag
            l_i_for_asns = l_i.detach() if self.detach_l_i else l_i
            candidate_mask, attention_weights, attention_scores = self.model.asns(
                l_i_for_asns,  # Local descriptors (detached if detach_l_i=True)
                g_i_for_asns,  # Global descriptors (detached if coverage_loss_weight=0)
                node_coords,
                valid_mask
            )
            # CRITICAL FIX: Detach candidate_mask when coverage loss is disabled
            # Prevents NaN gradients from ASNS from propagating through candidate_mask
            if self.loss_fn.coverage_weight == 0:
                candidate_mask = candidate_mask.detach()
            g_i_for_transformer = g_i.detach() if self.loss_fn.edge_weight == 0 else g_i
            # HYPERPARAMETER: Use the same l_i (detached or not) as used in ASNS for consistency
            l_i_for_transformer = l_i_for_asns
            
            # No coordinate noise during validation (self.training is False)
            pred_edge_logits = self.model.relation_transformer(
                l_i_for_transformer,
                g_i_for_transformer,
                node_coords,  # Use original coordinates (no noise in validation)
                z_star,
                candidate_mask,
                G_prime=G_prime,
                local_features=local_features,
                images=images,
                valid_mask=valid_mask,
                coords_in_image_space=False,  # Coordinates are in heatmap space
            )
        
        # ============================================================
        # Loss Computation in Order: Node ->Edge ->Coverage ->Budget
        # ============================================================
        
        # Step 1: Compute Node Loss
        target_node_mask = target_node_mask.to(
            device=pred_node_mask_logits.device, dtype=pred_node_mask_logits.dtype
        )
        node_loss = self.loss_fn.node_loss_fn(pred_node_mask_logits, target_node_mask)
        
        # Step 2: Compute Edge Loss (with hybrid supervision)
        # CRITICAL FIX: Skip edge loss computation if weight=0 to prevent NaN gradient propagation
        edge_loss = torch.tensor(0.0, device=pred_edge_logits.device, requires_grad=False)
        adjacency_loss_component = None
        pair_loss_component = None
        
        if self.loss_fn.edge_weight > 0:
            # Check if we have all data needed for hybrid edge loss (adjacency + pair-based)
            has_pair_data = (
                gt_edge_pairs is not None and
                gt_node_coords is not None and
                node_coords is not None and
                valid_mask is not None and
                all(pairs is not None for pairs in gt_edge_pairs)
            )
            
            if use_matched_nodes:
                N_gt = node_coords.shape[1]
                identity_permutation = torch.full((B, N_gt), -1, dtype=torch.long, device=node_coords.device)
                for b in range(B):
                    all_gt_positions = torch.where(valid_mask[b])[0]
                    if len(all_gt_positions) > 0:
                        identity_permutation[b, all_gt_positions] = all_gt_positions
                
                if has_pair_data:
                    edge_loss_dict = self.loss_fn.hybrid_edge_loss_fn(
                        pred_edge_logits,
                        target_adj,
                        gt_edge_pairs,
                        identity_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                        gt_node_coords=gt_node_coords_device,
                    )
                    edge_loss = edge_loss_dict['loss']
                    adjacency_loss_component = edge_loss_dict['adjacency_loss']
                    pair_loss_component = edge_loss_dict['pair_loss']
                else:
                    edge_loss = self.loss_fn.hybrid_edge_loss_fn.adjacency_loss(
                        pred_edge_logits,
                        target_adj,
                        identity_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                    )
            else:
                if has_pair_data:
                    edge_loss_dict = self.loss_fn.hybrid_edge_loss_fn(
                        pred_edge_logits,
                        target_adj,
                        gt_edge_pairs,
                        node_permutation,
                        valid_mask,
                        pred_node_coords_image=node_coords_image_for_loss,
                        gt_node_coords=gt_node_coords_device,
                    )
                    edge_loss = edge_loss_dict['loss']
                    adjacency_loss_component = edge_loss_dict['adjacency_loss']
                    pair_loss_component = edge_loss_dict['pair_loss']
                else:
                    if node_permutation is not None:
                        adj_for_edge = target_adj_permuted if target_adj_permuted is not None else target_adj
                        edge_loss = self.loss_fn.hybrid_edge_loss_fn.adjacency_loss(
                            pred_edge_logits,
                            adj_for_edge,
                            node_permutation,
                            valid_mask,
                            pred_node_coords_image=node_coords_image_for_loss,
                        )
                    else:
                        adj_for_edge = target_adj_permuted if target_adj_permuted is not None else target_adj
                        edge_loss = self.loss_fn.edge_loss_fn(
                            pred_edge_logits,
                            adj_for_edge,
                            candidate_mask,
                        )
        
        # Step 3: Compute ASNS Coverage Loss
        # CRITICAL FIX: Skip coverage loss computation if weight=0 to prevent NaN gradient propagation
        # Even with weight=0, computing the loss creates a computation graph through attention_weights
        # If the loss computation produces NaN, gradients propagate back to global_descriptor_head
        coverage_loss = torch.tensor(0.0, device=pred_edge_logits.device, requires_grad=False)
        if self.loss_fn.coverage_weight > 0:
            if candidate_mask is not None and (attention_weights is not None or attention_scores is not None):
                if use_matched_nodes:
                    adj_for_coverage = target_adj
                else:
                    adj_for_coverage = target_adj_permuted if target_adj_permuted is not None else target_adj
                
                coverage_loss = self.model.asns.compute_coverage_loss(
                    candidate_mask=candidate_mask,
                    target_adj=adj_for_coverage,
                    attention_weights=attention_weights,  # Preferred: normalized distribution from entmax/sparsemax
                    attention_scores=attention_scores,  # Fallback: pre-sparsemax scores
                    valid_mask=valid_mask,
                )
        
        # Step 4: Compute optional mask loss
        mask_loss = torch.tensor(0.0, device=pred_edge_logits.device)
        if self.loss_fn.use_mask_loss and target_edge_mask is not None:
            mask_loss = self.loss_fn.mask_loss_fn(pred_node_mask_logits, target_edge_mask)
        
        # Step 5: Compute node count error (for validation only)
        # CRITICAL: Count matched nodes (not all predicted nodes) for meaningful error metric
        # This measures how well the model detects and matches nodes to GT
        node_count_error = torch.tensor(0.0, device=pred_edge_logits.device)
        if gt_node_coords is not None:
            # Count matched nodes per sample
            # node_permutation[b, i] >= 0 means predicted node i is matched to GT node j
            if node_permutation is not None:
                num_matched_nodes = (node_permutation >= 0).sum(dim=1).float()  # [B] - number of matched nodes per sample
            else:
                # Fallback: if no matching was done, count valid predicted nodes
                num_matched_nodes = valid_mask_original.sum(dim=1).float()  # [B]
            
            # Count GT nodes
            if isinstance(gt_node_coords, list):
                num_gt_nodes = torch.tensor([len(coords) if coords is not None else 0 for coords in gt_node_coords], 
                                           device=pred_edge_logits.device, dtype=torch.float32)
            else:
                # If it's a tensor, count non-zero coordinates
                num_gt_nodes = (gt_node_coords.sum(dim=-1) > 0).sum(dim=1).float()  # [B]
            
            # Compute absolute error per sample, then average
            # Error = |num_matched_nodes - num_gt_nodes|
            node_count_error = (num_matched_nodes - num_gt_nodes).abs().mean()
            
            # DIAGNOSTIC: Log node count statistics (only for first validation batch)
            if batch_idx == 0:
                self._log_diag(f"[Node Count] Avg Matched: {num_matched_nodes.mean().item():.2f}, "
                          f"Avg GT: {num_gt_nodes.mean().item():.2f}, "
                          f"Error: {node_count_error.item():.2f}")
        
        # Step 6: Combine all losses with weights
        total_loss = (
            self.loss_fn.node_weight * node_loss +
            self.loss_fn.edge_weight * edge_loss +
            self.loss_fn.coverage_weight * coverage_loss +
            self.loss_fn.mask_weight * mask_loss
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'edge_adjacency_loss': adjacency_loss_component,
            'edge_pair_loss': pair_loss_component,
            'coverage_loss': coverage_loss,
            'mask_loss': mask_loss,
            'node_count_error': node_count_error,
        }
        
        # Log losses
        self.log('val/loss', loss_dict['total_loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/node_count_error', loss_dict['node_count_error'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/node_loss', loss_dict['node_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/edge_loss', loss_dict['edge_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        # Log edge loss components if available
        if 'edge_adjacency_loss' in loss_dict and loss_dict['edge_adjacency_loss'] is not None:
            self.log('val/edge_adjacency_loss', loss_dict['edge_adjacency_loss'], on_step=False, on_epoch=True, sync_dist=True)
        if 'edge_pair_loss' in loss_dict and loss_dict['edge_pair_loss'] is not None:
            self.log('val/edge_pair_loss', loss_dict['edge_pair_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        self.log('val/coverage_loss', loss_dict['coverage_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        if loss_dict['mask_loss'] > 0:
            self.log('val/mask_loss', loss_dict['mask_loss'], on_step=False, on_epoch=True, sync_dist=True)
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with different learning rates for different components."""
        # Base learning rate
        base_lr = self.learning_rate
        
        # Parameter groups with different learning rates
        param_groups = []
        self._base_param_lrs = {}
        
        # Group 1: Node Detector (mask_head + local_descriptor_head) - 2x base LR in phase1
        node_detector_params = []
        if hasattr(self.model, 'node_detector'):
            # Heatmap head parameters
            if hasattr(self.model.node_detector, 'mask_head'):
                node_detector_params.extend(self.model.node_detector.mask_head.parameters())
            # Local descriptor head parameters
            if hasattr(self.model.node_detector, 'local_descriptor_head'):
                node_detector_params.extend(self.model.node_detector.local_descriptor_head.parameters())
        
        if node_detector_params:
            num_node_params = sum(p.numel() for p in node_detector_params)
            param_groups.append({
                'params': node_detector_params,
                'lr': base_lr * 2.0,
                'weight_decay': self.weight_decay,
                'name': 'node_detector'
            })
            self._base_param_lrs['node_detector'] = base_lr * 2.0
            logger.info(f"Node Detector (mask_head + local_descriptor_head): {num_node_params:,} params, LR = {base_lr * 2.0:.2e}")
        
        # Group 2: LoRA parameters - 0.1x base LR
        lora_params = []
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'lora_w_As'):
            # LoRA A matrices (down projection)
            for w_A in self.model.encoder.lora_w_As:
                if w_A.weight.requires_grad:
                    lora_params.append(w_A.weight)
            # LoRA B matrices (up projection)
            for w_B in self.model.encoder.lora_w_Bs:
                if w_B.weight.requires_grad:
                    lora_params.append(w_B.weight)
        
        if lora_params:
            num_lora_params = sum(p.numel() for p in lora_params)
            param_groups.append({
                'params': lora_params,
                'lr': base_lr * 0.1,  # 0.1x base LR
                'weight_decay': self.weight_decay,
                'name': 'lora'
            })
            logger.info(f"LoRA parameters: {num_lora_params:,} params, LR = {base_lr * 0.1:.2e}")
            self._base_param_lrs['lora'] = base_lr * 0.1
        
        # Group 3: ASNS parameters - 1.0x base LR
        asns_params = []
        if hasattr(self.model, 'asns'):
            asns_params = list(self.model.asns.parameters())
        
        if asns_params:
            num_asns_params = sum(p.numel() for p in asns_params)
            param_groups.append({
                'params': asns_params,
                'lr': base_lr * 1.0,
                'weight_decay': self.weight_decay,
                'name': 'asns'
            })
            logger.info(f"ASNS parameters: {num_asns_params:,} params, LR = {base_lr * 1.0:.2e}")
            self._base_param_lrs['asns'] = base_lr * 1.0
        
        # Group 4: All other trainable parameters - base LR
        # Get all parameters as a set for efficient lookup
        all_param_set = set(self.parameters())
        
        # Remove node detector params
        if node_detector_params:
            node_detector_param_set = set(node_detector_params)
            all_param_set -= node_detector_param_set
        
        # Remove LoRA params
        if lora_params:
            lora_param_set = set(lora_params)
            all_param_set -= lora_param_set
        
        if asns_params:
            asns_param_set = set(asns_params)
            all_param_set -= asns_param_set
        
        # Convert back to list
        other_params = list(all_param_set)
        
        if other_params:
            num_other_params = sum(p.numel() for p in other_params)
            param_groups.append({
                'params': other_params,
                'lr': base_lr,  # Base LR
                'weight_decay': self.weight_decay,
                'name': 'other'
            })
            logger.info(f"Other parameters: {num_other_params:,} params, LR = {base_lr:.2e}")
            self._base_param_lrs['other'] = base_lr
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,  # Default LR (will be overridden by param groups)
            weight_decay=self.weight_decay,
        )
        
        # Composite LR schedule: 5% warmup ->40% constant ->55% cosine decay
        total_epochs = getattr(self.trainer, 'max_epochs', None)
        if total_epochs is None or total_epochs <= 0:
            total_epochs = getattr(self.hparams, 'num_epochs', 1)
        total_epochs = max(1, total_epochs)
        warmup_epochs = max(1, int(total_epochs * 0.05))
        constant_epochs = max(1, int(total_epochs * 0.40))
        remaining = total_epochs - warmup_epochs - constant_epochs
        cosine_epochs = max(1, remaining)
        cosine_denominator = max(1, cosine_epochs - 1)

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return (current_epoch + 1) / warmup_epochs
            constant_start = warmup_epochs
            constant_end = warmup_epochs + constant_epochs
            if current_epoch < constant_end:
                return 1.0
            cos_epoch = current_epoch - constant_end
            cos_progress = min(1.0, cos_epoch / cosine_denominator)
            return 0.5 * (1.0 + math.cos(math.pi * cos_progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.current_epoch < self.phase1_epochs:
            self._enter_phase1()
        else:
            self._enter_phase2()

    def _set_module_requires_grad(self, module, requires_grad: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _set_param_group_lr(self, group_name: str, lr: float):
        optimizers = self.optimizers()
        if optimizers is None:
            return
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                if pg.get('name') == group_name:
                    pg['lr'] = lr

    def _enter_phase1(self):
        if self._current_phase == 'phase1':
            return
        self._current_phase = 'phase1'
        self.loss_fn.edge_weight = 0.0
        self.loss_fn.coverage_weight = 0.0
        self._set_module_requires_grad(getattr(self.model, 'asns', None), False)
        self._set_module_requires_grad(getattr(self.model, 'relation_transformer', None), False)
        self._set_module_requires_grad(getattr(self.model, 'global_topology', None), False)
        self._set_module_requires_grad(getattr(self.model, 'node_detector', None), True)
        self._set_param_group_lr('node_detector', self._base_param_lrs.get('node_detector', 0.0))
        self._set_param_group_lr('asns', 0.0)
        logger.info(f"[Phase 1] Epoch {self.current_epoch}: node-only training (edge/coverage disabled).")

    def _enter_phase2(self):
        if self._current_phase == 'phase2':
            return
        self._current_phase = 'phase2'
        self.loss_fn.edge_weight = self._edge_weight_full
        self.loss_fn.coverage_weight = self._coverage_weight_full
        self._set_module_requires_grad(getattr(self.model, 'asns', None), True)
        self._set_module_requires_grad(getattr(self.model, 'relation_transformer', None), True)
        self._set_module_requires_grad(getattr(self.model, 'global_topology', None), True)
        node_base_lr = self._base_param_lrs.get('node_detector', 0.0)
        finetune_lr = node_base_lr * self.node_finetune_lr_scale
        self._set_param_group_lr('node_detector', finetune_lr)
        self._set_param_group_lr('asns', self._base_param_lrs.get('asns', 0.0))
        logger.info(
            f"[Phase 2] Epoch {self.current_epoch}: edge & coverage enabled. "
            f"Node LR scaled to {finetune_lr:.2e} (scale={self.node_finetune_lr_scale})."
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SAM Graph Split model')
    
    # Dataset parameters
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    
    # Model parameters
    parser.add_argument('--sam-version', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h', 'sam2_base_plus', 'sam2_large', 'sam2_tiny', 'sam2_small', 'sam3'],
                        help='SAM model version')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                        help='Path to SAM checkpoint (.pth/.pt). If omitted, defaults to sam_checkpoint/sam_vit_*.pth based on --sam-version.')
    parser.add_argument('--sam-config', type=str, default=None,
                        help='Path to SAM config (optional)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--heatmap-resolution', type=int, default=32,
                        help='Heatmap resolution')
    parser.add_argument('--heatmap-sigma', type=float, default=1.5,
                        help='Gaussian sigma for heatmap generation')
    parser.add_argument('--max-nodes', type=int, default=50,
                        help='Maximum number of nodes to detect')
    parser.add_argument('--k-neighbors', type=int, default=20,
                        help='Number of neighbors for ASNS (Attention-Sparse Neighbor Sampler)')
    parser.add_argument('--neighbor-radius', type=float, default=256.0,
                        help='Neighbor radius (legacy compatibility parameter, not used by ASNS)')
    parser.add_argument('--neighbor-sampler', type=str, choices=['asns', 'knn'], default='asns',
                        help="Neighbor sampler to use: 'asns' (learnable attention) or 'knn' (deterministic).")
    parser.add_argument('--asns-entmax-alpha', type=float, default=1.5,
                        help='Entmax alpha parameter for ASNS activation (1.0 = softmax, 1.5 = entmax/sparsemax, 2.0 = hardmax). '
                             'Set to 1.0 to use softmax. Set to 0.0 to use sparsemax (use_entmax=False). '
                             'Default: 1.5 (entmax with alpha=1.5)')
    parser.add_argument('--relation-transformer-layers', type=int, default=3,
                        help='Number of relation transformer layers (default: 3, matching successful toy model)')
    parser.add_argument('--relation-edge-dim', type=int, default=256,
                        help='Edge feature dimension')
    parser.add_argument('--relation-hidden-dim', type=int, default=256,
                        help='Hidden dimension for relation transformer (default: 128 for EdgeAwareGraphTransformer, matching toy model)')
    parser.add_argument('--relation-num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8, matching successful toy model)')
    parser.add_argument('--relation-dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--edge-model', type=str, default='edge_aware_transformer',
                        choices=['mlp', 'graph_transformer', 'edge_aware_transformer'],
                        help='Edge prediction model: mlp (PairwiseEdgeMLP), graph_transformer (node-based), or edge_aware_transformer (edge-based with line graph attention). Default: edge_aware_transformer')
    parser.add_argument('--rgb-feature-dim', type=int, default=32,
                        help='RGB feature dimension for EdgeAwareGraphTransformer (default: 32)')
    parser.add_argument('--rgb-sequence-model', type=str, default='transformer',
                        choices=['mean', 'max', '1d_cnn', 'transformer'],
                        help='Model used to aggregate RGB samples along an edge path (default: transformer)')
    parser.add_argument('--rgb-seq-layers', type=int, default=2,
                        help='Number of layers for RGB sequence transformer (only used when --rgb-sequence-model=transformer)')
    parser.add_argument('--rgb-seq-heads', type=int, default=4,
                        help='Number of attention heads for RGB sequence transformer (only used when --rgb-sequence-model=transformer)')
    parser.add_argument('--rgb-neighborhood-aggregation', type=str, default='center',
                        choices=['center', 'mean', 'median', 'min_r_min_g_max_b'],
                        help='How to aggregate RGB from neighborhood: center (use only center point), mean (average all), min_r_min_g_max_b (min R, min G, max B). Default: center')
    parser.add_argument('--rgb-neighborhood-radius', type=float, default=4.0,
                        help='Radius in pixels for RGB neighborhood sampling. Default: 4.0 (increased from 2.0 to handle coordinate offsets)')
    parser.add_argument('--coordinate-noise-std', type=float, default=2.0,
                        help='Standard deviation of coordinate noise in pixels during training (for robustness to prediction errors). Default: 2.0 (matches window=9 accuracy: mean 1.52px, std 0.73px). Set to 0 to disable.')
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank')
    
    # Loss parameters
    parser.add_argument('--node-loss-weight', type=float, default=2.0,
                        help='Weight for node loss')
    parser.add_argument('--edge-loss-weight', type=float, default=5.0,
                        help='Weight for edge loss')
    parser.add_argument('--coverage-loss-weight', type=float, default=0.1,
                        help='Weight for coverage loss')
    parser.add_argument('--coverage-label-smoothing', type=float, default=0.1,
                        help='Label smoothing for coverage loss (0.0 = no smoothing, 0.1 = 10%% smoothing, default: 0.1). '
                             'Smooths target distribution to prevent overconfidence.')
    parser.add_argument('--mask-loss-weight', type=float, default=0.0,
                        help='Weight for mask loss')
    parser.add_argument('--use-mask-loss', action='store_true',
                        help='Use mask loss')
    parser.add_argument('--edge-use-focal', action='store_true',
                        help='Use focal loss for edges')
    parser.add_argument('--edge-focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha (default: 0.25)')
    parser.add_argument('--edge-focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--edge-pos-weight', type=float, default=1.0,
                        help='Positive class weight for edges (default: 1.0)')
    parser.add_argument('--use-hard-negative-mining', action='store_true',
                        help='Use hard negative mining')
    parser.add_argument('--hard-negative-threshold', type=float, default=0.3,
                        help='Hard negative threshold')
    parser.add_argument('--max-hard-negatives-ratio', type=float, default=2.0,
                        help='Max hard negatives ratio')
    parser.add_argument('--no-edge-length-weighting', action='store_true',
                        help='Disable edge length weighting')
    parser.add_argument('--edge-length-weight-power', type=float, default=1.5,
                        help='Edge length weight power')
    parser.add_argument('--adjacency-weight', type=float, default=2.0,
                        help='Weight for adjacency-based edge loss component (default: 2.0)')
    parser.add_argument('--pair-weight', type=float, default=1.0,
                        help='Weight for pair-based edge loss component (default: 1.0)')
    
    # Node detection hyperparameters
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Sigmoid probability threshold for selecting mask peaks (default: 0.5).')
    parser.add_argument('--mask-pool-radius', type=int, default=16,
                        help='Radius for local max pooling when extracting nodes from the mask (default: 16 ->33×33 kernel).')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'ddp', 'ddp_spawn', 'dp'],
                        help='Training strategy')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16', 'bf16', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--gradient-clip-val', type=float, default=5.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--log-every-n-steps', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val-check-interval', type=float, default=1.0,
                        help='Validation check interval')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--enable-diagnostics', action='store_true',
                        help='Enable verbose diagnostic logging (slower training)')
    parser.add_argument('--no-detach-l-i', dest='detach_l_i', action='store_false', default=True,
                        help='Do not detach l_i (allows gradients from edge/coverage losses to flow back to local descriptor head). '
                             'Default: detach_l_i=True (gradients do NOT flow back)')
    parser.add_argument('--phase1-epochs', type=int, default=20,
                        help='Number of initial epochs to train node detector only before enabling edge/coverage losses.')
    parser.add_argument('--node-finetune-lr-scale', type=float, default=0.2,
                        help='LR scale applied to node detector param group after phase1 (default: 0.2).')
    parser.add_argument('--teacher-forcing-epochs', type=int, default=30,
                        help='Number of edge-training epochs over which teacher forcing probability linearly decays to zero. Set 0 to disable teacher forcing.')
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)

    # Resolve SAM checkpoint path
    sam_checkpoint_path = Path(args.sam_checkpoint) if args.sam_checkpoint else None
    sam_config_path = Path(args.sam_config) if args.sam_config else None

    checkpoint_root = Path(__file__).parent.parent
    sam_v1_dir = checkpoint_root / 'sam_checkpoint'
    sam2_dir = checkpoint_root / 'sam2_checkpoints'
    sam3_dir = checkpoint_root / 'sam3_checkpoints'

    version_to_filename = {
        'vit_b': ('sam_vit_b.pth', None),
        'vit_l': ('sam_vit_l.pth', None),
        'vit_h': ('sam_vit_h.pth', None),
        'sam2_base_plus': ('sam2.1_hiera_base_plus.pt', 'sam2.1_hiera_b+.yaml'),
        'sam2_large': ('sam2.1_hiera_large.pt', 'sam2.1_hiera_l.yaml'),
        'sam2_tiny': ('sam2.1_hiera_tiny.pt', 'sam2.1_hiera_t.yaml'),
        'sam2_small': ('sam2.1_hiera_small.pt', 'sam2.1_hiera_s.yaml'),
        'sam3': ('sam3.pt', None),
    }

    candidate_name, candidate_config_name = version_to_filename.get(
        args.sam_version,
        (f"sam_{args.sam_version}.pth", None)
    )

    if sam_checkpoint_path is None or not sam_checkpoint_path.is_file():
        if args.sam_version.startswith('sam3'):
            candidate_dir = sam3_dir
        elif args.sam_version.startswith('sam2'):
            candidate_dir = sam2_dir
        else:
            candidate_dir = sam_v1_dir
        candidate_path = candidate_dir / candidate_name
        if candidate_path.is_file():
            sam_checkpoint_path = candidate_path
        elif args.sam_version.startswith('sam3') and candidate_dir.is_dir():
            sam3_candidates = []
            for pattern in ('*.pt', '*.pth', '*.ckpt'):
                sam3_candidates.extend(sorted(candidate_dir.glob(pattern)))
            if sam3_candidates:
                sam_checkpoint_path = sam3_candidates[0]

    if args.sam_version.startswith('sam2'):
        if sam_config_path is None or not sam_config_path.is_file():
            if candidate_config_name:
                candidate_config_path = sam2_dir / candidate_config_name
                if candidate_config_path.is_file():
                    sam_config_path = candidate_config_path
        if sam_config_path is None or not sam_config_path.is_file():
            raise FileNotFoundError(
                "SAM2 configuration YAML not found. "
                f"Provide --sam-config or place {candidate_config_name} under sam2_checkpoints/."
            )

    if sam_checkpoint_path is None or not sam_checkpoint_path.is_file():
        if args.sam_version.startswith('sam3'):
            logger.warning(
                "SAM3 checkpoint not found under %s and --sam-checkpoint was not valid. "
                "The SAM3 builder will try its default/Hugging Face loading path.",
                sam3_dir,
            )
        else:
            raise FileNotFoundError(
                f"SAM checkpoint not found. "
                f"Checked provided path ({args.sam_checkpoint}) and inferred path ({candidate_path if 'candidate_path' in locals() else 'N/A'}). "
                "Please provide a valid --sam-checkpoint path."
            )
    if sam_checkpoint_path is not None and sam_checkpoint_path.is_file():
        args.sam_checkpoint = str(sam_checkpoint_path.resolve())
    else:
        args.sam_checkpoint = None
    if sam_config_path is not None:
        args.sam_config = str(sam_config_path.resolve())
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(output_dir / 'training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("SAM Graph Split - Full Model Training")
    logger.info("=" * 60)
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Heatmap resolution: {args.heatmap_resolution}×{args.heatmap_resolution}")
    logger.info(f"Heatmap sigma: {args.heatmap_sigma}")
    logger.info(f"Max nodes: {args.max_nodes} (should be set based on dataset max nodes)")
    logger.info(f"Neighbor sampler: {args.neighbor_sampler.upper()} | k_neighbors={args.k_neighbors}, radius={args.neighbor_radius}")
    if args.neighbor_sampler == 'asns':
        if args.asns_entmax_alpha == 1.0:
            logger.info(f"ASNS activation: softmax (entmax_alpha=1.0)")
        elif args.asns_entmax_alpha == 0.0:
            logger.info(f"ASNS activation: sparsemax (entmax_alpha=0.0 triggers sparsemax)")
        else:
            logger.info(f"ASNS activation: entmax (alpha={args.asns_entmax_alpha})")
    else:
        logger.info("KNN sampler selected: deterministic neighbor selection (no learnable params, coverage loss is metric-only).")
    logger.info(f"Edge model: {args.edge_model}")
    logger.info(f"Relation transformer layers: {args.relation_transformer_layers}")
    logger.info(f"Relation edge dim: {args.relation_edge_dim}")
    logger.info(f"Relation hidden dim: {args.relation_hidden_dim}")
    logger.info(f"Relation heads: {args.relation_num_heads}")
    logger.info(f"Relation dropout: {args.relation_dropout}")
    logger.info(f"RGB sequence model: {args.rgb_sequence_model} | feature_dim={args.rgb_feature_dim}, seq_layers={args.rgb_seq_layers}, seq_heads={args.rgb_seq_heads}")
    logger.info(f"Coverage loss weight: {args.coverage_loss_weight}")
    logger.info(f"Adjacency loss weight: {args.adjacency_weight}")
    logger.info(f"Pair loss weight: {args.pair_weight}")
    logger.info(f"Two-step training: phase1_epochs={args.phase1_epochs}, node_finetune_lr_scale={args.node_finetune_lr_scale}")
    
    if args.edge_use_focal:
        logger.info("Edge Focal Loss: Enabled")
        logger.info(f"  Focal alpha: {args.edge_focal_alpha} (increased from 0.25 for class imbalance)")
        logger.info(f"  Focal gamma: {args.edge_focal_gamma}")
        logger.info(f"  Positive weight: {args.edge_pos_weight} (for 81% negatives, 19% positives)")
    
    if args.use_hard_negative_mining:
        logger.info("Hard Negative Mining: Enabled")
        logger.info(f"  Hard negative threshold: {args.hard_negative_threshold} (probability > threshold)")
        logger.info(f"  Max hard negatives ratio: {args.max_hard_negatives_ratio} (max = ratio * num_positives)")
    
    if args.no_edge_length_weighting:
        logger.info("Edge Length Weighting: Disabled")
    else:
        logger.info(f"Edge Length Weighting: Enabled (power={args.edge_length_weight_power})")
    
    logger.info("=" * 60)
    
    # Create datasets
    train_dataset = Image2MatrixDataset(
        dataset_path=args.dataset_root,
        split='train',
        augment=False,
        image_size=args.image_size,
        heatmap_resolution=args.heatmap_resolution,
        heatmap_sigma=args.heatmap_sigma,
    )
    
    val_dataset = Image2MatrixDataset(
        dataset_path=args.dataset_root,
        split='val',
        augment=False,
        image_size=args.image_size,
        heatmap_resolution=args.heatmap_resolution,
        heatmap_sigma=args.heatmap_sigma,
    )
    
    # Create data loaders with custom collate function for variable-sized tensors
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    
    # Create model
    model = SAMGraphSplitLightning(
        sam_version=args.sam_version,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        image_size=args.image_size,
        heatmap_resolution=args.heatmap_resolution,
        heatmap_sigma=args.heatmap_sigma,
        max_nodes=args.max_nodes,
        k_neighbors=args.k_neighbors,
        neighbor_radius=args.neighbor_radius,
        neighbor_sampler=args.neighbor_sampler,
        asns_use_entmax=True if args.asns_entmax_alpha != 0.0 else False,  # Use entmax unless alpha=0.0 (sparsemax)
        asns_entmax_alpha=args.asns_entmax_alpha if args.asns_entmax_alpha != 0.0 else 2.0,  # If 0.0, use sparsemax (alpha=2.0)
        relation_transformer_layers=args.relation_transformer_layers,
        relation_edge_dim=args.relation_edge_dim,
        relation_hidden_dim=args.relation_hidden_dim,
        relation_num_heads=args.relation_num_heads,
        relation_dropout=args.relation_dropout,
        rgb_feature_dim=args.rgb_feature_dim,
        rgb_sequence_model=args.rgb_sequence_model,
        rgb_seq_layers=args.rgb_seq_layers,
        rgb_seq_heads=args.rgb_seq_heads,
        rgb_neighborhood_aggregation=args.rgb_neighborhood_aggregation,
        rgb_neighborhood_radius=args.rgb_neighborhood_radius,
        edge_model=args.edge_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        coordinate_noise_std=args.coordinate_noise_std,
        node_loss_weight=args.node_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        coverage_loss_weight=args.coverage_loss_weight,
        coverage_label_smoothing=args.coverage_label_smoothing,
        mask_loss_weight=args.mask_loss_weight,
        use_mask_loss=args.use_mask_loss,
        edge_use_focal=args.edge_use_focal,
        edge_focal_alpha=args.edge_focal_alpha,
        edge_focal_gamma=args.edge_focal_gamma,
        edge_pos_weight=args.edge_pos_weight,
        use_hard_negative_mining=args.use_hard_negative_mining,
        hard_negative_threshold=args.hard_negative_threshold,
        max_hard_negatives_ratio=args.max_hard_negatives_ratio,
        use_edge_length_weighting=not args.no_edge_length_weighting,
        edge_length_weight_power=args.edge_length_weight_power,
        adjacency_weight=args.adjacency_weight,
        pair_weight=args.pair_weight,
        # Node detection hyperparameters
        mask_threshold=args.mask_threshold,
        mask_pool_radius=args.mask_pool_radius,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        enable_diagnostics=args.enable_diagnostics,
        detach_l_i=args.detach_l_i,
        phase1_epochs=args.phase1_epochs,
        node_finetune_lr_scale=args.node_finetune_lr_scale,
        teacher_forcing_epochs=args.teacher_forcing_epochs,
    )
    
    # Create callbacks
    callbacks = [
        # Model checkpoint - save best and last
        PhaseAwareModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-epoch={epoch:03d}',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            phase_start_epoch=args.phase1_epochs,
        ),
        # Early stopping
        PhaseAwareEarlyStopping(
            monitor='val/loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True,
            phase_start_epoch=args.phase1_epochs,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
        # CSV logger
        CSVLogger(
            output_dir=log_dir,
        ),
        # TQDM progress bar with file logging
        FileTQDMProgressBar(log_file=output_dir / 'training.log'),
    ]
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='VisAdj',
        default_hp_metric=False,
    )
    
    # Configure strategy for multi-GPU training
    # Use DDPStrategy with find_unused_parameters=True to handle frozen SAM encoder
    if args.gpus > 1:
        if args.strategy == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'auto'
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=False,
        benchmark=True,
    )
    
    # Log GPU info
    if args.gpus > 0:
        logger.info(f"Training on {args.gpus} GPU(s)")
        logger.info(f"Using strategy: {trainer.strategy}")
    
    # Log l_i detachment setting
    logger.info(f"detach_l_i={args.detach_l_i} (gradients from edge/coverage losses {'will NOT' if args.detach_l_i else 'WILL'} flow back to local descriptor head)")
    
    # Train with output redirection to capture all output including progress bars
    logger.info("Starting training...")
    with TeeOutput(output_dir / 'training.log'):
        trainer.fit(model, train_loader, val_loader)
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

