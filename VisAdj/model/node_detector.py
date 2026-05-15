"""
Node Detector

Predicts a dense node mask directly in image space and extracts highly accurate
node coordinates by reading from the mask instead of a coarse heatmap.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_norm_2d import LayerNorm2d
from sam_graph_split.utils.nms import _nms_coords


class CoordConv(nn.Module):
    """Adds coordinate channels before a conv to inject positional information."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_cat = torch.cat([x, xs, ys], dim=1)
        return self.conv(x_cat)


class NodeDetector(nn.Module):
    """
    Detects nodes by predicting a full-resolution binary mask and extracting peaks.
    """

    def __init__(
        self,
        local_feature_dim: int = 256,
        global_feature_dim: int = 256,
        node_feature_dim: int = 128,
        mask_threshold: float = 0.5,
        mask_pool_radius: int = 16,
        max_nodes: int = 50,
        heatmap_resolution: int = 32,
        global_resolution: int = 8,
        image_size: int = 512,
        subpixel_refine: bool = True,
        subpixel_refine_window: int = 9,  # Optimal window size from analysis (41.6% improvement over window=1)
    ):
        super().__init__()

        self.mask_threshold = mask_threshold
        self.mask_pool_radius = max(1, int(mask_pool_radius))
        self.nms_radius = None  # If None, uses mask_pool_radius. Can be set separately for inference tuning.
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.heatmap_resolution = heatmap_resolution
        self.global_resolution = global_resolution
        self.image_size = image_size
        self.subpixel_refine = subpixel_refine
        self.subpixel_refine_window = max(1, int(subpixel_refine_window))
        self.pixel_to_local_scale = heatmap_resolution / float(image_size)
        self._latest_coords_pixel: Optional[torch.Tensor] = None

        self.mask_head = nn.Sequential(
            CoordConv(local_feature_dim, 64, 3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        for m in self.mask_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.local_descriptor_head = nn.Sequential(
            nn.Conv2d(local_feature_dim, node_feature_dim, 3, padding=1),
            LayerNorm2d(node_feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(node_feature_dim, node_feature_dim, 3, padding=1),
            LayerNorm2d(node_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.global_descriptor_head = nn.Sequential(
            nn.Conv2d(global_feature_dim, node_feature_dim, 3, padding=1),
            LayerNorm2d(node_feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(node_feature_dim, node_feature_dim, 3, padding=1),
            LayerNorm2d(node_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        return_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if local_features.shape[-1] != self.heatmap_resolution:
            local_features = F.interpolate(
                local_features,
                size=(self.heatmap_resolution, self.heatmap_resolution),
                mode="bilinear",
                align_corners=False,
            )
        if global_features.shape[-1] != self.global_resolution:
            global_features = F.interpolate(
                global_features,
                size=(self.global_resolution, self.global_resolution),
                mode="bilinear",
                align_corners=False,
            )

        # CRITICAL FIX: Predict masks at low resolution first, then upsample (matching sam_road)
        # This reduces memory usage by 64× (8×8 = 64) compared to upsampling features first
        # sam_road predicts at 64×64, then upscales output to 512×512
        node_mask_logits_lowres = self.mask_head(local_features)  # Predict at heatmap_resolution (64×64)
        mask_probs_lowres = torch.sigmoid(node_mask_logits_lowres)
        
        # Upsample mask logits to full resolution for loss computation and node extraction
        node_mask_logits = F.interpolate(
            node_mask_logits_lowres,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mask_probs = F.interpolate(
            mask_probs_lowres,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        local_processed = self.local_descriptor_head(local_features)
        global_processed = self.global_descriptor_head(global_features)

        node_coords, l_i, g_i = self._extract_nodes_from_mask(
            mask_probs, local_processed, global_processed
        )

        if return_mask:
            return node_coords, l_i, g_i, node_mask_logits
        return node_coords, l_i, g_i, None

    def _extract_nodes_from_mask(
        self,
        mask_probs: torch.Tensor,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h_img, w_img = mask_probs.shape
        _, _, h_global, w_global = global_features.shape
        device = mask_probs.device
        max_n = self.max_nodes if self.max_nodes > 0 else 1

        with torch.no_grad():
            kernel = 2 * self.mask_pool_radius + 1
            pooled = F.max_pool2d(
                mask_probs,
                kernel_size=kernel,
                stride=1,
                padding=self.mask_pool_radius,
            )
            peaks = (torch.abs(mask_probs - pooled) < 1e-6) & (
                mask_probs > self.mask_threshold
            )

            node_lists = []
            for b_idx in range(b):
                peak_coords = torch.nonzero(peaks[b_idx, 0], as_tuple=False).float()
                scores = mask_probs[b_idx, 0]

                if len(peak_coords) == 0:
                    peak_coords = torch.zeros(0, 2, device=device)

                # CRITICAL FIX: Sort peaks by score and apply NMS to suppress nearby duplicates
                # Max pooling alone doesn't work in flat mask regions where many pixels
                # have the same value - they all pass the equality check
                if len(peak_coords) > 0:
                    # Get scores at peak locations
                    peak_y = peak_coords[:, 0].long()
                    peak_x = peak_coords[:, 1].long()
                    # Clamp to valid range
                    peak_y = torch.clamp(peak_y, 0, scores.shape[0] - 1)
                    peak_x = torch.clamp(peak_x, 0, scores.shape[1] - 1)
                    peak_values = scores[peak_y, peak_x]
                    # Sort by score (descending) - keep highest-scoring peaks
                    sorted_indices = torch.argsort(peak_values, descending=True)
                    peak_coords = peak_coords[sorted_indices]
                    peak_values = peak_values[sorted_indices]
                    
                    # Apply NMS to suppress nearby peaks (in flat mask regions)
                    # Use nms_radius if set, otherwise use mask_pool_radius
                    # This allows separate tuning of peak detection vs peak suppression
                    nms_radius = float(self.nms_radius if self.nms_radius is not None else self.mask_pool_radius)
                    kept_indices = _nms_coords(peak_coords, nms_radius)
                    peak_coords = peak_coords[kept_indices]

                if self.subpixel_refine and len(peak_coords) > 0:
                    peak_coords = self._refine_subpixel_coords(
                        scores, peak_coords, window=self.subpixel_refine_window
                    )

                if len(peak_coords) > max_n:
                    peak_coords = peak_coords[:max_n]

                node_lists.append(peak_coords)

        node_coords = []
        node_coords_pixel_list = []
        coords_local_norm_list = []
        coords_global_norm_list = []

        for coords_pixel in node_lists:
            if len(coords_pixel) == 0:
                coords_pixel = torch.zeros(0, 2, device=device)

            # CRITICAL FIX: Swap from (y, x) to (x, y) format
            # torch.nonzero returns (row, col) = (y, x), but model expects (x, y)
            coords_pixel_xy = coords_pixel[:, [1, 0]]  # Swap to (x, y) format
            coords_pixel_xy = torch.clamp(
                coords_pixel_xy,
                min=0.0,
                max=self.image_size - 1,
            )
            coords_pixel_xy_to_store = coords_pixel_xy.clone()

            coords_local = coords_pixel_xy * self.pixel_to_local_scale
            coords_local = torch.clamp(
                coords_local,
                min=0.0,
                max=self.heatmap_resolution - 1,
            )

            coords_local_norm = coords_local.clone()
            if self.heatmap_resolution > 1:
                coords_local_norm[:, 0] = (
                    coords_local[:, 0] / (self.heatmap_resolution - 1)
                ) * 2.0 - 1.0
                coords_local_norm[:, 1] = (
                    coords_local[:, 1] / (self.heatmap_resolution - 1)
                ) * 2.0 - 1.0
            else:
                coords_local_norm.zero_()

            coords_global = coords_local.clone()
            if self.heatmap_resolution > 1:
                coords_global[:, 0] = coords_local[:, 0] * (w_global - 1) / (
                    self.heatmap_resolution - 1
                )
                coords_global[:, 1] = coords_local[:, 1] * (h_global - 1) / (
                    self.heatmap_resolution - 1
                )
            coords_global_norm = coords_global.clone()
            if w_global > 1:
                coords_global_norm[:, 0] = (coords_global[:, 0] / (w_global - 1)) * 2.0 - 1.0
            else:
                coords_global_norm[:, 0] = 0.0
            if h_global > 1:
                coords_global_norm[:, 1] = (coords_global[:, 1] / (h_global - 1)) * 2.0 - 1.0
            else:
                coords_global_norm[:, 1] = 0.0

            n = coords_local.shape[0]
            if n < max_n:
                pad = max_n - n
                coords_local = torch.cat(
                    [coords_local, torch.zeros(pad, 2, device=device)], dim=0
                )
                coords_local_norm = torch.cat(
                    [coords_local_norm, torch.full((pad, 2), -2.0, device=device)],
                    dim=0,
                )
                coords_global_norm = torch.cat(
                    [coords_global_norm, torch.full((pad, 2), -2.0, device=device)],
                    dim=0,
                )
                coords_pixel_xy_to_store = torch.cat(
                    [coords_pixel_xy_to_store, torch.zeros(pad, 2, device=device)], dim=0
                )
            elif n > max_n:
                coords_local = coords_local[:max_n]
                coords_local_norm = coords_local_norm[:max_n]
                coords_global_norm = coords_global_norm[:max_n]
                coords_pixel_xy_to_store = coords_pixel_xy_to_store[:max_n]

            node_coords.append(coords_local)
            node_coords_pixel_list.append(coords_pixel_xy_to_store)
            coords_local_norm_list.append(coords_local_norm)
            coords_global_norm_list.append(coords_global_norm)

        node_coords = torch.stack(node_coords, dim=0)
        node_coords_pixel = torch.stack(node_coords_pixel_list, dim=0)
        coords_local_norm = torch.stack(coords_local_norm_list, dim=0)
        coords_global_norm = torch.stack(coords_global_norm_list, dim=0)
        self._latest_coords_pixel = node_coords_pixel.detach()

        grid_local = coords_local_norm.unsqueeze(2)
        l_i = F.grid_sample(
            local_features,
            grid_local,
            mode="bilinear",
            align_corners=False,
        ).squeeze(-1).permute(0, 2, 1)

        grid_global = coords_global_norm.unsqueeze(2)
        g_i = F.grid_sample(
            global_features,
            grid_global,
            mode="bilinear",
            align_corners=False,
        ).squeeze(-1).permute(0, 2, 1)

        return node_coords, l_i, g_i

    @property
    def latest_coords_pixel(self) -> Optional[torch.Tensor]:
        return self._latest_coords_pixel

    def _refine_subpixel_coords(
        self,
        mask_probs: torch.Tensor,
        coords: torch.Tensor,
        window: int = 1,
    ) -> torch.Tensor:
        if coords.numel() == 0:
            return coords

        h, w = mask_probs.shape
        refined = []
        radius = max(1, int(window))
        eps = 1e-6

        for idx in range(coords.shape[0]):
            y = coords[idx, 0].item()
            x = coords[idx, 1].item()

            y0 = int(max(0, min(h - 1, round(y))))
            x0 = int(max(0, min(w - 1, round(x))))

            y_min = max(0, y0 - radius)
            y_max = min(h - 1, y0 + radius)
            x_min = max(0, x0 - radius)
            x_max = min(w - 1, x0 + radius)

            patch = mask_probs[y_min : y_max + 1, x_min : x_max + 1]
            if patch.numel() == 0:
                refined.append(coords[idx])
                continue

            weights = patch - patch.min() + eps
            total = weights.sum()
            if total <= eps:
                refined.append(coords[idx])
                continue

            y_indices = torch.arange(y_min, y_max + 1, device=mask_probs.device).unsqueeze(1).expand_as(patch)
            x_indices = torch.arange(x_min, x_max + 1, device=mask_probs.device).unsqueeze(0).expand_as(patch)
            refined_y = (y_indices * weights).sum() / total
            refined_x = (x_indices * weights).sum() / total
            refined.append(torch.stack([refined_y, refined_x]))

        return torch.stack(refined, dim=0)
