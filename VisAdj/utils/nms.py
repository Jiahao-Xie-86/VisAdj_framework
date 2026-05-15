"""
NMS (Non-Maximum Suppression) utilities
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def nms_heatmap(
    heatmap: torch.Tensor,
    threshold: float = 0.5,
    radius: int = 10,
    max_peaks: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply NMS to heatmap to extract peak locations.
    
    Args:
        heatmap: Heatmap [H, W] or [B, 1, H, W]
        threshold: Threshold for peak detection
        radius: NMS radius
        max_peaks: Maximum number of peaks to return
    
    Returns:
        coords: Peak coordinates [N, 2] or [B, N, 2] (x, y)
        values: Peak values [N] or [B, N]
    """
    if heatmap.dim() == 2:
        # Single heatmap
        return _nms_single(heatmap, threshold, radius, max_peaks)
    elif heatmap.dim() == 4:
        # Batch of heatmaps
        B = heatmap.shape[0]
        coords_list = []
        values_list = []
        
        for b in range(B):
            coords, values = _nms_single(heatmap[b, 0], threshold, radius, max_peaks)
            coords_list.append(coords)
            values_list.append(values)
        
        # Pad to same length
        max_n = max(len(c) for c in coords_list)
        max_n = min(max_n, max_peaks)
        
        coords_batch = torch.zeros(B, max_n, 2, device=heatmap.device)
        values_batch = torch.zeros(B, max_n, device=heatmap.device)
        
        for b in range(B):
            n = min(len(coords_list[b]), max_n)
            if n > 0:
                coords_batch[b, :n] = coords_list[b][:n]
                values_batch[b, :n] = values_list[b][:n]
        
        return coords_batch, values_batch
    else:
        raise ValueError(f"Invalid heatmap shape: {heatmap.shape}")


def _nms_single(
    heatmap: torch.Tensor,
    threshold: float,
    radius: int,
    max_peaks: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NMS for single heatmap."""
    H, W = heatmap.shape
    
    # Threshold
    thresholded = (heatmap > threshold).float()
    
    if thresholded.sum() == 0:
        return torch.zeros(0, 2, device=heatmap.device), torch.zeros(0, device=heatmap.device)
    
    # Max pooling for NMS
    kernel_size = 2 * radius + 1
    max_pooled = F.max_pool2d(
        heatmap.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius
    ).squeeze()
    
    # Peaks are where heatmap equals max_pooled
    peaks = (heatmap == max_pooled) & (heatmap > threshold)
    
    # Get peak coordinates
    peak_coords = torch.nonzero(peaks, as_tuple=False).float()  # [N, 2] (y, x)
    
    if len(peak_coords) == 0:
        return torch.zeros(0, 2, device=heatmap.device), torch.zeros(0, device=heatmap.device)
    
    # Sort by value (descending)
    peak_values = heatmap[peak_coords[:, 0].long(), peak_coords[:, 1].long()]
    sorted_indices = torch.argsort(peak_values, descending=True)
    peak_coords = peak_coords[sorted_indices]
    peak_values = peak_values[sorted_indices]
    
    # Apply additional NMS: Remove peaks that are too close
    kept_indices = _nms_coords(peak_coords, radius)
    peak_coords = peak_coords[kept_indices]
    peak_values = peak_values[kept_indices]
    
    # Limit to max_peaks
    if len(peak_coords) > max_peaks:
        peak_coords = peak_coords[:max_peaks]
        peak_values = peak_values[:max_peaks]
    
    # Convert from (y, x) to (x, y)
    coords = peak_coords[:, [1, 0]]
    
    return coords, peak_values


def _nms_coords(coords: torch.Tensor, radius: float) -> torch.Tensor:
    """Apply NMS to coordinates."""
    if len(coords) == 0:
        return torch.tensor([], dtype=torch.long, device=coords.device)
    
    kept = []
    for i in range(len(coords)):
        if len(kept) == 0:
            kept.append(i)
        else:
            kept_coords = coords[kept]
            dist_to_kept = torch.cdist(coords[i:i+1], kept_coords).squeeze()
            if (dist_to_kept > radius).all():
                kept.append(i)
    
    return torch.tensor(kept, dtype=torch.long, device=coords.device)

