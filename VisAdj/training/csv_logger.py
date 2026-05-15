"""
CSV Logger Callback for PyTorch Lightning

Saves training and validation metrics to CSV files for easy plotting.
"""

import csv
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CSVLogger(Callback):
    """Callback to log metrics to CSV files."""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save CSV files
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.train_csv = self.output_dir / "train_losses.csv"
        self.val_csv = self.output_dir / "val_losses.csv"
        
        # Track metrics
        self.train_metrics = []
        self.val_metrics = []
        
        # Track last logged epoch to avoid duplicate validation logs
        self.last_logged_train_epoch = -1
        self.last_logged_val_epoch = -1
        
        # Initialize CSV files with headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        train_headers = [
            'epoch', 'step', 'loss', 'node_loss', 'edge_loss',
            'edge_adjacency_loss', 'edge_pair_loss',
            'coverage_loss', 'mask_loss'
        ]
        val_headers = [
            'epoch', 'loss', 'node_loss', 'edge_loss',
            'edge_adjacency_loss', 'edge_pair_loss',
            'coverage_loss', 'mask_loss', 'node_count_error'
        ]
        
        # Write headers
        with open(self.train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)
        
        with open(self.val_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)
    
    def _get_metric_value(self, metrics: dict, keys: list, default=0.0):
        """Try multiple key formats to get metric value."""
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if hasattr(value, 'item'):
                    return value.item()
                elif isinstance(value, (int, float)):
                    return float(value)
        return default
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each validation run."""
        # CRITICAL FIX: Only log from rank 0 to avoid duplicate entries in DDP
        if trainer.global_rank != 0:
            return
        
        epoch = trainer.current_epoch
        
        # PyTorch Lightning calls this after EACH validation run
        # With val_check_interval=0.5, validation runs:
        #   - At 0.5 epochs (mid-epoch) -> current_epoch = 0
        #   - At 1.0 epochs (end of epoch 0) -> current_epoch = 0 (still!)
        #   - At 1.5 epochs (mid-epoch) -> current_epoch = 1
        #   - At 2.0 epochs (end of epoch 1) -> current_epoch = 1 (still!)
        #
        # We only want to log the FINAL validation of each epoch.
        # Strategy: Store the latest validation result for each epoch,
        # and only write it when we're sure it's the final one.
        # We can detect this by checking if the next validation would be in a new epoch,
        # or by storing and only writing when epoch changes.
        
        # Try both callback_metrics and logged_metrics
        metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
        
        val_metrics = {
            'epoch': epoch,
            'loss': self._get_metric_value(metrics, ['val/loss'], 0.0),
            'node_loss': self._get_metric_value(metrics, ['val/node_loss'], 0.0),
            'edge_loss': self._get_metric_value(metrics, ['val/edge_loss'], 0.0),
            'edge_adjacency_loss': self._get_metric_value(metrics, ['val/edge_adjacency_loss'], 0.0),
            'edge_pair_loss': self._get_metric_value(metrics, ['val/edge_pair_loss'], 0.0),
            'coverage_loss': self._get_metric_value(metrics, ['val/coverage_loss'], 0.0),
            'mask_loss': self._get_metric_value(metrics, ['val/mask_loss'], 0.0),
            'node_count_error': self._get_metric_value(metrics, ['val/node_count_error'], 0.0),
        }
        
        # Store the latest validation result for this epoch
        # We'll write it when the epoch changes (in on_train_epoch_end)
        if not hasattr(self, 'pending_val_metrics'):
            self.pending_val_metrics = {}
        self.pending_val_metrics[epoch] = val_metrics
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each training epoch."""
        # CRITICAL FIX: Only log from rank 0 to avoid duplicate entries in DDP
        # In DDP mode, this callback is called on each GPU process
        # We only want to log once from the main process (rank 0)
        if trainer.global_rank != 0:
            return
        
        epoch = trainer.current_epoch
        
        # Only log once per epoch (avoid duplicates from multiple validation runs)
        if epoch <= self.last_logged_train_epoch:
            return
        
        # Try both callback_metrics and logged_metrics
        metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
        
        # Extract epoch-level metrics (on_epoch=True)
        step = trainer.global_step
        
        train_metrics = {
            'epoch': epoch,
            'step': step,
            'loss': self._get_metric_value(metrics, ['train/loss_epoch', 'train/loss'], 0.0),
            'node_loss': self._get_metric_value(metrics, ['train/node_loss_epoch', 'train/node_loss'], 0.0),
            'edge_loss': self._get_metric_value(metrics, ['train/edge_loss_epoch', 'train/edge_loss'], 0.0),
            'edge_adjacency_loss': self._get_metric_value(metrics, ['train/edge_adjacency_loss_epoch', 'train/edge_adjacency_loss'], 0.0),
            'edge_pair_loss': self._get_metric_value(metrics, ['train/edge_pair_loss_epoch', 'train/edge_pair_loss'], 0.0),
            'coverage_loss': self._get_metric_value(metrics, ['train/coverage_loss_epoch', 'train/coverage_loss'], 0.0),
            'mask_loss': self._get_metric_value(metrics, ['train/mask_loss_epoch', 'train/mask_loss'], 0.0),
        }
        
        # Append training metrics to CSV
        with open(self.train_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                train_metrics['epoch'],
                train_metrics['step'],
                train_metrics['loss'],
                train_metrics['node_loss'],
                train_metrics['edge_loss'],
                train_metrics['edge_adjacency_loss'],
                train_metrics['edge_pair_loss'],
                train_metrics['coverage_loss'],
                train_metrics['mask_loss'],
            ])
        
        # Write the final validation result for the CURRENT epoch
        # (validation at end of epoch happens right before train_epoch_end)
        if hasattr(self, 'pending_val_metrics') and epoch in self.pending_val_metrics:
            val_metrics = self.pending_val_metrics[epoch]
            with open(self.val_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    val_metrics['epoch'],
                    val_metrics['loss'],
                    val_metrics['node_loss'],
                    val_metrics['edge_loss'],
                    val_metrics['edge_adjacency_loss'],
                    val_metrics['edge_pair_loss'],
                    val_metrics['coverage_loss'],
                    val_metrics['mask_loss'],
                    val_metrics['node_count_error'],
                ])
            # Clean up
            del self.pending_val_metrics[epoch]
        
        self.last_logged_train_epoch = epoch
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of training - write any remaining validation metrics."""
        # CRITICAL FIX: Only log from rank 0 to avoid duplicate entries in DDP
        if trainer.global_rank != 0:
            return
        
        # Write the final epoch's validation result if it hasn't been written yet
        if hasattr(self, 'pending_val_metrics') and self.pending_val_metrics:
            final_epoch = max(self.pending_val_metrics.keys())
            val_metrics = self.pending_val_metrics[final_epoch]
            with open(self.val_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    val_metrics['epoch'],
                    val_metrics['loss'],
                    val_metrics['node_loss'],
                    val_metrics['edge_loss'],
                    val_metrics['edge_adjacency_loss'],
                    val_metrics['edge_pair_loss'],
                    val_metrics['coverage_loss'],
                    val_metrics['mask_loss'],
                    val_metrics['node_count_error'],
                ])

