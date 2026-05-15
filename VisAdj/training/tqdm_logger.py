"""
Custom TQDM Progress Bar that logs to file
"""

import sys
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
import logging


class FileTQDMProgressBar(TQDMProgressBar):
    """TQDM progress bar that also writes progress updates to a log file."""
    
    def __init__(self, log_file: str, refresh_rate: int = 1):
        """
        Args:
            log_file: Path to log file to write progress to
            refresh_rate: How often to log progress (in steps)
        """
        super().__init__(refresh_rate=refresh_rate)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Setup file handler for this logger
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called when training epoch starts."""
        super().on_train_epoch_start(trainer, pl_module)
        epoch = trainer.current_epoch
        total_batches = len(trainer.train_dataloader)
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch}/{trainer.max_epochs} - Training Started")
        self.logger.info(f"Total batches: {total_batches}")
        self.logger.info(f"{'='*60}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after each training batch."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # Log progress periodically
        if batch_idx % 100 == 0 or batch_idx == len(trainer.train_dataloader) - 1:
            epoch = trainer.current_epoch
            metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
            train_loss = metrics.get('train/loss', 'N/A')
            if hasattr(train_loss, 'item'):
                train_loss = train_loss.item()
            self.logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(trainer.train_dataloader)} | Train Loss: {train_loss:.6f}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when training epoch ends."""
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch
        metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
        train_loss = metrics.get('train/loss_epoch', metrics.get('train/loss', 'N/A'))
        if hasattr(train_loss, 'item'):
            train_loss = train_loss.item()
        self.logger.info(f"Epoch {epoch} - Training Completed | Train Loss: {train_loss:.6f}")
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when validation epoch starts."""
        super().on_validation_epoch_start(trainer, pl_module)
        epoch = trainer.current_epoch
        # Handle both list and single DataLoader
        if trainer.val_dataloaders:
            if isinstance(trainer.val_dataloaders, list):
                total_batches = len(trainer.val_dataloaders[0])
            else:
                total_batches = len(trainer.val_dataloaders)
        else:
            total_batches = 0
        self.logger.info(f"\nEpoch {epoch} - Validation Started | Batches: {total_batches}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called after each validation batch."""
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        # Handle both list and single DataLoader
        if trainer.val_dataloaders:
            if isinstance(trainer.val_dataloaders, list):
                val_loader = trainer.val_dataloaders[0]
            else:
                val_loader = trainer.val_dataloaders
            total_val_batches = len(val_loader)
        else:
            total_val_batches = 0
        
        # Log progress periodically
        if batch_idx % 50 == 0 or (total_val_batches > 0 and batch_idx == total_val_batches - 1):
            epoch = trainer.current_epoch
            metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
            val_loss = metrics.get('val/loss', 'N/A')
            if hasattr(val_loss, 'item'):
                val_loss = val_loss.item()
            if isinstance(val_loss, (int, float)):
                self.logger.info(f"Epoch {epoch} | Val Batch {batch_idx}/{total_val_batches} | Val Loss: {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch} | Val Batch {batch_idx}/{total_val_batches} | Val Loss: {val_loss}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when validation epoch ends."""
        super().on_validation_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch
        # Get latest validation metrics
        metrics = {**trainer.callback_metrics, **trainer.logged_metrics}
        val_loss = metrics.get('val/loss', 'N/A')
        if hasattr(val_loss, 'item'):
            val_loss = val_loss.item()
        
        # Get other metrics
        node_loss = metrics.get('val/node_loss', 'N/A')
        edge_loss = metrics.get('val/edge_loss', 'N/A')
        if hasattr(node_loss, 'item'):
            node_loss = node_loss.item()
        if hasattr(edge_loss, 'item'):
            edge_loss = edge_loss.item()
        
        self.logger.info(f"Epoch {epoch} - Validation Completed")
        self.logger.info(f"  Val Loss: {val_loss:.6f} | Node Loss: {node_loss:.6f} | Edge Loss: {edge_loss:.6f}")
        self.logger.info(f"{'='*60}\n")

