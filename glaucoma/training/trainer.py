# glaucoma/training/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from glaucoma.utils.logging import PipelineLogger
from glaucoma.models.losses import get_loss_function

class Trainer:
    """Trainer class for managing the training process."""
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        logger: PipelineLogger,
        checkpoint_dir: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            logger: Logger for tracking progress
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Set up loss function
        self.loss_fn = get_loss_function(
            config.loss_function,
            dice_weight=getattr(config, 'dice_weight', 1.0),
            bce_weight=getattr(config, 'bce_weight', 1.0)
        )
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.checkpoint_dir = None
        
        # Initialize metrics tracking
        self.best_val_metric = float('inf') if config.monitor_mode == 'min' else -float('inf')
        self.early_stopping_counter = 0
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize wandb if enabled
        self._setup_wandb()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config.
        
        Returns:
            Initialized optimizer
        """
        optimizer_name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=getattr(self.config, 'momentum', 0.9),
                weight_decay=getattr(self.config, 'weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(), 
                lr=lr,
                weight_decay=getattr(self.config, 'weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config.
        
        Returns:
            Initialized scheduler or None
        """
        if not getattr(self.config, 'use_scheduler', False):
            return None
            
        scheduler_name = getattr(self.config, 'scheduler', '').lower()
        
        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=getattr(self.config, 'monitor_mode', 'min'),
                factor=getattr(self.config, 'scheduler_factor', 0.1),
                patience=getattr(self.config, 'scheduler_patience', 10),
                verbose=True,
                min_lr=getattr(self.config, 'min_lr', 1e-6)
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.config, 'scheduler_t_max', self.config.epochs),
                eta_min=getattr(self.config, 'min_lr', 0)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.config, 'scheduler_step_size', 30),
                gamma=getattr(self.config, 'scheduler_gamma', 0.1)
            )
        elif not scheduler_name:
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _setup_wandb(self):
        """Set up Weights & Biases if enabled."""
        if getattr(self.config, 'use_wandb', False):
            try:
                import wandb
                self.use_wandb = True
                
                # Only initialize if not already initialized
                if wandb.run is None:
                    wandb.init(
                        project=getattr(self.config, 'wandb_project', 'glaucoma-detection'),
                        name=getattr(self.config, 'run_name', None),
                        config=vars(self.config) if hasattr(self.config, '__dict__') else self.config,
                    )
                    
                    # Log model architecture
                    wandb.watch(self.model, log="all", log_freq=100)
            except ImportError:
                self.logger.warning("Weights & Biases (wandb) not installed. Disabling wandb logging.")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def train(self) -> Dict[str, Any]:
        """Run the training process.
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting training")
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'time_per_epoch': time.time() - epoch_start_time
            }
            
            self._log_metrics(metrics)
            
            # Update learning rate scheduler if using val loss
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for early stopping and model saving
            monitor_value = val_loss
            if hasattr(self.config, 'monitor_metric') and self.config.monitor_metric in val_metrics:
                monitor_value = val_metrics[self.config.monitor_metric]
                if self.config.monitor_mode == 'max':
                    monitor_value = -monitor_value  # Convert to minimization problem
            
            improved = False
            if (self.config.monitor_mode == 'min' and monitor_value < self.best_val_metric) or \
               (self.config.monitor_mode == 'max' and monitor_value > self.best_val_metric):
                self.best_val_metric = monitor_value
                improved = True
                self.early_stopping_counter = 0
                
                # Save model if improved
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, metrics, is_best=True)
                    self.logger.info(f"Saved best model checkpoint (epoch {epoch+1})")
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"No improvement in {self.early_stopping_counter} epochs")
            
            # Check for early stopping
            if getattr(self.config, 'early_stopping', False) and \
               self.early_stopping_counter >= getattr(self.config, 'patience', 10):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % getattr(self.config, 'save_every', 5) == 0:
                self._save_checkpoint(epoch, metrics)
                self.logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Return training results
        return {
            'total_epochs': epoch + 1,
            'best_val_metric': self.best_val_metric if self.config.monitor_mode == 'min' else -self.best_val_metric,
            'training_time': training_time,
        }
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run a single training epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        epoch_loss = 0
        
        # Metrics to track
        metrics = {
            'dice': 0.0,
            'iou': 0.0,
        }
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate batch metrics (dice and iou)
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                masks_binary = masks > 0.5
                
                # Update metrics (using simple implementations here, can be replaced with torchmetrics)
                batch_dice = self._dice_coefficient(preds, masks_binary)
                batch_iou = self._iou_coefficient(preds, masks_binary)
                
                metrics['dice'] += batch_dice.item()
                metrics['iou'] += batch_iou.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': batch_dice.item(),
                'iou': batch_iou.item()
            })
        
        # Calculate average metrics
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        
        # Average metrics
        for k in metrics:
            metrics[k] /= num_batches
        
        self.logger.info(f"Train Epoch: {epoch+1} | Loss: {avg_loss:.4f} | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")
        
        return avg_loss, metrics
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run validation for a single epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        val_loss = 0
        
        # Metrics to track
        metrics = {
            'dice': 0.0,
            'iou': 0.0,
        }
        
        # For tracking predictions for wandb visualization
        sample_images = []
        sample_masks = []
        sample_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.val_loader, desc=f"Validation epoch {epoch+1}")):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate batch metrics
                preds = torch.sigmoid(outputs) > 0.5
                masks_binary = masks > 0.5
                
                batch_dice = self._dice_coefficient(preds, masks_binary)
                batch_iou = self._iou_coefficient(preds, masks_binary)
                
                metrics['dice'] += batch_dice.item()
                metrics['iou'] += batch_iou.item()
                
                # Store samples for visualization (first batch only)
                if batch_idx == 0 and self.use_wandb:
                    max_samples = min(5, images.size(0))
                    sample_images.extend([images[i].cpu() for i in range(max_samples)])
                    sample_masks.extend([masks[i].cpu() for i in range(max_samples)])
                    sample_preds.extend([outputs[i].sigmoid().cpu() for i in range(max_samples)])
        
        # Calculate average metrics
        num_batches = len(self.val_loader)
        avg_loss = val_loss / num_batches
        
        # Average metrics
        for k in metrics:
            metrics[k] /= num_batches
        
        self.logger.info(f"Validation Epoch: {epoch+1} | Loss: {avg_loss:.4f} | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")
        
        # Log sample predictions to wandb
        if self.use_wandb and sample_images:
            self._log_predictions_to_wandb(sample_images, sample_masks, sample_preds)
        
        return avg_loss, metrics
    
    def _dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Calculate Dice coefficient.
        
        Args:
            pred: Prediction tensor (binary)
            target: Target tensor (binary)
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient
        """
        intersection = (pred * target).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        return dice.mean()
    
    def _iou_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Calculate IoU coefficient.
        
        Args:
            pred: Prediction tensor (binary)
            target: Target tensor (binary)
            smooth: Smoothing factor
            
        Returns:
            IoU coefficient
        """
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = (pred + target).sum(dim=(1, 2, 3)) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to logger and wandb.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Log to wandb if enabled
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
        
        # Log to pipeline logger
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_metric(name, value)
    
    def _log_predictions_to_wandb(self, images, masks, predictions):
        """Log sample predictions to wandb.
        
        Args:
            images: List of sample images
            masks: List of sample masks
            predictions: List of sample predictions
        """
        if not self.use_wandb:
            return
            
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a grid of images with masks and predictions
        fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))
        
        for i, (img, mask, pred) in enumerate(zip(images, masks, predictions)):
            # Convert to numpy and transpose if needed
            img_np = img.numpy().transpose(1, 2, 0)
            mask_np = mask.numpy().squeeze()
            pred_np = pred.numpy().squeeze()
            
            # Normalize image for display
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Plot
            if len(images) == 1:
                # Handle case of single sample
                axes[0].imshow(img_np)
                axes[0].set_title('Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask_np, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
            else:
                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title('Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask_np, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_np, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        wandb.log({"predictions": wandb.Image(fig)})
        plt.close(fig)
        
        # Also log using wandb's built-in mask visualization
        wandb_masks = []
        for i, (img, mask, pred) in enumerate(zip(images, masks, predictions)):
            img_np = img.numpy().transpose(1, 2, 0)
            mask_np = mask.numpy().squeeze()
            pred_np = (pred.numpy().squeeze() > 0.5).astype(np.float32)
            
            # Normalize image for display
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            wandb_masks.append(
                wandb.Image(
                    img_np,
                    masks={
                        "ground_truth": {"mask_data": mask_np, "class_labels": {0: "background", 1: "glaucoma"}},
                        "prediction": {"mask_data": pred_np, "class_labels": {0: "background", 1: "glaucoma"}}
                    }
                )
            )
        
        wandb.log({"mask_samples": wandb_masks})
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any], is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_metric': self.best_val_metric,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config,
        }
        
        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        
        # Also save epoch checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")