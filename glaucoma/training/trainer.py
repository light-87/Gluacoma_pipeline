"""
Trainer Module

Implements training loop for glaucoma detection models with improved efficiency.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from glaucoma.utils.logging import PipelineLogger
from glaucoma.models.losses import get_loss_function
from glaucoma.evaluation.metrics import calculate_metrics

class Trainer:
    """Trainer class for managing the training process."""
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        logger: Any,
        checkpoint_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            logger: Logger for tracking progress
            checkpoint_dir: Directory to save checkpoints
            wandb_logger: Optional Weights & Biases logger
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.wandb_logger = wandb_logger
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(config, 'use_gpu', True) else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Set up automatic mixed precision
        self.use_amp = getattr(config, 'use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using automatic mixed precision training")
        else:
            self.scaler = None
        
        # Set up loss function
        self.loss_fn = self._create_loss_function()
        
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
        self.monitor_metric = getattr(config, 'monitor_metric', 'val_loss')
        self.monitor_mode = getattr(config, 'monitor_mode', 'min')
        self.best_metric = float('inf') if self.monitor_mode == 'min' else -float('inf')
        self.early_stopping_counter = 0
        
        # Initialize metrics history
        self.metrics_history = {}
        
        # Set gradient accumulation steps
        self.grad_accum_steps = getattr(config, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1:
            self.logger.info(f"Using gradient accumulation with {self.grad_accum_steps} steps")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Log training configuration
        self._log_training_config()
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on config.
        
        Returns:
            Initialized loss function
        """
        loss_type = getattr(self.config, 'loss_function', 'combined')
        
        # Get loss parameters
        loss_params = {}
        
        if loss_type == 'combined':
            loss_params = {
                'dice_weight': getattr(self.config, 'dice_weight', 1.0),
                'bce_weight': getattr(self.config, 'bce_weight', 1.0),
                'focal_weight': getattr(self.config, 'focal_weight', 0.0),
                'focal_gamma': getattr(self.config, 'focal_gamma', 2.0),
                'focal_alpha': getattr(self.config, 'focal_alpha', 0.25)
            }
        elif loss_type == 'focal':
            loss_params = {
                'alpha': getattr(self.config, 'focal_alpha', 0.25),
                'gamma': getattr(self.config, 'focal_gamma', 2.0)
            }
        elif loss_type == 'tversky':
            loss_params = {
                'alpha': getattr(self.config, 'tversky_alpha', 0.7),
                'beta': getattr(self.config, 'tversky_beta', 0.3)
            }
        
        return get_loss_function(loss_type, **loss_params)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config.
        
        Returns:
            Initialized optimizer
        """
        optimizer_name = getattr(self.config, 'optimizer', 'adam').lower()
        lr = getattr(self.config, 'learning_rate', 0.001)
        weight_decay = getattr(self.config, 'weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.config, 'momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
            return optim.Adam(self.model.parameters(), lr=lr)
    
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
                mode=self.monitor_mode,
                factor=getattr(self.config, 'scheduler_factor', 0.1),
                patience=getattr(self.config, 'scheduler_patience', 5),
                verbose=True,
                min_lr=getattr(self.config, 'min_lr', 1e-6)
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.config, 'epochs', 30),
                eta_min=getattr(self.config, 'min_lr', 0)
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.config, 'scheduler_step_size', 10),
                gamma=getattr(self.config, 'scheduler_gamma', 0.1)
            )
        elif scheduler_name == 'one_cycle':
            # One cycle policy for super-convergence
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=getattr(self.config, 'max_lr', self.config.learning_rate * 10),
                epochs=getattr(self.config, 'epochs', 30),
                steps_per_epoch=len(self.train_loader),
                pct_start=getattr(self.config, 'pct_start', 0.3),
                div_factor=getattr(self.config, 'div_factor', 25),
                final_div_factor=getattr(self.config, 'final_div_factor', 1e4)
            )
        else:
            return None
    
    def _log_training_config(self) -> None:
        """Log training configuration."""
        # Log to regular logger
        self.logger.info(f"Training Configuration:")
        self.logger.info(f"- Model: {getattr(self.config, 'architecture', 'Unknown')}")
        self.logger.info(f"- Loss Function: {getattr(self.config, 'loss_function', 'combined')}")
        self.logger.info(f"- Optimizer: {getattr(self.config, 'optimizer', 'adam')}")
        self.logger.info(f"- Learning Rate: {getattr(self.config, 'learning_rate', 0.001)}")
        self.logger.info(f"- Batch Size: {getattr(self.config, 'batch_size', 16)}")
        self.logger.info(f"- Epochs: {getattr(self.config, 'epochs', 30)}")
        self.logger.info(f"- Mixed Precision: {self.use_amp}")
        self.logger.info(f"- Gradient Accumulation Steps: {self.grad_accum_steps}")
        
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            self.logger.info(f"- Gradient Clipping: {self.config.grad_clip}")
        
        # Log to wandb if available
        if self.wandb_logger:
            train_config = {
                'architecture': getattr(self.config, 'architecture', 'Unknown'),
                'loss_function': getattr(self.config, 'loss_function', 'combined'),
                'optimizer': getattr(self.config, 'optimizer', 'adam'),
                'learning_rate': getattr(self.config, 'learning_rate', 0.001),
                'batch_size': getattr(self.config, 'batch_size', 16),
                'epochs': getattr(self.config, 'epochs', 30),
                'device': str(self.device),
                'use_amp': self.use_amp,
                'grad_accum_steps': self.grad_accum_steps,
                'grad_clip': getattr(self.config, 'grad_clip', 0)
            }
            self.wandb_logger.log(train_config)
    
    def train(self) -> Dict[str, Any]:
        """Run the training process.
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting training")
        start_time = time.time()
        
        epochs = getattr(self.config, 'epochs', 30)
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_dice'].append(train_metrics['dice'])
            self.metrics_history['val_dice'].append(val_metrics['dice'])
            self.metrics_history['learning_rate'].append(current_lr)
            
            # Prepare metrics to log
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_metrics['dice'],
                'val_dice': val_metrics['dice'],
                'train_iou': train_metrics['iou'],
                'val_iou': val_metrics['iou'],
                'learning_rate': current_lr,
                'time_per_epoch': time.time() - epoch_start_time
            }
            
            # Log metrics
            self._log_metrics(metrics)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, use monitored metric
                    if self.monitor_metric == 'val_loss':
                        self.scheduler.step(val_loss)
                    else:
                        # For metrics where higher is better, negate the value
                        monitor_value = val_metrics.get(self.monitor_metric.replace('val_', ''), val_loss)
                        if self.monitor_mode == 'max':
                            monitor_value = -monitor_value
                        self.scheduler.step(monitor_value)
                else:
                    # For other schedulers, just step
                    self.scheduler.step()
            
            # Check for model improvement
            improved = self._check_improvement(val_loss, val_metrics)
            
            # Save checkpoint if improved
            if improved and self.checkpoint_dir:
                self._save_checkpoint(epoch, metrics, is_best=True)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= getattr(self.config, 'patience', 10) and getattr(self.config, 'early_stopping', True):
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % getattr(self.config, 'save_every', 5) == 0:
                self._save_checkpoint(epoch, metrics, is_best=False)
        
        # Save final model
        if self.checkpoint_dir:
            self._save_checkpoint(epochs-1, metrics, is_final=True)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Log final metrics to W&B
        if self.wandb_logger:
            self.wandb_logger.log_metrics_over_time(self.metrics_history)
            self.wandb_logger.log({
                'training_time': training_time,
                'total_epochs': epoch + 1,
                'best_val_metric': self.best_metric if self.monitor_mode == 'min' else -self.best_metric,
            })
        
        # Return training results
        return {
            'total_epochs': epoch + 1,
            'best_metric': self.best_metric if self.monitor_mode == 'min' else -self.best_metric,
            'best_metric_name': self.monitor_metric,
            'training_time': training_time,
            'early_stopped': self.early_stopping_counter >= getattr(self.config, 'patience', 10),
            'metrics_history': self.metrics_history
        }
    
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run a single training epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        epoch_loss = 0.0
        
        # For metric calculation
        all_preds = []
        all_targets = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")
        
        # Keep track of batches for gradient accumulation
        accum_loss = 0.0
        accum_batches = 0
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients only at the beginning of accumulation
            if accum_batches == 0:
                self.optimizer.zero_grad()
            
            # Compute loss and backward pass with gradient accumulation
            if self.use_amp:
                # Use automatic mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    
                    # Scale loss for gradient accumulation
                    if self.grad_accum_steps > 1:
                        loss = loss / self.grad_accum_steps
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                
                # Update weights only after accumulating gradients
                accum_batches += 1
                accum_loss += loss.item() * (self.grad_accum_steps if self.grad_accum_steps > 1 else 1)
                
                if accum_batches == self.grad_accum_steps:
                    # Gradient clipping if configured
                    if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    accum_batches = 0
                    
                    # Log the accumulated loss
                    loss_value = accum_loss
                    accum_loss = 0.0
                else:
                    # If we're still accumulating, use the current loss for logging
                    loss_value = loss.item() * self.grad_accum_steps
            else:
                # Regular forward/backward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                # Scale loss for gradient accumulation
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights only after accumulating gradients
                accum_batches += 1
                accum_loss += loss.item() * (self.grad_accum_steps if self.grad_accum_steps > 1 else 1)
                
                if accum_batches == self.grad_accum_steps:
                    # Gradient clipping if configured
                    if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    # Optimizer step
                    self.optimizer.step()
                    accum_batches = 0
                    
                    # Log the accumulated loss
                    loss_value = accum_loss
                    accum_loss = 0.0
                else:
                    # If we're still accumulating, use the current loss for logging
                    loss_value = loss.item() * self.grad_accum_steps
            
            # Update metrics
            epoch_loss += loss_value
            
            # Store predictions and targets for metric calculation
            preds = torch.sigmoid(outputs.detach())
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss_value})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics on a subset for efficiency
        max_samples = min(1000, len(all_preds))
        indices = torch.randperm(len(all_preds))[:max_samples]
        
        metrics = {}
        for i in indices:
            batch_metrics = calculate_metrics(all_preds[i], all_targets[i])
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v
        
        # Average metrics
        metrics = {k: v / len(indices) for k, v in metrics.items()}
        
        self.logger.info(f"Train Epoch: {epoch+1} | Loss: {avg_loss:.4f} | Dice: {metrics['dice']:.4f}")
        
        return avg_loss, metrics
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run validation for a single epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        val_loss = 0.0
        
        # For metric calculation
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.val_loader, desc=f"Validation epoch {epoch+1}")):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass with mixed precision if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                
                # Update loss
                val_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
        
        # Calculate average loss
        avg_loss = val_loss / len(self.val_loader)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics on a subset for efficiency
        max_samples = min(1000, len(all_preds))
        indices = torch.randperm(len(all_preds))[:max_samples]
        
        metrics = {}
        for i in indices:
            batch_metrics = calculate_metrics(all_preds[i], all_targets[i])
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v
        
        # Average metrics
        metrics = {k: v / len(indices) for k, v in metrics.items()}
        
        self.logger.info(f"Validation Epoch: {epoch+1} | Loss: {avg_loss:.4f} | Dice: {metrics['dice']:.4f}")
        
        return avg_loss, metrics
    
    def _check_improvement(self, val_loss: float, val_metrics: Dict[str, float]) -> bool:
        """Check if the current validation results are an improvement.
        
        Args:
            val_loss: Validation loss
            val_metrics: Validation metrics
            
        Returns:
            Whether current results are an improvement
        """
        # Determine metric to monitor
        if self.monitor_metric == 'val_loss':
            current = val_loss
        else:
            # Monitor a specific metric
            metric_name = self.monitor_metric.replace('val_', '')
            current = val_metrics.get(metric_name, val_loss)
            
            # For metrics where higher is better, negate the value
            if self.monitor_mode == 'max':
                current = -current
        
        # Check for improvement
        if (self.monitor_mode == 'min' and current < self.best_metric) or \
           (self.monitor_mode == 'max' and current > self.best_metric):
            self.best_metric = current
            return True
        
        return False
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to logger and wandb.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        # Log to wandb if enabled
        if self.wandb_logger:
            self.wandb_logger.log(metrics)
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, Any], 
        is_best: bool = False,
        is_final: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config,
            'metrics_history': self.metrics_history,
            'use_amp': self.use_amp,
            'grad_accum_steps': self.grad_accum_steps,
        }
        
        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Add AMP scaler state if using mixed precision
        if self.use_amp and self.scaler is not None:
            checkpoint['amp_scaler_state_dict'] = self.scaler.state_dict()
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
            self.logger.info(f"Saved best model checkpoint (epoch {epoch+1})")
            
            # Also save to wandb if available
            if self.wandb_logger:
                self.wandb_logger.save_model(
                    self.model, 
                    name="best_model",
                    metadata={
                        'epoch': epoch + 1, 
                        'best_metric': self.best_metric if self.monitor_mode == 'min' else -self.best_metric,
                        'metric_name': self.monitor_metric
                    }
                )
        
        if is_final:
            torch.save(checkpoint, self.checkpoint_dir / "final_model.pt")
            self.logger.info(f"Saved final model checkpoint (epoch {epoch+1})")
            
            # Also save to wandb if available
            if self.wandb_logger:
                self.wandb_logger.save_model(
                    self.model, 
                    name="final_model",
                    metadata={'epochs': epoch + 1}
                )
        
        # Also save epoch checkpoint if configured
        save_every = getattr(self.config, 'save_every', 5)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")