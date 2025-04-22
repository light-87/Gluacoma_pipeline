"""
Training Module

This module provides functionality for training glaucoma detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm import tqdm
import os
import numpy as np
import time
import copy

from evaluation_module import calculate_metrics

class Trainer:
    """Model trainer with advanced features."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config,
        device: torch.device,
        checkpoint_dir: str,
        wandb_logger = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            config: Training configuration
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            wandb_logger: Logger for Weights & Biases
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.wandb_logger = wandb_logger
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup automatic mixed precision if enabled
        self.use_amp = getattr(self.config, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation steps
        self.grad_accum_steps = getattr(self.config, 'grad_accum_steps', 1)
        
        # Gradient clipping
        self.grad_clip = getattr(self.config, 'grad_clip', 0.0)
        
        # Early stopping
        self.early_stopping = getattr(self.config, 'early_stopping', False)
        self.patience = getattr(self.config, 'patience', 10)
        self.monitor_metric = getattr(self.config, 'monitor_metric', 'val_dice')
        self.monitor_mode = getattr(self.config, 'monitor_mode', 'max')
        
        # Check monitor metric is valid
        valid_metrics = ['val_loss', 'val_dice', 'val_iou', 'val_accuracy', 'val_f1']
        if self.monitor_metric not in valid_metrics:
            raise ValueError(f"Invalid monitor_metric: {self.monitor_metric}. Must be one of {valid_metrics}")
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Print training setup
        print(f"Training setup:")
        print(f"  Model: {model.__class__.__name__}")
        print(f"  Optimizer: {self.optimizer.__class__.__name__}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Grad accum steps: {self.grad_accum_steps}")
        print(f"  Use AMP: {self.use_amp}")
        print(f"  Gradient clipping: {self.grad_clip}")
        print(f"  Early stopping: {self.early_stopping}")
        if self.early_stopping:
            print(f"  Monitor metric: {self.monitor_metric} ({self.monitor_mode})")
            print(f"  Patience: {self.patience}")
        print(f"  Device: {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer.
        
        Returns:
            PyTorch optimizer
        """
        optimizer_name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        weight_decay = getattr(self.config, 'weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.config, 'momentum', 0.9)
            return optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler.
        
        Returns:
            PyTorch learning rate scheduler
        """
        if not getattr(self.config, 'use_scheduler', True):
            return None
        
        scheduler_name = getattr(self.config, 'scheduler', 'reduce_on_plateau').lower()
        
        if scheduler_name == 'reduce_on_plateau':
            patience = getattr(self.config, 'scheduler_patience', 5)
            factor = getattr(self.config, 'scheduler_factor', 0.5)
            min_lr = getattr(self.config, 'min_lr', 1e-6)
            
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )
        elif scheduler_name == 'cosine':
            t_max = getattr(self.config, 'scheduler_t_max', self.config.epochs)
            eta_min = getattr(self.config, 'min_lr', 0)
            
            return CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _backward_step(self, loss: torch.Tensor):
        """Perform backward pass with support for gradient accumulation and mixed precision.
        
        Args:
            loss: Loss tensor
        """
        # Scale loss for gradient accumulation
        loss = loss / self.grad_accum_steps
        
        # Backward pass with AMP if enabled
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _optimizer_step(self):
        """Perform optimizer step with support for gradient accumulation, clipping, and mixed precision."""
        # Apply gradient clipping if enabled
        if self.grad_clip > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step with AMP if enabled
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0
        
        # Metrics for tracking
        all_preds = []
        all_masks = []
        step_count = 0
        
        # Use tqdm progress bar
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False, ncols=100)
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass with AMP if enabled
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            self._backward_step(loss)
            
            # Update metrics
            epoch_loss += loss.item() * self.grad_accum_steps
            
            # Store predictions and masks for metrics calculation
            preds = torch.sigmoid(outputs).detach()
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
            
            # Optimizer step (if grad_accum_steps reached)
            step_count += 1
            if step_count % self.grad_accum_steps == 0:
                self._optimizer_step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Final optimizer step for remaining gradients
        if step_count % self.grad_accum_steps != 0:
            self._optimizer_step()
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = {}
        threshold = getattr(self.config, 'threshold', 0.5)
        
        for i in range(len(all_preds)):
            batch_metrics = calculate_metrics(all_preds[i], all_masks[i], threshold=threshold)
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(all_preds)
        
        # Add loss to metrics
        metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0
        
        # Metrics for tracking
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            # Use tqdm progress bar
            progress_bar = tqdm(self.val_loader, desc='Validation', leave=False, ncols=100)
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass (no AMP needed for validation)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                val_loss += loss.item()
                
                # Store predictions and masks for metrics calculation
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = {}
        threshold = getattr(self.config, 'threshold', 0.5)
        
        for i in range(len(all_preds)):
            batch_metrics = calculate_metrics(all_preds[i], all_masks[i], threshold=threshold)
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = 0
                metrics[k] += v
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(all_preds)
        
        # Add loss to metrics
        metrics['loss'] = val_loss / len(self.val_loader)
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Dictionary with training results
        """
        # Get configuration
        epochs = self.config.epochs
        save_every = getattr(self.config, 'save_every', 5)
        
        # Metrics for tracking
        best_val_loss = float('inf')
        best_val_metric = float('-inf') if self.monitor_mode == 'max' else float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Save best model state
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Measure epoch time
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Measure epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Extract the monitoring values
            val_loss = val_metrics['loss']
            
            if self.monitor_metric == 'val_loss':
                current_metric = val_loss
                is_better = current_metric < best_val_metric if self.monitor_mode == 'min' else current_metric > best_val_metric
            else:
                metric_name = self.monitor_metric.replace('val_', '')
                current_metric = val_metrics[metric_name]
                is_better = current_metric > best_val_metric if self.monitor_mode == 'max' else current_metric < best_val_metric
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Dice: {train_metrics['dice']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch time: {epoch_time:.2f}s")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            history['learning_rates'].append(current_lr)
            
            # Log to WandB if enabled
            if self.wandb_logger:
                log_data = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                
                # Add all metrics
                for k, v in train_metrics.items():
                    if k != 'loss':
                        log_data[f'train_{k}'] = v
                
                for k, v in val_metrics.items():
                    if k != 'loss':
                        log_data[f'val_{k}'] = v
                
                self.wandb_logger.log(log_data)
            
            # Check for best model
            if is_better:
                best_val_metric = current_metric
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model state
                best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_metric': best_val_metric,
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics
                }, os.path.join(self.checkpoint_dir, 'best_model.pt'))
                
                print(f"Saved new best model with metric: {best_val_metric:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'val_loss': val_loss,
                }, os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Early stopping
            if self.early_stopping and patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'best_val_loss': best_val_loss,
        }, os.path.join(self.checkpoint_dir, 'final_model.pt'))
        
        print(f"\nTraining completed.")
        print(f"Best Validation Metric: {best_val_metric:.4f} at epoch {best_epoch+1}")
        
        # Return training results
        return {
            'best_val_metric': best_val_metric,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'history': history
        }