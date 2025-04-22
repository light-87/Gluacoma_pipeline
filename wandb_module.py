"""
Weights & Biases Module

This module provides functions for integrating with Weights & Biases for experiment tracking.
"""

import wandb
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import io

class WandBLogger:
    """Weights & Biases logger with enhanced integrations."""
    
    def __init__(self, config):
        """Initialize WandB logger.
        
        Args:
            config: Pipeline configuration with WandB settings
        """
        self.config = config
        self.wandb_config = config.pipeline.wandb
        self.initialized = False
        
        # Always initialize WandB
        self.init_wandb()
    
    def init_wandb(self):
        """Initialize WandB run."""
        if self.initialized:
            return
        
        # Generate unique run name if not provided
        if not self.wandb_config.name:
            self.wandb_config.name = self._generate_run_name()
        
        # Add model architecture and encoder to tags
        if not hasattr(self.wandb_config, 'tags'):
            self.wandb_config.tags = []
        
        model_tag = f"model_{self.config.model.architecture}"
        encoder_tag = f"encoder_{self.config.model.encoder}"
        
        if model_tag not in self.wandb_config.tags:
            self.wandb_config.tags.append(model_tag)
        
        if encoder_tag not in self.wandb_config.tags:
            self.wandb_config.tags.append(encoder_tag)
        
        # Get WandB configuration
        wandb_init_args = {
            'project': self.wandb_config.project,
            'name': self.wandb_config.name,
            'tags': self.wandb_config.tags,
            'config': self._get_wandb_config(),
            'notes': self.wandb_config.notes,
            'dir': os.path.join(self.config.pipeline.output_dir, 'wandb'),
        }
        
        # Add entity if specified
        if self.wandb_config.entity:
            wandb_init_args['entity'] = self.wandb_config.entity
        
        # Add group if specified
        if self.wandb_config.group:
            wandb_init_args['group'] = self.wandb_config.group
        
        # Initialize wandb
        wandb.init(**wandb_init_args)
        self.initialized = True
        
        print(f"Initialized WandB run: {self.wandb_config.name}")
    
    def _generate_run_name(self) -> str:
        """Generate a descriptive run name for WandB."""
        # Use the method from Config class
        if hasattr(self.config, 'generate_run_name'):
            return self.config.generate_run_name()
        
        # Fallback to a basic name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.model.architecture}_{self.config.model.encoder}_{timestamp}"
    
    def _get_wandb_config(self) -> Dict:
        """Get configuration for WandB.
        
        Returns:
            Dictionary with configuration for WandB
        """
        # Convert config to dictionary, removing any non-serializable objects
        config_dict = {}
        
        # Add data config
        config_dict['data'] = {
            'data_dirs': list(self.config.data.data_dirs.keys()),
            'validation_split': self.config.data.validation_split,
            'test_split': self.config.data.test_split
        }
        
        # Add cache size and prefetch size if they exist
        if hasattr(self.config.data, 'cache_size'):
            config_dict['data']['cache_size'] = self.config.data.cache_size
        if hasattr(self.config.data, 'prefetch_size'):
            config_dict['data']['prefetch_size'] = self.config.data.prefetch_size
        
        # Add model config
        config_dict['model'] = {
            'architecture': self.config.model.architecture,
            'encoder': self.config.model.encoder,
            'pretrained': self.config.model.pretrained,
            'in_channels': self.config.model.in_channels,
            'num_classes': self.config.model.num_classes
        }
        
        # Add preprocessing config
        config_dict['preprocessing'] = {
            'image_size': self.config.preprocessing.image_size,
            'normalization': self.config.preprocessing.normalization,
            'augmentation_enabled': self.config.preprocessing.augmentation_enabled
        }
        
        # Add training config
        config_dict['training'] = {
            'epochs': self.config.training.epochs,
            'batch_size': self.config.training.batch_size,
            'num_workers': self.config.training.num_workers,
            'learning_rate': self.config.training.learning_rate,
            'optimizer': self.config.training.optimizer,
            'weight_decay': self.config.training.weight_decay,
            'loss_function': self.config.training.loss.loss_function,
            'dice_weight': self.config.training.loss.dice_weight,
            'focal_weight': self.config.training.loss.focal_weight,
            'device': self.config.training.device,
            'use_amp': self.config.training.use_amp,
            'grad_accum_steps': self.config.training.grad_accum_steps
        }
        
        # Add evaluation config
        config_dict['evaluation'] = {
            'batch_size': self.config.evaluation.batch_size,
            'threshold': self.config.evaluation.threshold,
            'calculate_cdr': self.config.evaluation.calculate_cdr,
            'cdr_method': self.config.evaluation.cdr_method,
            'use_tta': self.config.evaluation.use_tta
        }
        
        # Add pipeline config
        config_dict['pipeline'] = {
            'run_id': self.config.pipeline.run_id,
            'steps': self.config.pipeline.steps,
            'description': self.config.pipeline.description
        }
        
        return config_dict
    
    def log(self, data: Dict[str, Any]):
        """Log data to WandB.
        
        Args:
            data: Dictionary with data to log
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping log")
                return
        
        wandb.log(data)
    
    def save_model(self, model: nn.Module, name: str = 'model', metadata: Optional[Dict[str, Any]] = None):
        """Save model to WandB artifacts.
        
        Args:
            model: Model to save
            name: Artifact name
            metadata: Additional metadata
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping model save")
                return
        
        # Create artifact
        artifact = wandb.Artifact(
            name=name,
            type='model',
            metadata=metadata or {}
        )
        
        # Save model to a file
        model_path = os.path.join(wandb.run.dir, f"{name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Add model file to artifact
        artifact.add_file(model_path)
        
        # Log artifact
        wandb.log_artifact(artifact)
    
    def log_model(self, model: nn.Module):
        """Log model architecture to WandB.
        
        Args:
            model: Model to log
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping model log")
                return
        
        # Use model architecture
        try:
            wandb.watch(
                model, 
                log="all", 
                log_freq=100, 
                log_graph=True
            )
        except Exception as e:
            print(f"Error logging model to WandB: {e}")
    
    def log_image(self, image: np.ndarray, caption: str = "Image"):
        """Log image to WandB.
        
        Args:
            image: Image to log (numpy array)
            caption: Image caption
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping image log")
                return
        
        wandb.log({caption: wandb.Image(image)})
    
    def log_figure(self, figure: plt.Figure, caption: str = "Figure"):
        """Log matplotlib figure to WandB.
        
        Args:
            figure: Figure to log
            caption: Figure caption
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping figure log")
                return
        
        wandb.log({caption: wandb.Image(figure)})
    
    def log_image_with_masks(self, image: np.ndarray, masks: Dict[str, np.ndarray], caption: str = "Segmentation"):
        """Log image with segmentation masks.
        
        Args:
            image: Image to log (numpy array)
            masks: Dictionary mapping mask names to mask arrays
            caption: Image caption
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping image log")
                return
        
        # Create W&B Image with masks
        wandb_image = wandb.Image(
            image,
            masks={
                name: {"mask_data": mask, "class_labels": {1: name}}
                for name, mask in masks.items()
            }
        )
        
        wandb.log({caption: wandb_image})
    
    def log_image_grid(self, images: List[np.ndarray], captions: List[str], title: str = "Images"):
        """Log a grid of images.
        
        Args:
            images: List of images to log
            captions: List of captions for each image
            title: Title for the grid
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping image grid log")
                return
        
        # Create figure
        n = len(images)
        fig, axs = plt.subplots(1, n, figsize=(5*n, 5))
        
        # Handle the case where there's only one image
        if n == 1:
            axs = [axs]
        
        # Add each image to the grid
        for i, (img, caption) in enumerate(zip(images, captions)):
            axs[i].imshow(img)
            axs[i].set_title(caption)
            axs[i].axis('off')
        
        plt.tight_layout()
        
        # Log the figure
        wandb.log({title: wandb.Image(fig)})
        plt.close(fig)
    
    def log_table(self, data: List[List[Any]], table_name: str, columns: List[str]):
        """Log a table to WandB.
        
        Args:
            data: Table data as list of rows
            table_name: Name for the table
            columns: Column names
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping table log")
                return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})
    
    def log_metrics_over_time(self, metrics_history: List[Dict[str, float]]):
        """Log metrics over time.
        
        Args:
            metrics_history: List of dictionaries with metrics for each epoch
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping metrics log")
                return
        
        # Convert metrics history to dictionary of lists
        metrics_dict = {}
        for epoch, metrics in enumerate(metrics_history):
            for key, value in metrics.items():
                if key not in metrics_dict:
                    metrics_dict[key] = []
                metrics_dict[key].append(value)
                
                # Log with epoch number
                wandb.log({key: value, 'epoch': epoch + 1})
    
    def log_code(self):
        """Log code to WandB for reproducibility."""
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping code log")
                return
        
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    def log_prediction_samples(self, images: torch.Tensor, masks: torch.Tensor, 
                             predictions: torch.Tensor, threshold: float = 0.5, 
                             max_samples: int = 8):
        """Log sample predictions as a grid.
        
        Args:
            images: Batch of input images
            masks: Batch of ground truth masks
            predictions: Batch of predicted masks (probabilities)
            threshold: Threshold for binary predictions
            max_samples: Maximum number of samples to log
        """
        if not self.initialized:
            # Try initializing again before giving up
            self.init_wandb()
            if not self.initialized:
                print("Failed to initialize WandB, skipping prediction samples log")
                return
        
        # Convert to numpy for processing
        images = images.cpu().detach().numpy()
        masks = masks.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        
        # Apply sigmoid if predictions are logits
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = 1 / (1 + np.exp(-predictions))
        
        # Create binary predictions
        binary_predictions = (predictions > threshold).astype(np.float32)
        
        # Determine number of samples to visualize
        num_samples = min(max_samples, len(images))
        
        for i in range(num_samples):
            # Get sample data
            img = images[i].transpose(1, 2, 0)  # Change from (C,H,W) to (H,W,C)
            mask = masks[i, 0]  # Take first channel of mask
            pred_prob = predictions[i, 0]
            pred_binary = binary_predictions[i, 0]
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            # Log with mask overlays
            self.log_image_with_masks(
                img,
                {
                    "ground_truth": mask,
                    "prediction": pred_binary
                },
                f"Sample {i+1}"
            )
    
    def finish(self):
        """Finish WandB run."""
        if self.initialized:
            wandb.finish()
            self.initialized = False