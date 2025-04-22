"""
Models Module

This module provides model creation, loading and ensemble functionality for the glaucoma detection pipeline.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, List, Optional, Union, Any
import os

def create_model(model_config) -> nn.Module:
    """Create a model based on the configuration.
    
    Args:
        model_config: Model configuration section
        
    Returns:
        PyTorch model
    """
    architecture = model_config.architecture.lower()
    encoder = model_config.encoder
    in_channels = model_config.in_channels
    num_classes = model_config.num_classes
    encoder_weights = 'imagenet' if model_config.pretrained else None
    
    # Select the architecture
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    elif architecture == 'unetplusplus' or architecture == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    elif architecture == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    elif architecture == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    print(f"Created {architecture} model with {encoder} encoder")
    return model

def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from a checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the model state dict itself
        model.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None, mode: str = 'average'):
        """Initialize an ensemble model.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model (for weighted average mode)
            mode: Ensembling mode ('average', 'weighted', or 'max')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if mode not in ['average', 'weighted', 'max']:
            raise ValueError(f"Unsupported ensemble mode: {mode}")
        
        self.mode = mode
        
        if mode == 'weighted' and weights is None:
            # Default to equal weights
            weights = [1.0 / len(models)] * len(models)
        
        if weights is not None and len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
        
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        
        if self.mode == 'max':
            # Take the maximum predicted probability
            predictions = torch.stack([torch.sigmoid(output) for output in outputs], dim=0)
            return torch.max(predictions, dim=0)[0]
        
        elif self.mode == 'weighted':
            # Weighted average of predictions
            weighted_outputs = [w * output for w, output in zip(self.weights, outputs)]
            return sum(weighted_outputs)
        
        else:  # 'average'
            # Simple average of predictions
            return sum(outputs) / len(outputs)
    
    def to(self, device):
        """Move ensemble to device."""
        for model in self.models:
            model.to(device)
        return super().to(device)

def create_ensemble(model_configs: List[Dict], checkpoints: List[str],
                   weights: Optional[List[float]] = None, mode: str = 'average') -> nn.Module:
    """Create an ensemble of models.
    
    Args:
        model_configs: List of model configurations
        checkpoints: List of checkpoint paths
        weights: Weights for each model (for weighted average mode)
        mode: Ensembling mode ('average', 'weighted', or 'max')
        
    Returns:
        Ensemble model
    """
    if len(model_configs) != len(checkpoints):
        raise ValueError(f"Number of model configs ({len(model_configs)}) must match "
                         f"number of checkpoints ({len(checkpoints)})")
    
    # Create individual models
    models = []
    for config, checkpoint_path in zip(model_configs, checkpoints):
        model = create_model(config)
        if checkpoint_path:
            model = load_checkpoint(model, checkpoint_path)
        models.append(model)
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights, mode)
    
    print(f"Created ensemble of {len(models)} models with mode '{mode}'")
    return ensemble