"""
Model Factory Module

Implements factory functions for creating various model architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union

def create_model(config: Any) -> nn.Module:
    """Factory function to create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    # Handle ensemble cases
    if hasattr(config, 'use_ensemble') and config.use_ensemble:
        return create_ensemble(config)
    
    # Get model parameters
    architecture = getattr(config, 'architecture', 'unet').lower()
    encoder_name = getattr(config, 'encoder', 'resnet34')
    encoder_weights = 'imagenet' if getattr(config, 'pretrained', True) else None
    in_channels = getattr(config, 'in_channels', 3)
    num_classes = getattr(config, 'num_classes', 1)
    activation = getattr(config, 'activation', None)
    
    # Import segmentation-models-pytorch
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "Please install segmentation_models_pytorch: "
            "pip install segmentation-models-pytorch"
        )
    
    # Create model based on architecture
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture in ['unetplusplus', 'unet++']:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture in ['deeplabv3', 'deeplab']:
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'pan':
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    elif architecture == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

def create_ensemble(config: Any) -> nn.Module:
    """Create ensemble model based on configuration.
    
    Args:
        config: Ensemble configuration
        
    Returns:
        Initialized ensemble model
    """
    from glaucoma.models.ensemble import ModelEnsemble, EnsembleFactory
    
    # Get ensemble method
    ensemble_method = getattr(config, 'ensemble_method', 'average')
    
    # Get weights if applicable
    weights = getattr(config, 'ensemble_weights', None)
    
    # Check ensemble type
    if hasattr(config, 'ensemble_type') and config.ensemble_type == 'cross_validation':
        # Create cross-validation ensemble
        return EnsembleFactory.create_cross_validation_ensemble(
            base_model_config=config.base_model,
            checkpoint_paths=config.checkpoint_paths,
            ensemble_method=ensemble_method,
            weights=weights
        )
    elif hasattr(config, 'ensemble_type') and config.ensemble_type == 'multi_architecture':
        # Create multi-architecture ensemble
        return EnsembleFactory.create_multi_architecture_ensemble(
            model_configs=config.model_configs,
            checkpoint_paths=getattr(config, 'checkpoint_paths', None),
            ensemble_method=ensemble_method,
            weights=weights
        )
    else:
        # Default to creating ensemble from model checkpoints
        model_checkpoints = []
        
        for model_config in config.models:
            model_checkpoints.append({
                'model': model_config.model,
                'checkpoint_path': getattr(model_config, 'checkpoint_path', None)
            })
        
        return EnsembleFactory.create_ensemble(
            model_checkpoints=model_checkpoints,
            ensemble_method=ensemble_method,
            weights=weights
        )

def load_checkpoint(
    model: nn.Module, 
    checkpoint_path: str, 
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load weights from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights on
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check for model_state_dict key
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Some checkpoints use 'state_dict' key
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return model

def get_feature_extractor(
    model: nn.Module, 
    layer_name: str
) -> Optional[nn.Module]:
    """Get feature extractor from model for a specific layer.
    
    Args:
        model: Model to extract features from
        layer_name: Name of layer to extract features from
        
    Returns:
        Feature extractor module or None if not found
    """
    # Find the module with the given name
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    
    return None

def get_model_summary(model: nn.Module, input_size: Optional[tuple] = None) -> str:
    """Get summary of model architecture.
    
    Args:
        model: Model to summarize
        input_size: Optional input size for calculating parameter counts
        
    Returns:
        String with model summary
    """
    try:
        from torchsummary import summary
        if input_size:
            return str(summary(model, input_size))
        else:
            return str(model)
    except ImportError:
        return str(model)