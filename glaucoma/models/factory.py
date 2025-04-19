# glaucoma/models/factory.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any

def create_model(config):
    """Factory function to create model based on configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized model
    """
    architecture = config.architecture.lower()
    encoder_name = config.encoder
    encoder_weights = 'imagenet' if config.pretrained else None
    in_channels = config.in_channels
    classes = config.num_classes
    
    if architecture == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # Apply activation in loss function
        )
    elif architecture == 'unetplusplus' or architecture == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif architecture == 'deeplabv3' or architecture == 'deeplab':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model