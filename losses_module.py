"""
Losses Module

This module provides loss functions for the glaucoma detection pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        """Initialize dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate dice loss.
        
        Args:
            inputs: Predicted tensor (B, C, H, W)
            targets: Target tensor (B, C, H, W)
            
        Returns:
            Dice loss
        """
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Calculate loss
        return 1 - dice

class FocalLoss(nn.Module):
    """Focal Loss implementation for dealing with class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter controls the down-weighting of well-classified examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        
        Args:
            inputs: Predicted tensor (B, C, H, W)
            targets: Target tensor (B, C, H, W)
            
        Returns:
            Focal loss
        """
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate BCE loss
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weights
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Calculate focal loss
        focal_loss = alpha_factor * modulating_factor * bce
        
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    """Tversky loss for handling imbalanced data better than Dice."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        """Initialize Tversky loss.
        
        Args:
            alpha: Weight of false positives
            beta: Weight of false negatives
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Tversky loss.
        
        Args:
            inputs: Predicted tensor (B, C, H, W)
            targets: Target tensor (B, C, H, W)
            
        Returns:
            Tversky loss
        """
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate true positives, false positives, and false negatives
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Calculate loss
        return 1 - tversky

class CombinedLoss(nn.Module):
    """Combined loss function using multiple component losses."""
    
    def __init__(
        self, 
        dice_weight: float = 1.0, 
        bce_weight: float = 0.0, 
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7
    ):
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
            tversky_weight: Weight for Tversky loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Alpha parameter for Focal loss
            tversky_alpha: Alpha parameter for Tversky loss
            tversky_beta: Beta parameter for Tversky loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss() if dice_weight > 0 else None
        self.bce_loss = nn.BCEWithLogitsLoss() if bce_weight > 0 else None
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if focal_weight > 0 else None
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta) if tversky_weight > 0 else None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            inputs: Predicted tensor (B, C, H, W)
            targets: Target tensor (B, C, H, W)
            
        Returns:
            Combined loss
        """
        # Calculate component losses
        loss = 0.0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(inputs, targets)
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(inputs, targets)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(inputs, targets)
        
        if self.tversky_weight > 0:
            loss += self.tversky_weight * self.tversky_loss(inputs, targets)
        
        return loss

def create_loss_function(config):
    """Create loss function based on configuration.
    
    Args:
        config: Loss configuration section
        
    Returns:
        Loss function
    """
    loss_type = config.loss_function.lower()
    
    # Use the weights directly from the config - they should be properly set now
    return CombinedLoss(
        dice_weight=config.dice_weight,
        bce_weight=config.bce_weight,
        focal_weight=config.focal_weight,
        tversky_weight=0.0,  # Use config.tversky_weight if available
        focal_gamma=config.focal_gamma,
        focal_alpha=config.focal_alpha
    )