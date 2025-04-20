# glaucoma/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List

class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """Initialize dice loss.
        
        Args:
            smooth: Smoothing factor
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate dice loss.
        
        Args:
            inputs: Prediction tensor (before sigmoid)
            targets: Target tensor
            
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
        loss = 1 - dice
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * targets.size(0)
        else:  # 'none'
            return loss

class FocalLoss(nn.Module):
    """Focal Loss implementation for dealing with class imbalance."""
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter for hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.
        
        Args:
            inputs: Prediction tensor (before sigmoid)
            targets: Target tensor
            
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
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss function using Dice and BCE/Focal losses."""
    
    def __init__(
        self, 
        dice_weight: float = 1.0, 
        bce_weight: float = 1.0, 
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        reduction: str = 'mean'
    ):
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            focal_alpha: Alpha parameter for Focal loss
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss(reduction=reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            inputs: Prediction tensor (before sigmoid)
            targets: Target tensor
            
        Returns:
            Combined loss value
        """
        # Calculate component losses
        loss = 0.0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(inputs, targets)
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(inputs, targets)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(inputs, targets)
        
        return loss

class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced segmentation tasks."""
    
    def __init__(
        self, 
        alpha: float = 0.7, 
        beta: float = 0.3, 
        smooth: float = 1.0, 
        reduction: str = 'mean'
    ):
        """Initialize Tversky loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Tversky loss.
        
        Args:
            inputs: Prediction tensor (before sigmoid)
            targets: Target tensor
            
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
        loss = 1 - tversky
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss * targets.size(0)
        else:  # 'mean'
            return loss

def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """Get loss function based on type.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Initialized loss function
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")