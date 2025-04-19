# glaucoma/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional

class CombinedLoss(nn.Module):
    """Combined loss function using Dice and BCE.
    
    Combines Dice loss and BCE loss for better segmentation results.
    """
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, mode: str = 'binary'):
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            mode: Loss mode ('binary' or 'multiclass')
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor
            
        Returns:
            Combined loss value
        """
        # Ensure inputs have proper shape for both losses
        if y_pred.dim() == 4 and y_pred.size(1) == 1:
            # Prepare inputs for BCE loss which expects [B, 1, H, W]
            y_pred_bce = y_pred
            y_true_bce = y_true
        else:
            # Handle other cases
            y_pred_bce = y_pred
            y_true_bce = y_true
        
        # Apply BCE loss
        bce = self.bce_loss(y_pred_bce, y_true_bce)
        
        # Apply Dice loss
        dice = self.dice_loss(y_pred, y_true)
        
        # Combine losses
        return self.dice_weight * dice + self.bce_weight * bce

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
        return smp.losses.DiceLoss(mode='binary')
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'focal':
        return smp.losses.FocalLoss(mode='binary')
    elif loss_type == 'jaccard':
        return smp.losses.JaccardLoss(mode='binary')
    elif loss_type == 'lovasz':
        return smp.losses.LovaszLoss(mode='binary')
    elif loss_type == 'tversky':
        return smp.losses.TverskyLoss(mode='binary')
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")