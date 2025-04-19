# glaucoma/evaluation/metrics.py
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union

def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Dice coefficient.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient
    """
    # Ensure inputs are binary
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # Calculate intersection and sum
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.mean()

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Calculate IoU (Jaccard index).
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing factor
        
    Returns:
        IoU coefficient
    """
    # Ensure inputs are binary
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def calculate_confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    """Calculate confusion matrix elements.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        
    Returns:
        Tuple of (tn, fp, fn, tp)
    """
    # Ensure inputs are binary and flattened
    pred = (pred > 0.5).float().view(-1)
    target = (target > 0.5).float().view(-1)
    
    # Calculate confusion matrix elements
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    return tn, fp, fn, tp

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate all evaluation metrics.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        threshold: Threshold for binary segmentation
        
    Returns:
        Dictionary with metrics
    """
    # Apply threshold
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Get confusion matrix elements
    tn, fp, fn, tp = calculate_confusion_matrix(pred_binary, target_binary)
    
    # Calculate metrics
    results = {}
    
    # Basic metrics
    results['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    results['precision'] = tp / (tp + fp + 1e-8)
    results['recall'] = tp / (tp + fn + 1e-8)
    results['specificity'] = tn / (tn + fp + 1e-8)
    results['f1'] = 2 * tp / (2 * tp + fp + fn + 1e-8)
    results['dice'] = 2 * tp / (2 * tp + fp + fn + 1e-8)  # Same as F1 for binary
    results['iou'] = tp / (tp + fp + fn + 1e-8)
    
    # Confusion matrix elements
    results['true_positives'] = tp
    results['true_negatives'] = tn
    results['false_positives'] = fp
    results['false_negatives'] = fn
    
    return results

def calculate_cdr(cup_mask, disc_mask) -> float:
    """Calculate Cup-to-Disc Ratio (CDR).
    
    Args:
        cup_mask: Binary mask of the optic cup
        disc_mask: Binary mask of the optic disc
        
    Returns:
        Cup-to-Disc Ratio
    """
    # Ensure masks are binary
    cup_mask = (cup_mask > 0.5).float()
    disc_mask = (disc_mask > 0.5).float()
    
    # Calculate areas
    cup_area = cup_mask.sum().item()
    disc_area = disc_mask.sum().item()
    
    # Calculate CDR
    cdr = cup_area / (disc_area + 1e-8)
    
    return cdr