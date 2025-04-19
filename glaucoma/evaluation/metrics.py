"""
Metrics Module

Implements comprehensive evaluation metrics for glaucoma detection.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
import sklearn.metrics as skmetrics

def calculate_dice_coefficient(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    smooth: float = 1e-6,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Calculate Dice coefficient.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing factor
        reduction: Reduction method ('mean', 'none')
        
    Returns:
        Dice coefficient
    """
    # Ensure inputs are binary
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # Handle different input shapes
    if pred.dim() == 4 and pred.size(1) == 1:  # (B, 1, H, W)
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:  # (B, 1, H, W)
        target = target.squeeze(1)
    
    # Flatten if needed
    if pred.dim() == 3:  # (B, H, W)
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
    
    # Calculate intersection and sum
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Apply reduction
    if reduction == 'mean':
        return dice.mean()
    else:
        return dice

def calculate_iou(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    smooth: float = 1e-6,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Calculate IoU (Jaccard index).
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        smooth: Smoothing factor
        reduction: Reduction method ('mean', 'none')
        
    Returns:
        IoU coefficient
    """
    # Ensure inputs are binary
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # Handle different input shapes
    if pred.dim() == 4 and pred.size(1) == 1:  # (B, 1, H, W)
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:  # (B, 1, H, W)
        target = target.squeeze(1)
    
    # Flatten if needed
    if pred.dim() == 3:  # (B, H, W)
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).sum(dim=1) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Apply reduction
    if reduction == 'mean':
        return iou.mean()
    else:
        return iou

def calculate_confusion_matrix_elements(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> Tuple[float, float, float, float]:
    """Calculate confusion matrix elements.
    
    Args:
        pred: Prediction tensor (binary)
        target: Target tensor (binary)
        
    Returns:
        Tuple of (tn, fp, fn, tp)
    """
    # Ensure inputs are binary and flattened
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # Handle different input shapes
    if pred.dim() == 4 and pred.size(1) == 1:  # (B, 1, H, W)
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:  # (B, 1, H, W)
        target = target.squeeze(1)
    
    # Flatten all dimensions
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate confusion matrix elements
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    return tn, fp, fn, tp

def calculate_segmentation_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate segmentation evaluation metrics.
    
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
    
    # Calculate basic metrics
    dice = calculate_dice_coefficient(pred_binary, target_binary).item()
    iou = calculate_iou(pred_binary, target_binary).item()
    
    # Get confusion matrix elements
    tn, fp, fn, tp = calculate_confusion_matrix_elements(pred_binary, target_binary)
    
    # Calculate additional metrics
    total = tp + tn + fp + fn
    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def calculate_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    mode: str = 'segmentation'
) -> Dict[str, float]:
    """Calculate all evaluation metrics.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        threshold: Threshold for binary segmentation
        mode: Evaluation mode ('segmentation' or 'classification')
        
    Returns:
        Dictionary with metrics
    """
    if mode == 'segmentation':
        return calculate_segmentation_metrics(pred, target, threshold)
    else:
        # For classification, use classification metrics
        return calculate_classification_metrics(pred, target, threshold)

def calculate_classification_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate classification evaluation metrics.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with metrics
    """
    # Ensure predictions and targets are flat
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Convert to numpy for sklearn metrics
    pred_np = pred.numpy()
    target_np = target.numpy()
    
    # Apply threshold
    pred_binary = (pred_np > threshold).astype(float)
    
    # Calculate metrics
    accuracy = skmetrics.accuracy_score(target_np, pred_binary)
    
    # Only calculate these if we have positive examples
    if np.sum(target_np) > 0:
        precision = skmetrics.precision_score(target_np, pred_binary)
        recall = skmetrics.recall_score(target_np, pred_binary)
        f1 = skmetrics.f1_score(target_np, pred_binary)
        try:
            roc_auc = skmetrics.roc_auc_score(target_np, pred_np)
        except:
            roc_auc = 0.5  # Default for random
        try:
            pr_auc = skmetrics.average_precision_score(target_np, pred_np)
        except:
            pr_auc = np.sum(target_np) / len(target_np)  # Default for random
    else:
        precision = recall = f1 = roc_auc = pr_auc = 0.0
    
    # Get confusion matrix
    tn, fp, fn, tp = skmetrics.confusion_matrix(target_np, pred_binary, labels=[0, 1]).ravel()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

def calculate_roc_curve(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate ROC curve.
    
    Args:
        pred: Prediction tensor/array
        target: Target tensor/array
        
    Returns:
        Tuple of (fpr, tpr, roc_auc)
    """
    # Convert to numpy if tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    
    # Flatten
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = skmetrics.roc_curve(target, pred)
    roc_auc = skmetrics.auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def calculate_pr_curve(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate precision-recall curve.
    
    Args:
        pred: Prediction tensor/array
        target: Target tensor/array
        
    Returns:
        Tuple of (precision, recall, pr_auc)
    """
    # Convert to numpy if tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    
    # Flatten
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate PR curve
    precision, recall, _ = skmetrics.precision_recall_curve(target, pred)
    pr_auc = skmetrics.average_precision_score(target, pred)
    
    return precision, recall, pr_auc

def calculate_cdr(
    cup_mask: Union[torch.Tensor, np.ndarray], 
    disc_mask: Union[torch.Tensor, np.ndarray]
) -> float:
    """Calculate Cup-to-Disc Ratio (CDR).
    
    Args:
        cup_mask: Binary mask of the optic cup
        disc_mask: Binary mask of the optic disc
        
    Returns:
        Cup-to-Disc Ratio
    """
    # Convert to numpy if tensors
    if isinstance(cup_mask, torch.Tensor):
        cup_mask = cup_mask.cpu().numpy()
    if isinstance(disc_mask, torch.Tensor):
        disc_mask = disc_mask.cpu().numpy()
    
    # Ensure masks are binary
    cup_mask = (cup_mask > 0.5).astype(float)
    disc_mask = (disc_mask > 0.5).astype(float)
    
    # Calculate areas
    cup_area = np.sum(cup_mask)
    disc_area = np.sum(disc_mask)
    
    # Calculate CDR
    cdr = cup_area / (disc_area + 1e-8)
    
    return cdr

def calculate_hausdorff_distance(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> float:
    """Calculate Hausdorff distance between prediction and target.
    
    Args:
        pred: Prediction tensor/array
        target: Target tensor/array
        
    Returns:
        Hausdorff distance
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # Convert to numpy if tensors
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # Ensure binary
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        
        # Get contours
        pred_points = np.argwhere(pred)
        target_points = np.argwhere(target)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate Hausdorff distance
        forward = directed_hausdorff(pred_points, target_points)[0]
        backward = directed_hausdorff(target_points, pred_points)[0]
        
        return max(forward, backward)
    except Exception as e:
        print(f"Error calculating Hausdorff distance: {e}")
        return float('inf')

def calculate_surface_distance(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """Calculate surface distance metrics.
    
    Args:
        pred: Prediction tensor/array
        target: Target tensor/array
        
    Returns:
        Dictionary with surface distance metrics
    """
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Convert to numpy if tensors
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # Ensure binary
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        
        # Get contours
        pred_contour = pred - np.logical_and(pred, binary_erosion(pred))
        target_contour = target - np.logical_and(target, binary_erosion(target))
        
        # Get distance transforms
        dt_pred = distance_transform_edt(~pred_contour)
        dt_target = distance_transform_edt(~target_contour)
        
        # Get surface distances
        pred_to_target = dt_target[pred_contour > 0]
        target_to_pred = dt_pred[target_contour > 0]
        
        # Calculate metrics
        metrics = {}
        metrics['avg_symmetric_surface_distance'] = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2
        metrics['max_symmetric_surface_distance'] = max(np.max(pred_to_target), np.max(target_to_pred))
        
        return metrics
    except Exception as e:
        print(f"Error calculating surface distance: {e}")
        return {'avg_symmetric_surface_distance': float('inf'), 'max_symmetric_surface_distance': float('inf')}

def binary_erosion(binary_mask):
    """Simple binary erosion for contour extraction."""
    from scipy import ndimage
    return ndimage.binary_erosion(binary_mask)