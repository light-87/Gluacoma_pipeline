"""
Evaluation Module

This module provides functions for evaluating glaucoma segmentation models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from tqdm import tqdm
import os
import cv2
import pandas as pd

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate evaluation metrics for segmentation.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        threshold: Threshold for binary segmentation
        
    Returns:
        Dictionary with metrics
    """
    # Convert to binary
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Calculate metrics
    # Dice coefficient / F1
    smooth = 1e-6
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    
    # IoU (Jaccard)
    iou = intersection / (pred_binary.sum() + target_binary.sum() - intersection + smooth)
    
    # Confusion matrix elements
    tp = (pred_binary * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    
    # Accuracy
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    f1 = dice  # Same as dice coefficient
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1.item(),
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def calculate_cdr(disc_mask: torch.Tensor, cup_mask: torch.Tensor, method: str = 'diameter') -> float:
    """Calculate Cup-to-Disc Ratio (CDR).
    
    Args:
        disc_mask: Binary disc mask
        cup_mask: Binary cup mask
        method: Method to calculate CDR ('diameter', 'area', or 'both')
        
    Returns:
        CDR value or dictionary of CDR values
    """
    # Ensure binary masks
    disc_mask = (disc_mask > 0.5).float()
    cup_mask = (cup_mask > 0.5).float()
    
    # Convert to numpy array
    disc_mask = disc_mask.cpu().numpy()
    cup_mask = cup_mask.cpu().numpy()
    
    if method == 'area':
        # Area-based CDR
        disc_area = disc_mask.sum()
        cup_area = cup_mask.sum()
        
        if disc_area == 0:
            return 0.0
        
        return cup_area / disc_area
    
    elif method == 'diameter':
        # Diameter-based CDR
        
        # Find contours
        disc_mask_np = (disc_mask[0] * 255).astype(np.uint8)
        cup_mask_np = (cup_mask[0] * 255).astype(np.uint8)
        
        # Find disc contour
        disc_contours, _ = cv2.findContours(disc_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not disc_contours:
            return 0.0
        
        # Find cup contour
        cup_contours, _ = cv2.findContours(cup_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cup_contours:
            return 0.0
        
        # Get largest disc and cup contours
        disc_contour = max(disc_contours, key=cv2.contourArea)
        cup_contour = max(cup_contours, key=cv2.contourArea)
        
        # Fit ellipse to disc and cup
        try:
            disc_ellipse = cv2.fitEllipse(disc_contour)
            cup_ellipse = cv2.fitEllipse(cup_contour)
            
            # Get vertical diameters
            disc_height = max(disc_ellipse[1])
            cup_height = max(cup_ellipse[1])
            
            if disc_height == 0:
                return 0.0
            
            return cup_height / disc_height
        except cv2.error:
            # If ellipse fitting fails, use area
            return calculate_cdr(disc_mask, cup_mask, method='area')
    
    elif method == 'both':
        # Return both methods
        area_cdr = calculate_cdr(disc_mask, cup_mask, method='area')
        try:
            diameter_cdr = calculate_cdr(disc_mask, cup_mask, method='diameter')
        except:
            diameter_cdr = area_cdr
        
        return {
            'area_cdr': area_cdr,
            'diameter_cdr': diameter_cdr,
            'average_cdr': (area_cdr + diameter_cdr) / 2
        }
    
    else:
        raise ValueError(f"Unsupported CDR method: {method}")

def evaluate_model(model: nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  threshold: float = 0.5,
                  calculate_cdr_flag: bool = False,
                  cdr_method: str = 'diameter',
                  use_tta: bool = False) -> Dict[str, Any]:
    """Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        threshold: Threshold for binary segmentation
        calculate_cdr_flag: Whether to calculate CDR
        cdr_method: Method to calculate CDR
        use_tta: Whether to use test-time augmentation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Metrics for tracking
    all_preds = []
    all_masks = []
    all_cdrs = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc='Evaluation')):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (with TTA if enabled)
            if use_tta:
                outputs = test_time_augmentation(model, images)
            else:
                outputs = model(images)
            
            # Get predictions
            preds = torch.sigmoid(outputs)
            
            # Store predictions and masks for metrics calculation
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
            
            # Calculate CDR if requested
            if calculate_cdr_flag:
                for i in range(len(preds)):
                    try:
                        # Assume the target is the disc and the prediction is the cup
                        # (for glaucoma, cup is the region of interest)
                        cdr = calculate_cdr(masks[i], preds[i], method=cdr_method)
                        all_cdrs.append(cdr)
                    except Exception as e:
                        print(f"Error calculating CDR: {e}")
    
    # Concatenate all predictions and masks
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Calculate metrics for each sample
    metrics = {}
    for i in range(len(all_preds)):
        sample_metrics = calculate_metrics(all_preds[i], all_masks[i], threshold=threshold)
        for k, v in sample_metrics.items():
            if k not in metrics:
                metrics[k] = 0
            metrics[k] += v
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(all_preds)
    
    # Add CDR metrics if calculated
    if calculate_cdr_flag and all_cdrs:
        metrics['cdr_mean'] = np.mean(all_cdrs)
        metrics['cdr_std'] = np.std(all_cdrs)
        metrics['cdr_median'] = np.median(all_cdrs)
    
    return metrics

def test_time_augmentation(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Apply test-time augmentation to improve prediction accuracy.
    
    Args:
        model: Model to use for predictions
        images: Input images
        
    Returns:
        Averaged predictions
    """
    # Original prediction
    outputs = model(images)
    
    # Horizontal flip
    flipped = torch.flip(images, dims=[3])
    outputs_flipped = model(flipped)
    outputs_flipped = torch.flip(outputs_flipped, dims=[3])
    
    # Vertical flip
    v_flipped = torch.flip(images, dims=[2])
    outputs_v_flipped = model(v_flipped)
    outputs_v_flipped = torch.flip(outputs_v_flipped, dims=[2])
    
    # Average predictions
    outputs_combined = (outputs + outputs_flipped + outputs_v_flipped) / 3.0
    
    return outputs_combined

def visualize_predictions(images: torch.Tensor, 
                         masks: torch.Tensor, 
                         preds: torch.Tensor,
                         threshold: float = 0.5,
                         max_samples: int = 8,
                         output_dir: Optional[str] = None) -> List[plt.Figure]:
    """Visualize model predictions.
    
    Args:
        images: Input images
        masks: Target masks
        preds: Predicted masks
        threshold: Threshold for binary segmentation
        max_samples: Maximum number of samples to visualize
        output_dir: Directory to save visualizations (if None, not saved)
        
    Returns:
        List of matplotlib figures
    """
    # Ensure tensors are on CPU and detached
    images = images.cpu().detach()
    masks = masks.cpu().detach()
    preds = preds.cpu().detach()
    
    # Apply sigmoid to predictions if they're not already probabilities
    if preds.max() > 1.0 or preds.min() < 0.0:
        preds = torch.sigmoid(preds)
    
    # Create binary predictions
    preds_binary = (preds > threshold).float()
    
    # Determine number of samples to visualize
    num_samples = min(max_samples, len(images))
    
    # Create figures for each sample
    figures = []
    
    for i in range(num_samples):
        # Get sample data
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i, 0].numpy()
        pred = preds[i, 0].numpy()
        pred_binary = preds_binary[i, 0].numpy()
        
        # Denormalize image if needed
        if img.max() <= 1.0:
            # Assuming ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        # Create a figure with subplots
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot original image
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')
        
        # Plot ground truth mask
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')
        
        # Plot prediction probability
        axs[2].imshow(pred, cmap='plasma')
        axs[2].set_title('Prediction (Prob)')
        axs[2].axis('off')
        
        # Plot binary prediction
        axs[3].imshow(pred_binary, cmap='gray')
        axs[3].set_title('Prediction (Binary)')
        axs[3].axis('off')
        
        plt.tight_layout()
        
        # Save figure if output_dir is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f'visualization_{i}.png')
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        
        figures.append(fig)
    
    return figures