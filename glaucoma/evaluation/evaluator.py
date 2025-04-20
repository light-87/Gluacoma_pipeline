"""
Evaluator Module

Implements comprehensive evaluation for glaucoma detection models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import torch.nn.functional as F

from glaucoma.evaluation.metrics import (
    calculate_metrics, calculate_roc_curve, calculate_pr_curve, 
    calculate_confusion_matrix_elements
)
from glaucoma.evaluation.visualization import VisualizationManager

class Evaluator:
    """Evaluator class for model evaluation."""
    
    def __init__(
        self, 
        model: Optional[torch.nn.Module], 
        dataloader: torch.utils.data.DataLoader, 
        config: Any, 
        logger: Any, 
        output_dir: str,
        wandb_logger: Optional[Any] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            dataloader: Test dataloader
            config: Evaluation configuration
            logger: Logger for tracking progress
            output_dir: Directory to save evaluation results
            wandb_logger: Optional Weights & Biases logger
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.wandb_logger = wandb_logger
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(config, 'use_gpu', True) else 'cpu')
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        # Initialize visualization manager
        self.visualization_manager = VisualizationManager(str(self.output_dir))
        
        # Set threshold for binary segmentation
        self.threshold = getattr(config, 'threshold', 0.5)
        
        # Test-time augmentation settings
        self.use_tta = getattr(config, 'use_tta', False)
        self.tta_scales = getattr(config, 'tta_scales', [0.75, 1.0, 1.25])
        self.tta_flips = getattr(config, 'tta_flips', True)
        self.tta_rotations = getattr(config, 'tta_rotations', [0, 90, 180, 270])
        
        if self.use_tta:
            self.logger.info(f"Using test-time augmentation with scales={self.tta_scales}, "
                          f"flips={self.tta_flips}, rotations={self.tta_rotations}")
    
    def _apply_tta(self, image: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple]]:
        """Apply test-time augmentations to an image.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (augmented_images, transforms_info)
            - augmented_images: List of augmented image tensors
            - transforms_info: List of tuples (scale, flip, rotation) for each augmentation
        """
        B, C, H, W = image.shape
        augmented_images = []
        transforms_info = []
        
        # Loop through scales, flips, and rotations
        for scale in self.tta_scales:
            # Scale the image
            if scale != 1.0:
                # Calculate new dimensions
                new_h, new_w = int(H * scale), int(W * scale)
                # Scale the image
                scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
                # Pad or crop to original size
                if scale > 1.0:  # Crop
                    diff_h, diff_w = new_h - H, new_w - W
                    start_h, start_w = diff_h // 2, diff_w // 2
                    scaled_img = scaled_img[:, :, start_h:start_h+H, start_w:start_w+W]
                else:  # Pad
                    diff_h, diff_w = H - new_h, W - new_w
                    pad_h1, pad_w1 = diff_h // 2, diff_w // 2
                    pad_h2, pad_w2 = diff_h - pad_h1, diff_w - pad_w1
                    scaled_img = F.pad(scaled_img, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect')
            else:
                scaled_img = image
            
            # Apply flips
            if self.tta_flips:
                # Original (no flip)
                augmented_images.append(scaled_img)
                transforms_info.append((scale, False, 0))
                
                # Horizontal flip
                flipped_img = torch.flip(scaled_img, dims=[-1])
                augmented_images.append(flipped_img)
                transforms_info.append((scale, True, 0))
            else:
                # Only use original (no flip)
                augmented_images.append(scaled_img)
                transforms_info.append((scale, False, 0))
            
            # Apply rotations
            if self.tta_rotations and len(self.tta_rotations) > 1:
                for angle in self.tta_rotations[1:]:  # Skip 0 rotation as it's already included
                    # Rotate image
                    k = angle // 90  # Number of 90-degree rotations
                    rotated_img = torch.rot90(scaled_img, k=k, dims=[-2, -1])
                    augmented_images.append(rotated_img)
                    transforms_info.append((scale, False, angle))
                    
                    # Rotate and flip
                    if self.tta_flips:
                        rotated_flipped_img = torch.flip(rotated_img, dims=[-1])
                        augmented_images.append(rotated_flipped_img)
                        transforms_info.append((scale, True, angle))
        
        return augmented_images, transforms_info
    
    def _reverse_augmentation(self, pred: torch.Tensor, transform_info: Tuple) -> torch.Tensor:
        """Reverse the augmentation to get the prediction in original space.
        
        Args:
            pred: Prediction tensor
            transform_info: Tuple of (scale, flip, rotation)
            
        Returns:
            Reversed prediction tensor
        """
        scale, flipped, angle = transform_info
        
        # Un-rotate
        if angle != 0:
            k = 4 - (angle // 90) % 4  # Inverse rotation (360 - angle) in terms of 90-degree units
            pred = torch.rot90(pred, k=k, dims=[-2, -1])
        
        # Un-flip
        if flipped:
            pred = torch.flip(pred, dims=[-1])
        
        # Un-scale
        if scale != 1.0:
            _, _, H, W = pred.shape
            orig_h, orig_w = int(H / scale), int(W / scale)
            
            if scale > 1.0:  # Was cropped, now pad
                diff_h, diff_w = int(H * scale) - H, int(W * scale) - W
                pad_h1, pad_w1 = diff_h // 2, diff_w // 2
                pad_h2, pad_w2 = diff_h - pad_h1, diff_w - pad_w1
                pred = F.pad(pred, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect')
                # Now resize back to original
                pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            else:  # Was padded, now crop
                # First resize back to original
                pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                # Then crop the padding
                diff_h, diff_w = orig_h - int(H * scale), orig_w - int(W * scale)
                start_h, start_w = diff_h // 2, diff_w // 2
                pred = pred[:, :, start_h:start_h+H, start_w:start_w+W]
        
        return pred
        
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on dataloader.
        
        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            self.logger.error("No model provided for evaluation")
            return {}
        
        self.logger.info("Starting evaluation")
        self.model.eval()
        
        # Storage for predictions and ground truth
        all_preds = []
        all_masks = []
        all_images = []
        
        # Metrics for accumulation
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0, 
            'f1': 0.0
        }
        
        # Total number of samples and pixels
        num_samples = 0
        confusion_matrix = np.zeros((2, 2), dtype=np.int64)  # [[tn, fp], [fn, tp]]
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                # Move inputs to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if self.use_tta:
                    # Get predictions with test-time augmentation
                    batch_preds = self._predict_with_tta(images)
                else:
                    # Get model predictions without TTA
                    outputs = self.model(images)
                    
                    # Convert to numpy for metrics calculation
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle models that return multiple outputs
                    
                    # Apply sigmoid for binary segmentation
                    batch_preds = torch.sigmoid(outputs)
                
                # Store for later visualization and metric calculation
                all_preds.append(batch_preds.cpu())
                all_masks.append(masks.cpu())
                all_images.append(images.cpu())
                
                # Calculate metrics for this batch
                for i in range(images.size(0)):
                    pred = batch_preds[i].cpu()
                    mask = masks[i].cpu()
                    
                    # Calculate metrics
                    batch_metrics = calculate_metrics(pred, mask, threshold=self.threshold)
                    
                    # Update metrics sum
                    for metric, value in batch_metrics.items():
                        if metric in metrics_sum:
                            metrics_sum[metric] += value
                    
                    # Update confusion matrix elements
                    tn, fp, fn, tp = calculate_confusion_matrix_elements(pred, mask)
                    confusion_matrix[0, 0] += tn
                    confusion_matrix[0, 1] += fp
                    confusion_matrix[1, 0] += fn
                    confusion_matrix[1, 1] += tp
                    
                    num_samples += 1
    
    def _predict_with_tta(self, images: torch.Tensor) -> torch.Tensor:
        """Make predictions with test-time augmentation.
        
        Args:
            images: Input image tensor
            
        Returns:
            Aggregated prediction after test-time augmentation
        """
        batch_size = images.size(0)
        output_shape = list(images.shape)
        output_shape[1] = 1  # Change channel dim to 1 for output mask
        
        # Initialize tensors for accumulated predictions
        accumulated_preds = torch.zeros(output_shape).to(self.device)
        accumulation_count = torch.zeros(output_shape).to(self.device)
        
        # Process each image in the batch
        for i in range(batch_size):
            img = images[i:i+1]  # Keep batch dimension
            
            # Apply augmentations
            augmented_imgs, transforms_info = self._apply_tta(img)
            
            # Get predictions for each augmentation
            for aug_img, transform_info in zip(augmented_imgs, transforms_info):
                # Pass through model
                aug_output = self.model(aug_img)
                
                if isinstance(aug_output, tuple):
                    aug_output = aug_output[0]
                    
                # Apply sigmoid
                aug_pred = torch.sigmoid(aug_output)
                
                # Revert the augmentation
                reverted_pred = self._reverse_augmentation(aug_pred, transform_info)
                
                # Accumulate predictions
                accumulated_preds[i:i+1] += reverted_pred
                accumulation_count[i:i+1] += 1
        
        # Average the predictions
        batch_preds = accumulated_preds / accumulation_count
        
        return batch_preds
        
        # Calculate average metrics
        metrics_avg = {k: v / num_samples for k, v in metrics_sum.items()}
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = confusion_matrix.flatten()
        total_pixels = np.sum(confusion_matrix)
        
        # Prepare final evaluation results
        eval_results = {
            'metrics': metrics_avg,
            'confusion_matrix': confusion_matrix.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': num_samples,
            'total_pixels': int(total_pixels),
            'threshold': self.threshold
        }
        
        # Flatten results for top-level access
        for metric, value in metrics_avg.items():
            eval_results[metric] = value
        
        # Log evaluation results
        self._log_results(eval_results)
        
        # Generate visualizations
        if getattr(self.config, 'generate_visualizations', True):
            self._generate_visualizations(all_images, all_masks, all_preds)
        
        # Save evaluation results to file
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(
                {k: v for k, v in eval_results.items() if not isinstance(v, np.ndarray)},
                f, indent=4
            )
        
        self.logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        return eval_results
    
    def evaluate_ensemble(self, models: List[torch.nn.Module]) -> Dict[str, Any]:
        """Evaluate ensemble of models.
        
        Args:
            models: List of models in the ensemble
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Starting ensemble evaluation with {len(models)} models")
        
        # Move models to device
        models = [model.to(self.device) for model in models]
        
        # Set models to evaluation mode
        for model in models:
            model.eval()
        
        # Storage for predictions and ground truth
        all_ensemble_preds = []
        all_masks = []
        all_images = []
        
        # Metrics for accumulation
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'accuracy': 0.0,
            'precision': 0.0, 
            'recall': 0.0,
            'f1': 0.0
        }
        
        # Total number of samples
        num_samples = 0
        confusion_matrix = np.zeros((2, 2), dtype=np.int64)  # [[tn, fp], [fn, tp]]
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.dataloader, desc="Evaluating ensemble")):
                # Move inputs to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions from each model
                model_preds = []
                for model in models:
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Apply sigmoid for binary segmentation
                    model_preds.append(torch.sigmoid(outputs))
                
                # Ensemble predictions (simple average)
                ensemble_preds = torch.stack(model_preds).mean(dim=0)
                
                # Store for later visualization and metric calculation
                all_ensemble_preds.append(ensemble_preds.cpu())
                all_masks.append(masks.cpu())
                all_images.append(images.cpu())
                
                # Calculate metrics for this batch
                for i in range(images.size(0)):
                    pred = ensemble_preds[i].cpu()
                    mask = masks[i].cpu()
                    
                    # Calculate metrics
                    batch_metrics = calculate_metrics(pred, mask, threshold=self.threshold)
                    
                    # Update metrics sum
                    for metric, value in batch_metrics.items():
                        if metric in metrics_sum:
                            metrics_sum[metric] += value
                    
                    # Update confusion matrix elements
                    tn, fp, fn, tp = calculate_confusion_matrix_elements(pred, mask)
                    confusion_matrix[0, 0] += tn
                    confusion_matrix[0, 1] += fp
                    confusion_matrix[1, 0] += fn
                    confusion_matrix[1, 1] += tp
                    
                    num_samples += 1
        
        # Calculate average metrics
        metrics_avg = {k: v / num_samples for k, v in metrics_sum.items()}
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = confusion_matrix.flatten()
        
        # Prepare final evaluation results
        eval_results = {
            'metrics': metrics_avg,
            'confusion_matrix': confusion_matrix.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_samples': num_samples,
            'threshold': self.threshold,
            'ensemble_size': len(models)
        }
        
        # Flatten results for top-level access
        for metric, value in metrics_avg.items():
            eval_results[metric] = value
        
        # Log evaluation results
        self._log_results(eval_results, is_ensemble=True)
        
        # Generate visualizations
        if getattr(self.config, 'generate_visualizations', True):
            self._generate_visualizations(
                all_images, all_masks, all_ensemble_preds, 
                prefix="ensemble_"
            )
        
        # Save evaluation results to file
        results_path = self.output_dir / "ensemble_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(
                {k: v for k, v in eval_results.items() if not isinstance(v, np.ndarray)},
                f, indent=4
            )
        
        self.logger.info(f"Ensemble evaluation completed. Results saved to {results_path}")
        
        return eval_results
    
    def _log_results(self, results: Dict[str, Any], is_ensemble: bool = False) -> None:
        """Log evaluation results.
        
        Args:
            results: Dictionary with evaluation results
            is_ensemble: Whether the results are from ensemble evaluation
        """
        # Log to regular logger
        prefix = "Ensemble " if is_ensemble else ""
        self.logger.info(f"{prefix}Evaluation Results:")
        
        # Log key metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in results['metrics'].items()])
        self.logger.info(f"Metrics: {metrics_str}")
        
        # Log confusion matrix
        cm = results['confusion_matrix']
        self.logger.info(f"Confusion Matrix: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
        
        # Log to wandb if available
        if self.wandb_logger is not None:
            log_dict = {
                f"{prefix.lower()}eval_{k}": v 
                for k, v in results.items() 
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            self.wandb_logger.log(log_dict)
            
            # Log confusion matrix
            self.wandb_logger.log_confusion_matrix(
                np.array(results['confusion_matrix']),
                class_names=['Background', 'Glaucoma']
            )
    
    def _generate_visualizations(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor], 
        preds: List[torch.Tensor],
        prefix: str = ""
    ) -> None:
        """Generate visualizations from evaluation results.
        
        Args:
            images: List of image tensors
            masks: List of ground truth mask tensors
            preds: List of prediction tensors
            prefix: Prefix for output filenames
        """
        # Calculate number of samples to visualize
        num_samples = min(
            getattr(self.config, 'sample_count', 10),
            len(self.dataloader.dataset)
        )
        
        # Concatenate tensors from batches
        all_images = torch.cat(images, dim=0).numpy()
        all_masks = torch.cat(masks, dim=0).numpy()
        all_preds = torch.cat(preds, dim=0).numpy()
        
        # Generate sample prediction visualizations
        self.visualization_manager.plot_sample_predictions(
            all_images[:num_samples],
            all_masks[:num_samples],
            all_preds[:num_samples],
            output_filename=f"{prefix}sample_predictions.png"
        )
        
        # Prepare data for ROC and PR curves
        all_masks_flat = all_masks.reshape(-1)
        all_preds_flat = all_preds.reshape(-1)
        
        # Generate ROC curve
        fpr, tpr, roc_auc = calculate_roc_curve(all_preds_flat, all_masks_flat)
        self.visualization_manager.plot_roc_curve(
            fpr, tpr, roc_auc,
            output_filename=f"{prefix}roc_curve.png"
        )
        
        # Generate PR curve
        precision, recall, pr_auc = calculate_pr_curve(all_preds_flat, all_masks_flat)
        self.visualization_manager.plot_pr_curve(
            precision, recall, pr_auc,
            output_filename=f"{prefix}pr_curve.png"
        )
        
        # Generate confusion matrix visualization
        tn, fp, fn, tp = calculate_confusion_matrix_elements(
            torch.tensor(all_preds_flat), 
            torch.tensor(all_masks_flat)
        )
        cm = np.array([[tn, fp], [fn, tp]])
        self.visualization_manager.plot_confusion_matrix(
            cm,
            output_filename=f"{prefix}confusion_matrix.png"
        )
        
        # Generate HTML report
        self.visualization_manager.generate_report(
            {
                'dice': np.mean([calculate_metrics(p, m, threshold=self.threshold)['dice'] 
                              for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'iou': np.mean([calculate_metrics(p, m, threshold=self.threshold)['iou'] 
                             for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'accuracy': np.mean([calculate_metrics(p, m, threshold=self.threshold)['accuracy'] 
                                 for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'precision': np.mean([calculate_metrics(p, m, threshold=self.threshold)['precision'] 
                                   for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'recall': np.mean([calculate_metrics(p, m, threshold=self.threshold)['recall'] 
                                for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'f1': np.mean([calculate_metrics(p, m, threshold=self.threshold)['f1'] 
                           for p, m in zip(all_preds[:num_samples], all_masks[:num_samples])]),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            },
            output_filename=f"{prefix}evaluation_report.html"
        )
        
        # Log to wandb if available
        if self.wandb_logger is not None:
            # Log ROC curve
            self.wandb_logger.log_roc_curve(fpr, tpr, roc_auc)
            
            # Log PR curve
            self.wandb_logger.log_pr_curve(precision, recall, pr_auc)
            
            # Log sample images
            indices = np.random.choice(len(all_images), min(5, len(all_images)), replace=False)
            self.wandb_logger.log_images(
                [all_images[i] for i in indices],
                [all_masks[i] for i in indices],
                [all_preds[i] for i in indices],
                num_samples=len(indices)
            )