"""
Evaluation Module

Implements comprehensive evaluation of glaucoma detection models.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

from glaucoma.evaluation.metrics import (
    calculate_dice_coefficient,
    calculate_iou,
    calculate_metrics,
    calculate_cdr
)
from glaucoma.evaluation.visualization import VisualizationManager
from glaucoma.utils.logging import PipelineLogger


class Evaluator:
    """Class for evaluating model performance."""
    
    def __init__(
        self, 
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        config: Any,
        logger: PipelineLogger,
        output_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            dataloader: Test/validation dataloader
            config: Evaluation configuration
            logger: Logger for tracking progress
            output_dir: Directory to save evaluation results
            wandb_logger: Optional WandB logger
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.logger = logger
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_dir if hasattr(config, 'output_dir') else "output")
        self.results_dir = self.output_dir / "evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_logger = wandb_logger
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(config, 'use_gpu', True) else 'cpu')
        
        # Set threshold
        self.threshold = getattr(config, 'threshold', 0.5)
        
        # Create visualization manager
        self.viz_manager = VisualizationManager(str(self.results_dir))
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on the dataloader.
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting evaluation")
        self.model.eval()
        
        # Lists to store predictions, masks, and images
        all_preds = []
        all_masks = []
        all_images = []
        
        # For cup-to-disc ratio calculation
        cdrs = []
        
        # For confusion matrix elements
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        
        batch_metrics = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                
                # Apply threshold to get binary predictions
                preds = (probs > self.threshold).float()
                
                # Calculate batch metrics
                batch_result = self._calculate_batch_metrics(preds, masks)
                batch_metrics.append(batch_result)
                
                # Update confusion matrix elements
                total_tp += batch_result.get('true_positives', 0)
                total_fp += batch_result.get('false_positives', 0)
                total_tn += batch_result.get('true_negatives', 0)
                total_fn += batch_result.get('false_negatives', 0)
                
                # Store predictions, masks, and images for later visualization
                # Detach and move to CPU to avoid GPU memory issues
                all_preds.append(probs.cpu())
                all_masks.append(masks.cpu())
                all_images.append(images.cpu())
                
                # Calculate cup-to-disc ratio if configured
                if getattr(self.config, 'calculate_cdr', False):
                    batch_cdrs = [calculate_cdr(pred, mask) for pred, mask in zip(preds, masks)]
                    cdrs.extend(batch_cdrs)
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Evaluated {batch_idx + 1} batches")
        
        # Concatenate all tensors
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        # Calculate overall metrics
        results = self._calculate_overall_metrics(batch_metrics)
        
        # Add confusion matrix elements
        results['confusion_matrix'] = [[total_tn, total_fp], [total_fn, total_tp]]
        
        # Add cup-to-disc ratio if calculated
        if cdrs:
            results['mean_cdr'] = np.mean(cdrs)
            results['median_cdr'] = np.median(cdrs)
            results['cdr_std'] = np.std(cdrs)
            results['cdr_values'] = cdrs
        
        # Generate visualizations if configured
        if getattr(self.config, 'generate_visualizations', True):
            self._generate_visualizations(all_images, all_masks, all_preds, results, cdrs)
        
        # Save results to CSV
        self._save_results(results)
        
        # Log results to wandb if available
        if self.wandb_logger:
            self.wandb_logger.log(
                {k: v for k, v in results.items() if k != 'cdr_values' and not isinstance(v, list)}
            )
            
            # Log confusion matrix
            self.wandb_logger.log_confusion_matrix(
                np.array(results['confusion_matrix']), 
                class_names=['Background', 'Glaucoma']
            )
            
            # Log sample predictions
            self.wandb_logger.log_images(
                all_images[:10], 
                all_masks[:10], 
                all_preds[:10],
                num_samples=min(10, len(all_images))
            )
        
        self.logger.info("Evaluation completed")
        return results
    
    def _calculate_batch_metrics(self, preds: torch.Tensor, masks: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for a batch.
        
        Args:
            preds: Prediction tensor (binary)
            masks: Ground truth tensor
            
        Returns:
            Dictionary with batch metrics
        """
        # Ensure inputs are on CPU for metric calculation
        preds_cpu = preds.cpu()
        masks_cpu = masks.cpu()
        
        # Calculate metrics
        return calculate_metrics(preds_cpu, masks_cpu, threshold=self.threshold)
    
    def _calculate_overall_metrics(self, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall metrics from batch metrics.
        
        Args:
            batch_metrics: List of batch metric dictionaries
            
        Returns:
            Dictionary with overall metrics
        """
        # Initialize results
        results = {}
        
        # Get metrics to average
        metrics_to_average = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity', 'f1']
        
        # Calculate average metrics
        for metric in metrics_to_average:
            values = [batch.get(metric, 0) for batch in batch_metrics if metric in batch]
            if values:
                results[metric] = np.mean(values)
        
        return results
    
    def _generate_visualizations(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor, 
        predictions: torch.Tensor,
        results: Dict[str, Any],
        cdrs: List[float] = None
    ) -> None:
        """Generate and save visualizations.
        
        Args:
            images: Batch of images
            masks: Batch of masks
            predictions: Batch of predictions
            results: Evaluation results
            cdrs: List of CDR values
        """
        self.logger.info("Generating visualizations")
        
        # Plot sample predictions
        num_samples = min(getattr(self.config, 'num_visualization_samples', 10), len(images))
        self.viz_manager.plot_sample_predictions(
            images.numpy(), 
            masks.numpy(), 
            predictions.numpy(),
            num_samples=num_samples
        )
        
        # Calculate and plot ROC curve
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(masks.numpy().flatten(), predictions.numpy().flatten())
            roc_auc = auc(fpr, tpr)
            self.viz_manager.plot_roc_curve(fpr, tpr, roc_auc)
        except Exception as e:
            self.logger.warning(f"Error calculating ROC curve: {e}")
        
        # Calculate and plot PR curve
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(masks.numpy().flatten(), predictions.numpy().flatten())
            pr_auc = average_precision_score(masks.numpy().flatten(), predictions.numpy().flatten())
            self.viz_manager.plot_pr_curve(precision, recall, pr_auc)
        except Exception as e:
            self.logger.warning(f"Error calculating PR curve: {e}")
        
        # Plot confusion matrix
        if 'confusion_matrix' in results:
            self.viz_manager.plot_confusion_matrix(np.array(results['confusion_matrix']))
        
        # Plot CDR distribution if available
        if cdrs and len(cdrs) > 0:
            # Create dummy labels - in a real scenario, we would have actual labels
            labels = np.zeros(len(cdrs))
            self.viz_manager.plot_cdr_distribution(cdrs, labels)
        
        # Generate HTML report
        report_path = self.viz_manager.generate_report(results)
        self.logger.info(f"Generated visualization report at {report_path}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to CSV.
        
        Args:
            results: Evaluation results
        """
        # Filter out non-scalar values
        scalar_results = {k: v for k, v in results.items() 
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
        
        # Create dataframe
        df = pd.DataFrame([scalar_results])
        
        # Save to CSV
        csv_path = self.results_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved evaluation results to {csv_path}")
        
        # Also save as JSON for easier loading
        json_path = self.results_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            import json
            # Handle numpy values
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_results[k] = float(v)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.integer, np.floating)):
                    serializable_results[k] = [float(x) for x in v]
                elif isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                else:
                    # Skip non-serializable values
                    if not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) == 0):
                        serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=4)
        self.logger.info(f"Saved evaluation results to {json_path}")
    
    def evaluate_ensemble(self, models: List[torch.nn.Module]) -> Dict[str, Any]:
        """Evaluate an ensemble of models.
        
        Args:
            models: List of models to ensemble
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating ensemble of {len(models)} models")
        
        # Lists to store predictions, masks, and images
        all_masks = []
        all_images = []
        all_model_preds = []
        
        # Move models to device
        models = [model.to(self.device) for model in models]
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.dataloader, desc="Evaluating Ensemble")):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Store masks and images
                all_masks.append(masks.cpu())
                all_images.append(images.cpu())
                
                # Get predictions from each model
                model_preds = []
                for model in models:
                    model.eval()
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    model_preds.append(probs.cpu())
                
                all_model_preds.append(model_preds)
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Evaluated {batch_idx + 1} batches")
        
        # Concatenate all tensors
        all_masks = torch.cat(all_masks, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        # Evaluate different ensemble methods
        ensemble_results = {}
        
        # Average ensemble
        avg_preds = self._average_ensemble_predictions(all_model_preds)
        avg_results = self._calculate_batch_metrics((avg_preds > self.threshold).float(), all_masks)
        ensemble_results['average'] = avg_results
        
        # Max ensemble
        max_preds = self._max_ensemble_predictions(all_model_preds)
        max_results = self._calculate_batch_metrics((max_preds > self.threshold).float(), all_masks)
        ensemble_results['maximum'] = max_results
        
        # Weighted ensemble (equal weights as a starting point)
        weighted_preds = self._weighted_ensemble_predictions(all_model_preds, weights=None)
        weighted_results = self._calculate_batch_metrics((weighted_preds > self.threshold).float(), all_masks)
        ensemble_results['weighted'] = weighted_results
        
        # Find best ensemble method
        best_method = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['dice'])
        best_results = ensemble_results[best_method]
        best_preds = locals()[f"{best_method.split('_')[0]}_preds"]
        
        self.logger.info(f"Best ensemble method: {best_method} with Dice score {best_results['dice']:.4f}")
        
        # Generate visualizations for the best method
        if getattr(self.config, 'generate_visualizations', True):
            self._generate_visualizations(all_images, all_masks, best_preds, best_results)
        
        # Save results
        self._save_results(best_results)
        
        # Add ensemble method to results
        best_results['ensemble_method'] = best_method
        
        return best_results
    
    def _average_ensemble_predictions(self, all_model_preds: List[List[torch.Tensor]]) -> torch.Tensor:
        """Average predictions from multiple models.
        
        Args:
            all_model_preds: List of model predictions for each batch
            
        Returns:
            Averaged predictions
        """
        # Initialize with the structure of the first batch
        batch_averaged_preds = []
        
        for batch_preds in all_model_preds:
            # Stack predictions from all models for this batch
            stacked = torch.stack(batch_preds)
            # Average along model dimension
            batch_averaged_preds.append(torch.mean(stacked, dim=0))
        
        # Concatenate all batches
        return torch.cat(batch_averaged_preds, dim=0)
    
    def _max_ensemble_predictions(self, all_model_preds: List[List[torch.Tensor]]) -> torch.Tensor:
        """Take maximum prediction from multiple models.
        
        Args:
            all_model_preds: List of model predictions for each batch
            
        Returns:
            Maximum predictions
        """
        batch_max_preds = []
        
        for batch_preds in all_model_preds:
            # Stack predictions from all models for this batch
            stacked = torch.stack(batch_preds)
            # Take maximum along model dimension
            batch_max_preds.append(torch.max(stacked, dim=0)[0])
        
        # Concatenate all batches
        return torch.cat(batch_max_preds, dim=0)
    
    def _weighted_ensemble_predictions(
        self, 
        all_model_preds: List[List[torch.Tensor]], 
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute weighted predictions from multiple models.
        
        Args:
            all_model_preds: List of model predictions for each batch
            weights: Optional list of weights for each model
            
        Returns:
            Weighted predictions
        """
        num_models = len(all_model_preds[0])
        
        # If weights not provided, use equal weights
        if weights is None:
            weights = [1.0 / num_models] * num_models
        
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]
        
        batch_weighted_preds = []
        
        for batch_preds in all_model_preds:
            # Compute weighted sum
            weighted_sum = sum(w * pred for w, pred in zip(weights, batch_preds))
            batch_weighted_preds.append(weighted_sum)
        
        # Concatenate all batches
        return torch.cat(batch_weighted_preds, dim=0)