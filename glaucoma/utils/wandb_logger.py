"""
Weights & Biases Integration Module

Implements comprehensive Weights & Biases integration for experiment tracking
with enhanced visualization capabilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

class WandBLogger:
    """Logger for Weights & Biases integration with enhanced visualizations."""
    
    def __init__(
        self, 
        config: Any, 
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """Initialize WandB logger.
        
        Args:
            config: Configuration object
            project_name: Optional project name (overrides config)
            run_name: Optional run name (overrides config)
            tags: Optional list of tags
            notes: Optional run notes
        """
        self.config = config
        self.enabled = getattr(config.logging, 'use_wandb', False)
        
        # Get project name
        if project_name is None:
            self.project_name = getattr(config.logging, 'wandb_project', 'glaucoma-detection')
        else:
            self.project_name = project_name
        
        # Get run name
        if run_name is None:
            self.run_name = getattr(config.logging, 'run_name', None)
        else:
            self.run_name = run_name
        
        # Get tags
        if tags is None:
            self.tags = getattr(config.logging, 'tags', [])
        else:
            self.tags = tags
        
        # Get notes
        self.notes = notes
        
        if self.enabled:
            self._initialize_wandb()
    
    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb
            if wandb.run is None:
                # Convert config to dictionary
                if hasattr(self.config, 'to_dict'):
                    config_dict = self.config.to_dict()
                elif hasattr(self.config, '__dict__'):
                    config_dict = vars(self.config)
                else:
                    config_dict = self.config
                
                wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                    config=config_dict,
                    tags=self.tags,
                    notes=self.notes
                )
                
                self.run = wandb.run
                print(f"Initialized Weights & Biases run: {self.run.name}")
            else:
                # Use existing run
                self.run = wandb.run
                print(f"Using existing Weights & Biases run: {self.run.name}")
        except ImportError:
            print("Warning: Weights & Biases (wandb) not installed. Disabling wandb logging.")
            self.enabled = False
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled:
            return
            
        try:
            # Filter out non-serializable values
            filtered_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)) or (isinstance(v, np.ndarray) and v.size == 1):
                    filtered_metrics[k] = v
                elif isinstance(v, np.ndarray) and v.size > 1:
                    # Skip large arrays
                    continue
                elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    # Allow lists of simple types
                    filtered_metrics[k] = v
            
            self.wandb.log(filtered_metrics, step=step)
        except Exception as e:
            print(f"Error logging to Weights & Biases: {e}")
    
    def log_model(self, model: torch.nn.Module, log_freq: int = 100) -> None:
        """Log model architecture to wandb.
        
        Args:
            model: PyTorch model
            log_freq: Frequency of gradient/parameter logging
        """
        if not self.enabled:
            return
            
        try:
            self.wandb.watch(model, log="all", log_freq=log_freq)
        except Exception as e:
            print(f"Error watching model in Weights & Biases: {e}")
    
    def log_images(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor, 
        predictions: torch.Tensor,
        num_samples: int = 4,
        class_labels: Optional[Dict[int, str]] = None
    ) -> None:
        """Log sample predictions to wandb.
        
        Args:
            images: Batch of images (B, C, H, W) or (B, H, W, C)
            masks: Batch of ground truth masks (B, H, W) or (B, 1, H, W)
            predictions: Batch of predictions (B, H, W) or (B, 1, H, W)
            num_samples: Number of samples to log
            class_labels: Optional class label mapping
        """
        if not self.enabled:
            return
            
        try:
            # Convert tensors to numpy if needed
            if isinstance(images, torch.Tensor):
                images_np = images.detach().cpu().numpy()
            else:
                images_np = images
                
            if isinstance(masks, torch.Tensor):
                masks_np = masks.detach().cpu().numpy()
            else:
                masks_np = masks
                
            if isinstance(predictions, torch.Tensor):
                preds_np = predictions.detach().cpu().numpy()
            else:
                preds_np = predictions
            
            # Ensure proper shapes
            if images_np.shape[1] == 3 and len(images_np.shape) == 4:  # (B, C, H, W)
                images_np = np.transpose(images_np, (0, 2, 3, 1))
            
            if len(masks_np.shape) == 4 and masks_np.shape[1] == 1:  # (B, 1, H, W)
                masks_np = masks_np.squeeze(1)
            
            if len(preds_np.shape) == 4 and preds_np.shape[1] == 1:  # (B, 1, H, W)
                preds_np = preds_np.squeeze(1)
            
            # Default class labels if not provided
            if class_labels is None:
                class_labels = {0: "background", 1: "glaucoma"}
            
            # Select random samples
            num_samples = min(num_samples, len(images_np))
            indices = np.random.choice(len(images_np), num_samples, replace=False)
            
            # 1. Log standard wandb mask images
            wandb_mask_images = []
            for i in indices:
                img = images_np[i]
                mask = masks_np[i]
                pred = (preds_np[i] > 0.5).astype(np.float32)
                
                # Normalize image for display
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Ensure mask is binary
                mask = (mask > 0.5).astype(np.float32)
                
                wandb_mask_images.append(
                    self.wandb.Image(
                        img,
                        masks={
                            "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                            "prediction": {"mask_data": pred, "class_labels": class_labels}
                        }
                    )
                )
            
            self.wandb.log({"segmentation_masks": wandb_mask_images})
            
            # 2. Create and log detailed visualization grid
            for idx, i in enumerate(indices):
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Get images for this sample
                img = images_np[i]
                mask = masks_np[i]
                pred_prob = preds_np[i]  # Probability map
                pred_binary = (pred_prob > 0.5).astype(np.float32)  # Binary prediction
                
                # Normalize image if needed
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Row 1: Original images and masks
                axes[0, 0].imshow(img)
                axes[0, 0].set_title("Original Image", fontsize=14)
                axes[0, 0].axis('off')
                
                # Ground truth mask
                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title("Ground Truth Mask", fontsize=14)
                axes[0, 1].axis('off')
                
                # Prediction probability
                im = axes[0, 2].imshow(pred_prob, cmap='plasma', vmin=0, vmax=1)
                axes[0, 2].set_title("Prediction Probability", fontsize=14)
                axes[0, 2].axis('off')
                plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
                
                # Row 2: Overlays and difference
                # Create overlay of prediction on image
                overlay_pred = self._create_overlay(img, pred_binary, alpha=0.5, color=[1, 0, 0])  # Red overlay
                axes[1, 0].imshow(overlay_pred)
                axes[1, 0].set_title("Prediction Overlay", fontsize=14)
                axes[1, 0].axis('off')
                
                # Create overlay of ground truth on image
                overlay_gt = self._create_overlay(img, mask, alpha=0.5, color=[0, 1, 0])  # Green overlay
                axes[1, 1].imshow(overlay_gt)
                axes[1, 1].set_title("Ground Truth Overlay", fontsize=14)
                axes[1, 1].axis('off')
                
                # Difference between prediction and ground truth
                diff = np.abs(pred_binary - mask)
                # Create a custom colormap: green for correct, red for false negative, blue for false positive
                overlay_diff = self._create_error_visualization(img, pred_binary, mask)
                axes[1, 2].imshow(overlay_diff)
                axes[1, 2].set_title("Error Visualization", fontsize=14)
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                
                # Log the figure
                self.wandb.log({f"detailed_viz_{idx}": self.wandb.Image(fig)})
                plt.close(fig)
                
            # 3. Log metrics visualization for this batch
            metrics_to_plot = ['Dice', 'IoU', 'Precision', 'Recall']
            values = []
            
            # Calculate metrics for each sample
            for i in indices:
                pred = (preds_np[i] > 0.5).astype(np.float32)
                mask = (masks_np[i] > 0.5).astype(np.float32)
                
                # Calculate Dice
                intersection = np.sum(pred * mask)
                dice = (2. * intersection) / (np.sum(pred) + np.sum(mask) + 1e-6)
                
                # Calculate IoU
                union = np.sum(pred) + np.sum(mask) - intersection
                iou = intersection / (union + 1e-6)
                
                # Calculate Precision & Recall
                tp = np.sum(pred * mask)
                fp = np.sum(pred * (1 - mask))
                fn = np.sum((1 - pred) * mask)
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                
                values.append([dice, iou, precision, recall])
            
            # Create a DataFrame for easy plotting
            df = pd.DataFrame(values, columns=metrics_to_plot)
            
            # Plot metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            df.plot(kind='bar', ax=ax)
            ax.set_title('Metrics per Sample', fontsize=14)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score')
            ax.set_xlabel('Sample Index')
            ax.grid(True, alpha=0.3)
            
            # Log the figure
            self.wandb.log({"metrics_visualization": self.wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"Error logging images to Weights & Biases: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color: List[float] = [1, 0, 0]) -> np.ndarray:
        """Create an overlay of a mask on an image.
        
        Args:
            image: Original image (HxWxC)
            mask: Binary mask (HxW)
            alpha: Transparency of the overlay
            color: RGB color for the overlay [R, G, B] in range [0, 1]
            
        Returns:
            Overlay image
        """
        # Ensure the mask is binary and has the right shape
        mask_bin = mask > 0.5
        
        # Create RGB mask
        h, w = mask.shape
        mask_color = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            mask_color[..., i] = mask_bin * color[i]
        
        # Create overlay
        overlay = image.copy()
        if overlay.dtype == np.uint8:
            overlay = overlay.astype(np.float32) / 255.0
            
        # Apply mask with alpha blending
        overlay = overlay * (1 - alpha * mask_bin[..., None]) + mask_color * alpha * mask_bin[..., None]
        
        # Ensure values are in range [0, 1]
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def _create_error_visualization(self, image: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a visualization of prediction errors.
        
        Args:
            image: Original image (HxWxC)
            pred: Binary prediction mask (HxW)
            mask: Binary ground truth mask (HxW)
            
        Returns:
            Error visualization image
        """
        # Ensure binary masks
        pred_bin = pred > 0.5
        mask_bin = mask > 0.5
        
        # Create RGB error visualization
        h, w = mask.shape
        error_viz = np.zeros((h, w, 3), dtype=np.float32)
        
        # True Positive (white)
        tp = np.logical_and(pred_bin, mask_bin)
        error_viz[tp] = [1, 1, 1]  # White
        
        # False Positive (red)
        fp = np.logical_and(pred_bin, np.logical_not(mask_bin))
        error_viz[fp] = [1, 0, 0]  # Red
        
        # False Negative (blue)
        fn = np.logical_and(np.logical_not(pred_bin), mask_bin)
        error_viz[fn] = [0, 0, 1]  # Blue
        
        # True Negative (transparent/dark)
        # We'll leave these as zeros (black)
        
        # Blend with original image (darken the image slightly to make errors visible)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Darken the image
        darkened_img = image * 0.6
        
        # Create alpha channel for error visualization (transparent for TN, opaque for others)
        alpha = np.logical_or(np.logical_or(tp, fp), fn).astype(np.float32) * 0.7
        
        # Blend
        blended = darkened_img * (1 - alpha[..., None]) + error_viz * alpha[..., None]
        
        # Ensure values are in range [0, 1]
        blended = np.clip(blended, 0, 1)
        
        return blended
    
    def log_confusion_matrix(
        self, 
        matrix: np.ndarray, 
        class_names: List[str] = ['Background', 'Glaucoma']
    ) -> None:
        """Log confusion matrix to wandb.
        
        Args:
            matrix: Confusion matrix array
            class_names: Names of classes
        """
        if not self.enabled:
            return
            
        try:
            import seaborn as sns
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Calculate total samples and percentages
            total = matrix.sum()
            percentage_matrix = (matrix / total) * 100
            
            # Create annotation text
            annot = [[f"{int(val)}\n({percentage_matrix[i, j]:.2f}%)" 
                     for j, val in enumerate(row)] for i, row in enumerate(matrix)]
            
            # Plot confusion matrix with advanced styling
            sns.heatmap(
                matrix, 
                annot=annot, 
                fmt='', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar=True,
                linewidths=1,
                linecolor='black'
            )
            
            # Add titles and labels
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('True', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
            # Calculate and add metrics to the plot
            tn, fp, fn, tp = matrix.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"
            plt.figtext(0.6, 0.01, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Log figure
            self.wandb.log({"confusion_matrix": self.wandb.Image(fig)})
            
            # Create metrics table
            metrics_table = self.wandb.Table(columns=["Metric", "Value"])
            metrics_table.add_data("Accuracy", accuracy)
            metrics_table.add_data("Precision", precision)
            metrics_table.add_data("Recall", recall)
            metrics_table.add_data("F1 Score", f1)
            
            self.wandb.log({"confusion_metrics": metrics_table})
            
            # Close figure
            plt.close(fig)
        except Exception as e:
            print(f"Error logging confusion matrix to Weights & Biases: {e}")
    
    def log_pr_curve(self, precision: np.ndarray, recall: np.ndarray, pr_auc: float) -> None:
        """Log precision-recall curve to wandb.
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under the PR curve
        """
        if not self.enabled:
            return
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot PR curve with gradient color based on threshold
            points = np.array([recall, precision]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a line collection
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis')
            lc.set_array(np.linspace(0, 1, len(precision)))
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label='Threshold')
            
            # Set limits and titles
            ax.set_xlim(0, 1.01)
            ax.set_ylim(0, 1.01)
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add a random guessing line
            ax.plot([0, 1], [np.mean(precision), np.mean(precision)], 'k--', lw=1, alpha=0.7, label='Random Guessing')
            
            # Add legend
            ax.legend(loc='best')
            
            # Log figure
            self.wandb.log({"pr_curve": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
            
            # Also log as interactive wandb plot
            try:
                pr_data = [[x, y] for x, y in zip(recall, precision)]
                table = self.wandb.Table(data=pr_data, columns=["recall", "precision"])
                self.wandb.log({"pr_curve_interactive": self.wandb.plot.line(
                    table, "recall", "precision",
                    title=f"Precision-Recall Curve (AUC={pr_auc:.4f})"
                )})
            except:
                pass  # Silently fail if interactive plot doesn't work
            
        except Exception as e:
            print(f"Error logging precision-recall curve to Weights & Biases: {e}")
    
    def log_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
        """Log ROC curve to wandb.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
        """
        if not self.enabled:
            return
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot ROC curve with gradient color based on threshold
            points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a line collection
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis')
            lc.set_array(np.linspace(0, 1, len(fpr)))
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax, label='Threshold')
            
            # Plot random guessing line
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random Guessing')
            
            # Set limits and titles
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='lower right')
            
            # Log figure
            self.wandb.log({"roc_curve": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
            
            # Also log as interactive wandb plot
            try:
                roc_data = [[x, y] for x, y in zip(fpr, tpr)]
                table = self.wandb.Table(data=roc_data, columns=["fpr", "tpr"])
                self.wandb.log({"roc_curve_interactive": self.wandb.plot.line(
                    table, "fpr", "tpr", 
                    title=f"ROC Curve (AUC={roc_auc:.4f})"
                )})
            except:
                pass  # Silently fail if interactive plot doesn't work
            
        except Exception as e:
            print(f"Error logging ROC curve to Weights & Biases: {e}")
    
    def log_table(
        self, 
        table_data: Union[List[List[Any]], Dict[str, List[Any]]],
        table_name: str = "results",
        columns: Optional[List[str]] = None
    ) -> None:
        """Log data as a table to wandb.
        
        Args:
            table_data: Table data as list of lists or dict of lists
            table_name: Name for the table
            columns: Optional column names
        """
        if not self.enabled:
            return
            
        try:
            # Create columns if not provided
            if columns is None:
                if isinstance(table_data, dict):
                    columns = list(table_data.keys())
                else:
                    columns = [f"col_{i}" for i in range(len(table_data[0]))]
            
            # Convert dict to list of lists if needed
            if isinstance(table_data, dict):
                # Check if all lists have the same length
                list_lengths = [len(v) for v in table_data.values()]
                if len(set(list_lengths)) > 1:
                    print(f"Warning: Lists in table_data have different lengths: {list_lengths}")
                    max_length = max(list_lengths)
                    # Pad shorter lists with None
                    for k, v in table_data.items():
                        if len(v) < max_length:
                            table_data[k] = v + [None] * (max_length - len(v))
                
                # Convert dict to list of lists
                rows = []
                for i in range(len(next(iter(table_data.values())))):
                    rows.append([table_data[col][i] if i < len(table_data[col]) else None for col in columns])
            else:
                rows = table_data
            
            # Create table
            table = self.wandb.Table(columns=columns, data=rows)
            
            # Log table
            self.wandb.log({table_name: table})
        except Exception as e:
            print(f"Error logging table to Weights & Biases: {e}")
    
    def log_histogram(self, values: Union[np.ndarray, List[float]], name: str) -> None:
        """Log histogram to wandb.
        
        Args:
            values: Values to create histogram from
            name: Name for the histogram
        """
        if not self.enabled:
            return
            
        try:
            # Convert to numpy if needed
            if isinstance(values, list):
                values = np.array(values)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram with KDE
            bins = min(50, len(values) // 10) if len(values) > 0 else 10
            n, bins, patches = ax.hist(values, bins=bins, alpha=0.7, color='skyblue', density=True)
            
            # Try to add KDE curve
            try:
                from scipy import stats
                kde_x = np.linspace(min(values), max(values), 1000)
                kde = stats.gaussian_kde(values)
                ax.plot(kde_x, kde(kde_x), 'r-', lw=2, label='KDE')
                ax.legend()
            except:
                pass  # Skip KDE if it fails
            
            # Add statistics
            if len(values) > 0:
                stats_text = (
                    f"Mean: {np.mean(values):.4f}\n"
                    f"Median: {np.median(values):.4f}\n"
                    f"Std: {np.std(values):.4f}\n"
                    f"Min: {np.min(values):.4f}\n"
                    f"Max: {np.max(values):.4f}"
                )
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add labels and title
            ax.set_xlabel(name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{name} Distribution', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Log figure
            self.wandb.log({f"{name}_histogram": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
            
            # Also log as wandb histogram
            self.wandb.log({f"{name}_hist": self.wandb.Histogram(values)})
            
        except Exception as e:
            print(f"Error logging histogram to Weights & Biases: {e}")
    
    def log_image_grid(
        self, 
        images: List[np.ndarray], 
        titles: List[str], 
        name: str,
        cols: int = 3
    ) -> None:
        """Log a grid of images to wandb.
        
        Args:
            images: List of images
            titles: List of titles for each image
            name: Name for the grid
            cols: Number of columns in the grid
        """
        if not self.enabled:
            return
            
        try:
            # Calculate rows needed
            n_images = len(images)
            rows = (n_images + cols - 1) // cols
            
            # Create figure
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            # Handle case of single subplot
            if n_images == 1:
                axes = np.array([axes])
            
            # Flatten axes for easy indexing
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Add each image to the grid
            for i in range(n_images):
                ax = axes[i]
                img = images[i]
                
                # Handle different image types
                if len(img.shape) == 2:  # Grayscale
                    ax.imshow(img, cmap='gray')
                else:  # RGB
                    ax.imshow(img)
                
                # Add title
                if i < len(titles):
                    ax.set_title(titles[i])
                
                # Remove axis ticks
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Log figure
            self.wandb.log({name: self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            print(f"Error logging image grid to Weights & Biases: {e}")
    
    def log_metrics_over_time(self, metrics_history: Dict[str, List[float]], name: str = "metrics_over_time") -> None:
        """Log metrics over time as a line plot.
        
        Args:
            metrics_history: Dictionary mapping metric names to lists of values
            name: Name for the plot
        """
        if not self.enabled:
            return
            
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each metric
            for metric_name, values in metrics_history.items():
                # Create x values (steps/epochs)
                steps = list(range(1, len(values) + 1))
                
                # Plot line
                ax.plot(steps, values, 'o-', label=metric_name)
            
            # Add labels and title
            ax.set_xlabel('Step/Epoch', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Metrics Over Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Log figure
            self.wandb.log({name: self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
            
            # Also log as interactive wandb plot if possible
            try:
                data = []
                for step_idx in range(len(next(iter(metrics_history.values())))):
                    step_data = [step_idx + 1]  # Step/epoch number
                    for metric_name in metrics_history.keys():
                        if step_idx < len(metrics_history[metric_name]):
                            step_data.append(metrics_history[metric_name][step_idx])
                        else:
                            step_data.append(None)
                    data.append(step_data)
                
                columns = ["step"] + list(metrics_history.keys())
                table = self.wandb.Table(data=data, columns=columns)
                
                self.wandb.log({f"{name}_interactive": self.wandb.plot.line(
                    table, "step", list(metrics_history.keys()),
                    title="Metrics Over Time"
                )})
            except Exception as e:
                # Skip interactive plot if it fails
                print(f"Warning: Could not create interactive metrics plot: {e}")
            
        except Exception as e:
            print(f"Error logging metrics over time to Weights & Biases: {e}")
    
    def save_model(
        self, 
        model: torch.nn.Module, 
        name: str = "model", 
        metadata: Optional[Dict[str, Any]] = None,
        save_state_dict: bool = True
    ) -> None:
        """Save model as wandb artifact.
        
        Args:
            model: PyTorch model
            name: Name for the artifact
            metadata: Optional metadata to include with the artifact
            save_state_dict: Whether to save just the state dict (True) or the entire model (False)
        """
        if not self.enabled:
            return
            
        try:
            # Create artifact
            artifact = self.wandb.Artifact(name=name, type="model", metadata=metadata)
            
            # Create temporary directory
            os.makedirs('wandb_artifacts', exist_ok=True)
            temp_path = f"wandb_artifacts/{name}_{int(time.time())}.pt"
            
            # Save model
            if save_state_dict:
                torch.save(model.state_dict(), temp_path)
            else:
                torch.save(model, temp_path)
            
            # Add model file to artifact
            artifact.add_file(temp_path)
            
            # Log artifact
            self.wandb.log_artifact(artifact)
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
        except Exception as e:
            print(f"Error saving model to Weights & Biases: {e}")
    
    def log_code(self, root_dir: str = "./", name: str = "code", exclude_dirs: Optional[List[str]] = None) -> None:
        """Log code to wandb as an artifact.
        
        Args:
            root_dir: Root directory of the code
            name: Name for the artifact
            exclude_dirs: List of directories to exclude
        """
        if not self.enabled:
            return
            
        try:
            # Create artifact
            code_artifact = self.wandb.Artifact(name=name, type="code")
            
            # Add code files
            if exclude_dirs is None:
                exclude_dirs = [".git", "__pycache__", "wandb", "logs", "output", "data"]
            
            for root, dirs, files in os.walk(root_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    if file.endswith((".py", ".md")):
                        file_path = os.path.join(root, file)
                        code_artifact.add_file(file_path, name=os.path.relpath(file_path, root_dir))
            
            # Log artifact
            self.wandb.log_artifact(code_artifact)
        except Exception as e:
            print(f"Error logging code to Weights & Biases: {e}")
    
    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"Error finishing Weights & Biases run: {e}")