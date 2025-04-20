"""
Visualization Module

Implements visualization utilities for glaucoma detection models.
This module combines the previous visualization.py and visualization_enhanced.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2

class VisualizationManager:
    """Manager for generating visualizations of model results."""
    
    def __init__(
        self, 
        output_dir: str,
        create_subdirs: bool = True,
        fig_size: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ):
        """Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualizations
            create_subdirs: Whether to create subdirectories for different visualization types
            fig_size: Default figure size for plots
            dpi: Default DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create visualization subdirectory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different visualization types
        if create_subdirs:
            self.prediction_dir = self.viz_dir / "predictions"
            self.curve_dir = self.viz_dir / "curves"
            self.matrix_dir = self.viz_dir / "matrices"
            self.distribution_dir = self.viz_dir / "distributions"
            
            self.prediction_dir.mkdir(exist_ok=True)
            self.curve_dir.mkdir(exist_ok=True)
            self.matrix_dir.mkdir(exist_ok=True)
            self.distribution_dir.mkdir(exist_ok=True)
        else:
            self.prediction_dir = self.viz_dir
            self.curve_dir = self.viz_dir
            self.matrix_dir = self.viz_dir
            self.distribution_dir = self.viz_dir
        
        # Set default plotting parameters
        self.fig_size = fig_size
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            # Try different seaborn style variants
            for style_name in ['seaborn-v0_8', 'seaborn-v0_8-darkgrid', 'seaborn', 'ggplot']:
                try:
                    plt.style.use(style_name)
                    break
                except:
                    continue
        except:
            # Final fallback to default style
            print("Warning: Could not set custom matplotlib style, using default")
        
        # Store paths to generated visualizations
        self.visualization_paths = {
            'predictions': [],
            'curves': [],
            'matrices': [],
            'distributions': []
        }
    
    def _prepare_save_path(self, output_dir: Path, filename: str) -> Path:
        """Prepare save path with timestamp if file exists.
        
        Args:
            output_dir: Directory to save file
            filename: Base filename
            
        Returns:
            Path to save file
        """
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # Check if file exists
        output_path = output_dir / filename
        if output_path.exists():
            # Add timestamp to filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{name}_{timestamp}{ext}"
        
        return output_path
    
    def plot_sample_predictions(
        self, 
        images: np.ndarray, 
        masks: np.ndarray, 
        predictions: np.ndarray,
        num_samples: int = 5,
        overlay_alpha: float = 0.5,
        output_filename: str = "sample_predictions.png"
    ) -> Path:
        """Plot sample predictions with ground truth and overlay.
        
        Args:
            images: Batch of images (b, h, w, 3) or (b, 3, h, w)
            masks: Batch of masks (b, h, w) or (b, 1, h, w)
            predictions: Batch of predictions (b, h, w) or (b, 1, h, w)
            num_samples: Number of samples to plot
            overlay_alpha: Alpha value for overlay
            output_filename: Name of output file
            
        Returns:
            Path to saved visualization
        """
        # Handle different input formats
        if images.shape[1] == 3 and images.ndim == 4:  # (b, 3, h, w)
            images = np.transpose(images, (0, 2, 3, 1))
        
        if masks.ndim == 4 and masks.shape[1] == 1:  # (b, 1, h, w)
            masks = masks.squeeze(1)
        
        if predictions.ndim == 4 and predictions.shape[1] == 1:  # (b, 1, h, w)
            predictions = predictions.squeeze(1)
        
        # Normalize images if needed
        if images.max() <= 1.0:
            images = images * 255
        
        # Apply threshold to predictions
        if np.issubdtype(predictions.dtype, np.floating):
            pred_binary = (predictions > 0.5).astype(np.float32)
        else:
            pred_binary = predictions.astype(np.float32)
        
        # Get indices for visualization
        num_samples = min(num_samples, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        
        # Adjust for single sample case
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
        
        # Plot each sample
        for i, idx in enumerate(indices):
            # Get data for this sample
            image = images[idx].astype(np.uint8)
            mask = masks[idx]
            pred = predictions[idx]
            pred_bin = pred_binary[idx]
            
            # Plot original image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Sample {idx}: Original Image')
            axes[i, 0].axis('off')
            
            # Plot ground truth mask
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot prediction (probability)
            axes[i, 2].imshow(pred, cmap='plasma')
            axes[i, 2].set_title('Prediction (Prob)')
            axes[i, 2].axis('off')
            
            # Plot overlay
            overlay = self._create_overlay(image, pred_bin, mask, alpha=overlay_alpha)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Green=GT, Red=Pred)')
            axes[i, 3].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self._prepare_save_path(self.prediction_dir, output_filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Store path
        self.visualization_paths['predictions'].append(output_path)
        
        return output_path
    
    def _create_overlay(
        self, 
        image: np.ndarray, 
        pred: np.ndarray, 
        target: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create overlay of prediction and ground truth on image.
        
        Args:
            image: Original image (HxWxC)
            pred: Prediction mask (HxW)
            target: Ground truth mask (HxW)
            alpha: Alpha value for overlay
            
        Returns:
            Overlay image
        """
        # Create copy of image and convert to float for blending
        overlay = image.copy().astype(np.float32) / 255.0
        
        # Create RGB masks
        h, w = pred.shape[:2]
        pred_mask = np.zeros((h, w, 3), dtype=np.float32)
        pred_mask[..., 0] = pred  # Red channel for predictions
        
        target_mask = np.zeros((h, w, 3), dtype=np.float32)
        target_mask[..., 1] = target  # Green channel for ground truth
        
        # Blend with image
        overlay = overlay * (1 - alpha) + pred_mask * alpha * 0.8  # Slightly reduce prediction intensity
        overlay = overlay * (1 - alpha) + target_mask * alpha
        
        # Clip to valid RGB range
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def plot_roc_curve(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        roc_auc: float, 
        output_filename: str = 'roc_curve.png'
    ) -> Path:
        """Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
            output_filename: Name of output file
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=self.fig_size)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self._prepare_save_path(self.curve_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Store path
        self.visualization_paths['curves'].append(output_path)
        
        return output_path
    
    def plot_pr_curve(
        self, 
        precision: np.ndarray, 
        recall: np.ndarray, 
        pr_auc: float, 
        output_filename: str = 'pr_curve.png'
    ) -> Path:
        """Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under the PR curve
            output_filename: Name of output file
            
        Returns:
            Path to saved visualization
        """
        plt.figure(figsize=self.fig_size)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=np.sum(recall > 0) / len(recall), color='red', linestyle='--', 
                  label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self._prepare_save_path(self.curve_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Store path
        self.visualization_paths['curves'].append(output_path)
        
        return output_path
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray, 
        class_names: List[str] = ['Background', 'Glaucoma'], 
        output_filename: str = 'confusion_matrix.png',
        normalize: bool = False
    ) -> Path:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array [[tn, fp], [fn, tp]]
            class_names: Names of classes
            output_filename: Name of output file
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Path to saved visualization
        """
        # Normalize if requested
        if normalize and cm.sum() > 0:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
        else:
            cm_display = cm
            fmt = 'd'
        
        plt.figure(figsize=self.fig_size)
        
        # Create heatmap
        sns.heatmap(
            cm_display, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar=True
        )
        
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        
        # Save figure
        output_path = self._prepare_save_path(self.matrix_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Store path
        self.visualization_paths['matrices'].append(output_path)
        
        return output_path
    
    def plot_metrics_over_epochs(
        self, 
        metrics_dict: Dict[str, List[float]],
        output_filename: str = 'training_metrics.png'
    ) -> Path:
        """Plot metrics over training epochs.
        
        Args:
            metrics_dict: Dictionary mapping metric names to lists of values
            output_filename: Name of output file
            
        Returns:
            Path to saved visualization
        """
        # Filter out non-scalar metrics
        metrics_to_plot = {}
        for name, values in metrics_dict.items():
            if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                metrics_to_plot[name] = values
        
        # Calculate number of metrics to plot
        n_metrics = len(metrics_to_plot)
        
        if n_metrics == 0:
            print("No valid metrics to plot.")
            return None
        
        # Calculate grid layout
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        
        # Convert to 2D array if needed
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.array(axes).reshape(n_rows, n_cols)
        
        # Plot each metric
        for i, (name, values) in enumerate(metrics_to_plot.items()):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]
            
            # Create x-axis values (epochs)
            epochs = np.arange(1, len(values) + 1)
            
            # Plot metric
            ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
            
            # Add trend line
            try:
                from scipy.signal import savgol_filter
                if len(values) > 5:
                    window_size = min(len(values)-2 if len(values) % 2 == 0 else len(values)-1, 11)
                    if window_size > 2:
                        smoothed = savgol_filter(values, window_size, 3)
                        ax.plot(epochs, smoothed, 'r--', linewidth=1.5, alpha=0.6)
            except ImportError:
                pass
            
            # Add labels and grid
            ax.set_title(name, fontsize=12)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add min/max markers
            if len(values) > 0:
                min_idx = np.argmin(values)
                max_idx = np.argmax(values)
                
                ax.plot(epochs[min_idx], values[min_idx], 'v', color='blue', markersize=8,
                       label=f'Min: {values[min_idx]:.4f}')
                ax.plot(epochs[max_idx], values[max_idx], '^', color='red', markersize=8,
                       label=f'Max: {values[max_idx]:.4f}')
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].axis('off')
        
        plt.suptitle('Training Metrics', fontsize=16)
        
        # Save figure
        output_path = self._prepare_save_path(self.curve_dir, output_filename)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Store path
        self.visualization_paths['curves'].append(output_path)
        
        return output_path
    
    def generate_report(
        self, 
        metrics: Dict[str, float], 
        output_filename: str = 'report.html'
    ) -> Path:
        """Generate HTML report with evaluation results.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_filename: Name of output file
            
        Returns:
            Path to saved report
        """
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Glaucoma Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 8px; margin-top: 30px; }}
                .metric {{ margin-bottom: 10px; display: flex; justify-content: space-between; }}
                .metric-name {{ font-weight: bold; }}
                .metrics-container {{ 
                    max-width: 600px; 
                    margin: 20px 0; 
                    padding: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px;
                    background-color: #f9f9f9;
                }}
                .visualization {{ margin: 30px 0; }}
                .visualization h3 {{ margin-bottom: 15px; color: #2980b9; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .timestamp {{ color: #7f8c8d; margin-top: 30px; font-style: italic; }}
                .row {{ display: flex; flex-wrap: wrap; margin: 0 -15px; }}
                .col {{ padding: 0 15px; box-sizing: border-box; }}
                .col-50 {{ width: 50%; }}
                @media (max-width: 768px) {{ 
                    .col-50 {{ width: 100%; }} 
                    .metrics-container {{ max-width: 100%; }}
                }}
            </style>
        </head>
        <body>
            <h1>Glaucoma Detection Evaluation Report</h1>
            <div class="timestamp">Generated on: {timestamp}</div>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-container">
        """
        
        # Add metrics
        for name, value in sorted(metrics.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                html += f'<div class="metric"><span class="metric-name">{name}</span><span class="metric-value">{formatted_value}</span></div>\n'
        
        html += """
            </div>
            
            <h2>Visualizations</h2>
            <div class="row">
        """
        
        # Add visualizations
        viz_types = [
            ('curves', 'Performance Curves'),
            ('matrices', 'Confusion Matrix'),
            ('predictions', 'Sample Predictions'),
            ('distributions', 'Distributions')
        ]
        
        for viz_type, title in viz_types:
            if viz_type in self.visualization_paths and self.visualization_paths[viz_type]:
                html += f'<div class="col col-50"><div class="visualization"><h3>{title}</h3>\n'
                
                # Add all visualizations of this type
                for path in self.visualization_paths[viz_type]:
                    rel_path = os.path.relpath(path, self.output_dir)
                    html += f'<img src="{rel_path}" alt="{os.path.basename(path)}">\n'
                
                html += '</div></div>\n'
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        output_path = self._prepare_save_path(self.output_dir, output_filename)
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path