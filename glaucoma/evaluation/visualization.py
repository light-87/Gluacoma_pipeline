# glaucoma/evaluation/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any

class VisualizationManager:
    """Manager for generating visualizations of model results."""
    
    def __init__(self, output_dir: str):
        """Initialize the visualization manager.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default matplotlib style
        plt.style.use('seaborn')
    
    def plot_training_history(self, history_file: Union[str, Path]) -> Optional[Path]:
        """Plot training metrics history from CSV logs.
        
        Args:
            history_file: Path to training history CSV file
            
        Returns:
            Path to saved visualization file
        """
        import pandas as pd
        history_path = Path(history_file)
        if not history_path.exists():
            print(f"Training history file not found: {history_path}")
            return None
        
        # Load training history
        try:
            history = pd.read_csv(history_path)
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None
        
        # Determine which metrics to plot
        metrics = []
        for col in history.columns:
            # Skip non-metric columns
            if col in ['epoch', 'step']:
                continue
            
            # Check if there's a validation version of this metric
            base_metric = col.replace('train_', '').replace('val_', '')
            if f"train_{base_metric}" in history.columns and f"val_{base_metric}" in history.columns:
                metrics.append(base_metric)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        if n_metrics == 0:
            print("No training metrics found in history file")
            return None
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), dpi=100)
        if n_metrics == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Get train and val columns
            train_col = f"train_{metric}"
            val_col = f"val_{metric}"
            
            # Plot if available
            if train_col in history.columns:
                history.plot(x='epoch', y=train_col, ax=ax, label=f'Training {metric}', color='blue')
            if val_col in history.columns:
                history.plot(x='epoch', y=val_col, ax=ax, label=f'Validation {metric}', color='orange')
            
            ax.set_title(f'{metric.capitalize()} over Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self.viz_dir / "training_history.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Training history visualization saved to {output_path}")
        return output_path
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, output_name: str = 'roc_curve.png') -> Path:
        """Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
            output_name: Name of output file
            
        Returns:
            Path to saved image
        """
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self.viz_dir / output_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_pr_curve(self, precision, recall, pr_auc, output_name: str = 'pr_curve.png') -> Path:
        """Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under the PR curve
            output_name: Name of output file
            
        Returns:
            Path to saved image
        """
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = self.viz_dir / output_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_confusion_matrix(self, cm, class_names=['Background', 'Glaucoma'], 
                            output_name: str = 'confusion_matrix.png') -> Path:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array [[tn, fp], [fn, tp]]
            class_names: Names of classes
            output_name: Name of output file
            
        Returns:
            Path to saved image
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save figure
        output_path = self.viz_dir / output_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_sample_predictions(self, 
                             images: np.ndarray, 
                             masks: np.ndarray, 
                             predictions: np.ndarray,
                             num_samples: int = 5) -> List[Path]:
        """Plot sample predictions with ground truth and overlay.
        
        Args:
            images: Batch of images (b, h, w, 3) or (b, 3, h, w)
            masks: Batch of masks (b, h, w) or (b, 1, h, w)
            predictions: Batch of predictions (b, h, w) or (b, 1, h, w)
            num_samples: Number of samples to plot
            
        Returns:
            List of paths to saved visualizations
        """
        # Handle different input formats
        if images.shape[1] == 3 and len(images.shape) == 4:  # (b, 3, h, w)
            images = np.transpose(images, (0, 2, 3, 1))
        
        if len(masks.shape) == 4 and masks.shape[1] == 1:  # (b, 1, h, w)
            masks = masks.squeeze(1)
        
        if len(predictions.shape) == 4 and predictions.shape[1] == 1:  # (b, 1, h, w)
            predictions = predictions.squeeze(1)
        
        # Normalize images if needed
        if images.max() <= 1.0:
            images = images * 255
        
        # Get indices for visualization
        num_samples = min(num_samples, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        output_paths = []
        
        # Create visualizations for each sample
        for i, idx in enumerate(indices):
            image = images[idx].astype(np.uint8)
            mask = masks[idx]
            pred = predictions[idx]
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Plot prediction
            axes[2].imshow(pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Plot overlay
            overlay = self._create_overlay(image, pred, mask)
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay (Green=GT, Red=Pred)')
            axes[3].axis('off')
            
            # Save figure
            output_path = self.viz_dir / f"sample_{i}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            output_paths.append(output_path)
        
        return output_paths
    
    def _create_overlay(self, image, pred, target):
        """Create overlay of prediction and ground truth on image.
        
        Args:
            image: Original image (HxWxC)
            pred: Prediction mask (HxW)
            target: Ground truth mask (HxW)
            
        Returns:
            Overlay image
        """
        # Copy image
        overlay = image.copy().astype(np.float32) / 255.0
        
        # Create mask for ground truth (green)
        gt_mask = np.zeros_like(overlay)
        gt_mask[:, :, 1] = target * 0.7  # Green channel
        
        # Create mask for prediction (red)
        pred_mask = np.zeros_like(overlay)
        pred_mask[:, :, 0] = pred * 0.7  # Red channel
        
        # Overlay
        overlay = np.clip(overlay * 0.7 + gt_mask + pred_mask, 0, 1)
        
        return overlay
    
    def generate_report(self, metrics: Dict[str, float], output_name: str = 'report.html') -> Path:
        """Generate HTML report with evaluation results.
        
        Args:
            metrics: Dictionary with evaluation metrics
            output_name: Name of output file
            
        Returns:
            Path to saved report
        """
        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Glaucoma Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .metric {{ margin-bottom: 10px; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ float: right; }}
                .metrics-container {{ 
                    max-width: 600px; 
                    margin: 20px 0; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px;
                }}
                .metrics-container:after {{ 
                    content: ""; 
                    display: table; 
                    clear: both; 
                }}
                .visualization {{ margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Glaucoma Detection Evaluation Report</h1>
            
            <h2>Performance Metrics</h2>
            <div class="metrics-container">
        """
        
        # Add metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f'<div class="metric"><span class="metric-name">{name}</span><span class="metric-value">{value:.4f}</span></div>\n'
        
        html += """
            </div>
            
            <h2>Visualizations</h2>
            
            <div class="visualization">
                <h3>Training History</h3>
                <img src="visualizations/training_history.png" alt="Training History">
            </div>
            
            <div class="visualization">
                <h3>Confusion Matrix</h3>
                <img src="visualizations/confusion_matrix.png" alt="Confusion Matrix">
            </div>
            
            <div class="visualization">
                <h3>ROC Curve</h3>
                <img src="visualizations/roc_curve.png" alt="ROC Curve">
            </div>
            
            <div class="visualization">
                <h3>Precision-Recall Curve</h3>
                <img src="visualizations/pr_curve.png" alt="Precision-Recall Curve">
            </div>
            
            <h2>Sample Predictions</h2>
        """
        
        # Add sample predictions
        sample_paths = list(self.viz_dir.glob("sample_*.png"))
        for i, path in enumerate(sorted(sample_paths)):
            html += f'<div class="visualization"><h3>Sample {i+1}</h3><img src="visualizations/{path.name}" alt="Sample {i+1}"></div>\n'
        
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        output_path = self.output_dir / output_name
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path