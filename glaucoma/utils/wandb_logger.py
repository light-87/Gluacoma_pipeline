"""
Weights & Biases Integration Module

Implements comprehensive Weights & Biases integration for experiment tracking.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
from typing import Dict, List, Optional, Any, Union, Tuple

class WandBLogger:
    """Logger for Weights & Biases integration."""
    
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
            
            # Log images with masks
            wandb_images = []
            for i in indices:
                img = images_np[i]
                mask = masks_np[i]
                pred = (preds_np[i] > 0.5).astype(np.float32)
                
                # Normalize image for display
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Ensure mask is binary
                mask = (mask > 0.5).astype(np.float32)
                
                wandb_images.append(
                    self.wandb.Image(
                        img,
                        masks={
                            "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                            "prediction": {"mask_data": pred, "class_labels": class_labels}
                        }
                    )
                )
            
            self.wandb.log({"predictions": wandb_images})
        except Exception as e:
            print(f"Error logging images to Weights & Biases: {e}")
    
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
            # Create figure
            fig, ax = plt.subplots(figsize=(7, 7))
            
            # Plot confusion matrix
            ax.matshow(matrix, cmap='Blues')
            
            # Add labels
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(matrix[i, j]), va='center', ha='center')
            
            # Set axis labels
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            # Log figure
            self.wandb.log({"confusion_matrix": self.wandb.Image(fig)})
            
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
            fig, ax = plt.subplots(figsize=(7, 7))
            
            # Plot PR curve
            ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Log figure
            self.wandb.log({"pr_curve": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
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
            fig, ax = plt.subplots(figsize=(7, 7))
            
            # Plot ROC curve
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Log figure
            self.wandb.log({"roc_curve": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
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
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Plot histogram
            ax.hist(values, bins=30)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
            
            # Log figure
            self.wandb.log({f"{name}_histogram": self.wandb.Image(fig)})
            
            # Close figure
            plt.close(fig)
        except Exception as e:
            print(f"Error logging histogram to Weights & Biases: {e}")
    
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