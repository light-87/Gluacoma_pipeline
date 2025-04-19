# glaucoma/utils/wandb_logger.py
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
from pathlib import Path
import time

class WandBLogger:
    """Logger for Weights & Biases integration."""
    
    def __init__(self, config: Any):
        """Initialize WandB logger.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = getattr(config, 'use_wandb', False)
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize wandb
                if wandb.run is None:
                    wandb.init(
                        project=getattr(config, 'wandb_project', 'glaucoma-detection'),
                        name=getattr(config, 'run_name', None),
                        config=vars(config) if hasattr(config, '__dict__') else config,
                        tags=getattr(config, 'tags', [])
                    )
                    
                self.run = wandb.run
            except ImportError:
                print("Warning: Weights & Biases (wandb) not installed. Disabling wandb logging.")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled:
            return
            
        self.wandb.log(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module) -> None:
        """Log model architecture to wandb.
        
        Args:
            model: PyTorch model
        """
        if not self.enabled:
            return
            
        self.wandb.watch(model, log="all", log_freq=100)
    
    def log_images(self, 
                 images: torch.Tensor, 
                 masks: torch.Tensor, 
                 predictions: torch.Tensor,
                 num_samples: int = 4) -> None:
        """Log sample predictions to wandb.
        
        Args:
            images: Batch of images
            masks: Batch of ground truth masks
            predictions: Batch of predictions
            num_samples: Number of samples to log
        """
        if not self.enabled:
            return
            
        # Convert tensors to numpy
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        
        # Ensure proper shapes
        if images_np.shape[1] == 3:  # (B, C, H, W)
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        if len(masks_np.shape) == 4 and masks_np.shape[1] == 1:  # (B, 1, H, W)
            masks_np = masks_np.squeeze(1)
        
        if len(preds_np.shape) == 4 and preds_np.shape[1] == 1:  # (B, 1, H, W)
            preds_np = preds_np.squeeze(1)
        
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
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            wandb_images.append(
                self.wandb.Image(
                    img,
                    masks={
                        "ground_truth": {"mask_data": mask, "class_labels": {0: "background", 1: "glaucoma"}},
                        "prediction": {"mask_data": pred, "class_labels": {0: "background", 1: "glaucoma"}}
                    }
                )
            )
        
        self.wandb.log({"predictions": wandb_images})
    
    def log_confusion_matrix(self, matrix: np.ndarray, class_names: List[str] = ['Background', 'Glaucoma']) -> None:
        """Log confusion matrix to wandb.
        
        Args:
            matrix: Confusion matrix array
            class_names: Names of classes
        """
        if not self.enabled:
            return
            
        self.wandb.log({
            "confusion_matrix": self.wandb.plot.confusion_matrix(
                preds=matrix.flatten(),
                y_true=np.array([0, 0, 1, 1]),  # For 2x2 matrix: [TN, FP, FN, TP]
                class_names=class_names
            )
        })
    
    def save_model(self, model: torch.nn.Module, name: str = "model") -> None:
        """Save model as wandb artifact.
        
        Args:
            model: PyTorch model
            name: Name for the artifact
        """
        if not self.enabled:
            return
            
        # Create artifact
        artifact = self.wandb.Artifact(name=name, type="model")
        
        # Save model to a temporary file
        temp_path = Path(f"/tmp/{name}_{time.time()}.pt")
        torch.save(model.state_dict(), temp_path)
        
        # Add model file to artifact
        artifact.add_file(str(temp_path))
        
        # Log artifact
        self.wandb.log_artifact(artifact)
        
        # Remove temporary file
        temp_path.unlink()
    
    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled:
            self.wandb.finish()