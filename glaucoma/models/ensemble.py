"""
Model Ensemble Module

Implements ensemble methods for glaucoma detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union

class ModelEnsemble(nn.Module):
    """Ensemble of multiple segmentation models."""
    
    def __init__(self, models: List[nn.Module], ensemble_method: str = 'average', weights: Optional[List[float]] = None):
        """Initialize ensemble.
        
        Args:
            models: List of models to ensemble
            ensemble_method: Method for combining predictions ('average', 'max', 'weighted')
            weights: Optional weights for weighted ensemble (only used if ensemble_method is 'weighted')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method.lower()
        
        if self.ensemble_method == 'weighted':
            if weights is None:
                # Equal weights initialization
                weights = torch.ones(len(models)) / len(models)
            else:
                weights = torch.tensor(weights, dtype=torch.float32)
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
            
            # Create learnable weights parameter
            self.weights = nn.Parameter(weights)
        else:
            self.weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all models and combine predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined output from all models
        """
        if self.ensemble_method == 'average':
            return self._average_ensemble(x)
        elif self.ensemble_method == 'max':
            return self._max_ensemble(x)
        elif self.ensemble_method == 'weighted':
            return self._weighted_ensemble(x)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _average_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged output
        """
        # Get outputs from all models
        outputs = [model(x) for model in self.models]
        
        # Average outputs
        return torch.mean(torch.stack(outputs), dim=0)
    
    def _max_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Take maximum prediction for each pixel.
        
        Args:
            x: Input tensor
            
        Returns:
            Maximum output
        """
        # Get outputs from all models
        outputs = [model(x) for model in self.models]
        
        # Take maximum
        return torch.max(torch.stack(outputs), dim=0)[0]
    
    def _weighted_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average of predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted output
        """
        # Get outputs from all models
        outputs = [model(x) for model in self.models]
        
        # Get normalized weights
        weights = F.softmax(self.weights, dim=0)
        
        # Compute weighted sum
        return sum(w * out for w, out in zip(weights, outputs))
    
    def get_model_weights(self) -> List[float]:
        """Get current weights for each model in the ensemble.
        
        Returns:
            List of weights for each model (normalized to sum to 1)
        """
        if self.weights is not None:
            weights = F.softmax(self.weights, dim=0)
            return weights.detach().cpu().tolist()
        else:
            # For non-weighted ensembles, return equal weights
            return [1.0 / len(self.models)] * len(self.models)

class EnsembleFactory:
    """Factory for creating model ensembles."""
    
    @staticmethod
    def create_ensemble(
        model_checkpoints: List[Dict[str, Any]],
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ) -> ModelEnsemble:
        """Create an ensemble from model checkpoints.
        
        Args:
            model_checkpoints: List of dictionaries with 'model' and 'checkpoint_path' keys
            ensemble_method: Method for combining predictions
            weights: Optional weights for weighted ensemble
            device: Device to load models onto
            
        Returns:
            Initialized ModelEnsemble
        """
        from glaucoma.models.factory import create_model
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        models = []
        for i, checkpoint_info in enumerate(model_checkpoints):
            # Create model
            model_config = checkpoint_info['model']
            model = create_model(model_config)
            
            # Load checkpoint
            checkpoint_path = checkpoint_info['checkpoint_path']
            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"Loaded checkpoint for model {i+1} from {checkpoint_path}")
                except Exception as e:
                    print(f"Error loading checkpoint for model {i+1}: {e}")
            
            # Move to device
            model = model.to(device)
            models.append(model)
        
        # Create ensemble
        return ModelEnsemble(models, ensemble_method=ensemble_method, weights=weights)
    
    @staticmethod
    def create_cross_validation_ensemble(
        base_model_config: Dict[str, Any],
        checkpoint_paths: List[str],
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ) -> ModelEnsemble:
        """Create an ensemble from cross-validation models.
        
        Args:
            base_model_config: Configuration for the base model
            checkpoint_paths: List of checkpoint paths from cross-validation folds
            ensemble_method: Method for combining predictions
            weights: Optional weights for weighted ensemble
            device: Device to load models onto
            
        Returns:
            Initialized ModelEnsemble with cross-validation models
        """
        # Create list of model checkpoint infos
        model_checkpoints = [
            {'model': base_model_config, 'checkpoint_path': path}
            for path in checkpoint_paths
        ]
        
        # Create ensemble
        return EnsembleFactory.create_ensemble(model_checkpoints, ensemble_method, weights, device)
    
    @staticmethod
    def create_multi_architecture_ensemble(
        model_configs: List[Dict[str, Any]],
        checkpoint_paths: Optional[List[str]] = None,
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ) -> ModelEnsemble:
        """Create an ensemble with different model architectures.
        
        Args:
            model_configs: List of model configurations
            checkpoint_paths: Optional list of checkpoint paths (if None, no checkpoints are loaded)
            ensemble_method: Method for combining predictions
            weights: Optional weights for weighted ensemble
            device: Device to load models onto
            
        Returns:
            Initialized ModelEnsemble with multiple architectures
        """
        # Create list of model checkpoint infos
        if checkpoint_paths is None:
            checkpoint_paths = [None] * len(model_configs)
        
        model_checkpoints = [
            {'model': config, 'checkpoint_path': path}
            for config, path in zip(model_configs, checkpoint_paths)
        ]
        
        # Create ensemble
        return EnsembleFactory.create_ensemble(model_checkpoints, ensemble_method, weights, device)