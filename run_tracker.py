"""
Run Tracker Module

This module provides functions to track and manage completed runs, allowing
for skipping repeated configurations and resuming batch runs.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

class RunTracker:
    """Tracks runs to avoid repeating configurations."""
    
    def __init__(self, tracker_file: str = "completed_runs.json"):
        """Initialize run tracker.
        
        Args:
            tracker_file: Path to file storing completed runs
        """
        self.tracker_file = tracker_file
        self.completed_runs = self._load_completed_runs()
    
    def _load_completed_runs(self) -> Dict[str, Any]:
        """Load completed runs from tracker file.
        
        Returns:
            Dictionary of completed runs with hashed configs as keys
        """
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading tracker file. Creating new one.")
                return {'runs': {}, 'metadata': {'last_updated': time.time()}}
        else:
            return {'runs': {}, 'metadata': {'last_updated': time.time()}}
    
    def _save_completed_runs(self):
        """Save completed runs to tracker file."""
        # Update last updated timestamp
        self.completed_runs['metadata']['last_updated'] = time.time()
        
        # Save to file
        with open(self.tracker_file, 'w') as f:
            json.dump(self.completed_runs, f, indent=2)
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a unique hash for a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Hash string representing the configuration
        """
        # Create a simplified version of the config with only relevant parameters
        # This prevents irrelevant changes from causing unnecessary reruns
        simplified_config = {
            'model': {
                'architecture': config.get('model', {}).get('architecture', ''),
                'encoder': config.get('model', {}).get('encoder', '')
            },
            'preprocessing': {
                'image_size': config.get('preprocessing', {}).get('image_size', 0),
                'augmentation_enabled': config.get('preprocessing', {}).get('augmentation_enabled', False)
            },
            'training': {
                'loss_function': config.get('training', {}).get('loss', {}).get('loss_function', ''),
                'batch_size': config.get('training', {}).get('batch_size', 0),
                'epochs': config.get('training', {}).get('epochs', 0),
                'learning_rate': config.get('training', {}).get('learning_rate', 0)
            },
            'evaluation': {
                'use_tta': config.get('evaluation', {}).get('use_tta', False),
                'calculate_cdr': config.get('evaluation', {}).get('calculate_cdr', False)
            }
        }
        
        # Convert to string and hash
        config_str = json.dumps(simplified_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def is_run_completed(self, config: Dict[str, Any]) -> bool:
        """Check if a run with this configuration has already been completed.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if run has already been completed, False otherwise
        """
        config_hash = self._generate_config_hash(config)
        return config_hash in self.completed_runs['runs']
    
    def mark_run_completed(self, config: Dict[str, Any], 
                          run_id: str = '', 
                          metadata: Optional[Dict[str, Any]] = None):
        """Mark a run as completed.
        
        Args:
            config: Configuration dictionary
            run_id: WandB run ID or other identifier
            metadata: Additional metadata about the run
        """
        config_hash = self._generate_config_hash(config)
        
        # Extract key info for human readability
        model_arch = config.get('model', {}).get('architecture', 'unknown')
        encoder = config.get('model', {}).get('encoder', 'unknown')
        loss_func = config.get('training', {}).get('loss', {}).get('loss_function', 'unknown')
        
        # Create run entry
        run_entry = {
            'completed_at': time.time(),
            'run_id': run_id,
            'config_summary': f"{model_arch}_{encoder}_{loss_func}",
            'metadata': metadata or {}
        }
        
        # Add to completed runs
        self.completed_runs['runs'][config_hash] = run_entry
        
        # Save to file
        self._save_completed_runs()
    
    def get_completed_run_count(self) -> int:
        """Get count of completed runs.
        
        Returns:
            Number of completed runs
        """
        return len(self.completed_runs['runs'])
    
    def get_completed_run_summaries(self) -> List[str]:
        """Get summaries of completed runs.
        
        Returns:
            List of run summaries
        """
        return [run['config_summary'] for run in self.completed_runs['runs'].values()]
    
    def clear_completed_runs(self):
        """Clear all completed runs from tracker."""
        self.completed_runs = {'runs': {}, 'metadata': {'last_updated': time.time()}}
        self._save_completed_runs()

def get_run_tracker() -> RunTracker:
    """Get a singleton instance of RunTracker.
    
    Returns:
        RunTracker instance
    """
    if not hasattr(get_run_tracker, 'instance'):
        get_run_tracker.instance = RunTracker()
    return get_run_tracker.instance