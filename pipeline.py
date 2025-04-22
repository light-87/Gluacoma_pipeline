"""
Glaucoma Detection Pipeline

This is the main file that orchestrates the glaucoma detection pipeline.
It coordinates data loading, model creation, training, and evaluation.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import modules
from config import Config, get_argument_parser, parse_args_and_create_config
from data_module import (
    DatasetLoader, save_dataset, create_dataloaders,
    get_augmentations
)
from models_module import create_model, load_checkpoint, create_ensemble
from losses_module import create_loss_function
from training_module import Trainer
from evaluation_module import (
    evaluate_model, visualize_predictions, calculate_cdr
)
from wandb_module import WandBLogger
from run_tracker import get_run_tracker

def setup_environment(config):
    """Set up the environment for running the pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Device to use for computations
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.data.random_state)
    np.random.seed(config.data.random_state)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.data.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Determine device
    if config.training.device.lower() == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.training.device)
    
    # Create output directories
    output_dir = Path(config.pipeline.output_dir) / f"run_{config.pipeline.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config.pipeline.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    config.save_json(config_path)
    
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    return device, output_dir, checkpoint_dir

def load_data(config):
    """Load dataset from the configured data directories.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        DataFrame with dataset
    """
    print("\n--- Loading Data ---")
    
    # Create data loader
    data_loader = DatasetLoader(config.data.data_dirs)
    
    # Load all datasets
    df = data_loader.load_all_datasets()
    
    if df.empty:
        print("No data loaded. Check data directories.")
        return None
    
    # Analyze dataset
    stats = data_loader.analyze_dataset(df)
    print(f"Dataset statistics: {stats}")
    
    # Save dataset
    dataset_path = Path(config.pipeline.output_dir) / f"run_{config.pipeline.run_id}" / "dataset.csv"
    save_dataset(
        df,
        dataset_path,
        create_splits=True,
        train_ratio=1 - config.data.validation_split - config.data.test_split,
        val_ratio=config.data.validation_split,
        test_ratio=config.data.test_split,
        random_state=config.data.random_state
    )
    
    print(f"Saved dataset to {dataset_path}")
    
    return df, dataset_path

def create_data_loaders(config, df=None):
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Pipeline configuration
        df: DataFrame with dataset (if None, load from dataset_file)
        
    Returns:
        Dictionary with data loaders
    """
    print("\n--- Creating Data Loaders ---")
    
    # Load dataset if not provided
    if df is None:
        dataset_path = config.data.dataset_file
        if not dataset_path:
            print("No dataset file specified")
            return None
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"Loaded dataset from {dataset_path} with {len(df)} samples")
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
            return None
    
    # Create dataloaders with augmentations
    # Temporarily disable prefetching and caching to avoid pickling issues
    dataloaders = create_dataloaders(
        df=df,
        transform_train=get_augmentations(config.preprocessing, is_train=True),
        transform_val=get_augmentations(config.preprocessing, is_train=False),
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        target_size=(config.preprocessing.image_size, config.preprocessing.image_size),
        cache_size=0,  # Disable caching
        prefetch_size=0,  # Disable prefetching
        mode='segmentation'
    )
    
    # Print dataset sizes
    print(f"Train: {len(dataloaders['train'].dataset)} samples, "
          f"{len(dataloaders['train'])} batches")
    print(f"Validation: {len(dataloaders['val'].dataset)} samples, "
          f"{len(dataloaders['val'])} batches")
    print(f"Test: {len(dataloaders['test'].dataset)} samples, "
          f"{len(dataloaders['test'])} batches")
    
    return dataloaders

def train_model(config, device, dataloaders, wandb_logger=None):
    """Train or load a model.
    
    Args:
        config: Pipeline configuration
        device: Device to use for training
        dataloaders: Data loaders for training and validation
        wandb_logger: WandB logger
        
    Returns:
        Trained model
    """
    print("\n--- Training Model ---")
    
    # Create model
    model = create_model(config.model)
    print(f"Created {config.model.architecture} model with {config.model.encoder} encoder")
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint if specified
    if config.model.checkpoint_path:
        model = load_checkpoint(model, config.model.checkpoint_path)
        print(f"Loaded checkpoint from {config.model.checkpoint_path}")
        return model
    
    # Create loss function
    criterion = create_loss_function(config.training.loss)
    print(f"Using {config.training.loss.loss_function} loss function")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        config=config.training,
        device=device,
        checkpoint_dir=config.pipeline.checkpoint_dir,
        wandb_logger=wandb_logger
    )
    
    # Train model
    result = trainer.train()
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Best validation metric: {result['best_val_metric']:.4f} at epoch {result['best_epoch']+1}")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    
    # Load best model
    best_checkpoint_path = os.path.join(config.pipeline.checkpoint_dir, 'best_model.pt')
    model = load_checkpoint(model, best_checkpoint_path)
    
    return model

def evaluate(config, device, model, dataloaders, output_dir, wandb_logger=None):
    """Evaluate a model.
    
    Args:
        config: Pipeline configuration
        device: Device to use for evaluation
        model: Model to evaluate
        dataloaders: Data loaders for evaluation
        output_dir: Directory to save evaluation results
        wandb_logger: WandB logger
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n--- Evaluating Model ---")
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create evaluation directory
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Evaluate on test set
    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
        device=device,
        threshold=config.evaluation.threshold,
        calculate_cdr_flag=config.evaluation.calculate_cdr,
        cdr_method=config.evaluation.cdr_method,
        use_tta=config.evaluation.use_tta
    )
    
    # Print evaluation metrics
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics
    metrics_path = eval_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save visualizations
    if config.evaluation.generate_visualizations:
        print("\nGenerating visualizations...")
        
        # Create visualizations directory
        vis_dir = eval_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Get a batch of test data
        test_iter = iter(dataloaders['test'])
        images, masks = next(test_iter)
        
        # Generate predictions
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
        
        # Visualize predictions
        figures = visualize_predictions(
            images=images,
            masks=masks,
            preds=preds,
            threshold=config.evaluation.threshold,
            max_samples=config.evaluation.sample_count,
            output_dir=str(vis_dir)
        )
        
        # Log visualizations to WandB
        if wandb_logger:
            for i, fig in enumerate(figures):
                wandb_logger.log_figure(fig, f"test_sample_{i}")
    
    # Log metrics to WandB
    if wandb_logger:
        wandb_logger.log(metrics)
    
    return metrics

def run_pipeline(config, skip_if_completed=True):
    """Run the glaucoma detection pipeline.
    
    Args:
        config: Pipeline configuration
        skip_if_completed: Whether to skip runs that have already been completed
        
    Returns:
        Status code (0 for success, 1 for failure)
    """
    # Check if run has already been completed
    if skip_if_completed:
        run_tracker = get_run_tracker()
        if run_tracker.is_run_completed(config.to_dict()):
            print("\n===== SKIPPING ALREADY COMPLETED RUN =====")
            print(f"Configuration: {config.model.architecture}_{config.model.encoder}_{config.training.loss.loss_function}")
            print("This configuration has already been run.")
            print("To force re-run, use --force-rerun")
            print("===== SKIPPING ALREADY COMPLETED RUN =====\n")
            return 0
    
    # Setup environment
    device, output_dir, checkpoint_dir = setup_environment(config)
    
    # Always initialize WandB logger
    wandb_logger = WandBLogger(config)
    
    # Get pipeline steps to execute
    steps = config.pipeline.steps
    print(f"Will execute steps: {', '.join(steps)}")
    
    try:
        # Step 1: Load data
        if 'load' in steps:
            df, dataset_path = load_data(config)
            if df is None:
                print("Failed to load data")
                wandb_logger.finish()
                return 1
            
            # Update config with dataset path
            config.data.dataset_file = str(dataset_path)
        
        # Step 2: Create data loaders
        dataloaders = None
        if 'preprocess' in steps:
            dataloaders = create_data_loaders(config, df if 'load' in steps else None)
            if dataloaders is None:
                print("Failed to create data loaders")
                wandb_logger.finish()
                return 1
        
        # Step 3: Train model
        model = None
        if 'train' in steps:
            if dataloaders is None:
                dataloaders = create_data_loaders(config)
                if dataloaders is None:
                    print("Failed to create data loaders")
                    wandb_logger.finish()
                    return 1
            
            model = train_model(config, device, dataloaders, wandb_logger)
        
        # Step 4: Evaluate model
        metrics = None
        if 'evaluate' in steps:
            if model is None:
                # Create model
                model = create_model(config.model)
                
                # Load checkpoint
                if config.model.checkpoint_path:
                    model = load_checkpoint(model, config.model.checkpoint_path)
                else:
                    # Try to find best model checkpoint
                    best_model_path = os.path.join(config.pipeline.checkpoint_dir, 'best_model.pt')
                    if os.path.exists(best_model_path):
                        model = load_checkpoint(model, best_model_path)
                    else:
                        print("No model checkpoint found for evaluation")
                        wandb_logger.finish()
                        return 1
            
            if dataloaders is None:
                dataloaders = create_data_loaders(config)
                if dataloaders is None:
                    print("Failed to create data loaders")
                    wandb_logger.finish()
                    return 1
            
            metrics = evaluate(config, device, model, dataloaders, output_dir, wandb_logger)
        
        # Step 5: Ensemble evaluation (not implemented in this version)
        
        # Mark run as completed in the tracker
        if skip_if_completed:
            run_tracker = get_run_tracker()
            run_metadata = {}
            if metrics:
                # Include key metrics in metadata
                for key in ['dice', 'iou', 'accuracy', 'precision', 'recall']:
                    if key in metrics:
                        run_metadata[key] = metrics[key]
            
            # Get wandb run ID if available
            run_id = ""
            if wandb_logger and hasattr(wandb_logger, 'run') and wandb_logger.run:
                run_id = wandb_logger.run.id
            
            run_tracker.mark_run_completed(
                config.to_dict(),
                run_id=run_id,
                metadata=run_metadata
            )
        
        # Finish WandB run
        wandb_logger.finish()
        
        print("\nPipeline completed successfully")
        return 0
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        if wandb_logger:
            wandb_logger.finish()
        return 1

def main():
    """Main entry point for the application."""
    # Parse arguments and create configuration
    parser = get_argument_parser()
    
    # Add argument to force rerun even if configuration has been run before
    parser.add_argument('--force-rerun', action='store_true', 
                      help='Force rerun even if this configuration has been run before')
    
    # Add argument to list completed runs
    parser.add_argument('--list-completed-runs', action='store_true',
                      help='List all completed runs and exit')
    
    # Add argument to clear run history
    parser.add_argument('--clear-run-history', action='store_true',
                      help='Clear history of completed runs and exit')
    
    args = parser.parse_args()
    
    # Handle run tracking commands
    if args.list_completed_runs:
        run_tracker = get_run_tracker()
        completed_runs = run_tracker.get_completed_run_summaries()
        count = run_tracker.get_completed_run_count()
        
        print(f"\nCompleted Runs ({count} total):")
        for i, run_summary in enumerate(completed_runs, 1):
            print(f"{i}. {run_summary}")
        return 0
    
    if args.clear_run_history:
        run_tracker = get_run_tracker()
        run_tracker.clear_completed_runs()
        print("Run history cleared.")
        return 0
    
    # Create configuration
    config = parse_args_and_create_config(args)
    
    # Run pipeline
    status = run_pipeline(config, skip_if_completed=not args.force_rerun)
    
    return status

if __name__ == "__main__":
    sys.exit(main())