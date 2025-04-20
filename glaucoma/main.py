"""
Main Entry Point for Glaucoma Detection Pipeline

This script serves as the entry point for the glaucoma detection pipeline,
with enhanced Weights & Biases integration for better visualization.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import datetime
import json
from typing import List, Dict, Any, Optional, Union

from glaucoma.config import Config, get_argument_parser, parse_args_and_create_config
from glaucoma.data.loader import DatasetLoader, save_dataset
from glaucoma.data.dataset import create_dataloaders
from glaucoma.data.augmentation import get_augmentations
from glaucoma.models.factory import create_model, load_checkpoint
from glaucoma.training.trainer import Trainer
from glaucoma.evaluation.evaluator import Evaluator
from glaucoma.utils.logging import get_logger
from glaucoma.utils.wandb_logger import WandBLogger


def main():
    """Main entry point for the pipeline."""
    # Parse arguments and create configuration
    parser = get_argument_parser()
    
    # Add specific wandb options
    parser.add_argument('--wandb-detailed-viz', action='store_true', help='Enable detailed W&B visualizations')
    parser.add_argument('--wandb-log-code', action='store_true', help='Log code to W&B')
    parser.add_argument('--wandb-notes', type=str, help='Notes for W&B run')
    
    args = parser.parse_args()
    config = parse_args_and_create_config(args)
    
    # Create run ID based on timestamp
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = Path(config.output_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = get_logger(name="glaucoma", level="info", log_dir=str(log_dir))
    logger.set_context(run_id=run_id)
    
    # Save configuration
    config_path = output_dir / "config.json"
    if hasattr(config, 'save_json'):
        config.save_json(config_path)
    else:
        with open(config_path, 'w') as f:
            json.dump(vars(config) if hasattr(config, '__dict__') else config, f, indent=4, default=str)
    logger.info(f"Saved configuration to {config_path}")
    
    # Get pipeline steps to execute
    steps = config.pipeline.steps
    logger.info(f"Will execute steps: {', '.join(steps)}")
    
    # Initialize W&B if enabled
    wandb_logger = None
    if hasattr(config, 'logging') and getattr(config.logging, 'use_wandb', False):
        # Get run name (use args or create one)
        run_name = args.run_name if hasattr(args, 'run_name') and args.run_name else f"glaucoma_{run_id}"
        
        # Get tags (automatic step tags + custom tags)
        tags = [f"step_{step}" for step in steps]
        if hasattr(config.logging, 'tags') and config.logging.tags:
            tags.extend(config.logging.tags)
        
        # Initialize W&B logger with possible notes
        wandb_logger = WandBLogger(
            config=config,
            project_name=getattr(config.logging, 'wandb_project', 'glaucoma-detection'),
            run_name=run_name,
            tags=tags,
            notes=args.wandb_notes if hasattr(args, 'wandb_notes') else None
        )
        logger.info("Initialized Weights & Biases logging")
        
        # Log code if requested
        if hasattr(args, 'wandb_log_code') and args.wandb_log_code:
            wandb_logger.log_code()
            logger.info("Logged code to Weights & Biases")
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    if 'load' in steps:
        logger.log_step_start("load")
        
        # Create data loader
        data_loader = DatasetLoader(config.data.data_dirs)
        
        # Load all datasets
        df = data_loader.load_all_datasets()
        
        if df.empty:
            logger.error("No data loaded. Check data directories.")
            if wandb_logger:
                wandb_logger.finish()
            return 1
        
        # Save consolidated dataset
        dataset_path = output_dir / "dataset.csv"
        config.data.dataset_file = str(dataset_path)
        save_dataset(
            df,
            dataset_path,
            create_splits=True,
            train_ratio=0.7,  # Default if not specified
            val_ratio=getattr(config.data, 'validation_split', 0.15),
            random_state=config.data.random_state
        )
        
        # Log dataset statistics
        stats = data_loader.analyze_dataset(df)
        logger.info(f"Dataset statistics: {stats}")
        
        # Log to W&B
        if wandb_logger:
            wandb_logger.log({"dataset_size": len(df), **stats})
            
            # Log dataset distribution as a table
            if 'dataset' in df.columns:
                dataset_counts = df['dataset'].value_counts().reset_index()
                dataset_counts.columns = ['Dataset', 'Count']
                dataset_counts['Percentage'] = 100 * dataset_counts['Count'] / len(df)
                
                dataset_table = []
                for _, row in dataset_counts.iterrows():
                    dataset_table.append([row['Dataset'], int(row['Count']), float(row['Percentage'])])
                
                wandb_logger.log_table(
                    dataset_table,
                    table_name="dataset_distribution",
                    columns=["Dataset", "Count", "Percentage"]
                )
        
        logger.log_step_end("load")
    
    # Step 2: Preprocess data
    if 'preprocess' in steps:
        logger.log_step_start("preprocess")
        
        # Load dataset from file if not loaded in previous step
        if 'load' not in steps:
            dataset_path = config.data.dataset_file
            if not dataset_path:
                logger.error("No dataset file specified and 'load' step not executed")
                if wandb_logger:
                    wandb_logger.finish()
                return 1
            
            try:
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded dataset from {dataset_path} with {len(df)} samples")
            except Exception as e:
                logger.error(f"Error loading dataset from {dataset_path}: {e}")
                if wandb_logger:
                    wandb_logger.finish()
                return 1
        
        # Create dataloaders with augmentations
        dataloaders = create_dataloaders(
            df=df,
            transform_train=get_augmentations(config.preprocessing, is_train=True),
            transform_val=get_augmentations(config.preprocessing, is_train=False),
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            target_size=(config.preprocessing.image_size, config.preprocessing.image_size),
            cache_size=getattr(config.data, 'cache_size', 100) if hasattr(config.data, 'cache_size') else 100,
            prefetch_size=getattr(config.data, 'prefetch_size', 0) if hasattr(config.data, 'prefetch_size') else 0,
            mode='segmentation'
        )
        
        # Log dataloader sizes
        logger.info(f"Created dataloaders: train={len(dataloaders['train'])} batches, "
                   f"val={len(dataloaders['val'])} batches, test={len(dataloaders['test'])} batches")
        
        # Save preprocessed dataset info
        preprocess_info = {
            'train_samples': len(dataloaders['train'].dataset),
            'val_samples': len(dataloaders['val'].dataset),
            'test_samples': len(dataloaders['test'].dataset),
            'batch_size': config.training.batch_size,
            'image_size': config.preprocessing.image_size,
            'augmentation_enabled': config.preprocessing.augmentation_enabled
        }
        
        # Save preprocessing info
        preprocess_path = output_dir / "preprocessing_info.json"
        with open(preprocess_path, 'w') as f:
            json.dump(preprocess_info, f, indent=4)
        
        # Log to W&B
        if wandb_logger:
            wandb_logger.log(preprocess_info)
            
            # Log data split distribution
            if 'split' in df.columns:
                split_counts = df['split'].value_counts().reset_index()
                split_counts.columns = ['Split', 'Count']
                split_counts['Percentage'] = 100 * split_counts['Count'] / len(df)
                
                split_table = []
                for _, row in split_counts.iterrows():
                    split_table.append([row['Split'], int(row['Count']), float(row['Percentage'])])
                
                wandb_logger.log_table(
                    split_table,
                    table_name="data_split_distribution",
                    columns=["Split", "Count", "Percentage"]
                )
            
            # Try to log sample augmentations if available
            try:
                # Get a few training samples
                train_iter = iter(dataloaders['train'])
                train_images, train_masks = next(train_iter)
                
                # Convert a few samples to numpy for visualization
                num_samples = min(5, len(train_images))
                sample_images = []
                sample_titles = []
                
                for i in range(num_samples):
                    # Get original image
                    img = train_images[i].numpy().transpose(1, 2, 0)
                    
                    # Denormalize if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    sample_images.append(img)
                    sample_titles.append(f"Sample {i+1}")
                
                # Log sample grid
                wandb_logger.log_image_grid(
                    sample_images,
                    sample_titles,
                    "training_samples"
                )
            except Exception as e:
                logger.warning(f"Could not log sample images: {e}")
        
        logger.log_step_end("preprocess")
    
    # Step 3: Train model
    if 'train' in steps:
        logger.log_step_start("train")
        
        # Create model
        model = create_model(config.model)
        logger.info(f"Created model: {config.model.architecture} with {config.model.encoder} encoder")
        
        # Load checkpoint if fine-tuning
        if hasattr(config.training, 'finetune') and config.training.finetune:
            if hasattr(config.training, 'checkpoint_path') and config.training.checkpoint_path:
                model = load_checkpoint(model, config.training.checkpoint_path)
                logger.info(f"Loaded checkpoint for fine-tuning: {config.training.checkpoint_path}")
        
        # Log model to W&B
        if wandb_logger:
            wandb_logger.log_model(model)
        
        # Make sure dataloaders are created
        if 'preprocess' not in steps:
            # Load dataset
            if not hasattr(locals(), 'df') or df is None:
                dataset_path = config.data.dataset_file
                if not dataset_path:
                    logger.error("No dataset file specified and 'load' step not executed")
                    if wandb_logger:
                        wandb_logger.finish()
                    return 1
                
                try:
                    df = pd.read_csv(dataset_path)
                    logger.info(f"Loaded dataset from {dataset_path} with {len(df)} samples")
                except Exception as e:
                    logger.error(f"Error loading dataset from {dataset_path}: {e}")
                    if wandb_logger:
                        wandb_logger.finish()
                    return 1
            
            # Create dataloaders
            dataloaders = create_dataloaders(
                df=df,
                transform_train=get_augmentations(config.preprocessing, is_train=True),
                transform_val=get_augmentations(config.preprocessing, is_train=False),
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
                target_size=(config.preprocessing.image_size, config.preprocessing.image_size),
                cache_size=getattr(config.data, 'cache_size', 100) if hasattr(config.data, 'cache_size') else 100,
                prefetch_size=getattr(config.data, 'prefetch_size', 0) if hasattr(config.data, 'prefetch_size') else 0,
                mode='segmentation'
            )
        
        # Create trainer (with wandb_logger)
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            config=config.training,
            logger=logger,
            checkpoint_dir=str(checkpoint_dir),
            wandb_logger=wandb_logger  # Pass wandb_logger to trainer
        )
        
        # Train model
        train_results = trainer.train()
        
        # Log results
        logger.info(f"Training completed: {train_results}")
        
        # Save final model
        final_model_path = checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(config) if hasattr(config, '__dict__') else config,
            'results': train_results
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Log to W&B
        if wandb_logger:
            wandb_logger.log({k: v for k, v in train_results.items() 
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
            wandb_logger.save_model(
                model, 
                name="final_model",
                metadata={'epochs': config.training.epochs, **{k: v for k, v in train_results.items() 
                                                              if isinstance(v, (int, float)) and not isinstance(v, bool)}}
            )
            
            # Log training metrics over time if available
            if 'metrics_history' in train_results:
                wandb_logger.log_metrics_over_time(train_results['metrics_history'])
        
        logger.log_step_end("train")
    
    # Step 4: Evaluate model
    if 'evaluate' in steps:
        logger.log_step_start("evaluate")
        
        # Get or create model
        if 'train' not in steps:
            # Create fresh model
            model = create_model(config.model)
            
            # Load checkpoint
            checkpoint_path = config.model.checkpoint_path
            if not checkpoint_path and checkpoint_dir.exists():
                # Try to find best model in checkpoint dir
                best_model_path = checkpoint_dir / "best_model.pt"
                if best_model_path.exists():
                    checkpoint_path = str(best_model_path)
                else:
                    final_model_path = checkpoint_dir / "final_model.pt"
                    if final_model_path.exists():
                        checkpoint_path = str(final_model_path)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                model = load_checkpoint(model, checkpoint_path)
                logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
            else:
                logger.warning("No checkpoint specified or found, using untrained model")
        
        # Make sure dataloaders are created
        if 'preprocess' not in steps and 'train' not in steps:
            # Load dataset
            if not hasattr(locals(), 'df') or df is None:
                dataset_path = config.data.dataset_file
                if not dataset_path:
                    logger.error("No dataset file specified and 'load' step not executed")
                    if wandb_logger:
                        wandb_logger.finish()
                    return 1
                
                try:
                    df = pd.read_csv(dataset_path)
                    logger.info(f"Loaded dataset from {dataset_path} with {len(df)} samples")
                except Exception as e:
                    logger.error(f"Error loading dataset from {dataset_path}: {e}")
                    if wandb_logger:
                        wandb_logger.finish()
                    return 1
            
            # Create dataloaders
            dataloaders = create_dataloaders(
                df=df,
                transform_train=get_augmentations(config.preprocessing, is_train=True),
                transform_val=get_augmentations(config.preprocessing, is_train=False),
                batch_size=config.evaluation.batch_size,
                num_workers=config.training.num_workers,
                target_size=(config.preprocessing.image_size, config.preprocessing.image_size),
                cache_size=getattr(config.data, 'cache_size', 100) if hasattr(config.data, 'cache_size') else 100,
                prefetch_size=getattr(config.data, 'prefetch_size', 0) if hasattr(config.data, 'prefetch_size') else 0,
                mode='segmentation'
            )
        
        # Create evaluator with wandb_logger
        evaluator = Evaluator(
            model=model,
            dataloader=dataloaders['test'],
            config=config.evaluation,
            logger=logger,
            output_dir=str(output_dir),
            wandb_logger=wandb_logger  # Pass wandb_logger to evaluator
        )
        
        # Evaluate model
        eval_results = evaluator.evaluate()
        
        # Log results
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save evaluation results
        eval_path = output_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump({k: v for k, v in eval_results.items() 
                      if not isinstance(v, dict) or k in ['confusion_matrix']}, 
                     f, indent=4, default=str)
        
        # Note: WandB logging is now handled inside the Evaluator class
        
        logger.log_step_end("evaluate")
    
    # Step 5: Ensemble evaluation (if configured)
    if 'ensemble' in steps:
        logger.log_step_start("ensemble")
        
        # Check if ensemble configuration exists
        if not hasattr(config, 'ensemble') or not hasattr(config.ensemble, 'models'):
            logger.error("Ensemble configuration not found")
            if wandb_logger:
                wandb_logger.finish()
            return 1
        
        # Create list of models
        ensemble_models = []
        
        for model_config in config.ensemble.models:
            # Create model
            model = create_model(model_config.model)
            
            # Load checkpoint
            if hasattr(model_config, 'checkpoint_path') and model_config.checkpoint_path:
                model = load_checkpoint(model, model_config.checkpoint_path)
                logger.info(f"Loaded ensemble model from {model_config.checkpoint_path}")
            
            ensemble_models.append(model)
        
        logger.info(f"Created ensemble with {len(ensemble_models)} models")
        
        # Make sure dataloaders are created
        if not hasattr(locals(), 'dataloaders') or dataloaders is None:
            # Load dataset
            if not hasattr(locals(), 'df') or df is None:
                dataset_path = config.data.dataset_file
                try:
                    df = pd.read_csv(dataset_path)
                    logger.info(f"Loaded dataset from {dataset_path} with {len(df)} samples")
                except Exception as e:
                    logger.error(f"Error loading dataset from {dataset_path}: {e}")
                    if wandb_logger:
                        wandb_logger.finish()
                    return 1
            
            # Create dataloaders
            dataloaders = create_dataloaders(
                df=df,
                transform_train=get_augmentations(config.preprocessing, is_train=True),
                transform_val=get_augmentations(config.preprocessing, is_train=False),
                batch_size=config.evaluation.batch_size,
                num_workers=config.training.num_workers,
                target_size=(config.preprocessing.image_size, config.preprocessing.image_size),
                cache_size=getattr(config.data, 'cache_size', 100) if hasattr(config.data, 'cache_size') else 100,
                prefetch_size=getattr(config.data, 'prefetch_size', 0) if hasattr(config.data, 'prefetch_size') else 0,
                mode='segmentation'
            )
        
        # Create evaluator with wandb_logger
        ensemble_output_dir = output_dir / "ensemble"
        ensemble_output_dir.mkdir(exist_ok=True)
        
        evaluator = Evaluator(
            model=None,  # No single model for ensemble evaluation
            dataloader=dataloaders['test'],
            config=config.evaluation,
            logger=logger,
            output_dir=str(ensemble_output_dir),
            wandb_logger=wandb_logger  # Pass wandb_logger to evaluator
        )
        
        # Evaluate ensemble
        ensemble_results = evaluator.evaluate_ensemble(ensemble_models)
        
        # Log results
        logger.info(f"Ensemble evaluation results: {ensemble_results}")
        
        # Save evaluation results
        ensemble_path = ensemble_output_dir / "ensemble_results.json"
        with open(ensemble_path, 'w') as f:
            json.dump({k: v for k, v in ensemble_results.items() 
                      if not isinstance(v, dict) or k in ['confusion_matrix', 'ensemble_comparison']}, 
                     f, indent=4, default=str)
        
        # Note: WandB logging is now handled inside the Evaluator class
        
        logger.log_step_end("ensemble")
    
    # Finish W&B run if enabled
    if wandb_logger:
        wandb_logger.finish()
    
    # Log run summary
    summary_path = output_dir / "summary.json"
    logger.log_run_summary(str(summary_path))
    
    # Update run notebook
    notebook_path = Path(config.output_dir) / "notebook.md"
    logger.update_notebook(str(notebook_path))
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())