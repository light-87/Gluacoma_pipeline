"""
Main Entry Point for Glaucoma Detection Pipeline

This script serves as the entry point for the glaucoma detection pipeline,
handling command-line arguments and coordinating the execution of pipeline steps.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import datetime
from typing import List

from glaucoma.config import parse_args_and_create_config, Config
from glaucoma.data.loader import DatasetLoader, save_dataset
from glaucoma.utils.logging import get_logger

def main():
    """Main entry point for the pipeline."""
    # Parse arguments and create configuration
    config = parse_args_and_create_config()
    
    # Create run ID based on timestamp
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = Path(config.pipeline.output_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = get_logger(name="glaucoma", level="info", log_dir=str(log_dir))
    logger.set_context(run_id=run_id)
    
    # Save configuration
    config_path = output_dir / "config.json"
    config.save(config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Get pipeline steps to execute
    steps = config.pipeline.steps
    logger.info(f"Will execute steps: {', '.join(steps)}")
    
    # Step 1: Load data
    if 'load' in steps:
        logger.log_step_start("load")
        
        # Create data loader
        data_loader = DatasetLoader(config.data.data_dirs)
        
        # Load all datasets
        df = data_loader.load_all_datasets()
        
        if df.empty:
            logger.error("No data loaded. Check data directories.")
            return 1
        
        # Save consolidated dataset
        dataset_path = output_dir / "dataset.csv"
        save_dataset(
            df,
            dataset_path,
            create_splits=True,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            random_state=config.data.random_state
        )
        
        # Log dataset statistics
        stats = data_loader.analyze_dataset(df)
        logger.info(f"Dataset statistics: {stats}")
        
        logger.log_step_end("load")
    
    # Step 2: Clean (to be implemented)
    if 'clean' in steps:
        logger.log_step_start("clean")
        # TODO: Implement data cleaning step
        logger.info("Data cleaning not yet implemented")
        logger.log_step_end("clean")
    
    # Step 3: Preprocess (to be implemented)
    if 'preprocess' in steps:
        logger.log_step_start("preprocess")
        # TODO: Implement preprocessing step
        logger.info("Preprocessing not yet implemented")
        logger.log_step_end("preprocess")
    
    # Step 4: Train (to be implemented)
    if 'train' in steps:
        logger.log_step_start("train")
        # TODO: Implement training step
        logger.info("Training not yet implemented")
        logger.log_step_end("train")
    
    # Step 5: Evaluate (to be implemented)
    if 'evaluate' in steps:
        logger.log_step_start("evaluate")
        # TODO: Implement evaluation step
        logger.info("Evaluation not yet implemented")
        logger.log_step_end("evaluate")
    
    # Log run summary
    summary_path = output_dir / "summary.json"
    logger.log_run_summary(str(summary_path))
    
    # Update run notebook
    notebook_path = Path(config.pipeline.output_dir) / "notebook.md"
    logger.update_notebook(str(notebook_path))
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())