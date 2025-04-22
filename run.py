"""
Run Script

Entry point for running the glaucoma detection pipeline with command line arguments,
including run tracking to avoid repeating configurations.
"""

import sys
from pipeline import run_pipeline
from config import get_argument_parser, parse_args_and_create_config
from run_tracker import get_run_tracker

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
    
    # Add custom help messages
    parser.description = """
    Glaucoma Detection Pipeline
    
    This script runs the glaucoma detection pipeline for segmentation of optic disc and cup in retinal images.
    It supports various model architectures, data augmentation, and evaluation metrics.
    
    The pipeline tracks completed runs to avoid repeating configurations, which is helpful for resuming 
    interrupted batch runs. Use --force-rerun to ignore this feature.
    
    Examples:
      # Run with default settings
      python run.py
      
      # Run with custom configuration file
      python run.py --config configs/my_config.json
      
      # Run specific steps only
      python run.py --steps load,train,evaluate
      
      # Train with specific model architecture and encoder
      python run.py --architecture deeplabv3plus --encoder efficientnet-b0
      
      # Run with Weights & Biases logging (enabled by default)
      python run.py --wandb-project glaucoma-detection --wandb-name "unet_resnet34_run1"
      
      # List all completed runs
      python run.py --list-completed-runs
      
      # Clear run history
      python run.py --clear-run-history
      
      # Force rerun of a configuration that has already been completed
      python run.py --force-rerun
    """
    
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