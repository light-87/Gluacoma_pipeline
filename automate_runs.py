"""
Automate Runs Script

This script automates multiple runs of the glaucoma detection pipeline with different configurations.
It can be used to perform hyperparameter tuning, architecture comparison, or ablation studies.
"""

import os
import sys
import json
import yaml
import argparse
import itertools
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Any, Optional

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automate multiple pipeline runs with different configurations")
    
    parser.add_argument("--config", type=str, required=True, help="Base configuration file (JSON or YAML)")
    parser.add_argument("--output-dir", type=str, default="output/runs", help="Output directory for all runs")
    parser.add_argument("--wandb-project", type=str, default="glaucoma-detection", help="WandB project name")
    parser.add_argument("--wandb-group", type=str, help="WandB group name for all runs")
    parser.add_argument("--param-grid", type=str, help="Path to parameter grid file (JSON or YAML)")
    parser.add_argument("--models", type=str, help="Path to models list file (JSON or YAML)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--run-name-prefix", type=str, default="", help="Prefix for run names")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith(('.yml', '.yaml')):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")

def load_param_grid(param_grid_path: str) -> Dict[str, List[Any]]:
    """Load parameter grid from file.
    
    Args:
        param_grid_path: Path to parameter grid file (JSON or YAML)
        
    Returns:
        Parameter grid dictionary
    """
    return load_config(param_grid_path)

def load_models_list(models_path: str) -> List[Dict[str, Any]]:
    """Load models list from file.
    
    Args:
        models_path: Path to models list file (JSON or YAML)
        
    Returns:
        List of model configurations
    """
    return load_config(models_path)

def generate_configs(base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate configurations from parameter grid.
    
    Args:
        base_config: Base configuration dictionary
        param_grid: Parameter grid dictionary
        
    Returns:
        List of configuration dictionaries
    """
    # Get parameter names and values
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations of parameters
    combinations = list(itertools.product(*param_values))
    
    # Create configurations for each combination
    configs = []
    for combo in combinations:
        # Create a copy of the base configuration
        config = {**base_config}
        
        # Update with parameter values
        for name, value in zip(param_names, combo):
            keys = name.split('.')
            target = config
            
            # Navigate to the correct nested dictionary
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set the value
            target[keys[-1]] = value
        
        configs.append(config)
    
    return configs

def generate_model_configs(base_config: Dict[str, Any], models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate configurations for different models.
    
    Args:
        base_config: Base configuration dictionary
        models: List of model configurations
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for model_config in models:
        # Create a copy of the base configuration
        config = {**base_config}
        
        # If 'model' is not in the base config, create it
        if 'model' not in config:
            config['model'] = {}
        
        # Update model configuration
        config['model'].update(model_config)
        
        configs.append(config)
    
    return configs

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration to
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
    elif output_path.endswith(('.yml', '.yaml')):
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {output_path}")

def generate_run_commands(configs: List[Dict[str, Any]], args) -> List[str]:
    """Generate run commands for each configuration.
    
    Args:
        configs: List of configuration dictionaries
        args: Command line arguments
        
    Returns:
        List of run commands
    """
    commands = []
    
    # Create timestamp for group of runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, config in enumerate(configs):
        # Create directory for this run
        run_id = f"run_{timestamp}_{i:03d}"
        run_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save configuration to file
        config_path = os.path.join(run_dir, "config.json")
        save_config(config, config_path)
        
        # Generate WandB run name
        if 'model' in config and 'architecture' in config['model'] and 'encoder' in config['model']:
            run_name = f"{args.run_name_prefix}{config['model']['architecture']}_{config['model']['encoder']}_{timestamp}_{i:03d}"
        else:
            run_name = f"{args.run_name_prefix}run_{timestamp}_{i:03d}"
        
        # Generate command
        cmd = [
            "python", "run.py",
            "--config", config_path,
            "--output-dir", run_dir
        ]
        
        # Add WandB arguments if not disabled
        if not args.no_wandb:
            cmd.extend([
                "--wandb-project", args.wandb_project,
                "--wandb-name", run_name
            ])
            
            if args.wandb_group:
                cmd.extend(["--wandb-group", args.wandb_group])
        else:
            cmd.append("--no-wandb")
        
        commands.append(' '.join(cmd))
    
    return commands

def generate_batch_file(commands: List[str], output_path: str):
    """Generate batch file with run commands.
    
    Args:
        commands: List of run commands
        output_path: Path to save batch file to
    """
    # Create batch file content
    if os.name == 'nt':  # Windows
        content = "@echo off\n"
        content += "echo Running %d configurations...\n" % len(commands)
        content += "echo.\n\n"
        
        for i, cmd in enumerate(commands):
            content += "echo Running configuration %d of %d...\n" % (i+1, len(commands))
            content += "%s\n" % cmd
            content += "if %ERRORLEVEL% neq 0 (\n"
            content += "    echo Configuration %d failed with error code %ERRORLEVEL%\n" % (i+1)
            content += "    echo Continuing with next configuration...\n"
            content += ")\n"
            content += "echo.\n\n"
        
        content += "echo All configurations completed.\n"
    else:  # Unix/Linux
        content = "#!/bin/bash\n"
        content += "echo \"Running ${#commands[@]} configurations...\"\n"
        content += "echo\n\n"
        
        for i, cmd in enumerate(commands):
            content += "echo \"Running configuration %d of %d...\"\n" % (i+1, len(commands))
            content += "%s\n" % cmd
            content += "if [ $? -ne 0 ]; then\n"
            content += "    echo \"Configuration %d failed with error code $?\"\n" % (i+1)
            content += "    echo \"Continuing with next configuration...\"\n"
            content += "fi\n"
            content += "echo\n\n"
        
        content += "echo \"All configurations completed.\"\n"
    
    # Save batch file
    with open(output_path, 'w') as f:
        f.write(content)
    
    # Make batch file executable on Unix/Linux
    if os.name != 'nt':
        os.chmod(output_path, 0o755)

def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_arguments()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Initialize configurations list
    configs = [base_config]
    
    # Apply parameter grid if specified
    if args.param_grid:
        param_grid = load_param_grid(args.param_grid)
        configs = generate_configs(base_config, param_grid)
    
    # Apply models list if specified
    if args.models:
        models = load_models_list(args.models)
        configs = generate_model_configs(base_config, models)
    
    # Generate run commands
    commands = generate_run_commands(configs, args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate batch file
    batch_file = os.path.join(args.output_dir, "run_batch.bat" if os.name == 'nt' else "run_batch.sh")
    generate_batch_file(commands, batch_file)
    
    print(f"Generated {len(configs)} configurations")
    print(f"Batch file saved to: {batch_file}")
    print(f"Run the batch file to execute all configurations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())