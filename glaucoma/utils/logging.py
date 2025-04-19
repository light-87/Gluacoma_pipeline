"""
Logging Utilities

Enhanced logging setup for the glaucoma detection pipeline.
"""

import os
import logging
import sys
from pathlib import Path
import datetime
from typing import Dict, List, Optional, Union, Any

# Define logging levels
LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

class LogFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m', # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[1;31m', # Bold Red
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Check if we're outputting to a terminal
        is_terminal = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        if is_terminal:
            levelname = record.levelname
            message = super().format(record)
            if levelname in self.COLORS:
                color = self.COLORS[levelname]
                reset = self.COLORS['RESET']
                # Only colorize the level name, not the whole message
                message = message.replace(levelname, f"{color}{levelname}{reset}", 1)
            return message
        else:
            return super().format(record)


def setup_logger(
    name: str,
    level: Union[str, int] = 'info',
    log_file: Optional[str] = None,
    console: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Logger name
        level: Logging level (debug, info, warning, error, critical)
        log_file: Log file name (if None, a default name is generated)
        console: Whether to log to console
        log_dir: Directory to store log files
        
    Returns:
        Configured logger
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Set level
    if isinstance(level, str):
        level = LEVELS.get(level.lower(), logging.INFO)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = LogFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested or log_dir is provided
    if log_file or log_dir:
        if log_dir:
            # Create log directory if it doesn't exist
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate default log file name if none provided
            if not log_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"{name.replace('.', '_')}_{timestamp}.log"
            
            log_path = log_dir / log_file
        else:
            log_path = log_file
        
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class PipelineLogger:
    """
    Centralized logging for the pipeline with metadata and run tracking.
    
    Features:
    - Automatic context tracking
    - Run metadata
    - Performance metrics logging
    - Integration with experiment tracking
    """
    
    def __init__(
        self,
        name: str = 'pipeline',
        level: str = 'info',
        log_dir: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        run_id: Optional[str] = None
    ):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Directory to store log files
            enable_console: Whether to log to console
            enable_file: Whether to log to file
            run_id: Optional run identifier
        """
        self.name = name
        self.level = level
        self.log_dir = Path(log_dir) if log_dir else None
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Generate run ID if not provided
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log file name
        log_file = f"{name}_{self.run_id}.log" if enable_file else None
        
        # Setup logger
        self.logger = setup_logger(
            name,
            level=level,
            log_file=log_file,
            console=enable_console,
            log_dir=str(self.log_dir) if self.log_dir else None
        )
        
        # Store context
        self.context = {'run_id': self.run_id}
        
        # Performance metrics
        self.metrics = {}
        
        # Store start time
        self.start_time = datetime.datetime.now()
        self.logger.info(f"Pipeline run {self.run_id} started at {self.start_time}")
    
    def set_context(self, **kwargs) -> None:
        """
        Set context information for logging.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context information except run_id."""
        run_id = self.context.get('run_id')
        self.context = {'run_id': run_id} if run_id else {}
    
    def _format_message(self, msg: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format message with context.
        
        Args:
            msg: Message to format
            context: Additional context specific to this message
            
        Returns:
            Formatted message with context
        """
        if context or self.context:
            combined_context = {**self.context, **(context or {})}
            context_str = ' | '.join(f"{k}={v}" for k, v in combined_context.items())
            return f"{msg} | {context_str}"
        return msg
    
    def debug(self, msg: str, **kwargs) -> None:
        """
        Log debug message.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.debug(self._format_message(msg, kwargs))
    
    def info(self, msg: str, **kwargs) -> None:
        """
        Log info message.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.info(self._format_message(msg, kwargs))
    
    def warning(self, msg: str, **kwargs) -> None:
        """
        Log warning message.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.warning(self._format_message(msg, kwargs))
    
    def error(self, msg: str, **kwargs) -> None:
        """
        Log error message.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.error(self._format_message(msg, kwargs))
    
    def critical(self, msg: str, **kwargs) -> None:
        """
        Log critical message.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.critical(self._format_message(msg, kwargs))
    
    def exception(self, msg: str, **kwargs) -> None:
        """
        Log exception message with traceback.
        
        Args:
            msg: Message to log
            **kwargs: Additional context for this message
        """
        self.logger.exception(self._format_message(msg, kwargs))
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if step is not None:
            self.metrics.setdefault(name, {})[step] = value
            self.info(f"Metric {name} = {value} at step {step}")
        else:
            self.metrics[name] = value
            self.info(f"Metric {name} = {value}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_step_start(self, step_name: str) -> None:
        """
        Log the start of a pipeline step.
        
        Args:
            step_name: Name of the step
        """
        self.set_context(step=step_name)
        self.info(f"Starting step: {step_name}")
    
    def log_step_end(self, step_name: str) -> None:
        """
        Log the end of a pipeline step.
        
        Args:
            step_name: Name of the step
        """
        self.info(f"Completed step: {step_name}")
    
    def log_run_summary(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a summary of the pipeline run and optionally save to file.
        
        Args:
            output_path: Path to save the summary
            
        Returns:
            Dictionary with run summary
        """
        # Calculate elapsed time
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        
        # Create summary
        summary = {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'elapsed_time': elapsed_time.total_seconds(),
            'context': self.context,
            'metrics': self.metrics
        }
        
        # Log summary
        self.info(f"Pipeline run {self.run_id} completed in {elapsed_time}")
        
        # Save to file if requested
        if output_path:
            import json
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.info(f"Run summary saved to {output_path}")
        
        return summary
    
    def update_notebook(self, notebook_path: str) -> None:
        """
        Update a Markdown notebook with run information.
        
        Args:
            notebook_path: Path to the notebook file
        """
        notebook_path = Path(notebook_path)
        
        # Create file if it doesn't exist
        if not notebook_path.exists():
            with open(notebook_path, 'w') as f:
                f.write("# Glaucoma Detection Pipeline Run Log\n\n")
                f.write("| Run ID | Start Time | Duration | Context | Metrics |\n")
                f.write("|--------|------------|----------|---------|--------|\n")
        
        # Calculate elapsed time
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        
        # Format context and metrics
        context_str = ', '.join(f"{k}={v}" for k, v in self.context.items() if k != 'run_id')
        metrics_str = ', '.join(f"{k}={v}" for k, v in self.metrics.items() if isinstance(v, (int, float)))
        
        # Format start time
        start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create row
        row = f"| {self.run_id} | {start_time_str} | {elapsed_time} | {context_str} | {metrics_str} |\n"
        
        # Append to file
        with open(notebook_path, 'a') as f:
            f.write(row)
        
        self.info(f"Updated run notebook at {notebook_path}")


def get_logger(
    name: str = 'glaucoma',
    level: str = 'info',
    log_dir: Optional[str] = None
) -> PipelineLogger:
    """
    Get a pipeline logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory to store log files
        
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name=name, level=level, log_dir=log_dir)


if __name__ == "__main__":
    # Example usage
    logger = get_logger(log_dir="logs")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Set context
    logger.set_context(dataset="ORIGA", model="UNet")
    logger.info("Processing dataset")
    
    # Log metrics
    logger.log_metric("accuracy", 0.95)
    logger.log_metric("loss", 0.05, step=1)
    
    # Log steps
    logger.log_step_start("preprocessing")
    # ... do preprocessing ...
    logger.log_step_end("preprocessing")
    
    # Log run summary
    logger.log_run_summary("logs/summary.json")
    
    # Update notebook
    logger.update_notebook("logs/notebook.md")