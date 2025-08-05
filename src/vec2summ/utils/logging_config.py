"""
Logging configuration for vec2summ experiments.

This module provides consistent logging setup across the project.
"""

import logging
import os
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None, log_dir="logs"):
    """
    Set up logging configuration for the experiment.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file name (if None, generates timestamp-based name)
        log_dir: Directory to save log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"vec2summ_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure logging to both file and console with proper formatting
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return log_path
