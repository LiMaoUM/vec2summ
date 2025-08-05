#!/usr/bin/env python3
"""
Command-line entry point for running Vec2Summ experiments.

This script provides a convenient way to run vec2summ experiments with
various configurations and datasets.
"""

import sys
import os

# Add src to path to import vec2summ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vec2summ.experiment import run_vec2summ_experiment, parse_args
from vec2summ.utils.logging_config import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Parse arguments and run experiment
    args = parse_args()
    run_vec2summ_experiment(args)
