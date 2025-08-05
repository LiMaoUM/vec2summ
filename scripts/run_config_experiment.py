#!/usr/bin/env python3
"""
Configuration-based experiment runner for Vec2Summ.

This script allows running experiments using YAML configuration files
instead of command-line arguments.
"""

import argparse
import sys
import os
import yaml
from dataclasses import dataclass, fields
from typing import Optional

# Add src to path to import vec2summ
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vec2summ.experiment import run_vec2summ_experiment
from vec2summ.utils.logging_config import setup_logging


@dataclass
class ExperimentConfig:
    """Configuration class matching the experiment arguments."""
    # Data parameters
    data_path: str
    text_column: Optional[str] = None
    max_samples: Optional[int] = None
    amazon_reviews: bool = False
    amazon_sample_size: int = 50
    
    # Embedding parameters
    embedding_type: str = "openai"
    openai_model: str = "text-embedding-ada-002"
    gtr_model: str = "sentence-transformers/gtr-t5-base"
    openai_api_key: Optional[str] = None
    
    # Sampling parameters
    n_samples: int = 10
    
    # Output parameters
    output_dir: str = "results"
    
    # Evaluation and visualization
    evaluate: bool = True
    evaluate_coverage: bool = True
    generate_summaries: bool = True
    visualize: bool = True
    
    # Misc parameters
    random_seed: int = 42
    cpu_only: bool = False


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten the nested structure
    flat_config = {}
    
    # Handle data section
    if 'data' in config_dict:
        flat_config.update({
            'data_path': config_dict['data']['path'],
            'text_column': config_dict['data'].get('text_column'),
            'max_samples': config_dict['data'].get('max_samples'),
            'amazon_reviews': config_dict['data'].get('amazon_reviews', False),
            'amazon_sample_size': config_dict['data'].get('amazon_sample_size', 50),
        })
    
    # Handle model section
    if 'model' in config_dict:
        flat_config.update({
            'embedding_type': config_dict['model'].get('embedding_type', 'openai'),
            'openai_model': config_dict['model'].get('openai_model', 'text-embedding-ada-002'),
            'gtr_model': config_dict['model'].get('gtr_model', 'sentence-transformers/gtr-t5-base'),
            'n_samples': config_dict['model'].get('n_samples', 10),
        })
    
    # Handle evaluation section
    if 'evaluation' in config_dict:
        flat_config.update({
            'evaluate': config_dict['evaluation'].get('enabled', True),
            'evaluate_coverage': config_dict['evaluation'].get('coverage_eval', True),
            'generate_summaries': config_dict['evaluation'].get('generate_summaries', True),
            'visualize': config_dict['evaluation'].get('visualize', True),
        })
    
    # Handle output section
    if 'output' in config_dict:
        flat_config.update({
            'output_dir': config_dict['output'].get('dir', 'results'),
        })
    
    # Handle misc settings
    flat_config.update({
        'random_seed': config_dict.get('random_seed', 42),
        'cpu_only': config_dict.get('cpu_only', False),
    })
    
    # Create config object
    return ExperimentConfig(**flat_config)


class SimpleNamespace:
    """Simple namespace object to hold configuration."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run Vec2Summ experiment from config file")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (overrides config)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override API key if provided
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    
    # Convert to namespace object (to match experiment function expectations)
    config_ns = SimpleNamespace(**config.__dict__)
    
    print(f"Running experiment with configuration from: {args.config}")
    print(f"Embedding type: {config.embedding_type}")
    print(f"Output directory: {config.output_dir}")
    
    # Run experiment
    run_vec2summ_experiment(config_ns)


if __name__ == "__main__":
    main()
