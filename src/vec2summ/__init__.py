"""
Vec2Summ: Text Summarization via Probabilistic Sentence Embeddings

A novel approach to text summarization that:
1. Embeds text into sentence embeddings
2. Calculates mean vector and covariance matrix
3. Samples from that distribution to get vectors
4. Reconstructs text from the sampled vectors

Author: Mao Li
"""

from .core.embeddings import get_gtr_embeddings, get_openai_embeddings
from .core.distribution import calculate_distribution_params, sample_from_distribution
from .core.reconstruction import reconstruct_text_from_embeddings
from .core.summarization import generate_vec2summ_summary, generate_llm_summary, compare_summaries_geval
from .data.loader import load_data, process_amazon_reviews
from .data.preprocessing import clean_text, TextDataset
from .evaluation.metrics import evaluate_reconstruction, evaluate_coverage_geval
from .utils.visualization import visualize_embeddings
from .utils.logging_config import setup_logging

__version__ = "1.0.0"
__author__ = "Mao Li"

__all__ = [
    "get_gtr_embeddings",
    "get_openai_embeddings", 
    "calculate_distribution_params",
    "sample_from_distribution",
    "reconstruct_text_from_embeddings",
    "generate_vec2summ_summary",
    "generate_llm_summary",
    "compare_summaries_geval",
    "load_data",
    "process_amazon_reviews",
    "clean_text",
    "TextDataset",
    "evaluate_reconstruction",
    "evaluate_coverage_geval",
    "visualize_embeddings",
    "setup_logging"
]
