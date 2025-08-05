"""
Main experiment runner for Vec2Summ.

This module orchestrates the complete vec2summ experiment pipeline:
1. Data loading and preprocessing
2. Embedding generation
3. Distribution parameter calculation
4. Sampling and reconstruction
5. Evaluation and comparison
"""

import argparse
import json
import logging
import os
import numpy as np
import torch
import vec2text
from transformers import AutoModel, AutoTokenizer

from .core.embeddings import get_gtr_embeddings, get_openai_embeddings
from .core.distribution import calculate_distribution_params, sample_from_distribution
from .core.reconstruction import reconstruct_text_from_embeddings
from .core.summarization import generate_vec2summ_summary, generate_llm_summary, compare_summaries_geval
from .data.loader import load_data
from .evaluation.metrics import evaluate_reconstruction, evaluate_coverage_geval
from .utils.visualization import visualize_embeddings
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_vec2summ_experiment(args):
    """
    Run the complete vec2summ experiment.
    
    Args:
        args: Parsed command-line arguments containing experiment configuration
    """
    logger.info("Starting Vec2Summ experiment")
    logger.info(f"Arguments: {args}")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    logger.info(f"Using device: {device}")
    
    # Set OpenAI API key if provided
    if args.openai_api_key:
        import openai
        openai.api_key = args.openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_path = args.data_path
    
    # Handle Amazon reviews dataset
    if args.amazon_reviews:
        logger.info("Processing Amazon book reviews dataset")
        texts = load_data(
            data_path, 
            data_type="tsv", 
            text_column="review_body",
            max_samples=args.max_samples,
            is_amazon_reviews=True,
            amazon_sample_size=args.amazon_sample_size,
            random_seed=args.random_seed
        )
        
        # Create basic metadata for Amazon reviews
        amazon_metadata = {
            'product_id': "amazon_reviews_processed",
            'total_reviews': len(texts),
            'cleaned_texts_count': len(texts)
        }
        
        # Save metadata to output directory
        with open(os.path.join(args.output_dir, "amazon_metadata.json"), "w") as f:
            json.dump(amazon_metadata, f, indent=2)
            
        logger.info(f"Processed Amazon reviews with {len(texts)} texts")
    else:
        # Handle standard data formats
        data_type = data_path.split(".")[-1].lower()
        if data_type == "tsv":
            texts = load_data(data_path, data_type="tsv", text_column=args.text_column, max_samples=args.max_samples)
        elif data_type == "csv":
            texts = load_data(data_path, data_type="csv", text_column=args.text_column, max_samples=args.max_samples)
        elif data_type in ["json", "jsonl"]:
            texts = load_data(data_path, data_type="json", text_column=args.text_column, max_samples=args.max_samples)
        else:
            logger.error(f"Unsupported data type: {data_type}")
            return
    
    if not texts:
        logger.error("No texts loaded. Exiting.")
        return
    
    # Initialize models based on embedding type
    if args.embedding_type == "openai":
        # For OpenAI embeddings
        embeddings = get_openai_embeddings(texts, model=args.openai_model)
        corrector = vec2text.load_pretrained_corrector(args.openai_model)
        embedding_model = "openai"
    else:
        # For GTR embeddings
        encoder = AutoModel.from_pretrained(args.gtr_model).encoder.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.gtr_model)
        embeddings = get_gtr_embeddings(texts, encoder, tokenizer, device=device)
        corrector = vec2text.load_pretrained_corrector("gtr-base")
        embedding_model = (encoder, tokenizer)
    
    # Calculate distribution parameters
    mean_vector, covariance_matrix = calculate_distribution_params(embeddings)
    
    # Sample from distribution
    sampled_vectors = sample_from_distribution(mean_vector, covariance_matrix, n_samples=args.n_samples)
    
    # Reconstruct text from sampled vectors
    reconstructed_texts = reconstruct_text_from_embeddings(sampled_vectors, corrector, device=device)
    
    # Save reconstructed texts
    with open(os.path.join(args.output_dir, "reconstructed_texts.txt"), "w") as f:
        for text in reconstructed_texts:
            f.write(text + "\n")
    
    # Generate Vec2Summ summary (GPT summarizing the reconstructed texts)
    if args.generate_summaries:
        # Pass Amazon metadata if available
        amazon_metadata = None
        if args.amazon_reviews and os.path.exists(os.path.join(args.output_dir, "amazon_metadata.json")):
            try:
                with open(os.path.join(args.output_dir, "amazon_metadata.json"), "r") as f:
                    amazon_metadata = json.load(f)
                logger.info(f"Using Amazon metadata for product {amazon_metadata['product_id']} in summary generation")
            except Exception as e:
                logger.warning(f"Failed to load Amazon metadata: {e}")
        
        vec2summ_summary = generate_vec2summ_summary(
            reconstructed_texts,
            openai_api_key=args.openai_api_key,
            amazon_metadata=amazon_metadata
        )
        
        if vec2summ_summary:
            # Save Vec2Summ summary
            with open(os.path.join(args.output_dir, "vec2summ_summary.txt"), "w") as f:
                f.write(vec2summ_summary)
            
            logger.info("Vec2Summ summary generated and saved")
            
            # Generate direct LLM summary for comparison
            direct_llm_summary = generate_llm_summary(
                texts, 
                max_length=len(vec2summ_summary),
                openai_api_key=args.openai_api_key
            )
            
            if direct_llm_summary:
                # Save direct LLM summary
                with open(os.path.join(args.output_dir, "direct_llm_summary.txt"), "w") as f:
                    f.write(direct_llm_summary)
                
                logger.info("Direct LLM summary generated and saved")
                
                # Compare summaries using G-Eval
                comparison_results = compare_summaries_geval(
                    vec2summ_summary,
                    direct_llm_summary,
                    texts,
                    openai_api_key=args.openai_api_key
                )
                
                if comparison_results:
                    # Save comparison results
                    with open(os.path.join(args.output_dir, "summary_comparison.json"), "w") as f:
                        json.dump(comparison_results, f, indent=2)
                    
                    logger.info(f"Summary comparison completed. Vec2Summ score: {comparison_results['vec2summ_score']}/5, Direct LLM score: {comparison_results['direct_llm_score']}/5")
    
    # Evaluate reconstruction quality
    if args.evaluate:
        # Use a subset of original texts for evaluation
        eval_texts = texts[:len(reconstructed_texts)]
        metrics = evaluate_reconstruction(eval_texts, reconstructed_texts, embedding_model)
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save metrics
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Add G-Eval coverage evaluation
        if args.evaluate_coverage:
            coverage_metrics = evaluate_coverage_geval(
                eval_texts, 
                reconstructed_texts,
                openai_api_key=args.openai_api_key
            )
            
            if coverage_metrics:
                if "normalized_score" in coverage_metrics and coverage_metrics['normalized_score'] is not None and coverage_metrics['coverage_score'] is not None:
                    logger.info(f"G-Eval Coverage score: {coverage_metrics['coverage_score']}/5 ({coverage_metrics['normalized_score']:.2f}%)")
                elif coverage_metrics.get('coverage_score') is not None:
                    logger.info(f"G-Eval Coverage score: {coverage_metrics['coverage_score']:.2f}%")
                else:
                    logger.info("G-Eval Coverage score: Unable to calculate score (received None values)")
                
                # Save coverage metrics
                with open(os.path.join(args.output_dir, "coverage_metrics.json"), "w") as f:
                    json.dump(coverage_metrics, f, indent=2)
    
    # Visualize embeddings
    if args.visualize:
        # Get embeddings for reconstructed texts for visualization
        if args.embedding_type == "openai":
            reconstructed_embeddings = get_openai_embeddings(reconstructed_texts, model=args.openai_model)
        else:
            reconstructed_embeddings = get_gtr_embeddings(reconstructed_texts, encoder, tokenizer, device=device)
        
        visualize_embeddings(
            embeddings, 
            torch.tensor(sampled_vectors), 
            reconstructed_embeddings,
            save_path=os.path.join(args.output_dir, "embeddings_visualization.png")
        )
    
    # Save distribution parameters
    np.save(os.path.join(args.output_dir, "mean_vector.npy"), mean_vector)
    np.save(os.path.join(args.output_dir, "covariance_matrix.npy"), covariance_matrix)
    
    logger.info("Vec2Summ experiment completed successfully")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Vec2Summ: Text Summarization via Probabilistic Sentence Embeddings")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Column name containing text (for CSV/TSV files)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    
    # Amazon reviews specific parameters
    parser.add_argument("--amazon_reviews", action="store_true",
                        help="Process data as Amazon book reviews dataset")
    parser.add_argument("--amazon_sample_size", type=int, default=50,
                        help="Number of reviews to sample for a selected book (for Amazon reviews)")

    
    # Embedding parameters
    parser.add_argument("--embedding_type", type=str, choices=["openai", "gtr"], default="openai",
                        help="Type of embeddings to use")
    parser.add_argument("--openai_model", type=str, default="text-embedding-ada-002",
                        help="OpenAI embedding model to use")
    parser.add_argument("--gtr_model", type=str, default="sentence-transformers/gtr-t5-base",
                        help="GTR model to use")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key (if not set in environment)")
    
    # Sampling parameters
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to generate from distribution")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    
    # Evaluation and visualization
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate reconstruction quality")
    parser.add_argument("--evaluate_coverage", action="store_true",
                        help="Evaluate coverage using G-Eval")
    parser.add_argument("--generate_summaries", action="store_true",
                        help="Generate and compare Vec2Summ and direct LLM summaries")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize embeddings")
    
    # Misc parameters
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Use CPU only, even if GPU is available")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    args = parse_args()
    run_vec2summ_experiment(args)
