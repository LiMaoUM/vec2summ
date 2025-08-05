"""
Evaluation metrics for vec2summ experiments.

This module provides various evaluation metrics including reconstruction quality
and coverage evaluation using G-Eval framework.
"""

import logging
import os
import re
import numpy as np
import openai
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from ..core.embeddings import get_gtr_embeddings, get_openai_embeddings

logger = logging.getLogger(__name__)


def evaluate_reconstruction(original_texts, reconstructed_texts, embedding_model):
    """
    Evaluate the quality of reconstructed texts compared to originals.
    
    Args:
        original_texts: List of original text strings
        reconstructed_texts: List of reconstructed text strings
        embedding_model: Either "openai" or tuple of (encoder, tokenizer) for GTR
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating reconstruction quality")
    
    # Get embeddings for both original and reconstructed texts
    if embedding_model == "openai":
        original_embeddings = get_openai_embeddings(original_texts)
        reconstructed_embeddings = get_openai_embeddings(reconstructed_texts)
    else:
        # Assume it's a tuple of (encoder, tokenizer) for GTR
        encoder, tokenizer = embedding_model
        original_embeddings = get_gtr_embeddings(original_texts, encoder, tokenizer)
        reconstructed_embeddings = get_gtr_embeddings(reconstructed_texts, encoder, tokenizer)
    
    # Convert to numpy if they're torch tensors
    if isinstance(original_embeddings, torch.Tensor):
        original_embeddings = original_embeddings.cpu().numpy()
    if isinstance(reconstructed_embeddings, torch.Tensor):
        reconstructed_embeddings = reconstructed_embeddings.cpu().numpy()
    
    # Calculate semantic similarity
    similarities = []
    for orig_emb in original_embeddings:
        # Find most similar reconstructed embedding
        sims = cosine_similarity([orig_emb], reconstructed_embeddings)[0]
        similarities.append(np.max(sims))
    
    # Calculate metrics
    metrics = {
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities))
    }
    
    return metrics


def evaluate_coverage_geval(original_texts, reconstructed_texts, openai_api_key=None, batch_size=200):
    """
    Evaluate coverage using G-Eval framework with gpt-4 as judge.
    
    G-Eval uses carefully designed prompts to evaluate summaries based on specific criteria.
    For large datasets, texts are divided into batches to respect GPT's context window limits.
    
    Args:
        original_texts: List of original texts to evaluate coverage for
        reconstructed_texts: List of reconstructed texts that form the summary
        openai_api_key: Optional API key for OpenAI
        batch_size: Maximum number of texts to include in a single batch (default: 200)
    
    Returns:
        Dictionary containing coverage evaluation metrics
    """
    logger.info("Evaluating coverage using G-Eval framework")
    
    if openai_api_key:
        openai.api_key = openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        logger.error("OpenAI API key not set. Cannot evaluate coverage.")
        return None
    
    # Combine all reconstructed texts into a single summary
    summary = " ".join(reconstructed_texts)
    
    # If the dataset is small enough to fit in a single batch
    if len(original_texts) <= batch_size:
        return _evaluate_single_batch(original_texts, summary)
    
    # For large datasets, break into batches
    logger.info(f"Large-scale evaluation with {len(original_texts)} texts, using batching approach")
    
    # Divide texts into batches of appropriate size
    num_batches = (len(original_texts) + batch_size - 1) // batch_size
    batches = [original_texts[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    logger.info(f"Divided into {num_batches} batches of maximum size {batch_size}")
    
    # Evaluate each batch separately
    batch_scores = []
    batch_results = []
    
    for i, batch in enumerate(tqdm(batches, desc="Evaluating batches")):
        logger.info(f"Evaluating batch {i+1}/{num_batches} with {len(batch)} texts")
        
        # Evaluate this batch
        batch_result = _evaluate_single_batch(batch, summary)
        
        if batch_result and "coverage_score" in batch_result:
            # For traditional g-eval method (1-5 scale)
            if "normalized_score" in batch_result:
                batch_scores.append(batch_result["coverage_score"])
                logger.info(f"Batch {i+1} coverage score: {batch_result['coverage_score']}/5")
            # For large-scale method (percentage)
            else:
                batch_scores.append(batch_result["coverage_score"])
                logger.info(f"Batch {i+1} coverage score: {batch_result['coverage_score']:.2f}%")
            
            batch_results.append(batch_result)
    
    # Calculate average score across all batches
    if batch_scores:
        avg_score = sum(batch_scores) / len(batch_scores)
        
        # Determine if we're using the 1-5 scale or percentage
        is_percentage = all(score > 5 for score in batch_scores if score is not None)
        
        if is_percentage:
            logger.info(f"Average coverage score across all batches: {avg_score:.2f}%")
            return {
                "method": "g-eval-batched",
                "coverage_score": avg_score,
                "batch_count": len(batch_scores),
                "total_corpus_size": len(original_texts),
                "batch_size": batch_size,
                "batch_results": batch_results
            }
        else:
            normalized_avg = (avg_score / 5 * 100) if avg_score else None
            logger.info(f"Average coverage score across all batches: {avg_score:.2f}/5 ({normalized_avg:.2f}%)")
            return {
                "method": "g-eval-batched",
                "coverage_score": avg_score,
                "normalized_score": normalized_avg,
                "batch_count": len(batch_scores),
                "total_corpus_size": len(original_texts),
                "batch_size": batch_size,
                "batch_results": batch_results
            }
    
    logger.error("Failed to calculate coverage scores for any batch")
    return None


def _evaluate_single_batch(texts, summary):
    """
    Helper function to evaluate coverage for a single batch of texts.
    
    Args:
        texts: List of texts in this batch
        summary: The summary to evaluate against
        
    Returns:
        Dictionary containing coverage evaluation metrics for this batch
    """
    # For small-scale evaluation (standard G-Eval approach)
    if len(texts) <= 100:
        # Combine all texts for source document
        source_text = "\n\n".join(texts)
        
        # G-Eval prompt for coverage evaluation
        prompt = f"""
        You are an expert evaluator for text summarization systems.
        
        I'll provide you with a source document and a summary. Your task is to evaluate how well the summary covers the important information in the source document.
        
        Source document:
        {source_text[:8000]}
        
        Summary:
        {summary}
        
        Please evaluate the COVERAGE of this summary on a scale of 1-5, where:
        1: The summary misses almost all important information
        2: The summary misses most important information
        3: The summary covers about half of the important information
        4: The summary covers most important information
        5: The summary covers all or almost all important information
        
        First, identify the key points in the source document.
        Then, check which of these key points are covered in the summary.
        Finally, provide your rating and explanation.
        
        Your response should follow this format:
        Key points in source: [list key points]
        Points covered in summary: [list covered points]
        Points missing in summary: [list missing points]
        Coverage score: [1-5]
        Explanation: [your explanation]
        
        IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        
        evaluation = response.choices[0].message.content.strip()
        
        # Remove any asterisks from the response to avoid parsing issues
        evaluation = evaluation.replace("*", "")
        
        # Extract score using regex
        score_match = re.search(r"Coverage score:\s*(\d+)", evaluation)
        coverage_score = int(score_match.group(1)) if score_match else None
        
        # Extract key points
        key_points_match = re.search(r"Key points in source:(.*?)Points covered", evaluation, re.DOTALL)
        key_points = key_points_match.group(1).strip() if key_points_match else ""
        
        # Extract covered points
        covered_points_match = re.search(r"Points covered in summary:(.*?)Points missing", evaluation, re.DOTALL)
        covered_points = covered_points_match.group(1).strip() if covered_points_match else ""
        
        # Extract missing points
        missing_points_match = re.search(r"Points missing in summary:(.*?)Coverage score", evaluation, re.DOTALL)
        missing_points = missing_points_match.group(1).strip() if missing_points_match else ""
        
        # Extract explanation
        explanation_match = re.search(r"Explanation:(.*)", evaluation, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        return {
            "method": "g-eval",
            "coverage_score": coverage_score,
            "normalized_score": (coverage_score / 5 * 100) if coverage_score is not None else None,
            "key_points": key_points,
            "covered_points": covered_points,
            "missing_points": missing_points,
            "explanation": explanation,
            "full_evaluation": evaluation
        }
    
    # For larger batches, use the sampling approach
    else:
        # Sample a subset of texts for evaluation
        sample_size = min(100, len(texts) // 10)  # 10% or max 100
        sampled_indices = np.random.choice(len(texts), size=sample_size, replace=False)
        sampled_texts = [texts[i] for i in sampled_indices]
        
        # Evaluate coverage for each sampled text
        covered_count = 0
        results = []
        
        for i, text in enumerate(tqdm(sampled_texts, desc="Evaluating coverage")):
            # Skip very long texts
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            prompt = f"""
            Here is a summary generated from a large collection of texts:
            
            {summary}
            
            Here is one of the original texts from the collection:
            
            {text}
            
            Question: Is the key information or sentiment expressed in this text captured or reflected in the summary above? Answer YES or NO, and explain briefly why.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=150
            )
            
            answer = response.choices[0].message.content.strip()
            is_covered = "YES" in answer.upper().split()
            
            if is_covered:
                covered_count += 1
            
            results.append({
                "text_id": sampled_indices[i],
                "is_covered": is_covered,
                "explanation": answer
            })
        
        coverage_score = covered_count / len(sampled_texts) * 100
        
        return {
            "method": "g-eval-large-scale",
            "coverage_score": coverage_score,
            "covered_count": covered_count,
            "total_evaluated": len(sampled_texts),
            "total_corpus_size": len(texts),
            "detailed_results": results
        }
