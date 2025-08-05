"""
Summary generation and comparison using LLMs.

This module provides functions for generating summaries from reconstructed texts
and comparing different summarization approaches.
"""

import logging
import os
import re
import openai
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def generate_vec2summ_summary(reconstructed_texts, openai_api_key=None, amazon_metadata=None):
    """
    Generate a summary from the reconstructed texts using an LLM (gpt-4).
    This is the final step in the Vec2Summ pipeline.
    
    Args:
        reconstructed_texts: List of texts to summarize
        openai_api_key: Optional API key for OpenAI
        amazon_metadata: Optional metadata for Amazon reviews
        
    Returns:
        str: Generated summary or None if failed
    """
    logger.info("Generating Vec2Summ summary from reconstructed texts")
    
    if openai_api_key:
        openai.api_key = openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        logger.error("OpenAI API key not set. Cannot generate Vec2Summ summary.")
        return None
    
    # Combine all reconstructed texts
    reconstructed_text = "\n\n".join(reconstructed_texts)
    
    # Create different prompts based on whether this is Amazon reviews or not
    if amazon_metadata:
        prompt = f"""
        Please summarize the following collection of Amazon book reviews. These reviews are all for the same product (Product ID: {amazon_metadata['product_id']}).
        
        Focus on capturing the main opinions, key praise and criticisms, and overall sentiment about the book.
        The reviews may be fragmented and disjointed, but try to create a coherent summary that would be helpful for someone deciding whether to purchase this book.
        
        IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
        
        Amazon book reviews to summarize:
        {reconstructed_text}
        """
    else:
        prompt = f"""
        Please summarize the following collection of texts. They are often fragmented and disjointed but just focus on capturing the main points, key information, and overall sentiment. The summary should be comprehensive yet concise.
        
        IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
        
        Texts to summarize:
        {reconstructed_text}
        """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    
    summary = response.choices[0].message.content.strip()
    # Remove any asterisks from the response to avoid parsing issues
    summary = summary.replace("*", "")
    return summary


def generate_llm_summary(original_texts, max_length=2000, openai_api_key=None):
    """
    Generate a summary of the original texts using an LLM (gpt-4).
    
    Args:
        original_texts: List of original texts to summarize
        max_length: Maximum length for the summary
        openai_api_key: Optional API key for OpenAI
        
    Returns:
        str: Generated summary or None if failed
    """
    logger.info("Generating LLM summary directly from original texts")
    
    if openai_api_key:
        openai.api_key = openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        logger.error("OpenAI API key not set. Cannot generate LLM summary.")
        return None
    
    # For small-scale evaluation (within context window)
    if len(original_texts) <= 200:
        # Combine all original texts for source document
        source_text = "\n\n".join(original_texts)
        
        prompt = f"""
        Please summarize the following collection of texts. Focus on capturing the main points, 
        key information, and overall sentiment. The summary should be comprehensive yet concise.
        
        IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
        
        Texts to summarize:
        {source_text}  
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        summary = response.choices[0].message.content.strip()
        # Remove any asterisks from the response to avoid parsing issues
        summary = summary.replace("*", "")
        logger.info("Direct LLM summary generated successfully")
        return summary
    
    # For large-scale evaluation (beyond context window)
    else:
        logger.info(f"Large-scale summarization with {len(original_texts)} texts")
        
        # Chunk the texts to fit within context window
        chunk_size = 100  # Adjust based on text length
        chunks = [original_texts[i:i+chunk_size] for i in range(0, len(original_texts), chunk_size)]
        
        chunk_summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc="Generating chunk summaries")):
            chunk_text = "\n\n".join(chunk)
            
            prompt = f"""
            Please summarize the following collection of texts. Focus on capturing the main points, 
            key information, and overall sentiment. The summary should be comprehensive yet concise.
            
            IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
            
            Texts to summarize (chunk {i+1} of {len(chunks)}):
            {chunk_text[:4000]}
            """
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            chunk_summary = response.choices[0].message.content.strip()
            # Remove any asterisks from the response to avoid parsing issues
            chunk_summary = chunk_summary.replace("*", "")
            chunk_summaries.append(chunk_summary)
        
        # Generate a meta-summary of all chunk summaries
        meta_prompt = f"""
        Below are summaries of different chunks of a larger collection of texts.
        Please create a unified summary that captures the key information across all these summaries.
        
        IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
        
        Chunk summaries:
        {' '.join(chunk_summaries)[:8000]}
        """
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        meta_summary = response.choices[0].message.content.strip()
        # Remove any asterisks from the response to avoid parsing issues
        meta_summary = meta_summary.replace("*", "")
        logger.info("Direct LLM summary generated successfully")
        return meta_summary


def compare_summaries_geval(vec2summ_summary, direct_llm_summary, original_texts, openai_api_key=None):
    """
    Compare Vec2Summ-generated summary with direct LLM-generated summary using G-Eval.
    
    Args:
        vec2summ_summary: Summary generated by Vec2Summ method
        direct_llm_summary: Summary generated directly by LLM
        original_texts: Original texts for reference
        openai_api_key: Optional API key for OpenAI
        
    Returns:
        dict: Comparison results including scores and detailed analysis
    """
    logger.info("Comparing Vec2Summ and direct LLM summaries using G-Eval")
    
    if openai_api_key:
        openai.api_key = openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        logger.error("OpenAI API key not set. Cannot compare summaries.")
        return None
    
    # Sample a subset of original texts for reference
    import numpy as np
    sample_size = min(20, len(original_texts))
    sampled_indices = np.random.choice(len(original_texts), size=sample_size, replace=False)
    sampled_texts = [original_texts[i] for i in sampled_indices]
    reference_text = "\n\n".join(sampled_texts)
    
    prompt = f"""
    You are an expert evaluator for text summarization systems. I'll provide you with:
    1. A sample of the original texts
    2. A summary generated by Method A (Vec2Summ)
    3. A summary generated by Method B (Direct LLM)
    
    Please compare these summaries on the following criteria:
    - Coverage: How well each summary captures the key information from the original texts
    - Conciseness: How efficiently each summary conveys information without redundancy
    - Coherence: How well-organized and fluent each summary is
    - Factual accuracy: Whether the summaries contain information not present in the original texts
    
    Sample of original texts:
    {reference_text[:2000]}
    
    Summary from Method A (Vec2Summ):
    {vec2summ_summary}
    
    Summary from Method B (Direct LLM):
    {direct_llm_summary}
    
    Please provide a detailed comparison and indicate which method performs better on each criterion.
    Finally, provide an overall assessment of which method produces the better summary and why.
    
    IMPORTANT: Do not use asterisks (*) in your response as they interfere with parsing.
    
    Your response should follow this format:
    Coverage comparison: [detailed comparison]
    Conciseness comparison: [detailed comparison]
    Coherence comparison: [detailed comparison]
    Factual accuracy comparison: [detailed comparison]
    Overall assessment: [your assessment]
    Method A score (1-5): [score]
    Method B score (1-5): [score]
    """
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500
    )
    
    comparison = response.choices[0].message.content.strip()
    # Remove any asterisks from the response to avoid parsing issues
    comparison = comparison.replace("*", "")
    
    # Extract scores using regex
    method_a_match = re.search(r"Method A score \(1-5\):\s*(\d+)", comparison)
    method_a_score = int(method_a_match.group(1)) if method_a_match else None
    
    method_b_match = re.search(r"Method B score \(1-5\):\s*(\d+)", comparison)
    method_b_score = int(method_b_match.group(1)) if method_b_match else None
    
    # Log the scores for better experiment output
    if method_a_score is not None and method_b_score is not None:
        logger.info(f"Summary comparison completed. Vec2Summ score: {method_a_score}/5, Direct LLM score: {method_b_score}/5")
    else:
        logger.warning("Could not extract scores from comparison response")
    
    return {
        "vec2summ_summary": vec2summ_summary,
        "direct_llm_summary": direct_llm_summary,
        "comparison": comparison,
        "vec2summ_score": method_a_score,
        "direct_llm_score": method_b_score
    }
