"""
Data loading utilities for various file formats.

This module handles loading data from CSV, TSV, JSON, and JSONL files,
with special support for Amazon reviews and other datasets.
"""

import json
import logging
import random
import pandas as pd
from tqdm.auto import tqdm

from .preprocessing import clean_text

logger = logging.getLogger(__name__)


def process_amazon_reviews(file_path, sample_size=50, random_seed=42):
    """
    Process Amazon book reviews dataset, identify top reviewed books, and extract samples.
    
    Args:
        file_path: Path to the Amazon reviews TSV file
        sample_size: Number of reviews to sample for a selected book
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing product info and sampled reviews, or None if processing fails
    """
    logger.info(f"Processing Amazon book reviews from {file_path}...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Count reviews per product_id using chunking to handle the large file
    product_counts = {}
    
    # Process the file in chunks to count reviews per product
    logger.info("Counting reviews per product...")
    chunk_size = 100000  # Adjust based on available memory
    
    try:
        for chunk in tqdm(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Count reviews per product in this chunk
            chunk_counts = chunk['product_id'].value_counts().to_dict()
            
            # Update the overall counts
            for product_id, count in chunk_counts.items():
                if product_id in product_counts:
                    product_counts[product_id] += count
                else:
                    product_counts[product_id] = count
    except Exception as e:
        logger.error(f"Error processing Amazon reviews file: {e}")
        return None
    
    # Convert to DataFrame for easier sorting
    product_counts_df = pd.DataFrame({
        'product_id': list(product_counts.keys()),
        'review_count': list(product_counts.values())
    })
    
    # Sort by review count in descending order
    product_counts_df = product_counts_df.sort_values('review_count', ascending=False).reset_index(drop=True)
    
    # Get the top 5 most reviewed products
    top_5_products = product_counts_df.head(5)
    logger.info("\nTop 5 most reviewed books:")
    for i, row in top_5_products.iterrows():
        logger.info(f"  {i+1}. Product ID: {row['product_id']} - {row['review_count']} reviews")
    
    # Randomly select one of the top 5 products
    selected_product = random.choice(top_5_products['product_id'].tolist())
    selected_product_count = top_5_products[top_5_products['product_id'] == selected_product]['review_count'].values[0]
    
    logger.info(f"\nRandomly selected product: {selected_product} with {selected_product_count} reviews")
    
    # Collect all reviews for the selected product
    logger.info("Collecting reviews for the selected product...")
    selected_reviews = []
    
    try:
        for chunk in tqdm(pd.read_csv(file_path, sep='\t', chunksize=chunk_size)):
            # Filter reviews for the selected product
            product_reviews = chunk[chunk['product_id'] == selected_product]
            
            if not product_reviews.empty:
                selected_reviews.append(product_reviews)
    except Exception as e:
        logger.error(f"Error collecting reviews for selected product: {e}")
        return None
    
    # Combine all chunks of reviews for the selected product
    if selected_reviews:
        all_product_reviews = pd.concat(selected_reviews, ignore_index=True)
        logger.info(f"Found {len(all_product_reviews)} reviews for product {selected_product}")
        
        # Take a random sample of reviews
        if len(all_product_reviews) > sample_size:
            sampled_reviews = all_product_reviews.sample(sample_size, random_state=random_seed)
        else:
            sampled_reviews = all_product_reviews
            logger.warning(f"Only {len(all_product_reviews)} reviews available, which is less than the requested sample size of {sample_size}")
        
        # Extract review texts
        review_texts = sampled_reviews['review_body'].tolist()
        
        # Clean the texts
        cleaned_texts = [clean_text(text) for text in review_texts]
        cleaned_texts = [text for text in cleaned_texts if len(text.split()) >= 5]  # Filter out very short texts
        
        logger.info(f"Extracted {len(cleaned_texts)} cleaned review texts for summarization")
        
        # Return the product info and cleaned texts
        return {
            'product_id': selected_product,
            'total_reviews': selected_product_count,
            'sampled_reviews': sampled_reviews,
            'texts': cleaned_texts
        }
    else:
        logger.error(f"No reviews found for product {selected_product}")
        return None


def load_data(data_path, data_type="csv", text_column="text", max_samples=None, is_amazon_reviews=False, amazon_sample_size=50, random_seed=42):
    """
    Load data from various file formats.
    
    Args:
        data_path: Path to the data file
        data_type: Type of data file (csv, tsv, json)
        text_column: Column name containing text data
        max_samples: Maximum number of samples to load
        is_amazon_reviews: Whether to process as Amazon reviews dataset
        amazon_sample_size: Number of reviews to sample for Amazon dataset
        random_seed: Random seed for reproducibility
        
    Returns:
        list: List of cleaned text strings
    """
    logger.info(f"Loading data from {data_path}")
    
    # Special handling for Amazon reviews dataset
    if is_amazon_reviews:
        result = process_amazon_reviews(data_path, sample_size=amazon_sample_size, random_seed=random_seed)
        if result:
            return result['texts']
        else:
            logger.error("Failed to process Amazon reviews dataset")
            return []
    
    texts = []
    
    if data_type == "csv":
        df = pd.read_csv(data_path)
        if text_column in df.columns:
            texts = df[text_column].dropna().tolist()
        else:
            # Try to find a column that might contain text
            potential_text_columns = ["text", "content", "message", "Message", "body", "description","review_body","Question"]
            for col in potential_text_columns:
                if col in df.columns:
                    texts = df[col].dropna().tolist()
                    logger.info(f"Using column '{col}' as text column")
                    break
            
            if not texts:
                logger.warning(f"No text column found in {data_path}. Available columns: {df.columns.tolist()}")
                return []
    
    elif data_type == "tsv":
        df = pd.read_csv(data_path, sep="\t")
        if text_column in df.columns:
            texts = df[text_column].dropna().tolist()
        else:
            # Try to find a column that might contain text
            potential_text_columns = ["text", "content", "message", "Message", "body", "description", "review_body","Question"]
            for col in potential_text_columns:
                if col in df.columns:
                    texts = df[col].dropna().tolist()
                    logger.info(f"Using column '{col}' as text column")
                    break
            
            if not texts:
                logger.warning(f"No text column found in {data_path}. Available columns: {df.columns.tolist()}")
                return []
    
    elif data_type == "json":
        # Check if file is JSONL (one JSON object per line) or regular JSON
        with open(data_path, 'r') as f:
            first_line = f.readline().strip()
            try:
                # Try to parse the first line as JSON
                json_obj = json.loads(first_line)
                # If we can parse the first line as JSON, it's likely JSONL format
                is_jsonl = True
            except json.JSONDecodeError:
                # If we can't parse the first line as JSON, assume it's a regular JSON file
                is_jsonl = False
        
        if is_jsonl:
            # Handle JSONL format (one JSON object per line)
            logger.info("Detected JSONL format (one JSON object per line)")
            data_items = []
            with open(data_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data_items.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON line: {e}")
                        continue
            
            # Try to find text field in the loaded objects
            if data_items:
                # Check if this is the TIFU dataset format based on the user's example
                tifu_fields = ["title", "selftext", "selftext_without_tldr"]
                is_tifu_format = all(field in data_items[0] for field in tifu_fields)
                
                if is_tifu_format:
                    logger.info("Detected TIFU dataset format")
                    # Use selftext_without_tldr as the main text field for TIFU dataset
                    texts = [item.get("selftext_without_tldr", "") for item in data_items if "selftext_without_tldr" in item]
                    logger.info(f"Using field 'selftext_without_tldr' as text field for TIFU dataset")
                else:
                    # Try standard text fields for other datasets
                    potential_text_fields = ["text", "content", "body", "message", "title", "selftext"]
                    for field in potential_text_fields:
                        if field in data_items[0]:
                            texts = [item.get(field, "") for item in data_items if field in item]
                            logger.info(f"Using field '{field}' as text field")
                            break
        else:
            # Handle regular JSON format
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of objects
                if len(data) > 0 and isinstance(data[0], dict):
                    # Try to find text field
                    potential_text_fields = ["text", "content", "body", "message", "title", "selftext"]
                    for field in potential_text_fields:
                        if field in data[0]:
                            texts = [item.get(field, "") for item in data if field in item]
                            logger.info(f"Using field '{field}' as text field")
                            break
            
            elif isinstance(data, dict):
                # Dictionary with potential text fields
                potential_text_fields = ["text", "content", "body", "message", "posts", "documents"]
                for field in potential_text_fields:
                    if field in data and isinstance(data[field], list):
                        if isinstance(data[field][0], str):
                            texts = data[field]
                        elif isinstance(data[field][0], dict):
                            # Try to find text field in nested objects
                            nested_fields = ["text", "content", "body", "message"]
                            for nested_field in nested_fields:
                                if nested_field in data[field][0]:
                                    texts = [item.get(nested_field, "") for item in data[field]]
                                    break
    
    # Clean and filter texts
    texts = [clean_text(text) for text in texts]
    texts = [text for text in texts if len(text.split()) >= 5]  # Filter out very short texts
    
    # Limit number of samples if specified
    if max_samples is not None and max_samples < len(texts):
        texts = texts[:max_samples]
    
    logger.info(f"Loaded {len(texts)} texts from {data_path}")
    return texts
