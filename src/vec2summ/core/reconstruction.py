"""
Text reconstruction from embeddings using vec2text.

This module handles the reconstruction of text from embedding vectors.
"""

import logging
import numpy as np
import torch
import vec2text

logger = logging.getLogger(__name__)


def reconstruct_text_from_embeddings(
    embeddings, 
    corrector,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Reconstruct text from embeddings using vec2text.
    
    Args:
        embeddings: Embeddings to reconstruct from (numpy array or torch tensor)
        corrector: The vec2text corrector model
        device: Device to run computations on
        
    Returns:
        list: List of reconstructed text strings
    """
    logger.info(f"Reconstructing text from {len(embeddings)} embeddings")
    
    # Convert to torch tensor if it's numpy array
    if isinstance(embeddings, np.ndarray):
        embeddings_tensor = torch.tensor(embeddings).to(device)
    else:
        embeddings_tensor = embeddings.to(device)
    
    # Use vec2text to invert embeddings
    reconstructed_texts = vec2text.invert_embeddings(
        embeddings=embeddings_tensor,
        corrector=corrector
    )
    
    return reconstructed_texts
