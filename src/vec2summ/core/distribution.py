"""
Distribution parameter calculation and sampling for vec2summ.

This module handles the core probabilistic aspects of vec2summ:
- Calculating mean and covariance from embeddings
- Sampling from the multivariate normal distribution
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_distribution_params(embeddings):
    """
    Calculate the mean vector and covariance matrix from a set of embeddings.
    
    Args:
        embeddings: Tensor or array of embeddings, shape (n_samples, embedding_dim)
        
    Returns:
        tuple: (mean_vector, covariance_matrix)
    """
    logger.info("Calculating distribution parameters")
    
    # Convert to numpy if it's a torch tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    mean_vector = np.mean(embeddings_np, axis=0)
    covariance_matrix = np.cov(embeddings_np, rowvar=False)
    
    # Ensure covariance matrix is positive definite
    min_eig = np.min(np.real(np.linalg.eigvals(covariance_matrix)))
    if min_eig < 0:
        logger.warning(f"Covariance matrix not positive definite. Min eigenvalue: {min_eig}")
        covariance_matrix -= 10 * min_eig * np.eye(*covariance_matrix.shape)
    
    return mean_vector, covariance_matrix


def sample_from_distribution(mean_vector, covariance_matrix, n_samples=10):
    """
    Sample vectors from the multivariate normal distribution.
    
    Args:
        mean_vector: Mean vector of the distribution
        covariance_matrix: Covariance matrix of the distribution
        n_samples: Number of samples to generate
        
    Returns:
        np.ndarray: Sampled vectors, shape (n_samples, embedding_dim)
    """
    logger.info(f"Sampling {n_samples} vectors from distribution")
    
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, size=n_samples)
    return samples
