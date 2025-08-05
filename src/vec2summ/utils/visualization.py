"""
Visualization utilities for vec2summ experiments.

This module provides functions for visualizing embeddings and experiment results.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def visualize_embeddings(
    original_embeddings, 
    sampled_embeddings, 
    reconstructed_embeddings=None,
    n_components=2,
    save_path="embeddings_visualization.png"
):
    """
    Visualize embeddings in lower-dimensional space using PCA.
    
    Args:
        original_embeddings: Original text embeddings
        sampled_embeddings: Sampled embeddings from distribution
        reconstructed_embeddings: Optional reconstructed embeddings
        n_components: Number of PCA components (2 or 3)
        save_path: Path to save the visualization
    """
    logger.info(f"Visualizing embeddings with {n_components} PCA components")
    
    # Convert to numpy if they're torch tensors
    if isinstance(original_embeddings, torch.Tensor):
        original_embeddings = original_embeddings.cpu().numpy()
    if isinstance(sampled_embeddings, torch.Tensor):
        sampled_embeddings = sampled_embeddings.cpu().numpy()
    if reconstructed_embeddings is not None and isinstance(reconstructed_embeddings, torch.Tensor):
        reconstructed_embeddings = reconstructed_embeddings.cpu().numpy()
    
    # Combine all embeddings for PCA
    all_embeddings = np.vstack([original_embeddings, sampled_embeddings])
    if reconstructed_embeddings is not None:
        all_embeddings = np.vstack([all_embeddings, reconstructed_embeddings])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(all_embeddings)
    
    # Split back into original, sampled, and reconstructed
    n_original = len(original_embeddings)
    n_sampled = len(sampled_embeddings)
    
    reduced_original = reduced_embeddings[:n_original]
    reduced_sampled = reduced_embeddings[n_original:n_original+n_sampled]
    
    if reconstructed_embeddings is not None:
        reduced_reconstructed = reduced_embeddings[n_original+n_sampled:]
    
    # Create visualization
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        
        # Plot original embeddings
        plt.scatter(
            reduced_original[:, 0], 
            reduced_original[:, 1], 
            c='blue', 
            alpha=0.5, 
            label='Original'
        )
        
        # Plot sampled embeddings
        plt.scatter(
            reduced_sampled[:, 0], 
            reduced_sampled[:, 1], 
            c='red', 
            alpha=0.5, 
            label='Sampled'
        )
        
        # Plot reconstructed embeddings if provided
        if reconstructed_embeddings is not None:
            plt.scatter(
                reduced_reconstructed[:, 0], 
                reduced_reconstructed[:, 1], 
                c='green', 
                alpha=0.5, 
                label='Reconstructed'
            )
        
        # Add labels and legend
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Embedding Space Visualization (2D)')
        plt.legend()
        plt.grid(alpha=0.3)
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original embeddings
        ax.scatter(
            reduced_original[:, 0], 
            reduced_original[:, 1], 
            reduced_original[:, 2],
            c='blue', 
            alpha=0.5, 
            label='Original'
        )
        
        # Plot sampled embeddings
        ax.scatter(
            reduced_sampled[:, 0], 
            reduced_sampled[:, 1], 
            reduced_sampled[:, 2],
            c='red', 
            alpha=0.5, 
            label='Sampled'
        )
        
        # Plot reconstructed embeddings if provided
        if reconstructed_embeddings is not None:
            ax.scatter(
                reduced_reconstructed[:, 0], 
                reduced_reconstructed[:, 1], 
                reduced_reconstructed[:, 2],
                c='green', 
                alpha=0.5, 
                label='Reconstructed'
            )
        
        # Add labels and legend
        ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_zlabel(f'PCA Component 3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
        ax.set_title('Embedding Space Visualization (3D)')
        ax.legend()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {save_path}")
    plt.close()
