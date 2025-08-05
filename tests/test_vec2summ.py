"""
Basic tests for vec2summ functionality.
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vec2summ.data.preprocessing import clean_text, TextDataset
from vec2summ.core.distribution import calculate_distribution_params, sample_from_distribution


class TestPreprocessing(unittest.TestCase):
    """Test text preprocessing functions."""
    
    def test_clean_text(self):
        """Test text cleaning function."""
        # Test normal text
        result = clean_text("This is a normal text.")
        self.assertEqual(result, "This is a normal text.")
        
        # Test text with URLs
        result = clean_text("Check this out: https://example.com")
        self.assertEqual(result, "Check this out:")
        
        # Test text with mentions and hashtags
        result = clean_text("Hello @user123 #hashtag")
        self.assertEqual(result, "Hello")
        
        # Test None input
        result = clean_text(None)
        self.assertEqual(result, "")
    
    def test_text_dataset(self):
        """Test TextDataset class."""
        texts = ["text1", "text2", "text3"]
        dataset = TextDataset(texts)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], "text1")
        self.assertEqual(dataset[2], "text3")


class TestDistribution(unittest.TestCase):
    """Test distribution-related functions."""
    
    def test_calculate_distribution_params(self):
        """Test distribution parameter calculation."""
        # Create sample embeddings
        embeddings = np.random.randn(10, 5)
        
        mean_vector, covariance_matrix = calculate_distribution_params(embeddings)
        
        # Check shapes
        self.assertEqual(mean_vector.shape, (5,))
        self.assertEqual(covariance_matrix.shape, (5, 5))
        
        # Check that mean is close to actual mean
        expected_mean = np.mean(embeddings, axis=0)
        np.testing.assert_array_almost_equal(mean_vector, expected_mean, decimal=5)
    
    def test_calculate_distribution_params_torch(self):
        """Test distribution parameter calculation with torch tensors."""
        # Create sample embeddings as torch tensor
        embeddings = torch.randn(10, 5)
        
        mean_vector, covariance_matrix = calculate_distribution_params(embeddings)
        
        # Check types (should be numpy)
        self.assertIsInstance(mean_vector, np.ndarray)
        self.assertIsInstance(covariance_matrix, np.ndarray)
        
        # Check shapes
        self.assertEqual(mean_vector.shape, (5,))
        self.assertEqual(covariance_matrix.shape, (5, 5))
    
    def test_sample_from_distribution(self):
        """Test sampling from distribution."""
        # Create simple distribution parameters
        mean_vector = np.zeros(3)
        covariance_matrix = np.eye(3)
        
        samples = sample_from_distribution(mean_vector, covariance_matrix, n_samples=5)
        
        # Check shape
        self.assertEqual(samples.shape, (5, 3))
        
        # Check that samples are different (very unlikely to be identical)
        self.assertFalse(np.array_equal(samples[0], samples[1]))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    @patch('vec2summ.core.embeddings.openai')
    def test_mock_openai_embeddings(self, mock_openai):
        """Test OpenAI embeddings with mocking."""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(2)]
        mock_openai.embeddings.create.return_value = mock_response
        mock_openai.api_key = "test_key"
        
        from vec2summ.core.embeddings import get_openai_embeddings
        
        texts = ["Test text 1", "Test text 2"]
        embeddings = get_openai_embeddings(texts)
        
        # Check that we got the expected shape
        self.assertEqual(embeddings.shape, (2, 3))
        self.assertTrue(torch.allclose(embeddings[0], torch.tensor([0.1, 0.2, 0.3])))


if __name__ == '__main__':
    unittest.main()
