"""
Text preprocessing utilities for vec2summ.

This module provides text cleaning and preprocessing functions.
"""

import re
import preprocessor as p
from torch.utils.data import Dataset

# Set preprocessor options
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)


class TextDataset(Dataset):
    """Dataset for loading and processing text data."""
    
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


def clean_text(text):
    """
    Minimal preprocessing for text to prepare for embeddings.
    
    Args:
        text: Input text string
        
    Returns:
        str: Cleaned text
    """
    if text is None:
        return ""
    
    # Use tweet preprocessor
    try:
        text = p.clean(text)
    except:
        pass
    
    # Remove "RT" at the start of the text
    text = re.sub(r"^RT\s+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r"@\w+", "", text)  # Mentions
    text = re.sub(r"#\w+", "", text)  # Hashtags
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
