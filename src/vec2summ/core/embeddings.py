"""
Embedding generation utilities for vec2summ.

This module provides functions to generate embeddings using different models:
- OpenAI embedding models (text-embedding-ada-002, etc.)
- GTR (Generative Text Retrieval) models
"""

import logging
import openai
import torch
import vec2text
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ..data.preprocessing import TextDataset

logger = logging.getLogger(__name__)


def get_compatible_corrector(embedding_type, openai_model="text-embedding-ada-002"):
    """
    Get the compatible corrector for a given embedding type.
    
    Args:
        embedding_type: Either "openai" or "gtr"
        openai_model: OpenAI model name (used if embedding_type="openai")
        
    Returns:
        corrector: The compatible vec2text corrector
        
    Raises:
        ValueError: If embedding_type is not supported
    """
    if embedding_type == "openai":
        return vec2text.load_pretrained_corrector(openai_model)
    elif embedding_type == "gtr":
        return vec2text.load_pretrained_corrector("gtr-base")
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Choose 'openai' or 'gtr'.")


def validate_embedding_corrector_pair(embedding_type, corrector_name):
    """
    Validate that an embedding type and corrector are compatible.
    
    Args:
        embedding_type: Either "openai" or "gtr"
        corrector_name: Name of the corrector model
        
    Returns:
        bool: True if compatible, False otherwise
    """
    if embedding_type == "openai":
        # OpenAI embeddings should use OpenAI correctors
        return corrector_name in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    elif embedding_type == "gtr":
        # GTR embeddings should use GTR correctors
        return corrector_name == "gtr-base"
    else:
        return False


def get_embeddings_and_corrector(
    text_list,
    embedding_type="openai",
    openai_model="text-embedding-ada-002",
    gtr_model="sentence-transformers/gtr-t5-base",
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """
    Unified interface to get embeddings and properly paired corrector.
    
    Args:
        text_list: List of text strings to embed
        embedding_type: Either "openai" or "gtr"
        openai_model: OpenAI embedding model name (used if embedding_type="openai")
        gtr_model: GTR model name (used if embedding_type="gtr")
        batch_size: Size of batches for processing
        device: Device to run computations on
        
    Returns:
        tuple: (embeddings, corrector, embedding_models) where embedding_models contains
               the encoder/tokenizer for GTR or model name for OpenAI
        
    Raises:
        ValueError: If embedding_type is not supported or required API keys are missing
    """
    if embedding_type == "openai":
        embeddings = get_openai_embeddings(text_list, model=openai_model, batch_size=batch_size)
        corrector = vec2text.load_pretrained_corrector(openai_model)
        embedding_models = openai_model
        return embeddings, corrector, embedding_models
    elif embedding_type == "gtr":
        encoder = AutoModel.from_pretrained(gtr_model).encoder.to(device)
        tokenizer = AutoTokenizer.from_pretrained(gtr_model)
        embeddings = get_gtr_embeddings(text_list, encoder, tokenizer, batch_size=batch_size, device=device)
        # Use the proper GTR corrector
        corrector = vec2text.load_pretrained_corrector("gtr-base")
        embedding_models = (encoder, tokenizer)
        return embeddings, corrector, embedding_models
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Choose 'openai' or 'gtr'.")


def get_embeddings(
    text_list,
    embedding_type="openai",
    openai_model="text-embedding-ada-002",
    gtr_model="sentence-transformers/gtr-t5-base",
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Unified interface to get embeddings using either OpenAI or GTR models.
    
    Args:
        text_list: List of text strings to embed
        embedding_type: Either "openai" or "gtr"
        openai_model: OpenAI embedding model name (used if embedding_type="openai")
        gtr_model: GTR model name (used if embedding_type="gtr")
        batch_size: Size of batches for processing
        device: Device to run computations on
        
    Returns:
        torch.Tensor: Embeddings for all input texts
        
    Raises:
        ValueError: If embedding_type is not supported or required API keys are missing
    """
    embeddings, _, _ = get_embeddings_and_corrector(
        text_list, embedding_type, openai_model, gtr_model, batch_size, device
    )
    return embeddings


def get_gtr_embeddings(
    text_list,
    encoder,
    tokenizer,
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Computes embeddings for a list of texts using GTR model with batching.
    
    Args:
        text_list: List of text strings to embed
        encoder: The GTR encoder model
        tokenizer: The tokenizer for the GTR model
        batch_size: Size of batches for processing
        device: Device to run computations on
        
    Returns:
        torch.Tensor: Embeddings for all input texts
    """
    logger.info(f"Computing GTR embeddings for {len(text_list)} texts")
    
    dataset = TextDataset(text_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    for batch_texts in tqdm(dataloader, desc="Computing GTR embeddings"):
        # Tokenize the current batch
        inputs = tokenizer(
            list(batch_texts),
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="longest",
        ).to(device)

        with torch.no_grad():
            # Forward pass through the encoder
            model_output = encoder(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            hidden_state = model_output.last_hidden_state
            # Mean pooling to obtain embeddings
            embeddings = vec2text.models.model_utils.mean_pool(
                hidden_state, inputs["attention_mask"]
            )

        all_embeddings.append(embeddings)

    # Concatenate all batch embeddings into a single tensor
    return torch.cat(all_embeddings, dim=0)


def get_openai_embeddings(
    text_list, 
    model="text-embedding-ada-002",
    batch_size=128
) -> torch.Tensor:
    """
    Computes embeddings for a list of texts using OpenAI API with batching.
    
    Args:
        text_list: List of text strings to embed
        model: OpenAI embedding model name
        batch_size: Size of batches for processing
        
    Returns:
        torch.Tensor: Embeddings for all input texts
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    logger.info(f"Computing OpenAI embeddings for {len(text_list)} texts")
    
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
    
    batches = (len(text_list) + batch_size - 1) // batch_size
    outputs = []
    
    for batch in tqdm(range(batches), desc="Computing OpenAI embeddings"):
        text_list_batch = text_list[batch * batch_size : (batch + 1) * batch_size]
        response = openai.embeddings.create(
            input=text_list_batch,
            model=model,
            encoding_format="float",  # override default base64 encoding
        )

        outputs.extend([e.embedding for e in response.data])
    
    return torch.tensor(outputs)
