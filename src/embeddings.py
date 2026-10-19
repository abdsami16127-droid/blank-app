"""Dense semantic embedding generation and caching module using ONNX-accelerated FastEmbed."""

from functools import lru_cache
import hashlib
from typing import List, Union
import onnxruntime  # Pre-loads Windows C++ ONNX runtime DLLs cleanly
from fastembed.text.text_embedding import TextEmbedding
import numpy as np

from src.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingEngine:
    """
    Generates dense vector embeddings using ONNX-accelerated sentence transformers.
    Runs locally on CPU with zero PyTorch DLL conflicts, low memory footprint, and high throughput.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of text strings into normalized dense vectors.

        Args:
            texts: List of review texts to embed.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        # FastEmbed returns a generator of numpy arrays
        embeddings_gen = self._model.embed(texts)
        embeddings = np.array(list(embeddings_gen), dtype=np.float32)

        # L2-normalize vectors for exact cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_embeddings = embeddings / norms

        return normalized_embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embeds a single string into a 1D normalized dense vector."""
        return self.embed_texts([text])[0]
