from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List

from app.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "as", "at", "from", "that", "this",
    "it", "what", "how", "when", "which", "does", "do", "can", "into", "using",
}


def _stem_token(tok: str) -> str:
    """Lightweight suffix stripping so morphological variants share vector dimensions."""
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _deterministic_feature_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Compute a deterministic L2-normalized feature-hashed vector (stems + bigrams + char n-grams).
    Used as an offline fallback when GEMINI_API_KEY is not configured.
    """
    vec = [0.0] * dim
    raw_tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    stems = [_stem_token(t) for t in raw_tokens]

    features: List[tuple[str, float]] = []
    for s in stems:
        features.append((f"u:{s}", 2.0))
        if len(s) >= 4:
            for i in range(len(s) - 2):
                features.append((f"c3:{s[i:i+3]}", 0.35))
    for i in range(len(stems) - 1):
        features.append((f"b:{stems[i]}_{stems[i+1]}", 1.5))

    if not features:
        features = [("empty", 1.0)]

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _normalize_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class EmbeddingService:
    """Generates document and query embeddings using Gemini API with offline fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.dim = self.settings.embedding_dim
        self.model = self.settings.embedding_model

    def _has_real_api_key(self) -> bool:
        key = (self.settings.gemini_api_key or "").strip()
        return bool(key and key != "your_gemini_api_key_here")

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embed a batch of document chunks for indexing."""
        if not texts:
            return []

        if not self._has_real_api_key():
            logger.info(
                "GEMINI_API_KEY not configured; using deterministic local feature embeddings (%d dims).",
                self.dim,
            )
            return [_deterministic_feature_embedding(t, self.dim) for t in texts]

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.settings.gemini_api_key)
        all_embeddings: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dim,
                ),
            )
            for emb in response.embeddings:
                all_embeddings.append(_normalize_vector(list(emb.values)))

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        if not self._has_real_api_key():
            return _deterministic_feature_embedding(query, self.dim)

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.settings.gemini_api_key)
        response = client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dim,
            ),
        )
        return _normalize_vector(list(response.embeddings[0].values))
