from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import urllib.request
from typing import Dict, List, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "as", "at", "from", "that", "this",
    "it", "what", "how", "when", "which", "does", "do", "can", "into", "using",
    "who", "where", "why", "must", "want", "wants", "should", "would", "could",
    "have", "has", "had", "will", "shall", "may", "might", "if", "then", "than",
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


def _normalize_vector(values: List[float]) -> List[float]:
    """L2-normalize a vector to unit length (||v||_2 = 1.0) for cosine similarity."""
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def is_l2_normalized(values: List[float], tol: float = 1e-3) -> bool:
    """Verify that an embedding vector has unit L2 norm within tolerance."""
    if not values:
        return False
    norm = math.sqrt(sum(v * v for v in values))
    return abs(norm - 1.0) <= tol


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity (1.0 - cosine_distance) matching pgvector `1 - (a <=> b)`.
    Both vectors are normalized to unit L2 norm so dot product == cosine similarity in [-1.0, 1.0].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    sim = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, sim))


def _deterministic_feature_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Compute a deterministic L2-normalized feature-hashed vector (stems + bigrams + char n-grams).
    Used as an offline fallback when GEMINI_API_KEY is not configured or network is unavailable.
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

    return _normalize_vector(vec)


_EMBEDDING_CACHE: Dict[Tuple[str, int, str, str], List[float]] = {}


class EmbeddingService:
    """Generates L2-normalized document and query embeddings using Gemini API with offline fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.dim = self.settings.embedding_dim
        self.model = self.settings.embedding_model

    def _has_real_api_key(self) -> bool:
        if os.getenv("DISABLE_GEMINI_EMBEDDINGS", "").lower() in ("1", "true", "yes"):
            return False
        key = (self.settings.gemini_api_key or "").strip()
        return bool(key and key != "your_gemini_api_key_here")

    def _rest_embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Call Gemini REST batchEmbedContents directly without requiring google-genai."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents"
        requests_payload = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": self.dim,
            }
            for text in texts
        ]
        req = urllib.request.Request(
            url,
            data=json.dumps({"requests": requests_payload}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.gemini_api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [_normalize_vector(list(emb["values"])) for emb in data["embeddings"]]

    def _rest_embed_single(self, text: str, task_type: str) -> List[float]:
        """Call Gemini REST embedContent directly without requiring google-genai."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dim,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.gemini_api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _normalize_vector(list(data["embedding"]["values"]))

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embed a batch of document chunks for indexing (L2-normalized)."""
        if not texts:
            return []

        if not self._has_real_api_key():
            return [_deterministic_feature_embedding(t, self.dim) for t in texts]

        missing_indices = [
            idx
            for idx, txt in enumerate(texts)
            if (self.model, self.dim, "RETRIEVAL_DOCUMENT", txt) not in _EMBEDDING_CACHE
        ]
        if missing_indices:
            missing_texts = [texts[idx] for idx in missing_indices]
            computed: List[List[float]] = []
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.settings.gemini_api_key)
                for start in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[start : start + batch_size]
                    response = client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.dim,
                        ),
                    )
                    for emb in response.embeddings:
                        computed.append(_normalize_vector(list(emb.values)))
            except Exception:
                try:
                    for start in range(0, len(missing_texts), batch_size):
                        batch = missing_texts[start : start + batch_size]
                        computed.extend(self._rest_embed_batch(batch, "RETRIEVAL_DOCUMENT"))
                except Exception as exc:
                    logger.warning("Gemini embed_documents failed (%s); using deterministic fallback.", exc)
                    computed = [_deterministic_feature_embedding(t, self.dim) for t in missing_texts]

            for txt, vec in zip(missing_texts, computed):
                _EMBEDDING_CACHE[(self.model, self.dim, "RETRIEVAL_DOCUMENT", txt)] = vec

        return [_EMBEDDING_CACHE[(self.model, self.dim, "RETRIEVAL_DOCUMENT", txt)] for txt in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query (L2-normalized)."""
        if not self._has_real_api_key():
            return _deterministic_feature_embedding(query, self.dim)

        cache_key = (self.model, self.dim, "RETRIEVAL_QUERY", query)
        if cache_key in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE[cache_key]

        vec: List[float]
        try:
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
            vec = _normalize_vector(list(response.embeddings[0].values))
        except Exception:
            try:
                vec = self._rest_embed_single(query, "RETRIEVAL_QUERY")
            except Exception as exc:
                logger.warning("Gemini embed_query failed (%s); using deterministic fallback.", exc)
                vec = _deterministic_feature_embedding(query, self.dim)

        _EMBEDDING_CACHE[cache_key] = vec
        return vec
