"""
ARIS-Idea: Embedding & Similarity Utilities
Cosine similarity computation and novelty checking with caching.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    sim(v1, v2) = (v1 · v2) / (‖v1‖ · ‖v2‖)

    Returns:
        Float in [-1, 1]. Higher = more similar.
    """
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def max_cosine_similarity(
    target_embedding: list[float],
    reference_embeddings: list[list[float]],
) -> float:
    """
    Compute the maximum cosine similarity between a target and a set of references.

    Args:
        target_embedding: The embedding to check.
        reference_embeddings: List of embeddings to compare against.

    Returns:
        Maximum similarity score (0.0 if no references).
    """
    if not reference_embeddings:
        return 0.0

    similarities = [
        cosine_similarity(target_embedding, ref)
        for ref in reference_embeddings
    ]
    return max(similarities)


def compute_keyword_overlap(text_a: str, text_b: str) -> float:
    """
    Compute keyword overlap ratio using simple TF-IDF-like approach.
    Uses word-level Jaccard similarity after stopword removal.

    Returns:
        Float in [0, 1]. Higher = more overlap.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how", "what",
        "which", "who", "whom", "this", "that", "these", "those", "it", "its",
        "we", "our", "they", "their", "them", "he", "she", "his", "her",
        "based", "using", "approach", "method", "propose", "proposed", "paper",
        "results", "show", "study", "model", "data", "also", "new",
    }

    def _extract_keywords(text: str) -> set[str]:
        words = text.lower().split()
        # Keep alphanumeric words, remove stopwords, min length 3
        return {
            w.strip(".,;:!?()[]{}\"'")
            for w in words
            if len(w) > 2 and w.lower() not in stopwords
        }

    keywords_a = _extract_keywords(text_a)
    keywords_b = _extract_keywords(text_b)

    if not keywords_a or not keywords_b:
        return 0.0

    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b

    return len(intersection) / len(union) if union else 0.0


def compute_novelty_score(
    semantic_sim: float,
    structural_overlap: float,
    keyword_overlap: float,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """
    Compute the multi-component novelty score.

    Score = α·(1 - semantic) + β·(1 - structural) + γ·(1 - keyword)

    Args:
        semantic_sim: Cosine similarity (0-1).
        structural_overlap: LLM-judged methodology overlap (0-1).
        keyword_overlap: Keyword Jaccard similarity (0-1).
        alpha, beta, gamma: Component weights (should sum to 1).

    Returns:
        Novelty score in [0, 1]. Higher = more novel.
    """
    score = (
        alpha * (1.0 - semantic_sim)
        + beta * (1.0 - structural_overlap)
        + gamma * (1.0 - keyword_overlap)
    )
    return max(0.0, min(1.0, score))
