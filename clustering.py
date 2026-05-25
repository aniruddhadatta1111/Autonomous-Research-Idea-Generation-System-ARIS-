"""
ARIS-Idea: Clustering Utilities
K-Means and silhouette-based optimal cluster selection for limitation embeddings.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as cfg

logger = logging.getLogger(__name__)


def find_optimal_k(
    embeddings: np.ndarray,
    k_min: int = cfg.CLUSTER_K_MIN,
    k_max: int = cfg.CLUSTER_K_MAX,
) -> int:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        embeddings: numpy array of shape (n_samples, n_features).
        k_min: Minimum k to try.
        k_max: Maximum k to try.

    Returns:
        Optimal k value.
    """
    n_samples = len(embeddings)

    # Can't cluster fewer samples than k
    k_max = min(k_max, n_samples - 1)
    k_min = min(k_min, k_max)

    if n_samples < 3 or k_max < 2:
        return min(2, n_samples)

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Silhouette score needs at least 2 clusters with >1 sample
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels)
            logger.info(f"k={k}, silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            logger.warning(f"Clustering with k={k} failed: {e}")
            continue

    logger.info(f"Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


def cluster_limitations(
    embeddings: list[list[float]],
    texts: list[str],
    k: int | None = None,
) -> list[dict]:
    """
    Cluster limitation embeddings and extract cluster metadata.

    Args:
        embeddings: List of embedding vectors.
        texts: Corresponding limitation text strings.
        k: Number of clusters (auto-detected if None).

    Returns:
        List of cluster dicts with keys:
        - cluster_id: int
        - centroid: list[float]
        - density: float (items / avg_distance_to_centroid)
        - size: int
        - representative_limitations: list[str] (closest to centroid)
        - all_limitations: list[str]
    """
    if not embeddings or len(embeddings) < 2:
        # Return everything as a single cluster
        return [{
            "cluster_id": 0,
            "centroid": embeddings[0] if embeddings else [],
            "density": 1.0,
            "size": len(embeddings),
            "representative_limitations": texts[:3] if texts else [],
            "all_limitations": texts,
        }]

    emb_array = np.array(embeddings, dtype=np.float64)

    # Find optimal k if not specified
    if k is None:
        k = find_optimal_k(emb_array)

    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emb_array)

    clusters = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = emb_array[mask]
        cluster_texts = [texts[i] for i in cluster_indices]

        centroid = kmeans.cluster_centers_[cluster_id]

        # Compute distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        avg_distance = float(np.mean(distances)) if len(distances) > 0 else 1.0

        # Density = size / avg_distance (higher = tighter, more recurring)
        density = float(len(cluster_texts) / max(avg_distance, 0.01))

        # Get representative limitations (closest to centroid)
        sorted_indices = np.argsort(distances)
        representative = [cluster_texts[i] for i in sorted_indices[:3]]

        clusters.append({
            "cluster_id": cluster_id,
            "centroid": centroid.tolist(),
            "density": round(density, 4),
            "size": len(cluster_texts),
            "representative_limitations": representative,
            "all_limitations": cluster_texts,
        })

    # Sort by density (most dense first = strongest gap signal)
    clusters.sort(key=lambda c: c["density"], reverse=True)

    return clusters
