"""
Node 5: Gap Clustering
Clusters limitation embeddings using K-Means with silhouette-based k selection.
High-density clusters indicate recurring, widespread limitations = strongest gap signals.
Actor: scikit-learn (K-Means)
"""

from __future__ import annotations

import logging

from state import ARISState
from utils.clustering import cluster_limitations

logger = logging.getLogger(__name__)


def gap_clustering_node(state: ARISState) -> dict:
    """
    Node 5: Cluster limitation embeddings.

    Input: limitations_embeddings, limitations_texts
    Output: limitation_clusters, status_message
    """
    embeddings = state.get("limitations_embeddings", [])
    texts = state.get("limitations_texts", [])

    if not embeddings or len(embeddings) < 2:
        logger.warning("Node 5: Not enough embeddings for clustering")
        # Create a single cluster with whatever we have
        return {
            "limitation_clusters": [{
                "cluster_id": 0,
                "centroid": embeddings[0] if embeddings else [],
                "density": 1.0,
                "size": len(texts),
                "representative_limitations": texts[:3],
                "all_limitations": texts,
            }] if texts else [],
            "status_message": "⚠️ Too few limitations for meaningful clustering",
        }

    logger.info(f"Node 5: Clustering {len(embeddings)} limitation embeddings")

    try:
        clusters = cluster_limitations(embeddings, texts)

        # Log cluster summary
        for c in clusters:
            logger.info(
                f"  Cluster {c['cluster_id']}: size={c['size']}, "
                f"density={c['density']:.2f}, "
                f"top_limitation='{c['representative_limitations'][0][:80]}...'"
            )

        return {
            "limitation_clusters": clusters,
            "status_message": f"✅ Found {len(clusters)} limitation clusters (sorted by density)",
        }

    except Exception as e:
        logger.error(f"Node 5 failed: {e}")
        return {
            "limitation_clusters": [],
            "status_message": f"❌ Clustering failed: {str(e)}",
            "error": str(e),
        }
