"""
Node 4: Limitation Embedding
Converts extracted limitation strings into vector embeddings
for downstream clustering and similarity analysis.
Actor: gemini-embedding-exp-03-07
"""

from __future__ import annotations

import logging

from state import ARISState
from utils.gemini_client import get_embeddings

logger = logging.getLogger(__name__)


def limitation_embedding_node(state: ARISState) -> dict:
    """
    Node 4: Embed all limitation strings.

    Input: extracted_knowledge
    Output: limitations_texts, limitations_embeddings, status_message
    """
    knowledge = state["extracted_knowledge"]

    if not knowledge:
        return {
            "limitations_texts": [],
            "limitations_embeddings": [],
            "status_message": "⚠️ No knowledge to embed",
        }

    # Flatten all limitations from all papers
    limitations_texts = []
    for paper in knowledge:
        paper_title = paper.get("paper_title", "Unknown")
        limitations = paper.get("limitations", [])
        for lim in limitations:
            if isinstance(lim, str) and lim.strip():
                # Prefix with paper title for context
                limitations_texts.append(f"[{paper_title}] {lim.strip()}")

    if not limitations_texts:
        return {
            "limitations_texts": [],
            "limitations_embeddings": [],
            "status_message": "⚠️ No limitations found in extracted knowledge",
        }

    logger.info(f"Node 4: Embedding {len(limitations_texts)} limitations")

    try:
        embeddings = get_embeddings(limitations_texts)

        logger.info(f"Node 4: Generated {len(embeddings)} embeddings (dim={len(embeddings[0]) if embeddings else 0})")

        return {
            "limitations_texts": limitations_texts,
            "limitations_embeddings": embeddings,
            "status_message": f"✅ Embedded {len(embeddings)} limitations into vectors",
        }

    except Exception as e:
        logger.error(f"Node 4 failed: {e}")
        return {
            "limitations_texts": limitations_texts,
            "limitations_embeddings": [],
            "status_message": f"❌ Embedding failed: {str(e)}",
            "error": str(e),
        }
