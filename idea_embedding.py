"""
Node 8: Idea Embedding
Converts generated ideas (title + methodology) into vector embeddings.
Actor: gemini-embedding-exp-03-07
"""

from __future__ import annotations

import logging

from state import ARISState
from utils.gemini_client import get_embeddings

logger = logging.getLogger(__name__)


def idea_embedding_node(state: ARISState) -> dict:
    """
    Node 8: Embed generated ideas.

    Input: generated_ideas
    Output: idea_embeddings, status_message
    """
    ideas = state.get("generated_ideas", [])

    if not ideas:
        return {
            "idea_embeddings": [],
            "status_message": "⚠️ No ideas to embed",
        }

    logger.info(f"Node 8: Embedding {len(ideas)} ideas")

    idea_texts = []
    for idea in ideas:
        title = idea.get("title", "")
        methodology = idea.get("methodology", "")
        motivation = idea.get("motivation", "")
        text = f"{title}. {motivation} {methodology}"
        idea_texts.append(text)

    try:
        embeddings = get_embeddings(idea_texts)
        logger.info(f"Node 8: Generated {len(embeddings)} idea embeddings")
        return {
            "idea_embeddings": embeddings,
            "status_message": f"✅ Embedded {len(embeddings)} ideas into vectors",
        }
    except Exception as e:
        logger.error(f"Node 8 failed: {e}")
        return {
            "idea_embeddings": [],
            "status_message": f"❌ Idea embedding failed: {str(e)}",
            "error": str(e),
        }
