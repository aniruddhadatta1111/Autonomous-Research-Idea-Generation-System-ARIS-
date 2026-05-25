"""
Node 1: Query Expansion
Converts a user topic into 3 targeted search queries to minimize
S2 API calls while covering key limitation areas.
Actor: Gemini 2.0 Flash
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from state import ARISState
from utils.gemini_client import call_flash
import config as cfg

logger = logging.getLogger(__name__)


class SearchQueries(BaseModel):
    """Schema for expanded search queries."""
    queries: list[str] = Field(
        description="List of exactly 3 technical search queries targeting limitations, "
        "scalability challenges, and open problems in the research area."
    )


EXPANSION_PROMPT = """You are a research strategist. Given a research topic, generate exactly 3 highly specific academic search queries designed to find papers that discuss:
1. Current limitations and bottlenecks
2. Scalability challenges and open problems
3. Recent failed or underperforming approaches

Research Topic: {topic}

CRITICAL RULES FOR QUERIES:
- Keep queries SHORT (max 3-5 words).
- DO NOT use quotes (") or boolean operators like AND/OR.
- Examples of good queries:
  "{topic} limitations"
  "{topic} scalability bottlenecks"
  "open problems {topic}"

Generate queries that would return papers discussing problems, NOT solutions.
Each query should target a different aspect of the topic's challenges.
"""


def query_expansion_node(state: ARISState) -> dict:
    """
    Node 1: Expand user topic into targeted search queries.

    Input: user_topic
    Output: search_queries, status_message
    """
    topic = state["user_topic"]
    logger.info(f"Node 1: Expanding topic '{topic}' into search queries")

    prompt = EXPANSION_PROMPT.format(topic=topic)

    try:
        result = call_flash(prompt, schema=SearchQueries)

        if isinstance(result, dict):
            queries = result.get("queries", [])
        else:
            # Fallback: try parsing as JSON
            queries = json.loads(result).get("queries", [])

        # Ensure we have at least some queries
        max_q = cfg.MAX_SEARCH_QUERIES
        if not queries:
            queries = [
                f"{topic} limitations challenges",
                f"{topic} scalability bottlenecks",
                f"open problems {topic} 2024 2025",
            ]

        # Post-process to sanitize: remove quotes, boolean operators, and truncate length
        sanitized_queries = []
        for q in queries:
            q_clean = q.replace('"', '').replace("'", "")
            q_clean = q_clean.replace(" AND ", " ").replace(" OR ", " ")
            # Truncate to max 6 words to guarantee S2 returns results
            words = q_clean.split()[:6]
            sanitized_queries.append(" ".join(words))

        queries = sanitized_queries[:max_q]
        logger.info(f"Node 1: Generated {len(queries)} queries: {queries}")

        return {
            "search_queries": queries,
            "status_message": f"✅ Generated {len(queries)} search queries",
        }

    except Exception as e:
        logger.error(f"Node 1 failed: {e}")
        # Fallback queries
        fallback = [
            f"{topic} limitations",
            f"{topic} challenges 2024",
            f"{topic} open problems",
        ]
        return {
            "search_queries": fallback,
            "status_message": f"⚠️ Query expansion partial failure, using fallback queries",
            "error": str(e),
        }
