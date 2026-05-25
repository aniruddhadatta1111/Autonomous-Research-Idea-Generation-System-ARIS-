"""
Node 2: Data Retrieval
Fetches top cited papers from Semantic Scholar for each search query.
Actor: Python requests + Semantic Scholar API
"""

from __future__ import annotations

import logging

from state import ARISState
from utils.semantic_scholar import search_papers_multi
import config as cfg

logger = logging.getLogger(__name__)


def data_retrieval_node(state: ARISState) -> dict:
    """
    Node 2: Retrieve papers from Semantic Scholar.

    Input: search_queries, config
    Output: retrieved_papers, status_message
    """
    queries = state["search_queries"]
    user_config = state.get("config", {})

    year_start = user_config.get("year_start", cfg.DEFAULT_YEAR_START)
    year_end = user_config.get("year_end", cfg.DEFAULT_YEAR_END)
    max_papers = user_config.get("max_papers", cfg.MAX_TOTAL_PAPERS)

    logger.info(f"Node 2: Searching S2 with {len(queries)} queries, years {year_start}-{year_end}")

    try:
        papers = search_papers_multi(
            queries=queries,
            year_start=year_start,
            year_end=year_end,
            limit_per_query=cfg.MAX_PAPERS_PER_QUERY,
            max_total=max_papers,
        )

        if not papers:
            return {
                "retrieved_papers": [],
                "status_message": "⚠️ No papers found. Try a broader topic.",
                "error": "No papers returned from Semantic Scholar",
            }

        logger.info(f"Node 2: Retrieved {len(papers)} unique papers")

        return {
            "retrieved_papers": papers,
            "status_message": f"✅ Retrieved {len(papers)} papers (sorted by citations)",
        }

    except Exception as e:
        logger.error(f"Node 2 failed: {e}")
        return {
            "retrieved_papers": [],
            "status_message": f"❌ Data retrieval failed: {str(e)}",
            "error": str(e),
        }
