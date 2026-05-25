"""
ARIS-Idea: LangGraph State Schema
Defines the shared state that flows through all 11 pipeline nodes.
Uses Annotated reducers for list fields that accumulate across nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ARISState(TypedDict):
    """Complete state schema for the ARIS-Idea LangGraph pipeline."""

    # ─── Input ────────────────────────────────────────────────────────────
    user_topic: str

    # ─── Node 1: Query Expansion ──────────────────────────────────────────
    search_queries: list[str]

    # ─── Node 2: Data Retrieval ───────────────────────────────────────────
    # Accumulates papers from multiple queries via operator.add
    retrieved_papers: Annotated[list[dict], operator.add]

    # ─── Node 3: Knowledge Extraction ─────────────────────────────────────
    # Accumulates structured knowledge from batched extractions
    extracted_knowledge: Annotated[list[dict], operator.add]

    # ─── Node 4: Limitation Embedding ─────────────────────────────────────
    limitations_texts: list[str]              # Flattened limitation strings
    limitations_embeddings: list[list[float]]  # 768-dim vectors per limitation

    # ─── Node 5: Gap Clustering ───────────────────────────────────────────
    # Each cluster: {cluster_id, centroid, density, size, representative_limitations}
    limitation_clusters: list[dict]

    # ─── Node 6: Gap Validation ───────────────────────────────────────────
    validated_gaps: list[str]

    # ─── Node 7: Idea Generation ──────────────────────────────────────────
    generated_ideas: list[dict]
    # Constraints accumulated from critic failures (operator.add)
    adaptive_constraints: Annotated[list[str], operator.add]

    # ─── Node 8: Idea Embedding ───────────────────────────────────────────
    idea_embeddings: list[list[float]]

    # ─── Node 9: Novelty Evaluation ───────────────────────────────────────
    # Per-idea: {idea_title, semantic_score, structural_score, keyword_score, final_score, passed}
    novelty_scores: list[dict]

    # ─── Node 10: Critic Loop ─────────────────────────────────────────────
    novelty_status: bool
    failure_reasons: Annotated[list[str], operator.add]
    iteration_count: int

    # ─── Node 11: Evaluation Metrics ──────────────────────────────────────
    # {average_novelty_score, pass_rate, gap_coverage, iterations_used, constraints_generated}
    evaluation_metrics: dict

    # ─── Output ───────────────────────────────────────────────────────────
    final_report: str
    status_message: str
    error: str

    # ─── UI Configuration (passed in at invocation) ───────────────────────
    config: dict
