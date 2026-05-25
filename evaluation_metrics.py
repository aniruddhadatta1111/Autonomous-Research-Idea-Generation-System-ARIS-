"""
Node 11: Evaluation Metrics
Computes summary metrics for the generation pipeline.
"""

from __future__ import annotations

import logging

from state import ARISState

logger = logging.getLogger(__name__)


def evaluation_metrics_node(state: ARISState) -> dict:
    """
    Node 11: Compute evaluation metrics.

    Input: novelty_scores, generated_ideas, validated_gaps, iteration_count, adaptive_constraints
    Output: evaluation_metrics, status_message
    """
    scores = state.get("novelty_scores", [])
    ideas = state.get("generated_ideas", [])
    gaps = state.get("validated_gaps", [])
    iteration = state.get("iteration_count", 0)
    constraints = state.get("adaptive_constraints", [])

    # Average novelty score
    final_scores = [s.get("final_score", 0) for s in scores]
    avg_novelty = sum(final_scores) / len(final_scores) if final_scores else 0.0

    # Pass rate
    passed = sum(1 for s in scores if s.get("passed", False))
    total = len(scores) if scores else 1
    pass_rate = passed / total

    # Gap coverage: how many unique gaps are addressed by the ideas
    addressed_gaps = set()
    for idea in ideas:
        target_gap = idea.get("target_gap", "")
        if target_gap:
            addressed_gaps.add(target_gap)
    gap_coverage = len(addressed_gaps) / len(gaps) if gaps else 0.0

    metrics = {
        "average_novelty_score": round(avg_novelty, 4),
        "pass_rate": round(pass_rate, 4),
        "gap_coverage": round(min(1.0, gap_coverage), 4),
        "iterations_used": iteration,
        "constraints_generated": len(constraints),
        "total_ideas_evaluated": total,
        "ideas_passed": passed,
        "total_gaps": len(gaps),
        "gaps_addressed": len(addressed_gaps),
    }

    logger.info(f"Node 11: Metrics — novelty={avg_novelty:.3f}, pass_rate={pass_rate:.1%}, gap_cov={gap_coverage:.1%}")

    return {
        "evaluation_metrics": metrics,
        "status_message": f"✅ Metrics: Novelty={avg_novelty:.2f}, Pass={pass_rate:.0%}, Coverage={gap_coverage:.0%}",
    }
