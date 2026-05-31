"""
Node 10: Critic Loop
Adaptive constraint feedback — extracts failure reasons, converts to constraints,
and decides whether to retry or proceed.
Actor: Decision logic + constraint extraction
"""

from __future__ import annotations

import logging

from state import ARISState
from utils.semantic_scholar import search_by_title
import config as cfg

logger = logging.getLogger(__name__)


def critic_loop_node(state: ARISState) -> dict:
    """
    Node 10: Critic loop with adaptive constraint generation.

    Input: novelty_scores, generated_ideas, iteration_count
    Output: novelty_status, failure_reasons, adaptive_constraints, iteration_count, status_message
    """
    scores = state.get("novelty_scores", [])
    ideas = state.get("generated_ideas", [])
    iteration = state.get("iteration_count", 0)
    max_iter = cfg.MAX_CRITIC_ITERATIONS

    logger.info(f"Node 10: Critic loop — iteration {iteration + 1}/{max_iter}")

    new_failure_reasons = []
    new_constraints = []

    for i, score_data in enumerate(scores):
        if score_data.get("passed", True):
            continue

        idea_title = score_data.get("idea_title", f"Idea {i+1}")
        idea = ideas[i] if i < len(ideas) else {}

        # Determine WHY it failed (which component was worst)
        sem = score_data.get("semantic_score", 0)
        struct = score_data.get("structural_score", 0)
        kw = score_data.get("keyword_score", 0)

        reasons = []
        if sem > 0.7:
            reasons.append(f"High semantic similarity ({sem:.2f}) — too close to existing papers")
        if struct > 0.6:
            reasons.append(f"High structural overlap ({struct:.2f}) — methodology too similar")
        if kw > 0.5:
            reasons.append(f"High keyword overlap ({kw:.2f}) — uses same terminology")

        reason = f"'{idea_title}' rejected: {'; '.join(reasons) if reasons else 'Overall novelty too low'}"
        new_failure_reasons.append(reason)

        # Convert to actionable constraint
        method = idea.get("methodology", "")
        constraint = (
            f"AVOID approach similar to '{idea_title}' — "
            f"{'; '.join(reasons)}. "
            f"Do not reuse: {method[:150]}"
        )
        new_constraints.append(constraint)

    # Check Semantic Scholar for title matches (all ideas)
    for idea in ideas:
        title = idea.get("title", "")
        if title:
            try:
                exists = search_by_title(title)
                if exists:
                    reason = f"'{title}' — exact/near title match found on Semantic Scholar"
                    new_failure_reasons.append(reason)
                    new_constraints.append(f"AVOID title '{title}' — already exists in literature")
            except Exception as e:
                logger.warning(f"S2 title check failed for '{title}': {e}")

    all_passed = len(new_failure_reasons) == 0
    new_iteration = iteration + 1

    if all_passed:
        logger.info("Node 10: ✅ All ideas passed novelty check!")
        return {
            "novelty_status": True,
            "failure_reasons": new_failure_reasons,
            "adaptive_constraints": new_constraints,
            "iteration_count": new_iteration,
            "status_message": f"✅ All ideas passed novelty verification!",
        }
    elif new_iteration >= max_iter:
        logger.warning(f"Node 10: Max iterations ({max_iter}) reached. Force-passing.")
        return {
            "novelty_status": True,  # Force pass
            "failure_reasons": new_failure_reasons,
            "adaptive_constraints": new_constraints,
            "iteration_count": new_iteration,
            "status_message": f"⚠️ Max iterations reached ({max_iter}). Proceeding with best ideas.",
        }
    else:
        logger.info(f"Node 10: {len(new_failure_reasons)} ideas failed. Retrying (iter {new_iteration}/{max_iter})")
        return {
            "novelty_status": False,
            "failure_reasons": new_failure_reasons,
            "adaptive_constraints": new_constraints,
            "iteration_count": new_iteration,
            "status_message": f"🔄 {len(new_failure_reasons)} ideas need revision (iteration {new_iteration}/{max_iter})",
        }


def critic_router(state: ARISState) -> str:
    """
    Router function for conditional edge after critic loop.
    Returns: "pass", "retry", or "force_pass"
    """
    novelty_status = state.get("novelty_status", False)
    iteration = state.get("iteration_count", 0)

    if novelty_status and iteration < cfg.MAX_CRITIC_ITERATIONS:
        return "pass"
    elif not novelty_status and iteration < cfg.MAX_CRITIC_ITERATIONS:
        return "retry"
    else:
        return "force_pass"
