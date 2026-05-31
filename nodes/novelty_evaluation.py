"""
Node 9: Novelty Evaluation
Multi-component novelty scoring: semantic + structural + keyword.
OPTIMIZED: Batches structural comparison into a single LLM call.
Actor: Embeddings + Gemini Flash + TF-IDF
"""

from __future__ import annotations

import json
import logging

from state import ARISState
from utils.gemini_client import call_flash, get_embeddings
from utils.embeddings import (
    max_cosine_similarity,
    compute_keyword_overlap,
    compute_novelty_score,
)
import config as cfg

logger = logging.getLogger(__name__)

BATCHED_STRUCTURAL_PROMPT = """Rate the structural similarity between each idea's methodology and the most relevant existing paper methodologies.
Score from 0.0 (completely different) to 1.0 (identical approach).

IDEAS:
{ideas_text}

EXISTING METHODOLOGIES:
{methods_text}

Return a JSON object: {{"scores": [<float>, <float>, ...]}}
One score per idea (the MAX similarity to any existing methodology).
"""


def _batch_structural_overlap(ideas: list[dict], papers: list[dict]) -> list[float]:
    """
    Compute structural overlap for ALL ideas in a single Flash call.
    Compares against top N papers (cfg.MAX_STRUCTURAL_COMPARISONS).
    """
    if not ideas or not papers:
        return [0.0] * len(ideas)

    # Format ideas
    ideas_text = ""
    for i, idea in enumerate(ideas, 1):
        method = idea.get("methodology", "N/A")[:200]
        ideas_text += f"Idea {i}: {idea.get('title', 'Untitled')} — {method}\n"

    # Format top paper methodologies (limited to save tokens)
    methods_text = ""
    for i, paper in enumerate(papers[:cfg.MAX_STRUCTURAL_COMPARISONS], 1):
        method = paper.get("core_methodology", "N/A")[:150]
        methods_text += f"Paper {i}: {paper.get('paper_title', 'Unknown')} — {method}\n"

    if not methods_text.strip():
        return [0.0] * len(ideas)

    try:
        prompt = BATCHED_STRUCTURAL_PROMPT.format(
            ideas_text=ideas_text,
            methods_text=methods_text,
        )
        result = call_flash(prompt)
        if isinstance(result, str):
            result = json.loads(result)
        scores = result.get("scores", [])

        # Ensure we have the right number of scores
        while len(scores) < len(ideas):
            scores.append(0.0)

        return [min(1.0, max(0.0, float(s))) for s in scores[:len(ideas)]]

    except Exception as e:
        logger.warning(f"Batched structural comparison failed: {e}")
        return [0.0] * len(ideas)


def novelty_evaluation_node(state: ARISState) -> dict:
    """
    Node 9: Multi-component novelty scoring.
    Uses batched structural comparison (1 API call for all ideas).

    Input: idea_embeddings, generated_ideas, retrieved_papers, extracted_knowledge
    Output: novelty_scores, status_message
    """
    ideas = state.get("generated_ideas", [])
    idea_embeddings = state.get("idea_embeddings", [])
    papers = state.get("retrieved_papers", [])
    knowledge = state.get("extracted_knowledge", [])
    user_config = state.get("config", {})

    alpha = user_config.get("alpha", cfg.DEFAULT_ALPHA)
    beta = user_config.get("beta", cfg.DEFAULT_BETA)
    gamma = user_config.get("gamma", cfg.DEFAULT_GAMMA)
    novelty_threshold = user_config.get("novelty_threshold", cfg.DEFAULT_NOVELTY_THRESHOLD)

    if not ideas:
        return {"novelty_scores": [], "status_message": "⚠️ No ideas to evaluate"}

    logger.info(f"Node 9: Evaluating novelty for {len(ideas)} ideas")

    # Get paper embeddings for semantic comparison (single batch call)
    paper_texts = [
        f"{p.get('title', '')}. {p.get('abstract', '')}" for p in papers
    ]
    try:
        paper_embeddings = get_embeddings(paper_texts) if paper_texts else []
    except Exception:
        paper_embeddings = []

    # Batched structural overlap — 1 Flash call for ALL ideas
    structural_scores = _batch_structural_overlap(ideas, knowledge)

    scores = []
    for i, idea in enumerate(ideas):
        idea_title = idea.get("title", f"Idea {i+1}")

        # 1) Semantic similarity (no API call — uses pre-computed embeddings)
        if i < len(idea_embeddings) and paper_embeddings:
            semantic_sim = max_cosine_similarity(idea_embeddings[i], paper_embeddings)
        else:
            semantic_sim = 0.0

        # 2) Structural overlap (from batched call)
        structural = structural_scores[i] if i < len(structural_scores) else 0.0

        # 3) Keyword overlap (no API call — pure Python)
        idea_text = f"{idea.get('title', '')} {idea.get('methodology', '')}"
        all_abstracts = " ".join(p.get("abstract", "") for p in papers)
        keyword = compute_keyword_overlap(idea_text, all_abstracts)

        # Final score
        final = compute_novelty_score(semantic_sim, structural, keyword, alpha, beta, gamma)
        passed = final >= novelty_threshold

        scores.append({
            "idea_title": idea_title,
            "semantic_score": round(semantic_sim, 4),
            "structural_score": round(structural, 4),
            "keyword_score": round(keyword, 4),
            "final_score": round(final, 4),
            "passed": passed,
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} '{idea_title}': sem={semantic_sim:.3f} str={structural:.3f} kw={keyword:.3f} → {final:.3f}")

    passed_count = sum(1 for s in scores if s["passed"])
    return {
        "novelty_scores": scores,
        "status_message": f"✅ Novelty evaluation: {passed_count}/{len(scores)} ideas passed",
    }
