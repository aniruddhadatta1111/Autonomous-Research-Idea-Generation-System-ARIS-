"""
Node 6: Gap Validation
Validates and ranks research gaps using cluster density as quantitative evidence
combined with Gemini Flash's reasoning.
Actor: Gemini 2.5 Flash + cluster density data
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from state import ARISState
from utils.gemini_client import call_flash

logger = logging.getLogger(__name__)


class ValidatedGaps(BaseModel):
    """Schema for validated research gaps."""
    gaps: list[str] = Field(
        description="List of 3-5 validated research gaps. Each gap should be a specific, "
        "actionable technical challenge. Include evidence of recurrence."
    )


GAP_VALIDATION_PROMPT = """You are a senior research scientist analyzing the limitations landscape of a research field.

Below are clusters of limitations extracted from {num_papers} recent papers, ranked by density (most recurring first).
High-density clusters represent limitations mentioned across MULTIPLE papers — these are the strongest gap signals.

LIMITATION CLUSTERS (ranked by density):
{clusters_text}

FULL EXTRACTED KNOWLEDGE:
{knowledge_summary}

Your task:
1. Cross-reference the limitation clusters with the paper methodologies
2. FILTER OUT gaps that are already solved (i.e., another paper's methodology directly addresses it)
3. RANK remaining gaps by: (a) cluster density, (b) number of papers affected, (c) potential impact
4. Return exactly 5 validated, unsolved research gaps

Each gap should be:
- SPECIFIC (not vague like "needs improvement")
- TECHNICAL (describe the actual bottleneck)
- EVIDENCE-BACKED (mention which cluster/papers support it)
"""


def gap_validation_node(state: ARISState) -> dict:
    """
    Node 6: Validate and rank gaps using cluster density + LLM reasoning.

    Input: limitation_clusters, extracted_knowledge
    Output: validated_gaps, status_message
    """
    clusters = state.get("limitation_clusters", [])
    knowledge = state.get("extracted_knowledge", [])

    if not clusters:
        return {
            "validated_gaps": ["No gaps could be identified — insufficient data"],
            "status_message": "⚠️ No clusters available for gap validation",
        }

    logger.info(f"Node 6: Validating gaps from {len(clusters)} clusters")

    # Format clusters for prompt
    clusters_text = ""
    for c in clusters:
        clusters_text += f"\n📊 Cluster {c['cluster_id']} (density={c['density']:.2f}, size={c['size']})\n"
        clusters_text += "   Representative limitations:\n"
        for lim in c.get("representative_limitations", [])[:3]:
            clusters_text += f"   • {lim}\n"

    # Format knowledge summary
    knowledge_summary = ""
    for paper in knowledge[:15]:
        knowledge_summary += f"\n📄 {paper.get('paper_title', 'Unknown')}\n"
        knowledge_summary += f"   Method: {paper.get('core_methodology', 'N/A')}\n"
        lims = paper.get("limitations", [])
        if lims:
            knowledge_summary += f"   Limitations: {'; '.join(lims[:3])}\n"

    prompt = GAP_VALIDATION_PROMPT.format(
        num_papers=len(knowledge),
        clusters_text=clusters_text,
        knowledge_summary=knowledge_summary,
    )

    try:
        result = call_flash(prompt, schema=ValidatedGaps)

        if isinstance(result, dict):
            gaps = result.get("gaps", [])
        else:
            gaps = json.loads(result).get("gaps", [])

        gaps = [g for g in gaps if isinstance(g, str) and g.strip()][:5]

        if not gaps:
            gaps = ["Could not validate specific gaps — review cluster data manually"]

        logger.info(f"Node 6: Validated {len(gaps)} gaps")
        for i, gap in enumerate(gaps):
            logger.info(f"  Gap {i+1}: {gap[:100]}...")

        return {
            "validated_gaps": gaps,
            "status_message": f"✅ Validated {len(gaps)} research gaps",
        }

    except Exception as e:
        logger.error(f"Node 6 failed: {e}")
        # Fallback: use top cluster representatives as gaps
        fallback_gaps = []
        for c in clusters[:5]:
            reps = c.get("representative_limitations", [])
            if reps:
                fallback_gaps.append(reps[0])

        return {
            "validated_gaps": fallback_gaps or ["Gap validation failed"],
            "status_message": f"⚠️ Gap validation partial failure, using cluster representatives",
            "error": str(e),
        }
