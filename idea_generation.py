"""
Node 7: Idea Generation
Proposes actionable research ideas that directly solve validated gaps.
Incorporates adaptive constraints from previous critic failures.
Actor: Gemini 2.5 Flash
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field
from typing import Literal

from state import ARISState
from utils.gemini_client import call_flash
import config as cfg

logger = logging.getLogger(__name__)


class ResearchIdea(BaseModel):
    """Schema for a single research idea."""
    title: str = Field(description="Concise, descriptive title for the research project")
    target_gap: str = Field(description="Which validated gap this idea addresses")
    motivation: str = Field(description="Why this is important and timely (2-3 sentences)")
    methodology: str = Field(
        description="Specific technical approach — algorithms, architectures, "
        "or frameworks to be used (3-5 sentences)"
    )
    approach_type: str = Field(
        description="One of: Optimization, Hybridization, Domain Transfer"
    )
    expected_impact: str = Field(description="What measurable improvement is expected")
    supporting_citations: list[str] = Field(
        description="Titles of papers from the literature that support this approach"
    )


class IdeaBatch(BaseModel):
    """Batch of generated ideas."""
    ideas: list[ResearchIdea]


IDEA_GENERATION_PROMPT = """You are a visionary research scientist proposing novel project ideas.

VALIDATED RESEARCH GAPS (sorted by importance):
{gaps_text}

EXISTING LITERATURE SUMMARY:
{knowledge_summary}

{constraints_section}

TASK: Generate exactly {num_ideas} actionable research project ideas.

Requirements for each idea:
1. DIRECTLY addresses one of the validated gaps above
2. Proposes a SPECIFIC technical methodology (not vague)
3. Is NOVEL — do not simply repackage existing methods
4. Falls into one category: Optimization, Hybridization, or Domain Transfer
5. Cites supporting papers from the literature above
6. Each idea should target a DIFFERENT gap (maximize gap coverage)

{iteration_context}
"""


def idea_generation_node(state: ARISState) -> dict:
    """
    Node 7: Generate research ideas targeting validated gaps.

    Input: validated_gaps, extracted_knowledge, adaptive_constraints, iteration_count
    Output: generated_ideas, status_message
    """
    gaps = state.get("validated_gaps", [])
    knowledge = state.get("extracted_knowledge", [])
    constraints = state.get("adaptive_constraints", [])
    iteration = state.get("iteration_count", 0)
    user_config = state.get("config", {})
    num_ideas = user_config.get("num_ideas", cfg.NUM_IDEAS)

    if not gaps:
        return {
            "generated_ideas": [],
            "status_message": "⚠️ No validated gaps to generate ideas for",
        }

    logger.info(f"Node 7: Generating {num_ideas} ideas (iteration {iteration + 1})")

    # Format gaps
    gaps_text = ""
    for i, gap in enumerate(gaps, 1):
        gaps_text += f"  {i}. {gap}\n"

    # Format knowledge summary
    knowledge_summary = ""
    for paper in knowledge[:15]:
        knowledge_summary += f"  • {paper.get('paper_title', 'Unknown')}: {paper.get('core_methodology', 'N/A')}\n"

    # Format constraints (from previous critic failures)
    constraints_section = ""
    if constraints:
        constraints_section = "⚠️ CONSTRAINTS FROM PREVIOUS ITERATIONS (you MUST avoid these):\n"
        for c in constraints:
            constraints_section += f"  ❌ {c}\n"
        constraints_section += "\nGenerate ideas that are DIFFERENT from previously rejected ones.\n"

    # Add iteration context
    iteration_context = ""
    if iteration > 0:
        iteration_context = (
            f"This is iteration {iteration + 1}. Previous ideas were rejected for "
            f"insufficient novelty. You MUST generate fundamentally different approaches."
        )

    prompt = IDEA_GENERATION_PROMPT.format(
        gaps_text=gaps_text,
        knowledge_summary=knowledge_summary,
        constraints_section=constraints_section,
        num_ideas=num_ideas,
        iteration_context=iteration_context,
    )

    try:
        result = call_flash(prompt, schema=IdeaBatch)

        if isinstance(result, dict):
            ideas = result.get("ideas", [])
        else:
            ideas = json.loads(result).get("ideas", [])

        # Ensure ideas are dicts
        parsed_ideas = []
        for idea in ideas:
            if isinstance(idea, dict):
                parsed_ideas.append(idea)

        logger.info(f"Node 7: Generated {len(parsed_ideas)} ideas")
        for idea in parsed_ideas:
            logger.info(f"  💡 {idea.get('title', 'Untitled')} [{idea.get('approach_type', '?')}]")

        return {
            "generated_ideas": parsed_ideas[:num_ideas],
            "status_message": f"✅ Generated {len(parsed_ideas[:num_ideas])} research ideas (iteration {iteration + 1})",
        }

    except Exception as e:
        logger.error(f"Node 7 failed: {e}")
        return {
            "generated_ideas": [],
            "status_message": f"❌ Idea generation failed: {str(e)}",
            "error": str(e),
        }
