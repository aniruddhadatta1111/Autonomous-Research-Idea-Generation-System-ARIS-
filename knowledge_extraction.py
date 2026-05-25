"""
Node 3: Knowledge Extraction
Reads paper abstracts and extracts structured knowledge using Gemini Flash.
BATCHES ALL PAPERS into a single API call to stay within call budget.
Actor: Gemini 2.0 Flash + Pydantic
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from state import ARISState
from utils.gemini_client import call_flash
import config as cfg

logger = logging.getLogger(__name__)


class PaperKnowledge(BaseModel):
    """Structured knowledge extracted from a single paper."""
    paper_title: str = Field(description="Title of the paper")
    core_methodology: str = Field(description="Main method or approach used (1 sentence)")
    key_findings: list[str] = Field(description="Key results (2-3 points, brief)")
    limitations: list[str] = Field(
        description="Limitations, weaknesses, and unresolved challenges (2-4 points). Be specific."
    )
    hardware_constraints: str = Field(
        description="Computational or hardware requirements. 'Not specified' if none."
    )
    domain: str = Field(description="Primary research domain/subfield")


class BatchExtraction(BaseModel):
    """Batch extraction result for multiple papers."""
    papers: list[PaperKnowledge]


EXTRACTION_PROMPT = """You are a research analyst. For each paper below, extract structured knowledge.
Focus heavily on LIMITATIONS — these are the most important for identifying research gaps.

Papers to analyze:
{papers_text}

Extract information for each paper. Be specific about limitations.
"""


def _truncate_abstract(abstract: str, max_words: int = cfg.MAX_ABSTRACT_WORDS) -> str:
    """Truncate abstract to max_words to save tokens."""
    words = abstract.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return abstract


def knowledge_extraction_node(state: ARISState) -> dict:
    """
    Node 3: Extract structured knowledge from ALL paper abstracts in ONE call.

    Input: retrieved_papers
    Output: extracted_knowledge, status_message
    """
    papers = state["retrieved_papers"]

    if not papers:
        return {
            "extracted_knowledge": [],
            "status_message": "⚠️ No papers to extract knowledge from",
        }

    logger.info(f"Node 3: Extracting knowledge from {len(papers)} papers (single batch call)")

    # Format ALL papers into one prompt (truncated abstracts)
    papers_text = ""
    for j, paper in enumerate(papers, 1):
        abstract = _truncate_abstract(paper.get("abstract", "No abstract available"))
        papers_text += f"\n--- Paper {j} ---\n"
        papers_text += f"Title: {paper.get('title', 'Unknown')}\n"
        papers_text += f"Year: {paper.get('year', 'Unknown')}\n"
        papers_text += f"Citations: {paper.get('citationCount', 0)}\n"
        papers_text += f"Abstract: {abstract}\n"

    prompt = EXTRACTION_PROMPT.format(papers_text=papers_text)

    try:
        result = call_flash(prompt, schema=BatchExtraction)

        all_knowledge = []
        if isinstance(result, dict):
            extracted = result.get("papers", [])
            for item in extracted:
                if isinstance(item, dict):
                    all_knowledge.append(item)

        logger.info(f"Node 3: Extracted knowledge from {len(all_knowledge)} papers (1 API call)")

        # If extraction returned fewer than expected, add fallbacks
        if len(all_knowledge) < len(papers):
            for i in range(len(all_knowledge), len(papers)):
                paper = papers[i] if i < len(papers) else {}
                all_knowledge.append({
                    "paper_title": paper.get("title", "Unknown"),
                    "core_methodology": "Extraction incomplete",
                    "key_findings": [],
                    "limitations": ["See original abstract for details"],
                    "hardware_constraints": "Not specified",
                    "domain": "Unknown",
                })

        return {
            "extracted_knowledge": all_knowledge,
            "status_message": f"✅ Extracted knowledge from {len(all_knowledge)} papers",
        }

    except Exception as e:
        logger.error(f"Node 3 failed: {e}")
        # Create minimal fallback entries for all papers
        fallback = []
        for paper in papers:
            fallback.append({
                "paper_title": paper.get("title", "Unknown"),
                "core_methodology": "Extraction failed",
                "key_findings": [],
                "limitations": ["Could not extract — see original abstract"],
                "hardware_constraints": "Not specified",
                "domain": "Unknown",
            })
        return {
            "extracted_knowledge": fallback,
            "status_message": f"⚠️ Extraction failed, using fallbacks",
            "error": str(e),
        }
