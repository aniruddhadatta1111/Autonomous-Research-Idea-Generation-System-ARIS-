"""
ARIS-Idea: Semantic Scholar API Helper
Handles paper search with caching, rate limiting, 429 progressive backoff,
and per-run call budgets. Designed for unauthenticated usage (~1 req/s).
"""

from __future__ import annotations

import os
import time
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

import requests

import config as cfg

logger = logging.getLogger(__name__)

# ─── Per-Run State ────────────────────────────────────────────────────────────

_call_count = 0
_query_cache: dict[str, list[dict]] = {}
_title_cache: dict[str, bool] = {}
_last_call_time = 0.0


def reset_s2_state():
    """Reset counters and caches at the start of each pipeline run."""
    global _call_count, _query_cache, _title_cache, _last_call_time
    _call_count = 0
    _query_cache = {}
    _title_cache = {}
    _last_call_time = 0.0


def _get_headers() -> dict[str, str]:
    """Build request headers, including API key if available."""
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _get_rate_delay() -> float:
    """Return appropriate rate limit delay based on auth status."""
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        return cfg.S2_RATE_LIMIT_DELAY_AUTH
    return cfg.S2_RATE_LIMIT_DELAY


def _enforce_rate_limit():
    """Enforce minimum delay between S2 API calls."""
    global _last_call_time
    now = time.time()
    delay = _get_rate_delay()
    elapsed = now - _last_call_time
    if elapsed < delay:
        wait = delay - elapsed
        logger.info(f"S2 rate limit: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_call_time = time.time()


def _check_budget() -> bool:
    """Check if we're within the per-run S2 call budget."""
    if _call_count >= cfg.S2_MAX_CALLS_PER_RUN:
        logger.warning(f"S2 call budget exhausted ({_call_count}/{cfg.S2_MAX_CALLS_PER_RUN}). Skipping.")
        return False
    return True


def _make_request(url: str, params: dict) -> dict | None:
    """Make an S2 API request with 429 handling and progressive backoff."""
    global _call_count

    if not _check_budget():
        return None

    _enforce_rate_limit()

    for attempt in range(cfg.S2_429_MAX_RETRIES):
        try:
            response = requests.get(
                url, params=params, headers=_get_headers(), timeout=30
            )

            if response.status_code == 429:
                delay = cfg.S2_429_BASE_DELAY * (2 ** attempt)
                logger.warning(f"S2 rate limited (429). Waiting {delay:.0f}s (attempt {attempt + 1}/{cfg.S2_429_MAX_RETRIES})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            _call_count += 1
            logger.info(f"S2 calls used: {_call_count}/{cfg.S2_MAX_CALLS_PER_RUN}")
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt < cfg.S2_429_MAX_RETRIES - 1:
                delay = 3.0 * (2 ** attempt)
                logger.warning(f"S2 request error: {e}. Retrying in {delay:.0f}s...")
                time.sleep(delay)
            else:
                logger.error(f"S2 request failed after {cfg.S2_429_MAX_RETRIES} attempts: {e}")
                return None

    return None


def _arxiv_fallback(query: str, limit: int) -> list[dict]:
    """Fallback to arXiv if Semantic Scholar is completely rate-limited or empty."""
    try:
        # Clean query for arXiv API
        clean_q = urllib.parse.quote(query.replace('"', '').replace("'", ""))
        url = f"http://export.arxiv.org/api/query?search_query=all:{clean_q}&start=0&max_results={limit}"
        
        response = urllib.request.urlopen(url, timeout=10)
        xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            abstract = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            paper_id = entry.find('atom:id', ns)
            
            if title is None or abstract is None:
                continue
                
            t_text = title.text.replace('\n', ' ')
            a_text = abstract.text.replace('\n', ' ')
            y_text = published.text[:4] if published is not None else "2025"
            pid_text = paper_id.text if paper_id is not None else t_text
            
            # Truncate abstract to save tokens
            words = a_text.split()
            if len(words) > cfg.MAX_ABSTRACT_WORDS:
                a_text = " ".join(words[:cfg.MAX_ABSTRACT_WORDS]) + "..."
                
            papers.append({
                "title": t_text,
                "abstract": a_text,
                "year": y_text,
                "citationCount": 50, # Mock citation count so it sorts nicely
                "paperId": pid_text
            })
            
        logger.info(f"ArXiv fallback found {len(papers)} papers for '{query}'")
        return papers
    except Exception as e:
        logger.error(f"ArXiv fallback failed: {e}")
        return []


def search_papers(
    query: str,
    year_start: int = cfg.DEFAULT_YEAR_START,
    year_end: int = cfg.DEFAULT_YEAR_END,
    limit: int = cfg.MAX_PAPERS_PER_QUERY,
) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.
    Results are cached to avoid duplicate API calls.
    """
    # Check cache first
    cache_key = f"{query}|{year_start}-{year_end}|{limit}"
    if cache_key in _query_cache:
        logger.info(f"S2 cache hit for '{query}'")
        return _query_cache[cache_key]

    params = {
        "query": query,
        "fields": cfg.S2_FIELDS,
        "year": f"{year_start}-{year_end}",
        "limit": limit,
    }

    data = _make_request(cfg.S2_SEARCH_ENDPOINT, params)
    
    # If S2 completely fails due to rate limits, fallback to ArXiv
    if data is None:
        logger.warning(f"S2 request failed for '{query}'. Falling back to ArXiv.")
        papers = _arxiv_fallback(query, limit)
        _query_cache[cache_key] = papers
        return papers

    papers = data.get("data", [])
    
    # Filter out papers with no abstract
    papers = [p for p in papers if p.get("abstract")]
    
    # If S2 returned no valid papers, fallback to ArXiv
    if not papers:
        logger.warning(f"S2 returned 0 valid papers for '{query}'. Falling back to ArXiv.")
        papers = _arxiv_fallback(query, limit)
        _query_cache[cache_key] = papers
        return papers

    # Truncate abstracts to save tokens downstream
    for p in papers:
        abstract = p.get("abstract", "")
        words = abstract.split()
        if len(words) > cfg.MAX_ABSTRACT_WORDS:
            p["abstract"] = " ".join(words[:cfg.MAX_ABSTRACT_WORDS]) + "..."

    # Cache the result
    _query_cache[cache_key] = papers
    logger.info(f"S2 search '{query}': found {len(papers)} papers with abstracts")
    return papers


def search_papers_multi(
    queries: list[str],
    year_start: int = cfg.DEFAULT_YEAR_START,
    year_end: int = cfg.DEFAULT_YEAR_END,
    limit_per_query: int = cfg.MAX_PAPERS_PER_QUERY,
    max_total: int = cfg.MAX_TOTAL_PAPERS,
) -> list[dict]:
    """
    Search for papers across multiple queries, deduplicate by paperId.
    Stops early if call budget is exhausted.
    """
    all_papers = []
    seen_ids = set()

    for i, query in enumerate(queries):
        if not _check_budget():
            logger.warning(f"Stopping search early — S2 budget exhausted after {i} queries")
            break

        papers = search_papers(query, year_start, year_end, limit_per_query)

        for paper in papers:
            pid = paper.get("paperId", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(paper)

    # Sort by citation count (most cited first)
    all_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)

    # Limit total
    all_papers = all_papers[:max_total]

    logger.info(f"Total unique papers after dedup: {len(all_papers)}")
    return all_papers


def search_by_title(title: str) -> bool:
    """
    Check if a paper with a similar title exists in Semantic Scholar.
    Results are cached to avoid duplicate API calls.
    """
    # Check cache first
    cache_key = title.lower().strip()
    if cache_key in _title_cache:
        logger.info(f"S2 title cache hit for '{title[:50]}'")
        return _title_cache[cache_key]

    # Skip if budget exhausted
    if not _check_budget():
        return False

    params = {
        "query": title,
        "fields": "title",
        "limit": 3,
    }

    data = _make_request(cfg.S2_SEARCH_ENDPOINT, params)
    if data is None:
        return False

    results = data.get("data", [])
    title_lower = title.lower().strip()

    found = False
    for paper in results:
        existing_title = paper.get("title", "").lower().strip()
        if existing_title == title_lower:
            found = True
            break
        if len(title_lower) > 10 and (
            title_lower in existing_title or existing_title in title_lower
        ):
            found = True
            break

    # Cache the result
    _title_cache[cache_key] = found
    return found
