"""
ARIS-Idea: Gemini Client Wrapper
Singleton client with rate limiting, back-off, and per-run call budgets.
Uses the google-genai SDK (not the deprecated google-generativeai).

Model strategy (free tier):
  Generation (primary)  : gemini-2.5-flash        (10 RPM / 250 RPD)
  Generation (fallback) : gemini-2.5-flash-lite   (used automatically on 503/429)
  Embeddings            : text-embedding-004      via a dedicated v1 client
  Embedding safety-net  : sklearn TF-IDF          (local, no API, always works)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

import config as cfg

logger = logging.getLogger(__name__)


# ─── Singleton Clients ────────────────────────────────────────────────────────
# Two separate clients:
#   _client      – default (v1beta) for generate_content
#   _emb_client  – stable v1 for embed_content (text-embedding-004 is v1-only)

_client: genai.Client | None = None
_emb_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Get or create the singleton v1beta Gemini client (for generation)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get a key at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _get_emb_client() -> genai.Client:
    """Get or create the singleton v1 Gemini client (for embeddings)."""
    global _emb_client
    if _emb_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        # text-embedding-004 only exists on the stable v1 endpoint, not v1beta
        _emb_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1"),
        )
    return _emb_client


# ─── Per-Run Call Counter ─────────────────────────────────────────────────────

_flash_calls = 0
_last_flash_time = 0.0


def reset_call_counters():
    """Reset counters at the start of each pipeline run."""
    global _flash_calls, _last_flash_time
    _flash_calls = 0
    _last_flash_time = 0.0


def _enforce_rate_limit():
    """Enforce minimum delay between Flash calls (10 RPM → 6s gap)."""
    global _last_flash_time
    now = time.time()
    elapsed = now - _last_flash_time
    if elapsed < cfg.GEMINI_FLASH_DELAY:
        wait = cfg.GEMINI_FLASH_DELAY - elapsed
        logger.info(f"Rate limit: waiting {wait:.1f}s before Flash call")
        time.sleep(wait)
    _last_flash_time = time.time()


# ─── Retry with 503 / 429 Handling ───────────────────────────────────────────

def _retry_with_backoff(
    func,
    max_retries: int = None,
    base_delay: float = None,
    fallback_func=None,
):
    """
    Retry wrapper with aggressive fallback for free tiers.

    - 503 / unavailable        → Jumps instantly to fallback_func (if provided)
    - 429 / resource_exhausted → Tries a quick 5s delay once. If it hits 429 again, jumps to fallback_func.
    - other errors             → 2s × 2^attempt
    """
    if max_retries is None:
        max_retries = cfg.GEMINI_429_MAX_RETRIES
    if base_delay is None:
        base_delay = cfg.GEMINI_429_BASE_DELAY

    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # 1. Server Overload (503) -> INSTANT FALLBACK
            if "503" in error_str or "unavailable" in error_str:
                if fallback_func is not None:
                    logger.warning(f"Service unavailable (503). Instantly jumping to fallback model...")
                    break  # Break retry loop, go to fallback
                else:
                    delay = 10.0 * (2 ** attempt)
                    logger.warning(f"Service unavailable (503). Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)

            # 2. Rate Limit (429) -> QUICK RETRY, THEN FALLBACK
            elif "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                if fallback_func is not None:
                    if attempt == 0:
                        delay = 5.0
                        logger.warning(f"Rate limited (429). Quick {delay}s wait before trying again...")
                        time.sleep(delay)
                    else:
                        logger.warning(f"Still rate limited (429). Jumping to fallback model...")
                        break  # Break retry loop, go to fallback
                else:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)

            # 3. Other Errors (500, etc.) -> Standard Backoff
            else:
                delay = 2.0 * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.0f}s...")
                time.sleep(delay)

    # All primary retries exhausted OR loop broken for instant fallback
    if fallback_func is not None:
        logger.info(f"Executing fallback function (using {cfg.FLASH_FALLBACK_MODEL})...")
        try:
            return fallback_func()
        except Exception as fe:
            logger.error(f"Fallback model also failed: {fe}")
            raise fe

    raise last_error


# ─── Flash Model (all generation tasks) ──────────────────────────────────────

def call_flash(prompt: str, schema: type[BaseModel] | None = None) -> dict | str:
    """
    Call Gemini 2.5 Flash with rate limiting.
    Instantly falls back to gemini-2.5-flash-lite on 503 or repeated 429s.
    Respects per-run call budget (cfg.GEMINI_MAX_FLASH_CALLS).
    """
    global _flash_calls

    if _flash_calls >= cfg.GEMINI_MAX_FLASH_CALLS:
        logger.warning(
            f"Flash call budget exhausted ({_flash_calls}/{cfg.GEMINI_MAX_FLASH_CALLS}). Skipping."
        )
        return {} if schema is not None else ""

    _enforce_rate_limit()
    client = get_client()

    gen_config: dict[str, Any] = {}
    if schema is not None:
        gen_config["response_mime_type"] = "application/json"
        gen_config["response_schema"] = schema

    cfg_obj = types.GenerateContentConfig(**gen_config) if gen_config else None

    def _primary():
        response = client.models.generate_content(
            model=cfg.FLASH_MODEL,
            contents=prompt,
            config=cfg_obj,
        )
        text = response.text
        return json.loads(text) if schema is not None else text

    def _fallback():
        logger.info(f"*** Using fallback model: {cfg.FLASH_FALLBACK_MODEL} ***")
        response = client.models.generate_content(
            model=cfg.FLASH_FALLBACK_MODEL,
            contents=prompt,
            config=cfg_obj,
        )
        text = response.text
        return json.loads(text) if schema is not None else text

    result = _retry_with_backoff(_primary, fallback_func=_fallback)
    _flash_calls += 1
    logger.info(f"Flash calls used: {_flash_calls}/{cfg.GEMINI_MAX_FLASH_CALLS}")
    return result


# ─── call_pro: transparent alias → Flash ─────────────────────────────────────

def call_pro(prompt: str, schema: type[BaseModel] | None = None) -> dict | str:
    """
    Alias for call_flash.
    Gemini 2.5 Pro is no longer used (free-tier quota too low).
    All tasks are routed to Gemini 2.5 Flash instead.
    """
    logger.debug("call_pro() → redirecting to call_flash()")
    return call_flash(prompt, schema=schema)


# ─── Embeddings ──────────────────────────────────────────────────────────────

def _tfidf_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Local stateless embeddings via sklearn HashingVectorizer — zero API calls, always works.
    Used as a safety-net fallback when all Gemini embedding calls fail.
    The resulting vectors are exactly 512 dimensions and L2-normalised, 
    so cosine similarity works correctly between separate embedding calls.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    logger.warning(
        "Using local Hashing embeddings (Gemini embedding API unavailable). "
        "Clustering and novelty checks will still work correctly."
    )

    vect = HashingVectorizer(n_features=512, norm='l2')
    mat = vect.transform(texts).toarray()
    return mat.tolist()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for a list of texts.

    Strategy:
      1. Try text-embedding-004 via the dedicated v1 client (stable endpoint).
      2. On any failure, fall back to local sklearn TF-IDF vectors.
    """
    if not texts:
        return []

    emb_client = _get_emb_client()

    def _call():
        response = emb_client.models.embed_content(
            model=cfg.EMBEDDING_MODEL,
            contents=texts,
        )
        return [e.values for e in response.embeddings]

    try:
        return _retry_with_backoff(_call, max_retries=2, base_delay=5.0)
    except Exception as e:
        logger.warning(f"Gemini embedding failed ({e}). Falling back to local TF-IDF.")
        return _tfidf_embeddings(texts)
