"""
ARIS-Idea: Centralized Configuration
All default values are overridable via the Streamlit sidebar UI.
"""

# ─── Gemini Model Names ──────────────────────────────────────────────────────
# Primary generation model (10 RPM / 250 RPD free tier)
FLASH_MODEL         = "gemini-2.5-flash"
# Fallback model used instantly when FLASH_MODEL hits 503 or repeated 429s
FLASH_FALLBACK_MODEL = "gemini-2.5-flash-lite"

# text-embedding-004 via the stable v1 endpoint
EMBEDDING_MODEL  = "text-embedding-004"

# ─── Novelty Scoring Weights ─────────────────────────────────────────────────
# Final Score = α·(1 - semantic) + β·(1 - structural) + γ·(1 - keyword)
DEFAULT_ALPHA = 0.5   # Semantic similarity weight
DEFAULT_BETA = 0.3    # Structural overlap weight
DEFAULT_GAMMA = 0.2   # Keyword overlap weight

# ─── Thresholds (configurable via UI) ────────────────────────────────────────
DEFAULT_COSINE_THRESHOLD = 0.85
DEFAULT_NOVELTY_THRESHOLD = 0.6
DEFAULT_RELEVANCE_THRESHOLD = 0.7

# ─── Pipeline Defaults ───────────────────────────────────────────────────────
MAX_CRITIC_ITERATIONS = 5
DEFAULT_YEAR_START = 2024
DEFAULT_YEAR_END = 2026
MAX_PAPERS_PER_QUERY = 5          # 5 per query × 3 queries, deduped to 15 max
MAX_TOTAL_PAPERS = 15             # Full 15 papers (doesn't add API calls)
NUM_IDEAS = 5
MAX_SEARCH_QUERIES = 3            # Reduced from 5 to save S2 calls

# ─── Clustering ──────────────────────────────────────────────────────────────
CLUSTER_K_MIN = 2
CLUSTER_K_MAX = 5

# ─── Semantic Scholar API ────────────────────────────────────────────────────
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_ENDPOINT = f"{S2_BASE_URL}/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,paperId"
S2_RATE_LIMIT_DELAY = 1.2         # seconds between requests (unauthenticated, with margin)
S2_RATE_LIMIT_DELAY_AUTH = 0.05   # seconds between requests (authenticated)
S2_MAX_CALLS_PER_RUN = 15        # Budget: 3 search + 5 title checks + 1 retry headroom
S2_429_BASE_DELAY = 5.0          # Base delay on 429 (progressive backoff)
S2_429_MAX_RETRIES = 3           # Max retries on 429

# ─── Gemini Rate Limits — Flash only (Free Tier: 10 RPM, 250 RPD) ───────────
# 10 RPM  →  minimum 6.0s inter-call gap; 6.5s adds a safety margin
GEMINI_FLASH_DELAY     = 6.5   # seconds between Flash calls
GEMINI_MAX_FLASH_CALLS = 8     # budget: query + extraction + gap-validation + idea-gen + structural
GEMINI_429_BASE_DELAY  = 15.0  # base back-off on 429
GEMINI_429_MAX_RETRIES = 3     # max retries on 429

# ─── Abstract Truncation ─────────────────────────────────────────────────────
MAX_ABSTRACT_WORDS = 150         # Truncate abstracts to save tokens

# ─── Structural Comparison Budget ─────────────────────────────────────────────
MAX_STRUCTURAL_COMPARISONS = 2   # Max papers to compare per idea (saves Flash calls)
