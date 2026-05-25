# 🔬 ARIS-Idea: Autonomous Research & Idea Generation System

A cloud-native, multi-agent AI system that autonomously discovers research gaps and generates novel, verified project ideas using a stateful LangGraph pipeline.

## ✨ Key Features

- **11-Node Agentic Pipeline** — Stateful LangGraph workflow, not a linear prompt chain
- **Critic-in-the-Loop** — Adaptive self-correction with multi-component novelty verification
- **Hybrid API Strategy** — Flash for speed, Pro for reasoning, Embeddings for similarity math
- **Gap Clustering** — K-Means clustering of limitation embeddings to find recurring bottlenecks
- **Multi-Component Novelty Score** — Semantic similarity + structural overlap + keyword analysis
- **Customizable Theme** — Dark/Light/Custom color schemes with CSS animations

## 🏗️ Architecture

```
User Topic → Query Expansion → Paper Retrieval → Knowledge Extraction
→ Limitation Embedding → Gap Clustering → Gap Validation
→ Idea Generation ⟷ Critic Loop → Evaluation Metrics → Report
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required:
- `GOOGLE_API_KEY` — [Get from Google AI Studio](https://aistudio.google.com/apikey)

Optional (recommended):
- `SEMANTIC_SCHOLAR_API_KEY` — [Get from Semantic Scholar](https://www.semanticscholar.org/product/api#api-key)

### 3. Run

```bash
streamlit run app.py
```

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (Python) |
| Fast LLM | Gemini 2.0 Flash |
| Reasoning LLM | Gemini 2.0 Pro |
| Embeddings | text-embedding-004 |
| Paper Data | Semantic Scholar API |
| Clustering | scikit-learn (K-Means) |
| Frontend | Streamlit |

## 📊 Novelty Scoring Formula

```
Score = α·(1 - semantic_similarity) + β·(1 - structural_overlap) + γ·(1 - keyword_overlap)
```

Default weights: α=0.5, β=0.3, γ=0.2 (configurable in UI)

## 📁 Project Structure

```
├── app.py                     # Streamlit frontend
├── graph.py                   # LangGraph pipeline definition
├── state.py                   # State schema
├── config.py                  # Configuration constants
├── styles.py                  # CSS themes & animations
├── nodes/                     # 11 pipeline nodes
│   ├── query_expansion.py     # Topic → Search queries
│   ├── data_retrieval.py      # Semantic Scholar API
│   ├── knowledge_extraction.py # Abstracts → Structured JSON
│   ├── limitation_embedding.py # Limitations → Vectors
│   ├── gap_clustering.py      # K-Means clustering
│   ├── gap_validation.py      # Density-based validation
│   ├── idea_generation.py     # Constraint-aware generation
│   ├── idea_embedding.py      # Ideas → Vectors
│   ├── novelty_evaluation.py  # Multi-component scoring
│   ├── critic_loop.py         # Adaptive feedback loop
│   └── evaluation_metrics.py  # Pipeline metrics
└── utils/                     # Shared utilities
    ├── gemini_client.py       # Gemini API wrapper
    ├── semantic_scholar.py    # S2 API helper
    ├── embeddings.py          # Similarity math
    └── clustering.py          # K-Means utilities
```

## 📜 License

MIT
