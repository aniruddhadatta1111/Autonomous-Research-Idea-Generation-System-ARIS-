"""
ARIS-Idea: Streamlit Frontend
Autonomous Research & Idea Generation System
Premium animated UI with customizable themes.
"""

import os
import logging
import streamlit as st
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

import config as cfg
from styles import get_theme_css, get_particles_html

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIS | AI Research Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Pipeline Step Definitions ────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("🚀", "Init"),
    ("🔍", "Query\nExpand"),
    ("📚", "Data\nRetrieve"),
    ("🧠", "Knowledge\nExtract"),
    ("📐", "Limit.\nEmbed"),
    ("🔬", "Gap\nCluster"),
    ("✅", "Gap\nValidate"),
    ("💡", "Idea\nGenerate"),
    ("📊", "Idea\nEmbed"),
    ("⚖️", "Novelty\nEval"),
    ("🔄", "Critic\nLoop"),
    ("📈", "Eval\nMetrics"),
    ("📄", "Report\nGen"),
]

NODE_KEYS = [
    "init_pipeline", "query_expansion", "data_retrieval", "knowledge_extraction",
    "limitation_embedding", "gap_clustering", "gap_validation",
    "idea_generation", "idea_embedding", "novelty_evaluation",
    "critic_loop", "evaluation_metrics", "report_generation",
]


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar with theme & config controls. Returns user config dict."""
    with st.sidebar:
        st.markdown("## 🎨 Theme")
        theme_mode = st.selectbox("Mode", ["Dark", "Light", "Custom"], index=0)

        primary = "#7C3AED"
        accent = "#06B6D4"
        bg = "#0F0F1A"
        surface = "#1A1A2E"
        text_color = "#E2E8F0"

        if theme_mode == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                primary = st.color_picker("Primary", "#7C3AED")
                bg = st.color_picker("Background", "#0F0F1A")
            with col2:
                accent = st.color_picker("Accent", "#06B6D4")
                text_color = st.color_picker("Text", "#E2E8F0")
            surface = bg  # auto-derive

        st.markdown("---")

        # ─── Collapsible: Pipeline Config ─────────────────────────────
        with st.expander("⚙️ Pipeline Configuration", expanded=False):
            year_range = st.slider(
                "Paper Year Range",
                min_value=2018, max_value=2026,
                value=(cfg.DEFAULT_YEAR_START, cfg.DEFAULT_YEAR_END),
            )
            num_ideas = st.slider("Number of Ideas", 1, 10, cfg.NUM_IDEAS)

        # ─── Collapsible: Thresholds ──────────────────────────────────
        with st.expander("🎚️ Thresholds", expanded=False):
            cosine_thresh = st.slider(
                "Cosine Similarity Threshold",
                0.50, 1.0, cfg.DEFAULT_COSINE_THRESHOLD, 0.05,
                help="Max allowed semantic similarity before flagging as non-novel",
            )
            novelty_thresh = st.slider(
                "Novelty Threshold",
                0.20, 1.0, cfg.DEFAULT_NOVELTY_THRESHOLD, 0.05,
                help="Minimum novelty score for an idea to pass",
            )
            relevance_thresh = st.slider(
                "Relevance Threshold",
                0.30, 1.0, cfg.DEFAULT_RELEVANCE_THRESHOLD, 0.05,
                help="Minimum relevance score for topic alignment",
            )

        # ─── Collapsible: Novelty Weights ─────────────────────────────
        with st.expander("⚖️ Novelty Weights (α, β, γ)", expanded=False):
            st.caption("α + β + γ will be normalized to 1.0")
            alpha_raw = st.slider("α (Semantic)", 0.0, 1.0, cfg.DEFAULT_ALPHA, 0.05)
            beta_raw = st.slider("β (Structural)", 0.0, 1.0, cfg.DEFAULT_BETA, 0.05)
            gamma_raw = st.slider("γ (Keyword)", 0.0, 1.0, cfg.DEFAULT_GAMMA, 0.05)

            total = alpha_raw + beta_raw + gamma_raw
            if total > 0:
                alpha = alpha_raw / total
                beta = beta_raw / total
                gamma = gamma_raw / total
            else:
                alpha, beta, gamma = 0.5, 0.3, 0.2

            st.caption(f"Normalized: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

    # Inject CSS and particles OUTSIDE sidebar (renders globally)
    css = get_theme_css(primary, accent, bg, surface, text_color, theme_mode.lower())
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(get_particles_html(primary, 15), unsafe_allow_html=True)

    return {
        "year_start": year_range[0],
        "year_end": year_range[1],
        "num_ideas": num_ideas,
        "cosine_threshold": cosine_thresh,
        "novelty_threshold": novelty_thresh,
        "relevance_threshold": relevance_thresh,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


# ─── Pipeline Stepper ────────────────────────────────────────────────────────

def render_pipeline_stepper(current_step: int = -1, completed: int = -1):
    """Render the horizontal pipeline visualization."""
    steps_html = '<div class="pipeline-stepper">'
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        if i <= completed:
            cls = "step done"
            icon_display = "✓"
        elif i == current_step:
            cls = "step active"
            icon_display = icon
        else:
            cls = "step"
            icon_display = icon

        steps_html += f'''
        <div class="{cls}">
            <span class="step-icon">{icon_display}</span>
            <span class="step-label">{label}</span>
        </div>'''
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)


# ─── Result Renderers ────────────────────────────────────────────────────────

def render_papers_tab(papers: list[dict]):
    """Render the papers table."""
    if not papers:
        st.info("No papers retrieved yet.")
        return

    st.markdown(f"### 📚 {len(papers)} Papers Retrieved")
    for i, p in enumerate(papers):
        with st.expander(f"**{i+1}. {p.get('title', 'N/A')}** — {p.get('year', '?')} ({p.get('citationCount', 0):,} citations)"):
            st.markdown(p.get("abstract", "_No abstract_"))


def render_gaps_tab(gaps: list[str], clusters: list[dict]):
    """Render gap analysis."""
    if clusters:
        st.markdown("### 🔬 Limitation Clusters")
        for c in clusters[:6]:
            density_bar = "█" * min(int(c.get("density", 0) * 2), 20)
            card_html = f"""
            <div class="gap-card">
                <strong>Cluster {c['cluster_id']}</strong>
                &nbsp;|&nbsp; Size: {c['size']}
                &nbsp;|&nbsp; Density: {c['density']:.2f} <span style="color:var(--accent)">{density_bar}</span>
                <ul style="margin-top:8px; padding-left:20px;">
            """
            for lim in c.get("representative_limitations", [])[:3]:
                card_html += f"<li style='margin:4px 0; font-size:0.9rem;'>{lim}</li>"
            card_html += "</ul></div>"
            st.markdown(card_html, unsafe_allow_html=True)

    if gaps:
        st.markdown("### 🎯 Validated Research Gaps")
        for i, gap in enumerate(gaps, 1):
            st.markdown(
                f'<div class="gap-card"><strong>Gap {i}:</strong> {gap}</div>',
                unsafe_allow_html=True,
            )


def render_ideas_tab(ideas: list[dict], scores: list[dict]):
    """Render idea blueprint cards."""
    if not ideas:
        st.info("No ideas generated yet.")
        return

    st.markdown(f"### 💡 {len(ideas)} Research Ideas")
    for i, idea in enumerate(ideas):
        score_data = scores[i] if i < len(scores) else {}
        final = score_data.get("final_score", 0)
        passed = score_data.get("passed", True)
        badge_cls = "score-pass" if passed else "score-fail"
        badge_text = f"✅ {final:.2f}" if passed else f"⚠️ {final:.2f}"

        card_html = f"""
        <div class="idea-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h4 style="margin:0; color:var(--text);">{idea.get('title', 'Untitled')}</h4>
                <span class="score-badge {badge_cls}">{badge_text}</span>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.9rem; margin-bottom:12px;">
                <div><strong>Approach:</strong> {idea.get('approach_type', 'N/A')}</div>
                <div><strong>Target Gap:</strong> {idea.get('target_gap', 'N/A')[:80]}</div>
            </div>
            <p style="margin:8px 0;"><strong>Motivation:</strong> {idea.get('motivation', 'N/A')}</p>
            <p style="margin:8px 0;"><strong>Methodology:</strong> {idea.get('methodology', 'N/A')}</p>
            <p style="margin:8px 0;"><strong>Expected Impact:</strong> {idea.get('expected_impact', 'N/A')}</p>
        """

        # Score breakdown
        if score_data:
            card_html += f"""
            <div style="margin-top:12px; padding-top:12px; border-top:1px solid var(--border); font-size:0.8rem; opacity:0.7;">
                Semantic: {score_data.get('semantic_score', 0):.3f} |
                Structural: {score_data.get('structural_score', 0):.3f} |
                Keyword: {score_data.get('keyword_score', 0):.3f}
            </div>"""

        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)


def render_metrics_tab(metrics: dict):
    """Render evaluation metrics dashboard."""
    if not metrics:
        st.info("No metrics available yet.")
        return

    st.markdown("### 📊 Evaluation Dashboard")

    cols = st.columns(4)
    metric_items = [
        ("Avg Novelty", f"{metrics.get('average_novelty_score', 0):.3f}"),
        ("Pass Rate", f"{metrics.get('pass_rate', 0):.0%}"),
        ("Gap Coverage", f"{metrics.get('gap_coverage', 0):.0%}"),
        ("Iterations", str(metrics.get('iterations_used', 0))),
    ]
    for col, (label, value) in zip(cols, metric_items):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("total_ideas_evaluated", 0)}</div>'
            f'<div class="metric-label">Ideas Evaluated</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{metrics.get("constraints_generated", 0)}</div>'
            f'<div class="metric-label">Constraints Generated</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    user_config = render_sidebar()

    # Header
    st.markdown(
        '<div class="aris-header">'
        '<h1 class="aris-title">ARIS</h1>'
        '<p class="aris-subtitle">Autonomous Research & Idea Generation System</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tagline
    st.markdown(
        '<p class="aris-tagline" style="text-align:center;">Find the gaps. Build the future.</p>',
        unsafe_allow_html=True,
    )

    # Input section
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Graph Neural Networks, Federated Learning, LLM Alignment...",
            label_visibility="collapsed",
        )
    with col2:
        run_button = st.button("Generate Ideas", use_container_width=True)

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "running" not in st.session_state:
        st.session_state.running = False

    # Pipeline stepper (show current progress)
    current_step = st.session_state.get("current_step", -1)
    completed_step = st.session_state.get("completed_step", -1)
    render_pipeline_stepper(current_step, completed_step)

    # Run pipeline
    if run_button and topic.strip():
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("⚠️ Please set `GOOGLE_API_KEY` in your `.env` file or Streamlit secrets.")
            return

        st.session_state.running = True
        st.session_state.results = None

        # Import graph here to avoid import errors on startup
        from graph import build_graph

        graph = build_graph()

        initial_state = {
            "user_topic": topic.strip(),
            "search_queries": [],
            "retrieved_papers": [],
            "extracted_knowledge": [],
            "limitations_texts": [],
            "limitations_embeddings": [],
            "limitation_clusters": [],
            "validated_gaps": [],
            "generated_ideas": [],
            "adaptive_constraints": [],
            "idea_embeddings": [],
            "novelty_scores": [],
            "novelty_status": False,
            "failure_reasons": [],
            "iteration_count": 0,
            "evaluation_metrics": {},
            "final_report": "",
            "status_message": "",
            "error": "",
            "config": user_config,
        }

        # Run with live progress
        progress_container = st.container()
        status_text = st.empty()

        with progress_container:
            with st.status("🔬 ARIS-Idea Pipeline Running...", expanded=True) as status:
                try:
                    step_idx = 0
                    # Accumulate state from stream events
                    accumulated_state = dict(initial_state)

                    for event in graph.stream(initial_state, stream_mode="updates"):
                        for node_name, node_output in event.items():
                            # Update pipeline stepper
                            if node_name in NODE_KEYS:
                                step_idx = NODE_KEYS.index(node_name)

                            st.session_state.current_step = step_idx
                            st.session_state.completed_step = step_idx

                            # Merge node output into accumulated state
                            if isinstance(node_output, dict):
                                for key, value in node_output.items():
                                    accumulated_state[key] = value

                                msg = node_output.get("status_message", "")
                            else:
                                msg = ""

                            icon, label = PIPELINE_STEPS[min(step_idx, len(PIPELINE_STEPS) - 1)]
                            st.write(f"{icon} **{label.replace(chr(10), ' ')}**: {msg}")

                            logger.info(f"Completed node: {node_name} — {msg}")

                    st.session_state.results = accumulated_state
                    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

                except Exception as e:
                    logger.error(f"Pipeline error: {e}", exc_info=True)
                    status.update(label=f"❌ Error: {str(e)[:100]}", state="error")
                    st.error(f"Pipeline failed: {str(e)}")

        st.session_state.running = False

    elif run_button and not topic.strip():
        st.warning("Please enter a research topic.")

    # Display results
    results = st.session_state.results
    if results:
        st.markdown("---")

        tabs = st.tabs(["📚 Papers", "🔬 Gap Analysis", "💡 Ideas", "📊 Metrics", "📄 Report"])

        with tabs[0]:
            render_papers_tab(results.get("retrieved_papers", []))

        with tabs[1]:
            render_gaps_tab(
                results.get("validated_gaps", []),
                results.get("limitation_clusters", []),
            )

        with tabs[2]:
            render_ideas_tab(
                results.get("generated_ideas", []),
                results.get("novelty_scores", []),
            )

        with tabs[3]:
            render_metrics_tab(results.get("evaluation_metrics", {}))

        with tabs[4]:
            report = results.get("final_report", "")
            if report:
                st.markdown(report)
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name=f"aris_report_{results.get('user_topic', 'research').replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("Report not yet generated.")


if __name__ == "__main__":
    main()
