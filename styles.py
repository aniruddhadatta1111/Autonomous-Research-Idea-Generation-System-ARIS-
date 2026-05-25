"""
ARIS-Idea: CSS Styles & Animations
Injects theme-aware CSS with animations into the Streamlit app.
"""


def get_theme_css(
    primary: str = "#7C3AED",
    accent: str = "#06B6D4",
    bg: str = "#0F0F1A",
    surface: str = "#1A1A2E",
    text: str = "#E2E8F0",
    mode: str = "dark",
) -> str:
    """Generate complete CSS for the app with theme variables and animations."""

    if mode == "light":
        bg = "#F8FAFC"
        surface = "#FFFFFF"
        text = "#1E293B"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
/* ─── CSS Variables ─────────────────────────────────────────────────── */
:root {{
    --primary: {primary};
    --accent: {accent};
    --bg: {bg};
    --surface: {surface};
    --text: {text};
    --glow: {primary}40;
    --border: {primary}30;
}}

/* ─── Global Overrides ──────────────────────────────────────────────── */
.stApp {{
    background: var(--bg) !important;
}}

section[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}}

/* ─── Animated Header ───────────────────────────────────────────────── */
.aris-header {{
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    margin-bottom: 0;
    position: relative;
}}

.aris-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 4.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, {primary}, {accent}, {primary});
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s ease-in-out infinite;
    margin: 0;
    letter-spacing: 6px;
}}

.aris-subtitle {{
    color: {text}99;
    font-size: 1.1rem;
    margin-top: 0.3rem;
    animation: fadeInUp 0.8s ease-out;
}}

.aris-tagline {{
    font-size: 1.6rem;
    font-weight: 300;
    color: {text}60;
    margin: 1rem 0 1.5rem;
    letter-spacing: 2px;
    animation: fadeInUp 1s ease-out;
}}

/* ─── Pipeline Stepper ──────────────────────────────────────────────── */
.pipeline-stepper {{
    display: flex;
    gap: 4px;
    padding: 1rem 0;
    overflow-x: auto;
    justify-content: center;
    flex-wrap: wrap;
}}

.step {{
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 70px;
    padding: 8px 6px;
    border-radius: 12px;
    transition: all 0.3s ease;
    background: {surface};
    border: 1px solid transparent;
}}

.step.active {{
    border-color: {primary};
    background: {primary}20;
    animation: pulse 1.5s ease-in-out infinite;
}}

.step.done {{
    border-color: #22C55E;
    background: #22C55E15;
}}

.step-icon {{
    font-size: 1.3rem;
    margin-bottom: 4px;
}}

.step-label {{
    font-size: 0.6rem;
    color: {text}80;
    text-align: center;
    line-height: 1.2;
}}

/* ─── Cards ─────────────────────────────────────────────────────────── */
.idea-card {{
    background: {surface};
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
    animation: slideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}}

.idea-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, {primary}, {accent});
    border-radius: 4px 0 0 4px;
}}

.idea-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px {primary}20;
    border-color: {primary}60;
}}

.gap-card {{
    background: linear-gradient(135deg, {surface}, {primary}08);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0;
    animation: fadeInUp 0.4s ease-out;
    transition: all 0.3s ease;
}}

.gap-card:hover {{
    border-color: {accent}60;
    transform: translateX(4px);
}}

/* ─── Metric Gauge ──────────────────────────────────────────────────── */
.metric-card {{
    background: {surface};
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease-out;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    transform: scale(1.03);
    box-shadow: 0 8px 30px {primary}15;
}}

.metric-value {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, {primary}, {accent});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.metric-label {{
    color: {text}70;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ─── Score Badge ───────────────────────────────────────────────────── */
.score-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
}}

.score-pass {{
    background: #22C55E20;
    color: #22C55E;
    border: 1px solid #22C55E40;
}}

.score-fail {{
    background: #EF444420;
    color: #EF4444;
    border: 1px solid #EF444440;
}}

/* ─── Animations ────────────────────────────────────────────────────── */
@keyframes shimmer {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}

@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(20px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes slideUp {{
    from {{
        opacity: 0;
        transform: translateY(30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 {primary}40; }}
    50% {{ box-shadow: 0 0 0 8px {primary}00; }}
}}

@keyframes glow {{
    0%, 100% {{ box-shadow: 0 0 20px {primary}30; }}
    50% {{ box-shadow: 0 0 40px {primary}50; }}
}}

/* ─── Custom Button ─────────────────────────────────────────────────── */
div.stButton > button {{
    background: linear-gradient(135deg, {primary}, {accent}) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    transition: all 0.3s ease !important;
    animation: glow 2s ease-in-out infinite;
}}

div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px {primary}40 !important;
}}

/* ─── Progress Bar ──────────────────────────────────────────────────── */
.stProgress > div > div {{
    background: linear-gradient(90deg, {primary}, {accent}) !important;
    border-radius: 10px;
}}

/* ─── Tabs ──────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 8px 20px !important;
    transition: all 0.3s ease;
}}

.stTabs [aria-selected="true"] {{
    background: {primary}20 !important;
    border-color: {primary} !important;
}}

/* ─── Particles Background ──────────────────────────────────────────── */
.particles {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}}

.particle {{
    position: absolute;
    width: 3px; height: 3px;
    background: {primary}30;
    border-radius: 50%;
    animation: float linear infinite;
}}

@keyframes float {{
    0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
    10% {{ opacity: 1; }}
    90% {{ opacity: 1; }}
    100% {{ transform: translateY(-10vh) rotate(720deg); opacity: 0; }}
}}

/* ─── Expander ──────────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
    background: {surface} !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}}

/* ─── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 6px;
}}
::-webkit-scrollbar-track {{
    background: {bg};
}}
::-webkit-scrollbar-thumb {{
    background: {primary}40;
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {primary}60;
}}
</style>
"""


def get_particles_html(primary: str = "#7C3AED", count: int = 20) -> str:
    """Generate HTML for floating particle background."""
    import random
    particles = ""
    for i in range(count):
        left = random.randint(0, 100)
        size = random.uniform(2, 5)
        duration = random.uniform(15, 30)
        delay = random.uniform(0, 15)
        particles += (
            f'<div class="particle" style="'
            f"left:{left}%; width:{size}px; height:{size}px; "
            f"animation-duration:{duration}s; animation-delay:{delay}s;"
            f'"></div>'
        )
    return f'<div class="particles">{particles}</div>'
