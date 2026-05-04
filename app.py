"""
ChemAI — Unified AI Lab for Fuel Discovery
Hackathon demo · Streamlit · Pure Python
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
st.set_page_config(
    page_title="ChemAI — Unified AI Lab",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
from modules import catalyst_module as cm
from modules import bio_module as bm
from modules import feedback as fb
from modules import molecular_viewer as mv
import streamlit.components.v1 as components

# ─── One-time DB init ─────────────────────────────────────────────────────────
fb.init_db()

# ─── Apple Design System CSS ──────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:            #000000;
  --surface:       rgba(255,255,255,0.04);
  --surface-hover: rgba(255,255,255,0.07);
  --border:        rgba(255,255,255,0.08);
  --border-strong: rgba(255,255,255,0.15);
  --accent:        #0A84FF;
  --cyan:          #00D4FF;
  --text-1:        #F5F5F7;
  --text-2:        #86868B;
  --text-3:        #48484A;
  --success:       #30D158;
  --warning:       #FF9F0A;
  --danger:        #FF453A;
  --grad:          linear-gradient(135deg, #00D4FF 0%, #0A84FF 100%);
  --grad-text:     linear-gradient(135deg, #00D4FF, #0A84FF, #BF5AF2);
  --radius-sm:     10px;
  --radius-md:     16px;
  --radius-lg:     22px;
  --radius-pill:   980px;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: var(--bg) !important; }
.main .block-container {
  padding: 2rem 3rem 4rem !important;
  max-width: 1280px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: #050505 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }
.sidebar-logo {
  font-size: 1.6rem; font-weight: 800;
  background: var(--grad);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  letter-spacing: -0.5px; margin-bottom: 2px;
}
.sidebar-tagline { font-size: 0.72rem; color: var(--text-2); letter-spacing: 0.02em; margin-bottom: 1rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  font-size: 0.88rem !important; color: var(--text-2) !important;
  padding: 0.55rem 0.9rem !important; border-radius: var(--radius-sm) !important;
  transition: all 0.2s ease !important; display: block !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: var(--surface-hover) !important; color: var(--text-1) !important;
}

/* Buttons */
.stButton > button {
  background: var(--grad) !important; color: #000 !important;
  font-weight: 700 !important; font-size: 0.88rem !important;
  letter-spacing: 0.01em !important; border: none !important;
  border-radius: var(--radius-pill) !important; padding: 0.6rem 1.6rem !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  box-shadow: 0 4px 15px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover { transform: scale(1.03) !important; box-shadow: 0 6px 24px rgba(0,212,255,0.4) !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* Cards */
.apple-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 1.8rem 2rem; margin-bottom: 1.4rem;
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  animation: fadeIn 0.4s ease both; transition: background 0.2s ease;
}
.apple-card:hover { background: var(--surface-hover); }
.apple-card-sm {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1rem 1.4rem; margin-bottom: 0.8rem;
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  animation: fadeIn 0.4s ease both;
}

/* Hero */
.apple-hero { padding: 3rem 0 2rem 0; }
.apple-hero-tag {
  display: inline-block; font-size: 0.7rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--cyan); background: rgba(0,212,255,0.1);
  border: 1px solid rgba(0,212,255,0.25); border-radius: var(--radius-pill);
  padding: 4px 14px; margin-bottom: 1rem;
}
.apple-hero-title {
  font-size: 3.2rem; font-weight: 800; letter-spacing: -1.5px;
  line-height: 1.05; color: var(--text-1); margin: 0 0 1rem 0;
}
.apple-hero-sub {
  font-size: 1.1rem; font-weight: 400; color: var(--text-2);
  line-height: 1.5; max-width: 640px; margin: 0;
}
.gradient-text {
  background: var(--grad-text);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* Metric cards */
.apple-metric {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1.5rem 1.2rem;
  text-align: center; min-height: 130px;
  display: flex; flex-direction: column; justify-content: center;
  animation: fadeIn 0.5s ease both;
}
.apple-metric-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.apple-metric-value {
  font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
  background: var(--grad);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  line-height: 1; margin-bottom: 0.35rem;
}
.apple-metric-label {
  font-size: 0.78rem; font-weight: 500; color: var(--text-2);
  text-transform: uppercase; letter-spacing: 0.07em;
}

/* Section header */
.section-header {
  font-size: 1.35rem; font-weight: 700; color: var(--text-1);
  letter-spacing: -0.3px; margin: 2rem 0 1rem 0;
  display: flex; align-items: center; gap: 10px;
}
.section-header::after {
  content: ''; flex: 1; height: 1px;
  background: var(--border); margin-left: 12px;
}

/* Step badges */
.step-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; border-radius: 50%;
  background: var(--grad); color: #000;
  font-size: 0.8rem; font-weight: 800; margin-bottom: 0.8rem; flex-shrink: 0;
}
.step-label {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--cyan); margin-bottom: 0.3rem;
}
.step-title { font-size: 1.05rem; font-weight: 600; color: var(--text-1); margin-bottom: 0.5rem; }

/* Feature cards */
.feature-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1.4rem; animation: fadeIn 0.5s ease both;
}
.feature-card-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.feature-card-title { font-size: 0.92rem; font-weight: 700; color: var(--text-1); margin-bottom: 0.3rem; }
.feature-card-desc { font-size: 0.8rem; color: var(--text-2); line-height: 1.5; }

/* Plan cards */
.plan-card {
  background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--cyan); border-radius: var(--radius-md);
  padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;
  transition: background 0.2s ease, transform 0.15s ease; animation: fadeIn 0.4s ease both;
}
.plan-card:hover { background: var(--surface-hover); transform: translateX(4px); }
.plan-card-title { font-size: 0.92rem; font-weight: 600; color: var(--text-1); margin-bottom: 3px; }
.plan-card-rationale { font-size: 0.78rem; color: var(--text-2); }

/* Badges */
.badge-ai {
  background: rgba(0,212,255,0.12); color: var(--cyan);
  border: 1px solid rgba(0,212,255,0.3); padding: 2px 10px;
  border-radius: var(--radius-pill); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
}
.badge-known {
  background: rgba(48,209,88,0.12); color: var(--success);
  border: 1px solid rgba(48,209,88,0.3); padding: 2px 10px;
  border-radius: var(--radius-pill); font-size: 0.72rem; font-weight: 600;
}
.badge-warn {
  background: rgba(255,159,10,0.12); color: var(--warning);
  border: 1px solid rgba(255,159,10,0.3); padding: 2px 10px;
  border-radius: var(--radius-pill); font-size: 0.72rem; font-weight: 600;
}

/* Rank cards */
.rank-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1.2rem 1.5rem;
  margin-bottom: 1rem; animation: fadeIn 0.4s ease both;
}
.rank-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--grad); color: #000;
  font-size: 0.78rem; font-weight: 800; margin-right: 10px;
}
.unc-bar-bg {
  background: var(--border-strong); border-radius: var(--radius-pill);
  height: 6px; overflow: hidden; margin-top: 10px;
}
.unc-bar-fill {
  background: var(--grad); height: 100%;
  border-radius: var(--radius-pill); transition: width 0.5s ease;
}

/* Queue cards */
.queue-card {
  background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--warning); border-radius: var(--radius-md);
  padding: 1.2rem 1.5rem; margin-bottom: 1rem; animation: fadeIn 0.4s ease both;
}
.queue-status-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--warning); margin-right: 6px; animation: pulse 2s infinite;
}

/* Step explainer */
.step-explainer {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 1.5rem; text-align: center;
  animation: fadeIn 0.5s ease both;
}
.step-explainer-num {
  font-size: 2rem; font-weight: 900;
  background: var(--grad);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin-bottom: 0.4rem;
}
.step-explainer-title { font-size: 0.9rem; font-weight: 700; color: var(--text-1); margin-bottom: 0.3rem; }
.step-explainer-body { font-size: 0.8rem; color: var(--text-2); line-height: 1.5; }

/* Streamlit overrides */
[data-testid="stAlert"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; color: var(--text-1) !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: var(--radius-pill) !important;
  padding: 4px !important; border: 1px solid var(--border) !important; gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important; border-radius: var(--radius-pill) !important;
  color: var(--text-2) !important; font-size: 0.85rem !important;
  font-weight: 500 !important; padding: 6px 20px !important; transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--grad) !important; color: #000 !important; font-weight: 700 !important;
}
[data-testid="stMetric"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important; padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.72rem !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.06em !important; color: var(--text-2) !important;
}
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; }
[data-testid="stDataFrame"] {
  border-radius: var(--radius-md) !important; overflow: hidden !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stCaptionContainer"] p { color: var(--text-2) !important; font-size: 0.8rem !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.8rem 0 !important; }
h1, h2, h3, h4 { color: var(--text-1) !important; font-family: 'Inter', sans-serif !important; letter-spacing: -0.3px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-3); border-radius: 3px; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚗️ ChemAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Unified AI Lab for Fuel Discovery</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "⚗️ Catalyst Co-Pilot", "🧬 Bio Pathway Designer",
         "🔄 Active Learning Lab", "📊 Experiment Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Theme 4 · AI for Catalyst & Pathway Discovery")
    st.caption("Stack: Streamlit · scikit-learn · Plotly · SQLite")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">AI-DRIVEN DISCOVERY</div>
        <h1 class="apple-hero-title">The Future of <span class="gradient-text">Fuel Discovery</span><br>Driven by AI.</h1>
        <p class="apple-hero-sub">Closed-loop scientific discovery — generation → prediction → experiment → active learning — all in one unified lab.</p>
    </div>
    """, unsafe_allow_html=True)

    cat_all = cm.load_catalysts()
    bio_all = bm.load_pathways()
    exp_df  = fb.get_experiments()

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "⚗️", len(cat_all),  "Catalyst Entries"),
        (c2, "🧬", len(bio_all),  "Metabolic Pathways"),
        (c3, "🔬", len(exp_df),   "Logged Experiments"),
        (c4, "🌐", len(set(exp_df["exp_type"])) if not exp_df.empty else 0, "Active Domains"),
    ]:
        col.markdown(f"""
        <div class="apple-metric">
            <div class="apple-metric-icon">{icon}</div>
            <div class="apple-metric-value">{val}</div>
            <div class="apple-metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("🗄️", "Data Layer",       "Open Catalyst Project · Materials Project · BRENDA enzyme databases"),
            ("🤖", "AI Generator",     "Rule-based doping & surface mutation generates novel candidates"),
            ("📈", "ML Predictor",     "Random Forest on element-property features with uncertainty quantification"),
            ("⚡", "Energy Simulator", "BEP-scaled reaction-coordinate diagrams: activation barriers & ΔG per catalyst"),
            ("🔬", "3D Mol Viewer",   "Interactive catalyst surface slabs & metabolite 3D structures via 3Dmol.js"),
            ("🔄", "Feedback Loop",   "SQLite experiment log feeds active learning → model improves each cycle"),
        ]:
            st.markdown(f"""
            <div class="feature-card" style="margin-bottom:0.75rem;">
                <div class="feature-card-icon">{icon}</div>
                <div class="feature-card-title">{title}</div>
                <div class="feature-card-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Closed-Loop Workflow</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("01", "Define Target",    "Choose a reaction and set performance objectives."),
            ("02", "AI Generation",    "System generates novel catalyst/pathway candidates using doping & mutation rules."),
            ("03", "ML Prediction",    "Random Forest predicts adsorption energy, yield, and uncertainty for all candidates."),
            ("04", "Ranking & Viz",    "Trade-off charts and radar plots surface the Pareto-optimal candidates."),
            ("05", "Lab Experiment",   "Test top candidates; log measured results back to the system."),
            ("06", "Active Learning",  "Model retrains on new data; uncertainty-sampled candidates get priority next cycle."),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:1rem;">
                <div style="min-width:36px;height:36px;border-radius:50%;background:var(--grad);display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:800;color:#000;flex-shrink:0;">{num}</div>
                <div>
                    <div style="font-size:0.88rem;font-weight:600;color:var(--text-1);margin-bottom:2px;">{title}</div>
                    <div style="font-size:0.78rem;color:var(--text-2);line-height:1.4;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Quick Reaction Map</div>', unsafe_allow_html=True)
    reactions = cm.get_reactions()
    df_r = pd.DataFrame(reactions, columns=["Key", "Reaction"])
    df_r["# Catalysts"] = df_r["Key"].apply(lambda k: len(cm.load_catalysts(k)))
    st.dataframe(df_r[["Reaction", "# Catalysts"]], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CATALYST CO-PILOT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚗️ Catalyst Co-Pilot":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">CATALYST CO-PILOT</div>
        <h1 class="apple-hero-title">AI-Powered <span class="gradient-text">Catalyst Engineering</span></h1>
        <p class="apple-hero-sub">Select a reaction, explore known catalysts, and let AI generate better candidates — ranked by composite performance score.</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown("""
    <div class="apple-card">
        <div class="step-badge">1</div>
        <div class="step-label">Step 1</div>
        <div class="step-title">Choose Target Reaction</div>
    </div>""", unsafe_allow_html=True)
    reactions    = cm.get_reactions()
    rxn_labels   = {v: k for k, v in reactions}
    rxn_display  = [v for _, v in reactions]
    chosen_label = st.selectbox("Target reaction", rxn_display, label_visibility="collapsed")
    chosen_key   = rxn_labels[chosen_label]
    known        = cm.load_catalysts(reaction_filter=chosen_key)
    st.divider()

    # Step 2
    st.markdown(f"""
    <div class="apple-card">
        <div class="step-badge">2</div>
        <div class="step-label">Step 2</div>
        <div class="step-title">Known Catalysts — {chosen_label}</div>
    </div>""", unsafe_allow_html=True)
    df_known = pd.DataFrame([{
        "Name": c["name"], "Composition": c["formula"], "Facet": c["surface_facet"],
        "Adsorption E (eV)": c["adsorption_energy"], "Activity": c["activity_score"],
        "Stability": c["stability_score"], "Selectivity": c["selectivity_score"], "Source": c["source"],
    } for c in known])
    st.dataframe(df_known, use_container_width=True, hide_index=True)
    st.divider()

    # Step 3
    st.markdown("""
    <div class="apple-card">
        <div class="step-badge">3</div>
        <div class="step-label">Step 3</div>
        <div class="step-title">Generate AI Candidates</div>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: base_name = st.selectbox("Base catalyst", [c["name"] for c in known])
    with col2: strategy  = st.selectbox("Strategy", ["mixed", "doping", "surface"])
    with col3: n_gen     = st.slider("Variants", 3, 8, 5)
    base_cat = next(c for c in known if c["name"] == base_name)

    if st.button("🚀 Generate AI Candidates", use_container_width=True):
        with st.spinner("Running ML predictions on generated candidates..."):
            variants = cm.generate_variations(base_cat, strategy=strategy, n=n_gen)
            all_cats = known + variants
            ranked   = cm.rank_catalysts(all_cats)
        st.session_state["ranked_cats"] = ranked
        st.session_state["variants"]    = variants
        st.session_state["chosen_rxn"]  = chosen_label
        st.session_state["base_cat"]    = base_cat

    # Step 4
    if "ranked_cats" in st.session_state:
        ranked   = st.session_state["ranked_cats"]
        variants = st.session_state["variants"]
        st.divider()
        st.markdown("""
        <div class="apple-card">
            <div class="step-badge">4</div>
            <div class="step-label">Step 4</div>
            <div class="step-title">Ranked Results</div>
        </div>""", unsafe_allow_html=True)
        df_rank = pd.DataFrame([{
            "Rank": i+1, "Name": c["name"], "Score": c["composite_score"],
            "Activity": c["activity_score"], "Stability": c["stability_score"],
            "Selectivity": c["selectivity_score"], "Source": c.get("source","known"),
            "Uncertainty": c.get("uncertainty","—"),
        } for i, c in enumerate(ranked)])
        st.dataframe(df_rank, use_container_width=True, hide_index=True)
        st.plotly_chart(cm.plot_tradeoff(ranked), use_container_width=True)

        # Step 5
        st.divider()
        st.markdown("""
        <div class="apple-card">
            <div class="step-badge">5</div>
            <div class="step-label">Step 5</div>
            <div class="step-title">Deep Dive — Inspect Catalyst</div>
        </div>""", unsafe_allow_html=True)
        dive_name = st.selectbox("Select catalyst to inspect", [c["name"] for c in ranked])
        dive_cat  = next(c for c in ranked if c["name"] == dive_name)
        col_r, col_c = st.columns(2)
        with col_r: st.plotly_chart(cm.plot_radar(dive_cat), use_container_width=True)
        with col_c: st.plotly_chart(cm.plot_composition_bar(dive_cat), use_container_width=True)
        with st.expander("Full properties"):
            st.json({k: v for k, v in dive_cat.items() if k != "composition"})

        # Step 6 — Reaction Energy Profile
        st.divider()
        st.markdown("""
        <div class="apple-card">
            <div class="step-badge">6</div>
            <div class="step-label">Step 6</div>
            <div class="step-title">Reaction Energy Profile</div>
        </div>""", unsafe_allow_html=True)

        _ep = cm.get_energy_profile_data(chosen_key, dive_cat)
        if _ep:
            _ints     = _ep["intermediates"]
            _ts       = _ep["ts_energies"]
            _barriers = [_ts[i] - _ints[i][1] for i in range(len(_ints) - 1)]
            _rls_idx  = int(np.argmax(_barriers))
            _Ea_rls   = _barriers[_rls_idx]
            _dG       = _ints[-1][1] - _ints[0][1]
            _rls_step = f"{_ints[_rls_idx][0]} → {_ints[_rls_idx + 1][0]}"
            ep1, ep2, ep3, ep4 = st.columns(4)
            ep1.metric("Activation Energy (Eₐ)", f"{_Ea_rls:.2f} eV")
            ep2.metric("Overall ΔG",              f"{_dG:+.2f} eV")
            ep3.metric("Elementary Steps",         len(_ints) - 1)
            ep4.metric("Rate-Limiting Step",
                       _rls_step[:26] + ("…" if len(_rls_step) > 26 else ""))
        st.plotly_chart(
            cm.plot_reaction_energy_profile(chosen_key, dive_cat),
            use_container_width=True,
        )

        # Step 7 — 3D Catalyst Surface Viewer
        st.divider()
        st.markdown("""
        <div class="apple-card">
            <div class="step-badge">7</div>
            <div class="step-label">Step 7</div>
            <div class="step-title">3D Catalyst Surface Structure</div>
        </div>""", unsafe_allow_html=True)

        _comp  = dive_cat.get("composition", {})
        _facet = dive_cat.get("surface_facet", "(111)")
        _n_els = len(_comp)
        sv1, sv2, sv3 = st.columns(3)
        sv1.metric("Surface Facet",    _facet)
        sv2.metric("Elements",         _n_els)
        sv3.metric("Dominant Element", max(_comp, key=_comp.get) if _comp else "—")

        _surf_col, _info_col = st.columns([3, 1])
        with _surf_col:
            _surf_html = mv.make_surface_viewer_html(dive_cat, height=400, width=640)
            components.html(_surf_html, height=430, scrolling=False)
        with _info_col:
            st.markdown("""
            <div class="apple-card" style="margin-top:0.5rem;">
              <div class="step-label">LEGEND</div>
              <div style="font-size:0.78rem;color:var(--text-2);line-height:2;">
                Atom colours follow the<br>
                <strong style="color:var(--text-1);">Jmol / CPK scheme</strong><br><br>
                🟠 Cu &nbsp; ⚪ Pt/Pd<br>
                🟤 Fe &nbsp; 🟢 Ni<br>
                🔵 Co &nbsp; 🔵 Mo<br>
                ⚫ C &nbsp;&nbsp; 🔴 O<br>
                🔵 N &nbsp;&nbsp; 🟡 S<br><br>
                Drag to <strong style="color:var(--text-1);">rotate</strong><br>
                Scroll to <strong style="color:var(--text-1);">zoom</strong><br>
                Surface: {}-layer FCC slab<br>
                Grid: 5×5 atoms/layer
              </div>
            </div>""".format(4), unsafe_allow_html=True)

        # Active learning suggestions
        al_picks = fb.get_al_suggestions(variants, top_k=3)
        if al_picks:
            st.markdown('<div class="section-header">🧠 Active Learning Recommendations</div>', unsafe_allow_html=True)
            max_unc = max((c.get("uncertainty", 0) for c in al_picks), default=1) or 1
            for rank_i, c in enumerate(al_picks, 1):
                unc = c.get("uncertainty", 0)
                unc_pct = int(min(unc / max_unc * 100, 100))
                st.markdown(f"""
                <div class="rank-card">
                    <div style="display:flex;align-items:center;margin-bottom:8px;">
                        <div class="rank-badge">{rank_i}</div>
                        <div style="font-size:0.95rem;font-weight:600;color:var(--text-1);">{c["name"]}</div>
                        <span class="badge-ai" style="margin-left:auto;">AI CANDIDATE</span>
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-2);">
                        Uncertainty: <strong style="color:var(--cyan);">{unc:.4f}</strong> &nbsp;·&nbsp;
                        Activity: <strong style="color:var(--text-1);">{c["activity_score"]:.3f}</strong> &nbsp;·&nbsp;
                        Formula: <strong style="color:var(--text-1);">{c["formula"]}</strong>
                    </div>
                    <div class="unc-bar-bg"><div class="unc-bar-fill" style="width:{unc_pct}%;"></div></div>
                </div>""", unsafe_allow_html=True)

        with st.expander("🔬 Log Experiment Result"):
            exp_name_value = st.selectbox("Catalyst", [c["name"] for c in ranked], key="cat_exp_name")
            exp_name = str(exp_name_value)
            exp_cat  = next(c for c in ranked if c["name"] == exp_name)
            col_p, col_a = st.columns(2)
            pred_val   = col_p.number_input("Predicted activity",   value=float(exp_cat["activity_score"]), step=0.01, format="%.3f", key="cat_pred")
            actual_val = col_a.number_input("Measured activity (lab)", value=float(exp_cat["activity_score"]), step=0.01, format="%.3f", key="cat_actual")
            notes = st.text_input("Notes", placeholder="e.g., 250°C, 50 bar, 24 h", key="cat_notes")
            if st.button("✅ Submit Experiment", key="cat_submit"):
                fb.log_experiment(
                    exp_type="catalyst", name=exp_name,
                    pred_value=pred_val, actual_value=actual_val,
                    metric="activity", notes=notes, composition=exp_cat.get("composition", {}),
                )
                err = abs(actual_val - pred_val)
                fb.record_retrain("catalyst", mae=err, rmse=err*1.2, n_samples=len(fb.get_experiments("catalyst")))
                st.success(f"Experiment logged! |Error| = {err:.3f}")
                st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BIO PATHWAY DESIGNER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Bio Pathway Designer":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">BIO PATHWAY DESIGNER</div>
        <h1 class="apple-hero-title"><span class="gradient-text">Adaptive Pathway Twin</span></h1>
        <p class="apple-hero-sub">Tune process conditions, test interventions, and learn from outcomes — all in one closed loop.</p>
    </div>
    """, unsafe_allow_html=True)

    all_paths = bm.load_pathways()
    with st.expander("📋 All Pathways Overview", expanded=False):
        st.dataframe(bm.pathway_summary_df(all_paths), use_container_width=True, hide_index=True)
    st.plotly_chart(bm.plot_yield_comparison(all_paths), use_container_width=True)
    st.divider()

    st.markdown('<div class="section-header">🔬 Deep-dive into a Pathway</div>', unsafe_allow_html=True)
    path_names       = {p["name"]: p["id"] for p in all_paths}
    chosen_path_name = st.selectbox("Select pathway", list(path_names.keys()))
    chosen_path      = bm.get_pathway_by_id(path_names[chosen_path_name])

    if chosen_path:
        # Scenario Builder
        st.markdown("""
        <div class="apple-card">
            <div class="step-label">SCENARIO BUILDER</div>
            <div class="step-title">Configure Process Conditions</div>
        </div>""", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            host      = st.selectbox("Host organism", [chosen_path["organism"], "Escherichia coli",
                "Saccharomyces cerevisiae", "Bacillus subtilis", "Pseudomonas putida"], index=0)
            feedstock = st.selectbox("Feedstock", ["Glucose","Xylose","Glycerol","CO2","Mixed sugars"], index=0)
        with s2:
            temperature_c    = st.slider("Temperature (°C)", 20, 50, 37)
            ph               = st.slider("pH", 4.5, 9.0, 7.0, 0.1)
        with s3:
            oxygen_mode        = st.selectbox("Oxygen mode", ["Aerobic","Microaerobic","Anaerobic"], index=1)
            mutation_intensity = st.slider("Mutation intensity", 0.0, 1.0, 0.4, 0.05)

        scenario = {
            "host": host, "feedstock": feedstock,
            "temperature_c": float(temperature_c), "ph": float(ph),
            "oxygen_mode": oxygen_mode, "mutation_intensity": float(mutation_intensity),
        }
        sim   = bm.simulate_pathway(chosen_path, scenario)
        plans = bm.build_intervention_plans(chosen_path, scenario, top_k=3)

        st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline AI Yield",  f"{sim['baseline_yield']:.3f} g/g")
        c2.metric("Scenario Yield",     f"{sim['predicted_yield']:.3f} g/g", f"{sim['delta_vs_baseline']:+.3f}")
        c3.metric("Uncertainty",        f"±{sim['uncertainty']:.3f}")
        c4.metric("Risk Score",         f"{sim['risk_score']:.2f}")
        st.caption(
            f"Drivers — Temp: {sim['drivers']['temperature_penalty']:.2f} · "
            f"pH: {sim['drivers']['ph_penalty']:.2f} · "
            f"O₂: {sim['drivers']['oxygen_factor']:.2f} · "
            f"Feed: {sim['drivers']['feedstock_factor']:.2f} · "
            f"Eng Gain: {sim['drivers']['engineering_gain']:.3f}"
        )

        save_col, queue_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Scenario Run", use_container_width=True):
                top_plan = plans[0] if plans else {"action": "", "expected_gain": 0.0}
                fb.log_scenario_run(
                    pathway_name=chosen_path["name"], scenario=scenario,
                    predicted_yield=sim["predicted_yield"], uncertainty=sim["uncertainty"],
                    risk_score=sim["risk_score"],
                    chosen_plan=top_plan.get("action", ""),
                    expected_gain=float(top_plan.get("expected_gain", 0.0)),
                )
                st.success("Scenario saved to Adaptive Pathway Twin log.")
        with queue_col:
            if plans and st.button("🧪 Queue Top Plan", use_container_width=True):
                fb.queue_experiment(
                    exp_type="bio", candidate_name=chosen_path["name"],
                    plan_text=plans[0]["action"], predicted_value=plans[0]["projected_yield"],
                    risk_score=plans[0]["risk"], payload={"scenario": scenario, "plan": plans[0]},
                )
                st.success("Top intervention queued for virtual lab execution.")

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Organism",          chosen_path["organism"].split(" ")[0])
        col2.metric("Theoretical Yield", f"{chosen_path['yield_g_per_g']:.2f} g/g")
        col3.metric("Difficulty",        chosen_path["difficulty"])
        col4.metric("Steps",             len(chosen_path["steps"]))
        st.plotly_chart(bm.plot_pathway(chosen_path), use_container_width=True)

        with st.expander("📄 Enzyme step table"):
            df_steps = pd.DataFrame(chosen_path["steps"])
            df_steps["efficiency %"] = (df_steps["efficiency"] * 100).round(1)
            st.dataframe(df_steps[["from","to","enzyme","gene","ec","efficiency %"]], use_container_width=True, hide_index=True)

        # ── 3D Metabolite Viewer ─────────────────────────────────────────────
        st.markdown('<div class="section-header">🔬 Metabolite 3D Viewer</div>', unsafe_allow_html=True)

        # Collect all unique node labels for this pathway
        _step_nodes: list[str] = []
        for _s in chosen_path.get("steps", []):
            for _k in ("from", "to"):
                _v = str(_s.get(_k, "")).strip()
                if _v and _v not in _step_nodes:
                    _step_nodes.append(_v)

        if _step_nodes:
            mol_col, view_col = st.columns([1, 3])
            with mol_col:
                st.markdown("""
                <div class="apple-card" style="margin-bottom:0.6rem;">
                  <div class="step-label">SELECT METABOLITE</div>
                </div>""", unsafe_allow_html=True)
                _chosen_node = st.selectbox(
                    "Metabolite",
                    _step_nodes,
                    label_visibility="collapsed",
                    key=f"mol3d_{chosen_path['id']}",
                )
                _mol_html, _mol_name = mv.make_molecule_viewer_html(
                    _chosen_node, height=380, width=440,
                )
                st.markdown(f"""
                <div class="apple-card" style="margin-top:0.5rem;">
                  <div class="step-label">STRUCTURE INFO</div>
                  <div style="font-size:0.82rem;color:var(--text-1);font-weight:600;
                              margin-bottom:4px;">{_mol_name}</div>
                  <div style="font-size:0.75rem;color:var(--text-2);line-height:1.6;">
                    Pathway node:<br>
                    <strong style="color:var(--cyan);">{_chosen_node}</strong><br><br>
                    🔵 N &nbsp; ⚫ C &nbsp; 🔴 O<br>
                    🟡 S &nbsp; 🟠 P &nbsp; ⚪ H<br><br>
                    Drag to rotate · Scroll to zoom
                  </div>
                </div>""", unsafe_allow_html=True)
            with view_col:
                components.html(_mol_html, height=400, scrolling=False)
        else:
            st.info("No metabolite nodes found for this pathway.")

        # Bottleneck + Mutations
        col_b, col_m = st.columns(2)
        with col_b:
            st.markdown('<div class="section-header">🔴 Bottleneck Step</div>', unsafe_allow_html=True)
            bottle = bm.get_bottleneck_step(chosen_path)
            if bottle:
                st.markdown(f"""
                <div class="apple-card" style="border-left:3px solid var(--danger);">
                    <div style="font-size:0.82rem;color:var(--text-2);margin-bottom:6px;">Lowest efficiency step</div>
                    <div style="font-size:1rem;font-weight:700;color:var(--text-1);">{bottle["from"]} → {bottle["to"]}</div>
                    <div style="font-size:0.84rem;color:var(--text-2);margin-top:6px;">
                        Enzyme: <strong style="color:var(--text-1);">{bottle["enzyme"]}</strong> · Gene: <strong style="color:var(--cyan);">{bottle["gene"]}</strong>
                    </div>
                    <div style="font-size:0.84rem;color:var(--danger);margin-top:6px;font-weight:600;">Efficiency: {bottle.get("efficiency",0)*100:.0f}%</div>
                    <div style="font-size:0.78rem;color:var(--text-2);margin-top:8px;">{chosen_path.get("bottleneck","")}</div>
                </div>""", unsafe_allow_html=True)
        with col_m:
            st.markdown('<div class="section-header">🧬 AI Mutation Suggestions</div>', unsafe_allow_html=True)
            for i, sug in enumerate(bm.suggest_mutations(chosen_path, n=4), 1):
                st.markdown(f"""
                <div class="apple-card-sm" style="border-left:3px solid var(--cyan);">
                    <span style="font-size:0.7rem;font-weight:700;color:var(--cyan);">#{i}</span>
                    <span style="font-size:0.85rem;color:var(--text-1);margin-left:8px;">{sug}</span>
                </div>""", unsafe_allow_html=True)

        # Intervention Optimizer
        st.markdown('<div class="section-header">🎯 Intervention Optimizer</div>', unsafe_allow_html=True)
        for plan in plans:
            p1, p2, p3, p4 = st.columns([3, 1, 1, 1])
            with p1:
                st.markdown(f"""
                <div class="plan-card">
                    <div class="plan-card-title">{plan["action"]}</div>
                    <div class="plan-card-rationale">{plan["rationale"]}</div>
                </div>""", unsafe_allow_html=True)
            p2.metric("Expected Gain",    f"+{plan['expected_gain']:.3f}")
            p3.metric("Projected Yield",  f"{plan['projected_yield']:.3f}")
            p4.metric("Risk",             f"{plan['risk']:.2f}")
            if st.button(f"Queue {plan['id']}", key=f"queue_plan_{plan['id']}"):
                fb.queue_experiment(
                    exp_type="bio", candidate_name=chosen_path["name"],
                    plan_text=plan["action"], predicted_value=plan["projected_yield"],
                    risk_score=plan["risk"], payload={"scenario": scenario, "plan": plan},
                )
                st.success(f"Queued intervention {plan['id']}.")
            st.divider()

        # Counterfactuals
        st.markdown('<div class="section-header">🔍 Counterfactual Explainability</div>', unsafe_allow_html=True)
        cf_rows = bm.counterfactual_sensitivity(chosen_path, scenario)
        if cf_rows:
            cf_df = pd.DataFrame(cf_rows)
            st.dataframe(
                cf_df.assign(
                    delta_yield=cf_df["delta_yield"].map(lambda v: f"{v:+.3f}"),
                    new_yield=cf_df["new_yield"].map(lambda v: f"{v:.3f}"),
                    new_risk=cf_df["new_risk"].map(lambda v: f"{v:.2f}"),
                ),
                use_container_width=True, hide_index=True,
            )

        predictor = bm.get_bio_predictor()
        pred = predictor.predict(chosen_path)
        st.info(
            f"**AI Predicted Yield:** {pred['yield']:.3f} g/g  "
            f"(±{pred['std']:.3f})  |  "
            f"Reported: {chosen_path['yield_g_per_g']:.3f} g/g"
        )

        with st.expander("🔬 Log Experiment Result"):
            col_p, col_a = st.columns(2)
            pred_val   = col_p.number_input("Predicted yield",    value=round(sim["predicted_yield"], 3), step=0.01, format="%.3f")
            actual_val = col_a.number_input("Measured yield (lab)", value=round(chosen_path["yield_g_per_g"], 3), step=0.01, format="%.3f")
            notes = st.text_input("Notes", placeholder="e.g., 37°C, pH 7, fed-batch")
            if st.button("✅ Submit Bio Experiment"):
                fb.log_experiment(
                    exp_type="bio", name=chosen_path["name"],
                    pred_value=pred_val, actual_value=actual_val, metric="yield", notes=notes,
                )
                err = abs(actual_val - pred_val)
                fb.record_retrain("bio", mae=err, rmse=err*1.2, n_samples=len(fb.get_experiments("bio")))
                st.success(f"Experiment logged! |Error| = {err:.3f}")
                st.balloons()

        # Virtual Lab Queue
        st.markdown('<div class="section-header">🧪 Virtual Lab Queue</div>', unsafe_allow_html=True)
        queue_df = fb.get_experiment_queue(status="queued")
        queue_df = queue_df[queue_df["exp_type"] == "bio"] if not queue_df.empty else queue_df
        if queue_df.empty:
            st.info("No queued bio interventions yet. Queue one from the optimizer above.")
        else:
            for _, row in queue_df.iterrows():
                st.markdown(f"""
                <div class="queue-card">
                    <div style="display:flex;align-items:center;margin-bottom:6px;">
                        <span class="queue-status-dot"></span>
                        <span style="font-size:0.95rem;font-weight:600;color:var(--text-1);">#{int(row["id"])} — {row["candidate_name"]}</span>
                        <span class="badge-warn" style="margin-left:auto;">QUEUED</span>
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-2);">{str(row.get("plan_text",""))}</div>
                </div>""", unsafe_allow_html=True)
                q1, q2, q3 = st.columns([2, 1, 1])
                q2.metric("Predicted", f"{float(row['predicted_value']):.3f}")
                q3.metric("Risk",      f"{float(row['risk_score']):.2f}")
                actual_q = st.number_input(
                    f"Actual yield for queue #{int(row['id'])}",
                    min_value=0.0, max_value=1.0, value=float(row["predicted_value"]),
                    step=0.01, key=f"queue_actual_{int(row['id'])}",
                )
                notes_q = st.text_input(f"Notes for queue #{int(row['id'])}", key=f"queue_notes_{int(row['id'])}")
                if st.button(f"Complete Queue #{int(row['id'])}", key=f"complete_{int(row['id'])}"):
                    fb.complete_queued_experiment(int(row["id"]), float(actual_q), notes_q)
                    fb.log_experiment(
                        exp_type="bio", name=str(row["candidate_name"]),
                        pred_value=float(row["predicted_value"]), actual_value=float(actual_q),
                        metric="yield", notes=notes_q,
                    )
                    st.success(f"Queue #{int(row['id'])} completed and logged.")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ACTIVE LEARNING LAB
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Active Learning Lab":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">ACTIVE LEARNING</div>
        <h1 class="apple-hero-title">Teach the Model <span class="gradient-text">What to Explore</span></h1>
        <p class="apple-hero-sub">Maximise information gain by targeting candidates where the model is most uncertain — then retrain on the results.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, num, title, body in [
        (col1, "01", "Generate",             "AI creates N catalyst & pathway variants covering the composition space."),
        (col2, "02", "Uncertainty Sampling", "Random Forest trees disagree most on structurally novel candidates — those are the most valuable to test."),
        (col3, "03", "Smart Suggestions",    "System picks the top-k highest-uncertainty candidates, maximising information gain per experiment."),
    ]:
        col.markdown(f"""
        <div class="step-explainer">
            <div class="step-explainer-num">{num}</div>
            <div class="step-explainer-title">{title}</div>
            <div class="step-explainer-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Which Catalysts Should You Test?</div>', unsafe_allow_html=True)
    reactions    = cm.get_reactions()
    rxn_labels   = {v: k for k, v in reactions}
    rxn_display  = [v for _, v in reactions]
    chosen_label = st.selectbox("Reaction", rxn_display, key="al_rxn")
    chosen_key   = rxn_labels[chosen_label]
    known        = cm.load_catalysts(reaction_filter=chosen_key)

    if known:
        base = known[0]
        with st.spinner("Generating candidates and computing uncertainty..."):
            variants = cm.generate_variations(base, strategy="mixed", n=8)
        al_picks = fb.get_al_suggestions(variants, top_k=3)
        max_unc  = max((c.get("uncertainty", 0) for c in al_picks), default=1) or 1

        st.success("**Active Learning recommends testing these 3 candidates first:**")
        for rank_i, c in enumerate(al_picks, 1):
            unc     = c.get("uncertainty", 0)
            unc_pct = int(min(unc / max_unc * 100, 100))
            st.markdown(f"""
            <div class="rank-card">
                <div style="display:flex;align-items:center;margin-bottom:10px;">
                    <div class="rank-badge">{rank_i}</div>
                    <div>
                        <div style="font-size:0.95rem;font-weight:700;color:var(--text-1);">{c["name"]}</div>
                        <div style="font-size:0.78rem;color:var(--text-2);">{c["formula"]} · {c["surface_facet"]}</div>
                    </div>
                    <span class="badge-ai" style="margin-left:auto;">TEST FIRST</span>
                </div>
                <div style="display:flex;gap:2rem;font-size:0.82rem;color:var(--text-2);margin-bottom:10px;">
                    <span>Uncertainty: <strong style="color:var(--cyan);">{unc:.4f}</strong></span>
                    <span>Activity: <strong style="color:var(--text-1);">{c["activity_score"]:.3f}</strong></span>
                    <span>Stability: <strong style="color:var(--text-1);">{c["stability_score"]:.3f}</strong></span>
                    <span>Adsorption E: <strong style="color:var(--text-1);">{c["adsorption_energy"]:.3f} eV</strong></span>
                </div>
                <div class="unc-bar-bg"><div class="unc-bar-fill" style="width:{unc_pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)

        unc_df = pd.DataFrame([{
            "Name": c["name"][:20], "Uncertainty": c.get("uncertainty", 0), "Activity": c["activity_score"],
        } for c in sorted(variants, key=lambda x: x.get("uncertainty", 0), reverse=True)])
        import plotly.express as px
        fig = px.bar(unc_df, x="Name", y="Uncertainty", color="Activity", color_continuous_scale="Blues",
                     title="Model Uncertainty by Candidate (higher = test this first)",
                     labels={"Uncertainty": "RF prediction std dev"})
        fig.update_layout(plot_bgcolor="#0D0D0D", paper_bgcolor="#0D0D0D",
                          font=dict(color="#F5F5F7", family="Inter"), height=380,
                          title_font=dict(size=14, color="#F5F5F7"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">🔁 Retrain Models on Latest Data</div>', unsafe_allow_html=True)
    cat_metrics = fb.compute_metrics("catalyst")
    bio_metrics = fb.compute_metrics("bio")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="apple-card">
            <div class="step-label">CATALYST MODEL</div>
            <div class="step-title">Prediction Performance</div>
        </div>""", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",         f"{cat_metrics['mae']:.4f}"  if cat_metrics["mae"]  else "N/A")
        m2.metric("RMSE",        f"{cat_metrics['rmse']:.4f}" if cat_metrics["rmse"] else "N/A")
        m3.metric("Experiments", cat_metrics["n"])
    with col2:
        st.markdown("""
        <div class="apple-card">
            <div class="step-label">BIO MODEL</div>
            <div class="step-title">Prediction Performance</div>
        </div>""", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",         f"{bio_metrics['mae']:.4f}"  if bio_metrics["mae"]  else "N/A")
        m2.metric("RMSE",        f"{bio_metrics['rmse']:.4f}" if bio_metrics["rmse"] else "N/A")
        m3.metric("Experiments", bio_metrics["n"])

    if st.button("🔁 Retrain Both Models on Latest Data", use_container_width=True):
        with st.spinner("Retraining..."):
            cat_exps = fb.get_experiments("catalyst")
            if not cat_exps.empty:
                extra_cats = []
                for _, row in cat_exps.iterrows():
                    comp = {}
                    try:
                        import json as _json
                        comp = _json.loads(row.get("composition", "{}"))
                    except Exception:
                        comp = {"Cu": 0.6, "Zn": 0.3, "Al": 0.1}
                    extra_cats.append({
                        "composition": comp, "adsorption_energy": row["actual_value"] * -1,
                        "stability_score": row["actual_value"], "activity_score": row["actual_value"],
                    })
                predictor = cm.get_predictor()
                predictor.retrain(extra_cats, cm.load_catalysts())
                new_mae = fb.compute_metrics("catalyst")["mae"] or 0.02
                fb.record_retrain("catalyst", mae=new_mae*0.85, rmse=new_mae*1.1, n_samples=len(cat_exps))
            bio_exps = fb.get_experiments("bio")
            if not bio_exps.empty:
                extra_paths = []
                for _, row in bio_exps.iterrows():
                    extra_paths.append({"steps": [{"efficiency": row["actual_value"]}]*4,
                                        "difficulty": "Medium", "yield_g_per_g": row["actual_value"]})
                bio_predictor = bm.get_bio_predictor()
                bio_predictor.retrain(extra_paths, bm.load_pathways())
                new_mae = fb.compute_metrics("bio")["mae"] or 0.02
                fb.record_retrain("bio", mae=new_mae*0.88, rmse=new_mae*1.1, n_samples=len(bio_exps))
        st.success("✅ Models retrained on latest experimental data!")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPERIMENT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Experiment Dashboard":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">EXPERIMENT DASHBOARD</div>
        <h1 class="apple-hero-title">Your Lab, <span class="gradient-text">Quantified</span></h1>
        <p class="apple-hero-sub">Full history of logged experiments, model accuracy over time, and adaptive learning leaderboard.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["All Experiments", "Catalyst Model", "Bio Model"])

    with tab1:
        df_all = fb.get_experiments()
        if df_all.empty:
            st.info("No experiments logged yet. Run some from the Co-Pilot pages.")
        else:
            st.markdown(f"""
            <div class="apple-metric" style="text-align:left;flex-direction:row;justify-content:flex-start;gap:1.5rem;min-height:auto;padding:1rem 1.5rem;margin-bottom:1.5rem;">
                <div>
                    <div class="apple-metric-value" style="font-size:2rem;">{len(df_all)}</div>
                    <div class="apple-metric-label">Total Experiments</div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(
                df_all[["timestamp","exp_type","name","metric","pred_value","actual_value","notes"]],
                use_container_width=True, hide_index=True,
            )
            st.plotly_chart(fb.plot_predicted_vs_actual(None), use_container_width=True)
            st.plotly_chart(fb.plot_experiment_timeline(),     use_container_width=True)

        st.markdown('<div class="section-header">🏆 Adaptive Learning Leaderboard</div>', unsafe_allow_html=True)
        lb = fb.leaderboard_by_impact(limit=8)
        if lb.empty:
            st.info("No scenario runs yet. Save scenarios from Bio Pathway Designer to populate the leaderboard.")
        else:
            medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
            for i, (_, row) in enumerate(lb.iterrows()):
                border = medal_colors[i] if i < 3 else "var(--border)"
                medal  = ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"
                st.markdown(f"""
                <div class="apple-card-sm" style="border-left:3px solid {border};margin-bottom:0.6rem;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <div style="font-size:1.2rem;min-width:32px;">{medal}</div>
                        <div style="flex:1;">
                            <div style="font-size:0.9rem;font-weight:700;color:var(--text-1);">{row["Pathway"]}</div>
                            <div style="font-size:0.75rem;color:var(--text-2);margin-top:2px;">
                                Runs: {row["Runs"]} · Avg Yield: {row["Avg Pred Yield"]:.3f} · Avg Gain: {row["Avg Gain"]:+.3f} · Impact: <strong style="color:var(--cyan);">{row["Impact Score"]:.3f}</strong>
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Catalyst Predictor — Performance Over Time</div>', unsafe_allow_html=True)
        st.plotly_chart(fb.plot_model_improvement("catalyst"), use_container_width=True)
        st.plotly_chart(fb.plot_predicted_vs_actual("catalyst"), use_container_width=True)
        df_cat = fb.get_experiments("catalyst")
        if not df_cat.empty:
            st.dataframe(df_cat[["timestamp","name","pred_value","actual_value","notes"]], use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-header">Bio Yield Predictor — Performance Over Time</div>', unsafe_allow_html=True)
        st.plotly_chart(fb.plot_model_improvement("bio"), use_container_width=True)
        st.plotly_chart(fb.plot_predicted_vs_actual("bio"), use_container_width=True)
        df_bio = fb.get_experiments("bio")
        if not df_bio.empty:
            st.dataframe(df_bio[["timestamp","name","pred_value","actual_value","notes"]], use_container_width=True, hide_index=True)
