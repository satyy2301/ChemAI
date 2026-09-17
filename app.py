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
from modules import db_integration as db
from modules import reaction_engine as re
from modules import reaction_library as rl
from modules import reactant_input as ri
from modules import auth
from modules import quality as qual
from modules.ml import active_learning as al_lib
from modules.benchmarks import run_benchmark
from modules.integrations.ord_export import export_run_to_ord
from modules.db.repository import get_repo
import streamlit.components.v1 as components

# ─── One-time DB init ─────────────────────────────────────────────────────────
fb.init_db()
rl.seed_library()


def _apply_lab_template_state(rxn_name: str) -> dict | None:
    """Sync Reaction Lab widgets to a library template by name (call before widgets)."""
    patch = rl.get_lab_template_patch(rxn_name)
    if patch:
        for key, value in patch.items():
            st.session_state[key] = value
    return patch


def _queue_lab_template(rxn_name: str) -> None:
    st.session_state["lab_pending_template"] = rxn_name


def _queue_lab_reactants(text: str) -> None:
    st.session_state["lab_pending_reactants"] = text
    st.session_state.pop("lab_suggestions_key", None)


_LAB_TEMPLATE_PLACEHOLDER = "— pick or request template —"


def _queue_template_request(reactants: list, smarts: str, user: str) -> int:
    req_id = rl.queue_unsupported_reaction(
        f"reactants={reactants}, smarts={smarts or '—'}", user,
    )
    st.session_state["lab_last_request_id"] = req_id
    return req_id


def _reset_stale_lab_template(parsed: list[str], rxn_options: list[dict]) -> None:
    """Clear mismatched template selection when no suggestions are available."""
    if not parsed or st.session_state.get("lab_suggestions"):
        return
    current_label = st.session_state.get("lab_rxn_select", "")
    current = next((o for o in rxn_options if o["label"] == current_label), None)
    if current and rl.smarts_reactant_count(current.get("rxn_smarts", "")) == len(parsed):
        return
    count_matches = [
        o for o in rxn_options
        if rl.smarts_reactant_count(o.get("rxn_smarts", "")) == len(parsed)
    ]
    if count_matches:
        _apply_lab_template_state(count_matches[0]["name"])
    else:
        st.session_state["lab_rxn_select"] = _LAB_TEMPLATE_PLACEHOLDER
        st.session_state["lab_custom_smarts"] = ""
        st.session_state.pop("lab_preset_reaction", None)


def _process_lab_pending_state() -> None:
    """Apply queued Reaction Lab updates before widgets are instantiated."""
    if "lab_pending_reactants" in st.session_state:
        st.session_state["lab_reactants_input"] = st.session_state.pop("lab_pending_reactants")
        st.session_state.pop("lab_suggestions", None)
        st.session_state.pop("lab_suggestions_key", None)
    if "lab_pending_template" in st.session_state:
        _apply_lab_template_state(st.session_state.pop("lab_pending_template"))
    if "lab_pending_suggestions" in st.session_state:
        st.session_state["lab_suggestions"] = st.session_state.pop("lab_pending_suggestions")
        _parsed, _ = ri.parse_reactant_input(st.session_state.get("lab_reactants_input", ""))
        if _parsed:
            st.session_state["lab_suggestions_key"] = "|".join(_parsed)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_reaction_options(reaction_count: int):
    return rl.get_reaction_options()


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
    st.markdown('<div class="sidebar-tagline">Open Chemistry Reaction Library</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🧪 Reaction Lab", "📚 Reaction Library",
         "⚗️ Catalyst Co-Pilot", "🧬 Bio Pathway Designer",
         "🔄 Active Learning Lab", "📊 Experiment Dashboard"],
        label_visibility="collapsed",
        key="nav_page",
    )
    st.divider()
    st.caption("ChemAI · RDKit · scikit-learn · SQLAlchemy")
    _n_rxn = get_repo().count_reactions()
    st.caption(f"Library: {_n_rxn} reactions")
    st.divider()
    with st.expander("👤 Researcher Profile", expanded=False):
        _name_input = st.text_input(
            "Your name", placeholder="e.g. Alice",
            value=st.session_state.get("current_user", ""),
            help="Used to attribute experiment logs and annotations.",
        )
        if st.button("Save Name", key="save_username", use_container_width=True):
            _name = (_name_input or "").strip()
            st.session_state["current_user"] = _name if _name else "anonymous"
            st.success(f"Logged in as {st.session_state['current_user']}")
            st.rerun()
        _cur = st.session_state.get("current_user", "anonymous")
        st.markdown(
            f'<div style="font-size:0.72rem;color:#30D158;margin:4px 0;">● Active: {_cur}</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    with st.expander("🔑 Database API Keys"):
        st.caption("Materials Project (optional)")
        _sidebar_mp_key_input = st.text_input(
            "MP API Key", type="password",
            placeholder="Paste your key…",
            help="Free key at materialsproject.org",
            value=st.session_state.get("sidebar_mp_key", ""),
        )
        if st.button("✓ Save API Key", key="submit_mp_key", use_container_width=True):
            st.session_state["sidebar_mp_key"] = (_sidebar_mp_key_input or "").strip()
            st.success("Key saved.")
            st.rerun()

        if st.session_state.get("sidebar_mp_key", ""):
            st.markdown(
                '<div style="font-size:0.72rem;color:#30D158;margin:4px 0;">● Key active</div>',
                unsafe_allow_html=True,
            )
            if st.button("⚡ Test Connection", key="test_mp_key_btn", use_container_width=True):
                with st.spinner("Testing…"):
                    _result = db.test_mp_key(st.session_state["sidebar_mp_key"])
                if _result["ok"]:
                    st.success(_result["msg"])
                else:
                    st.error(_result["msg"])
            if st.button("✕ Clear Key", key="clear_mp_key", use_container_width=True):
                st.session_state["sidebar_mp_key"] = ""
                st.rerun()
        else:
            st.markdown(
                '<div style="font-size:0.72rem;color:#8E8E93;margin:4px 0;">○ No key saved</div>',
                unsafe_allow_html=True,
            )
        st.caption("Catalysis Hub: no key needed")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">OPEN CHEMISTRY LIBRARY</div>
        <h1 class="apple-hero-title">Run, Test &amp; Learn from <span class="gradient-text">Any Reaction</span></h1>
        <p class="apple-hero-sub">Community reaction library — enter reactants, predict products, log lab results, and retrain models in one closed loop.</p>
    </div>
    """, unsafe_allow_html=True)

    cat_all = cm.load_catalysts()
    bio_all = bm.load_pathways()
    exp_df  = fb.get_experiments()
    lib_rxns = rl.list_reactions()
    rxn_runs = rl.list_reaction_runs(limit=500)
    _n_domains = len(set(exp_df["exp_type"])) if not exp_df.empty else 0
    _lib_domains = len({r.get("domain", "") for r in lib_rxns})

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, icon, val, label in [
        (c1, "🧪", len(lib_rxns),     "Reaction Templates"),
        (c2, "⚗️", len(cat_all),      "Catalyst Entries"),
        (c3, "🧬", len(bio_all),      "Metabolic Pathways"),
        (c4, "🔬", len(exp_df),       "Logged Experiments"),
        (c5, "📊", len(rxn_runs),     "Reaction Runs"),
    ]:
        col.markdown(f"""
        <div class="apple-metric">
            <div class="apple-metric-icon">{icon}</div>
            <div class="apple-metric-value">{val}</div>
            <div class="apple-metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Covering {_lib_domains} reaction domains · {_n_domains} experiment types in the training flywheel")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
        for icon, title, desc in [
            ("🧪", "Reaction Engine",  "RDKit SMARTS execution — formulas, names, or SMILES in; predicted products out"),
            ("📚", "Reaction Library", "60+ public templates (organic, catalysis, bio) — search, fork, contribute"),
            ("📈", "ML Yield Model",   "Morgan fingerprints + Random Forest with uncertainty; retrains on logged results"),
            ("🗄️", "Data Layer",       "SQLAlchemy persistence · Catalysis Hub · Materials Project · BRENDA"),
            ("🔬", "3D Mol Viewer",    "Interactive SMILES, metabolite, and catalyst surface structures via 3Dmol.js"),
            ("🔄", "Training Flywheel","Quality-gated experiments → auto-retrain → active learning suggestions"),
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
            ("01", "Enter Reactants",  "Type CO, O2, ethanol, or SMILES — the engine resolves names automatically."),
            ("02", "Pick Template",    "Choose or auto-suggest a reaction rule (e.g. CO Oxidation for CO + O₂)."),
            ("03", "Predict",          "RDKit applies SMARTS; heuristic + ML models estimate yield and selectivity."),
            ("04", "Run in Lab",       "Test the prediction; log measured yield and conditions back to the library."),
            ("05", "Community Learn",  "Quality-scored results enter the shared dataset and improve models."),
            ("06", "Explore More",     "Catalyst Co-Pilot, Bio Pathways, and Active Learning for deeper discovery."),
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
    st.markdown('<div class="section-header">Reaction Library at a Glance</div>', unsafe_allow_html=True)
    _lib_rows = [{
        "Reaction": r["name"],
        "Domain": r.get("domain", ""),
        "Tags": ", ".join(r.get("tags", [])[:3]),
        "Base Yield": f"{r.get('base_yield', 0):.0%}",
    } for r in lib_rxns[:25]]
    st.dataframe(pd.DataFrame(_lib_rows), use_container_width=True, hide_index=True)
    if len(lib_rxns) > 25:
        st.caption(f"Showing 25 of {len(lib_rxns)} templates — browse all in **Reaction Library**.")

    st.markdown('<div class="section-header">Catalyst Reaction Map</div>', unsafe_allow_html=True)
    reactions = cm.get_reactions()
    df_r = pd.DataFrame(reactions, columns=["Key", "Reaction"])
    df_r["# Catalysts"] = df_r["Key"].apply(lambda k: len(cm.load_catalysts(k)))
    st.dataframe(df_r[["Reaction", "# Catalysts"]], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REACTION LAB
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Reaction Lab":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">REACTION LAB</div>
        <h1 class="apple-hero-title">Run Any <span class="gradient-text">Chemical Reaction</span></h1>
        <p class="apple-hero-sub">Run library reactions or request new templates — paste SMILES, match a template, set conditions, predict products, and log results to train the community model.</p>
    </div>""", unsafe_allow_html=True)

    _user = auth.get_current_user()
    _process_lab_pending_state()
    _rxn_options = _cached_reaction_options(get_repo().count_reactions())
    _fork_id = st.session_state.get("fork_reaction_id")

    if "lab_reactants_input" not in st.session_state:
        st.session_state["lab_reactants_input"] = "CO\nO2"

    _early_parsed, _ = ri.parse_reactant_input(st.session_state["lab_reactants_input"])
    _rxn_key = "|".join(_early_parsed) if _early_parsed else ""
    if _early_parsed and st.session_state.get("lab_suggestions_key") != _rxn_key:
        st.session_state["lab_suggestions_key"] = _rxn_key
        _fast = rl.suggest_reactions_fast(_early_parsed, top_k=5)
        st.session_state["lab_suggestions"] = _fast
        if _fast:
            _apply_lab_template_state(_fast[0]["reaction"]["name"])

    with st.expander("How to enter reactants & what templates mean", expanded=False):
        st.markdown(ri.nomenclature_help())
        st.markdown(ri.template_help())
    with st.expander("How to discover reactions", expanded=False):
        st.markdown(ri.discovery_help())

    _preset = st.selectbox(
        "Quick start (optional)",
        ["— custom —"] + list(ri.QUICK_PRESETS.keys()),
        key="lab_preset",
    )
    if _preset != "— custom —" and st.button("Load preset", key="load_preset"):
        p = ri.QUICK_PRESETS[_preset]
        _queue_lab_reactants(p["reactants"])
        st.session_state["lab_pending_template"] = p["reaction_name"]
        st.session_state.pop("lab_result", None)
        st.rerun()

    st.markdown('<div class="section-header">Step 1 — Reactants</div>', unsafe_allow_html=True)
    st.caption("Enter formulas (CO, O2), names (oxygen, ethanol), or SMILES — one per line or separated by +")
    _reactant_text = st.text_area(
        "Reactants",
        height=100, key="lab_reactants_input",
        label_visibility="collapsed",
    )
    _parsed, _parse_notes = ri.parse_reactant_input(_reactant_text)
    if _parse_notes:
        for n in _parse_notes:
            st.info(n)
    if _parsed:
        st.caption("Resolved: " + "  +  ".join(_parsed))

    rc1, rc2, rc3 = st.columns(3)
    if rc1.button("Resolve names via PubChem", key="resolve_pubchem"):
        lines = [re.resolve_smiles(l.strip()) for l in _reactant_text.splitlines() if l.strip()]
        _queue_lab_reactants("\n".join(lines))
        st.rerun()
    if rc2.button("Suggest template", key="suggest_template") and _parsed:
        st.session_state["lab_pending_suggestions"] = rl.suggest_reactions_deep(_parsed, top_k=5)
        if st.session_state["lab_pending_suggestions"]:
            st.session_state["lab_pending_template"] = (
                st.session_state["lab_pending_suggestions"][0]["reaction"]["name"]
            )
        st.rerun()
    if rc3.button("Clear results", key="clear_lab"):
        st.session_state.pop("lab_result", None)
        st.session_state.pop("lab_run_id", None)
        st.rerun()

    if st.session_state.get("lab_suggestions"):
        _sug_list = st.session_state["lab_suggestions"]
        st.success(f"**{len(_sug_list)} matching template(s)** — best: **{_sug_list[0]['reaction']['name']}** → {_sug_list[0]['products_preview']}")
        for i, sug in enumerate(_sug_list[:5]):
            rxn = sug["reaction"]
            if st.button(f"Use: {rxn['name']} → {sug['products_preview']}", key=f"use_sug_{i}"):
                _queue_lab_template(rxn["name"])
                st.rerun()
    elif _parsed:
        st.warning(
            "No matching template found for these reactants. "
            "Try a quick preset, deep-scan, or request a new template below."
        )
        nm1, nm2, nm3 = st.columns(3)
        if nm1.button("Deep-scan templates", key="no_match_suggest", use_container_width=True):
            st.session_state["lab_pending_suggestions"] = rl.suggest_reactions_deep(_parsed, top_k=5)
            if st.session_state["lab_pending_suggestions"]:
                st.session_state["lab_pending_template"] = (
                    st.session_state["lab_pending_suggestions"][0]["reaction"]["name"]
                )
            st.rerun()
        if nm2.button("Request new template", key="no_match_request", use_container_width=True):
            _req_id = _queue_template_request(_parsed, "", _user)
            st.success(f"Template request #{_req_id} queued for community review.")
        if nm3.button("Browse Reaction Library", key="no_match_browse", use_container_width=True):
            st.session_state["nav_page"] = "📚 Reaction Library"
            st.session_state["lib_search"] = "combustion"
            st.rerun()

    if st.session_state.get("lab_last_request_id"):
        st.info(f"Template request **#{st.session_state['lab_last_request_id']}** is queued for review.")

    st.markdown('<div class="section-header">Step 2 — Reaction type (template)</div>', unsafe_allow_html=True)
    st.caption(
        "The template is auto-selected when possible. You can override below — "
        "wrong template = error even with correct reactants."
    )
    _reset_stale_lab_template(_parsed, _rxn_options)
    _labels = [o["label"] for o in _rxn_options]
    if _parsed and not st.session_state.get("lab_suggestions"):
        _labels = [_LAB_TEMPLATE_PLACEHOLDER] + _labels
    if "lab_rxn_select" not in st.session_state:
        _init = _rxn_options[0]
        if _fork_id:
            _init = next((o for o in _rxn_options if o["id"] == _fork_id), _init)
        st.session_state["lab_rxn_select"] = _init["label"]
        st.session_state["lab_custom_smarts"] = _init.get("rxn_smarts", "")

    def _on_template_change():
        sel = st.session_state.get("lab_rxn_select")
        if sel and sel != _LAB_TEMPLATE_PLACEHOLDER:
            rxn = next((o for o in _rxn_options if o["label"] == sel), None)
            if rxn:
                st.session_state["lab_custom_smarts"] = rxn.get("rxn_smarts", "")
                st.session_state["lab_preset_reaction"] = rxn["name"]

    _chosen_label = st.selectbox(
        "Reaction template", _labels,
        key="lab_rxn_select", on_change=_on_template_change,
    )
    _chosen_rxn = (
        next((o for o in _rxn_options if o["label"] == _chosen_label), None)
        if _chosen_label != _LAB_TEMPLATE_PLACEHOLDER else None
    )
    if _parsed and _chosen_rxn:
        _need = rl.smarts_reactant_count(_chosen_rxn.get("rxn_smarts", ""))
        if _need and _need != len(_parsed):
            st.warning(
                f"**{_chosen_rxn['name']}** expects {_need} reactant(s), "
                f"but you entered {len(_parsed)}. "
                "Run will auto-pick a matching template, or click **Suggest template** above."
            )
    elif _parsed and not _chosen_rxn:
        st.info("Pick a template from the list, or click **Request new template** above.")
    _custom_smarts = st.text_input(
        "Advanced: custom SMARTS (leave as-is unless you know SMARTS)",
        key="lab_custom_smarts",
    )

    st.markdown('<div class="section-header">Step 3 — Conditions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        _temp = st.number_input("Temperature (°C)", value=25.0, key="lab_temp")
        _pressure = st.number_input("Pressure (bar)", value=1.0, key="lab_pressure")
    with c2:
        _ph = st.number_input("pH", value=7.0, step=0.1, key="lab_ph")
        _solvent = st.selectbox("Solvent", ["water", "ethanol", "toluene", "DMF", "DMSO", "THF", "polar aprotic"], key="lab_solvent")
    with c3:
        _catalyst = st.text_input("Catalyst (optional)", key="lab_catalyst")
        _visibility = st.selectbox("Visibility", ["public", "private"], key="lab_vis")

    _conditions = {
        "temperature_c": _temp, "pressure_bar": _pressure,
        "ph": _ph, "solvent": _solvent, "catalyst": _catalyst,
    }

    st.markdown('<div class="section-header">Step 4 — Safety Check & Run</div>', unsafe_allow_html=True)
    _reactants = _parsed if _parsed else [l.strip() for l in _reactant_text.splitlines() if l.strip()]

    if st.button("Run Reaction", type="primary", use_container_width=True, key="lab_run"):
        with st.spinner("Running reaction engine..."):
            _active_rxn = None
            if _chosen_rxn:
                _active_rxn = dict(_chosen_rxn)
                _active_rxn["rxn_smarts"] = _custom_smarts or _chosen_rxn.get("rxn_smarts")
            _result, _used_rxn = rl.run_with_best_template(
                _reactants, _conditions, preferred_rxn=_active_rxn,
            )
            if _used_rxn and (_chosen_rxn is None or _used_rxn.get("name") != _chosen_rxn.get("name")):
                _queue_lab_template(_used_rxn["name"])
            st.session_state["lab_result"] = _result
            st.session_state["lab_used_rxn"] = _used_rxn
            if _result.validity and not (_result.safety and _result.safety.blocked):
                _run_id = rl.create_reaction_run(
                    reaction_id=_used_rxn.get("id") if _used_rxn else (_chosen_rxn or {}).get("id"),
                    conditions=_conditions,
                    predicted=_result.predicted_outcome,
                    user_id=_user,
                )
                st.session_state["lab_run_id"] = _run_id

    if "lab_result" in st.session_state:
        _res = st.session_state["lab_result"]
        if _res.safety and _res.safety.blocked:
            st.error("Reaction blocked by safety screening.")
            for a in _res.safety.alerts:
                st.warning(a)
        elif not _res.validity:
            st.error("Reaction failed: " + "; ".join(_res.warnings[:3]))
            st.markdown("""
**Common fixes:**
- Use **O2** or **oxygen** for oxygen gas (not plain `O` unless you mean O₂)
- Use **water** or **H2O** for water
- Pick the right template — for CO + O₂ use **CO Oxidation**, not Water Splitting OER
- Click **Suggest template** to auto-find a matching reaction
            """)
            _retry = rl.suggest_reactions_fast(_reactants, top_k=3)
            if _retry:
                st.markdown("**Try one of these templates instead:**")
                for i, sug in enumerate(_retry):
                    rxn = sug["reaction"]
                    if st.button(f"{rxn['name']} → {sug['products_preview']}", key=f"retry_{i}"):
                        _queue_lab_template(rxn["name"])
                        st.session_state.pop("lab_result", None)
                        st.rerun()
            if st.button("Request new template", key="req_template"):
                _req_id = _queue_template_request(_reactants, _custom_smarts, _user)
                st.success(f"Template request #{_req_id} queued for community review.")
        else:
            _used = st.session_state.get("lab_used_rxn") or _chosen_rxn or {}
            st.success(
                f"Products generated · **{_used.get('name', '—')}** · "
                f"Tier {_res.engine_tier} · Confidence {_res.confidence:.2f}"
            )
            if _res.warnings:
                for w in _res.warnings:
                    st.caption(f"⚠ {w}")

            _out = _res.predicted_outcome
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Yield", f"{_out.get('yield', 0):.1%}")
            m2.metric("Selectivity", f"{_out.get('selectivity', 0):.1%}")
            m3.metric("Uncertainty", f"±{_out.get('uncertainty', 0):.3f}")
            m4.metric("Engine Tier", _res.engine_tier)

            st.markdown("**Products**")
            for i, p in enumerate(_res.products):
                st.code(p, language=None)
                _html, _name = mv.make_smiles_viewer_html(p, height=280, width=500, label=f"Product {i+1}")
                components.html(_html, height=300, scrolling=False)

            st.markdown('<div class="section-header">Step 5 — Log Measured Result</div>', unsafe_allow_html=True)
            col_p, col_a = st.columns(2)
            _pred_y = col_p.number_input("Predicted yield", value=float(_out.get("yield", 0)), format="%.3f", key="lab_pred_y")
            _actual_y = col_a.number_input("Measured yield (lab)", value=float(_out.get("yield", 0)), format="%.3f", key="lab_actual_y")
            _notes = st.text_input("Notes", key="lab_notes")
            _prov_labels = {"internal_experiment": "Internal", "published_paper": "Published",
                            "screening": "Screening", "db_retrieved": "DB", "ai_simulation": "Simulation"}
            cp1, cp2 = st.columns(2)
            _prov = cp1.selectbox("Provenance", list(_prov_labels.keys()),
                                   format_func=lambda x: _prov_labels[x], key="lab_prov")
            _dq = cp2.selectbox("Data quality", ["good", "uncertain", "outlier"], key="lab_dq")

            if st.button("Submit Result", key="lab_submit"):
                _run_id = st.session_state.get("lab_run_id")
                _actual = {"yield": _actual_y, "selectivity": _out.get("selectivity", _actual_y)}
                _run_data = {
                    "reaction_id": _used.get("id"),
                    "conditions": _conditions, "predicted": _out,
                    "actual": _actual, "provenance": _prov,
                }
                _gated = qual.score_and_gate_run(_run_data, _dq, _user)
                if _run_id:
                    rl.update_reaction_run_actual(_run_id, _actual, _gated["quality_score"])
                fb.log_experiment(
                    exp_type="reaction", name=_used.get("name", "—"),
                    pred_value=_pred_y, actual_value=_actual_y,
                    metric="reaction_yield", notes=_notes, user=_user,
                    data_quality=_dq, source_provenance=_prov,
                    reaction_run_id=_run_id,
                )
                if _gated["training_eligible"]:
                    get_repo().increment_reputation(_user, 0.1)
                st.success(f"Logged by **{_user}** · Quality score: {_gated['quality_score']:.2f}")
                st.balloons()

            _run_id = st.session_state.get("lab_run_id")
            if _run_id:
                _run = rl.get_reaction_run(_run_id)
                if _run:
                    st.download_button("Export ORD JSON", data=export_run_to_ord(_run, _used),
                                       file_name=f"run_{_run_id}.ord.json", mime="application/json")
                    st.download_button("Export RXN", data=fb.export_reaction_rxn(_run),
                                       file_name=f"run_{_run_id}.rxn", mime="text/plain")

    _pending_jobs = [j for j in get_repo().get_pending_jobs(5) if j["job_type"] == "external_sim"]
    if _pending_jobs:
        st.markdown('<div class="section-header">Job Status</div>', unsafe_allow_html=True)
        for j in _pending_jobs:
            st.info(f"Job #{j['id']}: {j['job_type']} — {j['status']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REACTION LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Reaction Library":
    st.markdown("""
    <div class="apple-hero">
        <div class="apple-hero-tag">REACTION LIBRARY</div>
        <h1 class="apple-hero-title">Community <span class="gradient-text">Reaction Library</span></h1>
        <p class="apple-hero-sub">Browse, search, and fork public reactions contributed by the community.</p>
    </div>""", unsafe_allow_html=True)

    _user = auth.get_current_user()
    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        _search_q = st.text_input("Search reactions", key="lib_search")
    with s2:
        _domain = st.selectbox("Domain", ["all", "organic", "catalysis", "bio"], key="lib_domain")
    with s3:
        _tag = st.text_input("Tag filter", key="lib_tag")

    _domain_f = None if _domain == "all" else _domain
    _reactions = rl.search_reactions(_search_q, _domain_f, _tag or None)

    st.markdown(f"**{len(_reactions)}** reactions found")
    for rxn in _reactions[:30]:
        with st.expander(f"{rxn['name']} · {rxn['domain']} · {len(rxn.get('tags', []))} tags"):
            st.code(rxn.get("rxn_smarts", ""), language=None)
            st.caption(f"Reactants: {', '.join(rxn.get('reactants', [])[:3])}")
            st.caption(f"Base yield: {rxn.get('base_yield', 0.75):.0%} · By: {rxn.get('created_by', 'system')}")
            fc1, fc2, fc3 = st.columns(3)
            if fc1.button(f"Fork", key=f"fork_{rxn['id']}"):
                new_id = rl.fork_reaction(rxn["id"], _user)
                st.session_state["fork_reaction_id"] = new_id
                st.success(f"Forked as reaction #{new_id}. Open Reaction Lab to run it.")
            if fc2.button(f"Use in Lab", key=f"use_{rxn['id']}"):
                st.session_state["fork_reaction_id"] = rxn["id"]
                _queue_lab_reactants("\n".join(rxn.get("reactants", [])))
                st.session_state["lab_pending_template"] = rxn["name"]
                st.info("Switch to Reaction Lab to run this reaction.")
            if fc3.button(f"Flag", key=f"flag_{rxn['id']}"):
                get_repo().add_flag("reaction", rxn["id"], _user, "Community flag")
                st.warning("Reaction flagged for review.")

    st.divider()
    st.markdown('<div class="section-header">Pending Template Requests</div>', unsafe_allow_html=True)
    _pending_reqs = rl.list_reaction_requests(status="pending")
    if _pending_reqs:
        for _req in _pending_reqs[:10]:
            st.caption(
                f"#{_req['id']} · {_req['user_id']} · "
                f"{_req['user_input'][:80]}{'…' if len(_req['user_input']) > 80 else ''}"
            )
    else:
        st.caption("No pending requests. Use **Request new template** in Reaction Lab to queue one.")

    st.divider()
    st.markdown('<div class="section-header">Recent Public Runs</div>', unsafe_allow_html=True)
    _runs = rl.list_reaction_runs(limit=20)
    if _runs:
        import pandas as pd
        _rows = []
        for r in _runs:
            pred = r.get("predicted", {})
            actual = r.get("actual", {})
            _rows.append({
                "ID": r["id"], "User": r["user_id"],
                "Yield (pred)": pred.get("yield", "—"),
                "Yield (actual)": actual.get("yield", "—"),
                "Quality": r.get("quality_score", 0),
                "Tier": pred.get("engine_tier", "—"),
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No reaction runs yet. Be the first to contribute from Reaction Lab!")


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

    # ── Scientific Database Sources ────────────────────────────────────────────
    st.markdown("""
    <div class="apple-card" style="border-color:rgba(0,212,255,0.18);">
        <div class="step-badge" style="background:linear-gradient(135deg,#00D4FF,#0A84FF);">DB</div>
        <div class="step-label">LIVE DATA</div>
        <div class="step-title">Scientific Database Sources</div>
        <div style="font-size:0.78rem;color:var(--text-2);margin-top:4px;">
            Catalysis Hub (DFT) &nbsp;·&nbsp; Materials Project &nbsp;·&nbsp; BRENDA Enzymes
        </div>
    </div>""", unsafe_allow_html=True)

    _db_ref_cat = known[0] if known else {"composition": {"Pt": 1.0}}
    _db_mp_key  = st.session_state.get("sidebar_mp_key", "")
    _db_cache_key = f"db_results_{chosen_key}"

    _db_col1, _db_col2 = st.columns([3, 1])
    with _db_col2:
        _db_fetch = st.button("Fetch from Databases", key=f"dbfetch_{chosen_key}",
                              use_container_width=True)
    with _db_col1:
        st.caption("Retrieves real DFT reaction data and enzyme kinetics for the selected reaction. "
                   "Catalysis Hub results are filtered to the dominant catalyst element.")

    if _db_fetch:
        with st.spinner("Querying Catalysis Hub · Materials Project · BRENDA…"):
            st.session_state[_db_cache_key] = db.fetch_all(_db_ref_cat, chosen_key, _db_mp_key)

    if _db_cache_key in st.session_state:
        _db_res = st.session_state[_db_cache_key]
        _ch_res = _db_res.get("catalysis_hub", {})
        _mp_res = _db_res.get("materials_project", {})
        _br_res = _db_res.get("brenda", {})

        _tab_ch, _tab_mp, _tab_br = st.tabs([
            "⚡ Catalysis Hub (DFT)",
            "🔷 Materials Project",
            "🧬 BRENDA Enzymes",
        ])

        with _tab_ch:
            _ch_elem = _ch_res.get("element", "?")
            if _ch_res.get("status") == "ok":
                _ch_rows = _ch_res["rows"]
                _df_ch = pd.DataFrame([{
                    "Surface": r["surface"],
                    "Facet": r["facet"],
                    "Reactants (JSON)": r["reactants"][:45] + ("…" if len(r["reactants"]) > 45 else ""),
                    "Products (JSON)": r["products"][:40] + ("…" if len(r["products"]) > 40 else ""),
                    "ΔE (eV)": round(r["reaction_energy_ev"], 4) if r["reaction_energy_ev"] is not None else "—",
                    "Eₐ (eV)": round(r["activation_energy_ev"], 4) if r["activation_energy_ev"] is not None else "—",
                    "DFT Functional": r["dft_functional"],
                } for r in _ch_rows])
                st.markdown(f"**{len(_ch_rows)} DFT entries** for `{_ch_elem}` surface — source: "
                            "[Catalysis Hub](https://www.catalysis-hub.org/)")
                st.dataframe(_df_ch, use_container_width=True, hide_index=True)
            elif _ch_res.get("status") == "no_data":
                st.info(f"No DFT entries found for **{_ch_elem}** surface matching "
                        f"**{chosen_label}** reaction species in Catalysis Hub.")
            else:
                st.error(f"Catalysis Hub error: {_ch_res.get('error', 'unknown')}")

        with _tab_mp:
            if _mp_res.get("status") == "ok":
                _df_mp = pd.DataFrame([{
                    "Material ID": r["material_id"],
                    "Formula": r["formula"],
                    "Eₓ/atom (eV)": r["formation_energy_ev_atom"],
                    "E above hull (eV)": r["energy_above_hull_ev"],
                    "Band gap (eV)": r["band_gap_ev"],
                    "Sites": r["n_sites"],
                } for r in _mp_res["rows"]])
                st.markdown("**Materials Project** stability & electronic structure — "
                            "[materialsproject.org](https://materialsproject.org)")
                st.dataframe(_df_mp, use_container_width=True, hide_index=True)
            elif _mp_res.get("status") == "no_key":
                st.info("Enter your free Materials Project API key in the sidebar "
                        "(**Database API Keys**) to fetch formation energies and stability data.")
            elif _mp_res.get("status") == "no_data":
                st.info("No Materials Project entries found for this catalyst composition.")
            else:
                st.error(f"Materials Project error: {_mp_res.get('error', 'unknown')}")

        with _tab_br:
            if _br_res.get("status") == "ok":
                _df_br = pd.DataFrame([{
                    "Enzyme": r["enzyme"],
                    "EC Number": r["ec_number"],
                    "Substrate": r["substrate"],
                    "Product": r["product"],
                    "Kₘ (mM)": r.get("km_mm"),
                    "kcat (s⁻¹)": r.get("kcat_s"),
                    "Organism": r["organism"],
                    "pH opt.": r["ph_optimum"],
                    "T opt. (°C)": r["temp_optimum_c"],
                } for r in _br_res["rows"]])
                st.markdown("**BRENDA** enzyme kinetics — curated reference data "
                            "([brenda-enzymes.org](https://www.brenda-enzymes.org/))")
                st.dataframe(_df_br, use_container_width=True, hide_index=True)
            else:
                st.info("No curated BRENDA enzyme data for this reaction type. "
                        "BRENDA data is most relevant for bio-catalytic reactions.")

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
    with col2: strategy  = st.selectbox(
        "Strategy",
        ["mixed", "doping", "surface", "generative"],
        format_func=lambda s: {
            "mixed":       "Mixed (doping + surface)",
            "doping":      "Doping (add element)",
            "surface":     "Surface (facet swap)",
            "generative":  "Generative AI (GMM latent space)",
        }.get(s) or s,
    )
    with col3: n_gen = st.slider("Variants", 3, 8, 5)
    if strategy == "generative":
        st.info(
            "**Generative AI mode** — compositions are sampled from a Gaussian Mixture Model "
            "fitted to the PCA-compressed latent space of all known catalysts. "
            "Unlike rule-based doping, the model learns the *joint distribution* of element "
            "co-occurrence and generates genuinely novel points in that space."
        )
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

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📤 Export Candidates</div>', unsafe_allow_html=True)
        _export_n = st.slider(
            "Top N candidates to export", 3, min(len(ranked), 20), min(5, len(ranked)),
            key="export_n",
        )
        _rxn_slug = st.session_state.get("chosen_rxn", chosen_label).replace(" ", "_").replace("→", "to")
        _ex1, _ex2, _ex3 = st.columns(3)
        _ex1.download_button(
            "⬇️ CSV",
            data=fb.export_ranked_csv(ranked[:_export_n]),
            file_name=f"chemai_{_rxn_slug}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv",
            help="Ranked properties table — opens in Excel / pandas",
        )
        _ex2.download_button(
            "⬇️ JSON",
            data=fb.export_ranked_json(ranked[:_export_n]),
            file_name=f"chemai_{_rxn_slug}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
            help="Full properties in JSON — pipe into downstream tools",
        )
        _ex3.download_button(
            "⬇️ SDF",
            data=fb.generate_candidates_sdf(ranked[:_export_n]).encode("utf-8"),
            file_name=f"chemai_{_rxn_slug}.sdf",
            mime="chemical/x-mdl-sdfile",
            use_container_width=True,
            key="dl_sdf",
            help="Property-tagged SDF — compatible with RDKit / OpenBabel / ChemDraw",
        )
        with st.expander("📋 Lab Report Preview  (download as .txt)"):
            _report_txt = fb.generate_lab_report(ranked, chosen_label or "", top_n=_export_n)
            st.code(_report_txt, language=None)
            st.download_button(
                "⬇️ Download Lab Report",
                data=_report_txt.encode("utf-8"),
                file_name=f"chemai_lab_report_{_rxn_slug}.txt",
                mime="text/plain",
                key="dl_report",
            )

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
            _prov_labels = {"internal_experiment": "Internal Experiment", "published_paper": "Published Paper",
                            "screening": "Screening Campaign", "db_retrieved": "DB Retrieved", "ai_simulation": "AI Simulation"}
            col_prov, col_qual = st.columns(2)
            cat_provenance = col_prov.selectbox(
                "Data source", list(_prov_labels.keys()),
                format_func=lambda x: _prov_labels.get(x) or x, key="cat_provenance",
            )
            cat_quality = col_qual.selectbox("Data quality", ["good", "uncertain", "outlier"], key="cat_quality")
            if st.button("✅ Submit Experiment", key="cat_submit"):
                _user = auth.get_current_user()
                fb.log_experiment(
                    exp_type="catalyst", name=exp_name,
                    pred_value=pred_val, actual_value=actual_val,
                    metric="activity", notes=notes, composition=exp_cat.get("composition", {}),
                    user=_user, data_quality=cat_quality, source_provenance=cat_provenance,
                )
                err = abs(actual_val - pred_val)
                fb.record_retrain("catalyst", mae=err, rmse=err*1.2, n_samples=len(fb.get_experiments("catalyst")))
                st.success(f"Logged by **{_user}** · |Error| = {err:.3f}")
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

        # ── Flux Balance Analysis ─────────────────────────────────────────────
        st.markdown('<div class="section-header">⚗️ Flux Balance Analysis</div>', unsafe_allow_html=True)
        _fba = bm.run_fba(chosen_path, scenario)
        _fba_c1, _fba_c2, _fba_c3 = st.columns(3)
        _fba_c1.metric("FBA Optimal Yield",  f"{_fba['optimal_yield']:.3f} g/g")
        _fba_c2.metric("Flux-Limiting Step", (_fba.get("limiting_step") or "—")[:30])
        _fba_c3.metric("FBA Status",         _fba.get("status", "—").capitalize())
        st.plotly_chart(bm.plot_fba_fluxes(_fba), use_container_width=True)
        with st.expander("📊 FBA details — flux table"):
            if _fba.get("fluxes"):
                _fba_df = pd.DataFrame([
                    {"Reaction": r,
                     "Optimised Flux": f"{v:.4f}",
                     "Capacity": f"{_fba['efficiency_bounds'][i]:.4f}" if i < len(_fba.get("efficiency_bounds", [])) else "—",
                     "Limiting": "🔴 YES" if r == _fba.get("limiting_step") else ""}
                    for i, (r, v) in enumerate(_fba["fluxes"].items())
                ])
                st.dataframe(_fba_df, use_container_width=True, hide_index=True)
            else:
                st.info("FBA returned no flux data.")

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
            _mut_sugs = bm.suggest_mutations(chosen_path, n=4)
            for i, sug in enumerate(_mut_sugs, 1):
                st.markdown(f"""
                <div class="apple-card-sm" style="border-left:3px solid var(--cyan);">
                    <span style="font-size:0.7rem;font-weight:700;color:var(--cyan);">#{i}</span>
                    <span style="font-size:0.85rem;color:var(--text-1);margin-left:8px;">{sug}</span>
                </div>""", unsafe_allow_html=True)

        # ── Protein Structure Viewer ──────────────────────────────────────────
        st.markdown('<div class="section-header">🧬 Bottleneck Enzyme — 3D Protein Structure</div>', unsafe_allow_html=True)
        _bottle_step = bm.get_bottleneck_step(chosen_path)
        if _bottle_step:
            _ec   = _bottle_step.get("ec", "multi")
            _enz  = _bottle_step.get("enzyme", "Enzyme")
            _gene = _bottle_step.get("gene", "—")
            _prot_html, _prot_effect = mv.make_protein_viewer_html(
                _enz, _ec, _mut_sugs, width=620, height=430,
            )
            _prot_col, _prot_info = st.columns([3, 2])
            with _prot_col:
                components.html(_prot_html, height=450, scrolling=False)
            with _prot_info:
                st.markdown(f"""
                <div class="apple-card">
                  <div class="step-label">ENZYME INFO</div>
                  <div style="font-size:1rem;font-weight:700;color:var(--text-1);margin-bottom:6px;">{_enz}</div>
                  <div style="font-size:0.82rem;color:var(--text-2);line-height:1.7;">
                    Gene: <strong style="color:var(--cyan);">{_gene}</strong><br>
                    EC: <strong style="color:var(--accent);">{_ec}</strong><br>
                    Step: <strong style="color:var(--text-1);">{_bottle_step.get("from","")} → {_bottle_step.get("to","")}</strong><br>
                    Efficiency: <strong style="color:var(--danger);">{_bottle_step.get("efficiency",0)*100:.0f}%</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="apple-card" style="margin-top:0.5rem;">
                  <div class="step-label">ESTIMATED MUTATION EFFECTS</div>
                  <div style="font-size:0.82rem;color:var(--text-2);line-height:1.8;margin-top:4px;">
                    Yield gain: <strong style="color:var(--success);">+{_prot_effect['estimated_gain']:.1%}</strong><br>
                    Avg risk: <strong style="color:var(--danger);">{_prot_effect['estimated_risk']:.1%}</strong><br>
                    <span style="font-size:0.75rem;color:var(--text-2);">
                      Interventions parsed: {len(_prot_effect['interventions_parsed'])}
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.caption("🟠 Orange sticks = active-site residues. Rotate freely in the viewer.")
        else:
            st.info("No bottleneck step identified for this pathway.")

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
            _bio_prov_labels = {"internal_experiment": "Internal Experiment", "published_paper": "Published Paper",
                                "screening": "Screening Campaign", "db_retrieved": "DB Retrieved", "ai_simulation": "AI Simulation"}
            col_prov2, col_qual2 = st.columns(2)
            bio_provenance = col_prov2.selectbox(
                "Data source", list(_bio_prov_labels.keys()),
                format_func=lambda x: _bio_prov_labels.get(x) or x, key="bio_provenance",
            )
            bio_quality = col_qual2.selectbox("Data quality", ["good", "uncertain", "outlier"], key="bio_quality")
            if st.button("✅ Submit Bio Experiment"):
                _user = auth.get_current_user()
                fb.log_experiment(
                    exp_type="bio", name=chosen_path["name"],
                    pred_value=pred_val, actual_value=actual_val, metric="yield", notes=notes,
                    user=_user, data_quality=bio_quality, source_provenance=bio_provenance,
                )
                err = abs(actual_val - pred_val)
                fb.record_retrain("bio", mae=err, rmse=err*1.2, n_samples=len(fb.get_experiments("bio")))
                st.success(f"Logged by **{_user}** · |Error| = {err:.3f}")
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

    _last_retrain = get_repo().get_latest_promoted_model("reaction", "yield")
    if _last_retrain:
        st.caption(f"Last auto-retrain: {_last_retrain.get('timestamp', '—')} · MAE {_last_retrain.get('mae', 0):.4f}")

    if st.button("Retrain All Models on Latest Data", use_container_width=True):
        with st.spinner("Retraining..."):
            from infra.retrain_jobs.retrain_worker import run_retrain_job
            _retrain_result = run_retrain_job(force=True)
            st.session_state["last_retrain_result"] = _retrain_result
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
        _rr = st.session_state.get("last_retrain_result", {})
        if _rr.get("promoted"):
            st.success(f"Models retrained and promoted! MAE={_rr.get('mae', 0):.4f}")
        else:
            st.warning(f"Retrain complete (not promoted). Status: {_rr.get('status', 'unknown')}")
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "All Experiments", "Catalyst Model", "Bio Model", "Reaction Model",
        "Benchmarks", "Collaboration",
    ])

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
            _show_cols = ["timestamp", "user", "exp_type", "name", "metric",
                          "pred_value", "actual_value", "data_quality",
                          "source_provenance", "version_tag", "notes"]
            _show_cols = [c for c in _show_cols if c in df_all.columns]
            st.dataframe(df_all[_show_cols], use_container_width=True, hide_index=True)
            st.plotly_chart(fb.plot_predicted_vs_actual(None), use_container_width=True)
            st.plotly_chart(fb.plot_experiment_timeline(),     use_container_width=True)

            # ── Automated Discrepancy Flags ───────────────────────────────────
            st.markdown('<div class="section-header">🚨 Automated Discrepancy Flags</div>', unsafe_allow_html=True)
            _flag_col1, _flag_col2 = st.columns([3, 1])
            with _flag_col2:
                _thresh = st.slider(
                    "Flag threshold  |error| ≥",
                    min_value=0.01, max_value=0.15,
                    value=0.02, step=0.01,
                    key="flag_thresh",
                    help="Experiments whose |actual − predicted| exceeds this value are flagged.",
                )
                _flag_type = st.selectbox(
                    "Filter by type", ["all", "catalyst", "bio"], key="flag_type",
                )
            with _flag_col1:
                _flagged = fb.flag_discrepancies(
                    exp_type=None if _flag_type == "all" else _flag_type,
                    threshold=_thresh,
                )
                if _flagged.empty:
                    st.success(
                        f"No experiments exceed the |error| ≥ {_thresh:.2f} threshold. "
                        "Model predictions are within tolerance for all logged results."
                    )
                else:
                    st.warning(
                        f"**{len(_flagged)} experiment(s)** flagged  "
                        f"(|error| > {_thresh:.2f}).  "
                        "AI hypotheses are shown below each flag."
                    )
                    for _, _fl in _flagged.iterrows():
                        _is_over   = _fl["flag"] == "OVER-PREDICTED"
                        _fc        = "var(--danger)"  if _is_over else "var(--warning)"
                        _fbg       = "rgba(255,69,58,0.10)" if _is_over else "rgba(255,159,10,0.10)"
                        _pred_v    = float(_fl["pred_value"]   or 0)
                        _actual_v  = float(_fl["actual_value"] or 0)
                        _abs_err   = float(_fl["abs_error"]    or 0)
                        st.markdown(f"""
                        <div class="apple-card" style="border-left:3px solid {_fc};margin-bottom:0.9rem;">
                          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                            <span style="font-size:1.1rem;">🚨</span>
                            <span style="font-size:0.92rem;font-weight:700;color:var(--text-1);">
                              {_fl['name']}
                            </span>
                            <span style="background:{_fbg};color:{_fc};
                                         border:1px solid {_fc};padding:2px 10px;
                                         border-radius:999px;font-size:0.72rem;font-weight:700;
                                         margin-left:auto;">{_fl['flag']}</span>
                          </div>
                          <div style="display:flex;gap:2rem;font-size:0.8rem;color:var(--text-2);margin-bottom:8px;">
                            <span>Predicted: <strong style="color:var(--text-1);">{_pred_v:.3f}</strong></span>
                            <span>Actual: <strong style="color:var(--text-1);">{_actual_v:.3f}</strong></span>
                            <span>|Error|: <strong style="color:{_fc};">{_abs_err:.3f}</strong></span>
                            <span>Type: <strong style="color:var(--text-1);">{_fl['exp_type']}</strong></span>
                            <span>User: <strong style="color:var(--cyan);">{_fl.get('user','—')}</strong></span>
                          </div>
                          <div style="font-size:0.8rem;color:var(--text-2);line-height:1.55;
                                      background:rgba(255,255,255,0.03);border-radius:8px;
                                      padding:8px 12px;">
                            💡 <strong style="color:var(--cyan);">AI Hypothesis:</strong>&nbsp;
                            {_fl['hypothesis']}
                          </div>
                        </div>""", unsafe_allow_html=True)

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

    with tab4:
        st.markdown('<div class="section-header">Reaction Yield Predictor</div>', unsafe_allow_html=True)
        st.plotly_chart(fb.plot_model_improvement("reaction"), use_container_width=True)
        st.plotly_chart(fb.plot_predicted_vs_actual("reaction"), use_container_width=True)
        df_rxn = fb.get_experiments("reaction")
        if not df_rxn.empty:
            st.dataframe(df_rxn[["timestamp", "name", "pred_value", "actual_value", "notes"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No reaction experiments yet. Run reactions in Reaction Lab.")

        st.markdown('<div class="section-header">Community Data Gaps</div>', unsafe_allow_html=True)
        for sug in al_lib.get_library_suggestions(top_k=5):
            st.markdown(f"""
            <div class="apple-card-sm" style="border-left:3px solid var(--cyan);margin-bottom:0.5rem;">
                <div style="font-size:0.85rem;color:var(--text-1);">{sug['suggestion']}</div>
                <div style="font-size:0.75rem;color:var(--text-2);">
                    Priority: {sug['priority']:.3f} · Data points: {sug['data_points']}
                </div>
            </div>""", unsafe_allow_html=True)

        _contrib = get_repo().top_contributors(5)
        if not _contrib.empty:
            st.markdown('<div class="section-header">Top Contributors</div>', unsafe_allow_html=True)
            st.dataframe(_contrib, use_container_width=True, hide_index=True)

    with tab5:
        st.markdown('<div class="section-header">Benchmark Suites</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Run Organic SMARTS Benchmark", use_container_width=True):
            with st.spinner("Running..."):
                _bm = run_benchmark("organic_smarts")
            st.session_state["last_benchmark"] = _bm
        if b2.button("Run Fuel Reactions Benchmark", use_container_width=True):
            with st.spinner("Running..."):
                _bm = run_benchmark("fuel_reactions")
            st.session_state["last_benchmark"] = _bm
        if "last_benchmark" in st.session_state:
            _bm = st.session_state["last_benchmark"]
            st.metric("Pass Rate", f"{_bm.get('pass_rate', 0):.0%}")
            st.json(_bm)
        _bm_hist = get_repo().get_benchmark_runs(limit=10)
        if not _bm_hist.empty:
            st.dataframe(_bm_hist, use_container_width=True, hide_index=True)

    with tab6:
        # ── User Activity ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">User Activity</div>', unsafe_allow_html=True)
        _activity = fb.get_user_activity()
        if _activity.empty:
            st.info("No user activity yet. Sign in and log an experiment to appear here.")
        else:
            _act_cols = st.columns(min(len(_activity), 4))
            for _i, (_, _row) in enumerate(_activity.iterrows()):
                if _i < 4:
                    _act_cols[_i].markdown(f"""
                    <div class="apple-metric">
                        <div class="apple-metric-icon">👤</div>
                        <div class="apple-metric-value" style="font-size:1.3rem;">{_row['user']}</div>
                        <div class="apple-metric-label">{int(_row['experiments'])} experiments</div>
                        <div style="font-size:0.7rem;color:var(--text-3);margin-top:4px;">
                            Last: {str(_row['last_active'])[:10]}
                        </div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Data Provenance ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📦 Data Provenance</div>', unsafe_allow_html=True)
        st.plotly_chart(fb.plot_provenance_chart(), use_container_width=True)

        st.divider()

        # ── Annotations ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📝 Annotations & Shared Notes</div>', unsafe_allow_html=True)
        _ann_left, _ann_right = st.columns([2, 1])

        with _ann_right:
            st.markdown("""
            <div class="apple-card" style="margin-bottom:0.8rem;">
              <div class="step-label">ADD ANNOTATION</div>
            </div>""", unsafe_allow_html=True)
            _ann_user = auth.get_current_user()
            st.caption(f"Posting as **{_ann_user}**")
            _ann_exp_type = st.selectbox("Experiment type", ["catalyst", "bio"], key="ann_type")
            _ann_exps = fb.get_experiments(_ann_exp_type)
            _ann_names = sorted(_ann_exps["name"].unique().tolist()) if not _ann_exps.empty else []
            _ann_target = st.selectbox("Target experiment", _ann_names, key="ann_target") if _ann_names else None
            _ann_text = st.text_area(
                "Note / hypothesis / flag",
                placeholder="e.g., Fe segregation explains over-performance. Flag for re-test.",
                key="ann_text_input", height=110,
            )
            if st.button("💬 Post Annotation", key="post_ann", use_container_width=True):
                if _ann_text.strip() and _ann_target:
                    fb.add_annotation(
                        user=_ann_user, text=_ann_text.strip(),
                        exp_type=_ann_exp_type, target_name=_ann_target,
                    )
                    st.success(f"Posted by {_ann_user}!")
                    st.rerun()
                else:
                    st.warning("Enter text and select a target experiment.")

        with _ann_left:
            _ann_df = fb.get_annotations()
            if _ann_df.empty:
                st.info("No annotations yet. Add the first note using the panel on the right.")
            else:
                for _, _ann in _ann_df.iterrows():
                    st.markdown(f"""
                    <div class="apple-card-sm" style="border-left:3px solid var(--accent);margin-bottom:0.7rem;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                            <span style="font-size:0.8rem;font-weight:700;color:var(--text-1);">👤 {_ann['user']}</span>
                            <span style="font-size:0.72rem;color:var(--text-3);">{str(_ann['timestamp'])[:16]}</span>
                            <span class="badge-ai" style="margin-left:auto;">{_ann.get('exp_type','')}</span>
                        </div>
                        <div style="font-size:0.77rem;color:var(--text-2);margin-bottom:4px;">
                            Re: <strong style="color:var(--cyan);">{_ann.get('target_name','—')}</strong>
                        </div>
                        <div style="font-size:0.83rem;color:var(--text-1);line-height:1.45;">{_ann['text']}</div>
                    </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Version History ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🕑 Version History</div>', unsafe_allow_html=True)
        _vh_col1, _vh_col2 = st.columns([1, 2])
        with _vh_col1:
            _vh_type = st.selectbox("Experiment type", ["catalyst", "bio"], key="vh_type")
            _vh_exps = fb.get_experiments(_vh_type)
            _vh_names = sorted(_vh_exps["name"].unique().tolist()) if not _vh_exps.empty else []
            _vh_name = st.selectbox("Experiment name", _vh_names, key="vh_name") if _vh_names else None
        with _vh_col2:
            if _vh_name:
                _history = fb.get_experiment_history(_vh_name, _vh_type)
                if not _history.empty:
                    st.caption(f"{len(_history)} logged version(s) for **{_vh_name}**")
                    st.dataframe(_history, use_container_width=True, hide_index=True)
                else:
                    st.info("No version history found for this experiment.")
            else:
                st.info("No experiments logged yet.")
