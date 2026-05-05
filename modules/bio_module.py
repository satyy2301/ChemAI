import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# ─── Database path ────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "bio_db.json"


def load_pathways(category_filter: str | None = None) -> list:
    with open(_DB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if category_filter:
        data = [p for p in data if p.get("category") == category_filter]
    return data


def get_categories() -> list:
    data = load_pathways()
    seen = []
    for p in data:
        c = p.get("category", "Other")
        if c not in seen:
            seen.append(c)
    return seen


def get_pathway_by_id(pid: str) -> dict | None:
    for p in load_pathways():
        if p["id"] == pid:
            return p
    return None


# ─── Pathway graph visualisation ─────────────────────────────────────────────

def plot_pathway(pathway: dict, highlight_bottleneck: bool = True) -> go.Figure:
    steps = pathway.get("steps", [])
    bottleneck_enzyme = pathway.get("bottleneck", "")

    G = nx.DiGraph()
    edge_labels = {}
    edge_efficiencies = {}

    for step in steps:
        G.add_node(step["from"])
        G.add_node(step["to"])
        G.add_edge(step["from"], step["to"])
        edge_labels[(step["from"], step["to"])] = step["enzyme"]
        edge_efficiencies[(step["from"], step["to"])] = step.get("efficiency", 1.0)

    # Layout
    pos = nx.spring_layout(G, seed=7, k=2.0)

    # Build edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        eff = edge_efficiencies.get(edge, 1.0)
        enzyme = edge_labels.get(edge, "")
        is_bottleneck = bottleneck_enzyme.split("(")[0].strip() in enzyme

        color = "#FF6B6B" if (highlight_bottleneck and is_bottleneck) else "#00D4FF"
        width = 3 if is_bottleneck else 1.5

        # Arrow line
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Mid-point enzyme label
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_traces.append(go.Scatter(
            x=[mx], y=[my],
            mode="markers+text",
            marker=dict(size=0),
            text=[f"{enzyme}<br>({eff*100:.0f}%)"],
            textfont=dict(size=8, color="#AAAAAA"),
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node trace
    node_x, node_y, node_text, node_hover, node_colors = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_hover.append(f"<b>{node}</b>")
        if node == pathway.get("input"):
            node_colors.append("#A8FF78")
        elif node == pathway.get("output") or node.split(" ")[0] in pathway.get("output", ""):
            node_colors.append("#FFD700")
        else:
            node_colors.append("#4A90E2")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=22, color=node_colors,
                    line=dict(width=2, color="#ffffff")),
        text=node_text,
        textposition="bottom center",
        textfont=dict(size=9),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )

    # Legend items
    legend_items = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(color="#A8FF78", size=10), name="Input substrate"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(color="#FFD700", size=10), name="Target product"),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(color="#4A90E2", size=10), name="Intermediate"),
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color="#FF6B6B", width=3), name="Bottleneck step"),
    ]

    fig = go.Figure(data=edge_traces + [node_trace] + legend_items)
    fig.update_layout(
        title=f"Metabolic Pathway: {pathway.get('name', '')}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#333"),
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─── Yield predictor ─────────────────────────────────────────────────────────

class BioYieldPredictor:
    """Random-forest model trained on synthetic pathway features."""

    def __init__(self, pathways: list):
        self._scaler = StandardScaler()
        self._model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                            random_state=42)
        self._fit(pathways)

    @staticmethod
    def _featurise(pathway: dict) -> np.ndarray:
        steps = pathway.get("steps", [])
        n_steps = len(steps)
        avg_eff = np.mean([s.get("efficiency", 0.85) for s in steps]) if steps else 0.85
        min_eff = min((s.get("efficiency", 0.85) for s in steps), default=0.85)
        difficulty_map = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
        diff = difficulty_map.get(pathway.get("difficulty", "Medium"), 0.5)
        return np.array([n_steps, avg_eff, min_eff, diff])

    def _fit(self, pathways: list):
        X, y = [], []
        rng = np.random.default_rng(42)
        for p in pathways:
            feat = self._featurise(p)
            target_yield = p.get("yield_g_per_g", 0.5)
            X.append(feat)
            y.append(target_yield)
        # Augment
        X_aug, y_aug = list(X), list(y)
        for _ in range(8):
            for i, feat in enumerate(X):
                X_aug.append(feat + rng.normal(0, 0.02, feat.shape))
                y_aug.append(y[i] + rng.normal(0, 0.02))
        self._scaler.fit(np.array(X_aug))
        self._model.fit(self._scaler.transform(np.array(X_aug)), np.array(y_aug))

    def predict(self, pathway: dict) -> dict:
        feat = self._featurise(pathway).reshape(1, -1)
        feat_sc = self._scaler.transform(feat)
        preds = np.array([t.predict(feat_sc)[0] for t in self._model.estimators_])
        return {
            "yield": float(np.clip(np.mean(preds), 0, 1)),
            "std": float(np.std(preds)),
        }

    def retrain(self, extra: list, base: list):
        self._fit(base + extra)


_PREDICTOR: BioYieldPredictor | None = None


def get_bio_predictor() -> BioYieldPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = BioYieldPredictor(load_pathways())
    return _PREDICTOR


# ─── Mutation / improvement suggestions ──────────────────────────────────────

_GENERIC_MUTATIONS = [
    "Codon-optimise the rate-limiting enzyme gene for host expression",
    "Screen enzyme homologs from thermophilic organisms (higher Kcat)",
    "Apply directed evolution (error-prone PCR) on bottleneck enzyme",
    "Delete competing pathways draining the key precursor",
    "Engineer cofactor (NADH/NADPH) regeneration system",
    "Use dynamic metabolic regulation (biosensor-controlled expression)",
    "Overexpress membrane transporter for faster substrate uptake",
    "Apply adaptive laboratory evolution (ALE) under selection pressure",
]


def suggest_mutations(pathway: dict, n: int = 4) -> list:
    """Return mutation suggestions: pathway-specific first, then generic."""
    suggestions = []

    # Specific improvements from DB
    known = pathway.get("known_improvements", [])
    suggestions.extend(known[:n])

    # Fill rest with generic if needed
    rng = random.Random(pathway.get("id", "x"))
    if len(suggestions) < n:
        pool = [m for m in _GENERIC_MUTATIONS if m not in suggestions]
        suggestions.extend(rng.sample(pool, min(n - len(suggestions), len(pool))))

    return suggestions[:n]


def get_bottleneck_step(pathway: dict) -> dict | None:
    """Return the step dict that corresponds to the bottleneck."""
    bottleneck = pathway.get("bottleneck", "")
    for step in pathway.get("steps", []):
        if step["enzyme"].split(" ")[0] in bottleneck or step["from"] in bottleneck:
            return step
    # fallback: lowest efficiency
    steps = [s for s in pathway.get("steps", []) if isinstance(s, dict)]
    if steps:
        return min(steps, key=lambda s: float(s.get("efficiency", 1.0)))
    return None


# ─── Scenario simulator + optimizer ──────────────────────────────────────────

def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def simulate_pathway(pathway: dict, scenario: dict) -> dict:
    """Simulate pathway yield under user-defined process conditions."""
    predictor = get_bio_predictor()
    base = predictor.predict(pathway)

    temperature_c = float(scenario.get("temperature_c", 37.0))
    ph = float(scenario.get("ph", 7.0))
    oxygen_mode = str(scenario.get("oxygen_mode", "Microaerobic"))
    feedstock = str(scenario.get("feedstock", "Glucose"))
    mutation_intensity = float(scenario.get("mutation_intensity", 0.4))
    mutation_intensity = float(np.clip(mutation_intensity, 0.0, 1.0))

    # Temperature and pH penalty around biological sweet spot.
    temp_penalty = max(0.0, 1.0 - abs(temperature_c - 37.0) * 0.015)
    ph_penalty = max(0.0, 1.0 - abs(ph - 7.0) * 0.10)

    oxygen_factor_map = {
        "Aerobic": 0.95,
        "Microaerobic": 1.00,
        "Anaerobic": 0.92,
    }
    oxygen_factor = oxygen_factor_map.get(oxygen_mode, 1.0)

    feedstock_factor_map = {
        "Glucose": 1.00,
        "Xylose": 0.90,
        "Glycerol": 0.93,
        "CO2": 0.78,
        "Mixed sugars": 0.96,
    }
    feedstock_factor = feedstock_factor_map.get(feedstock, 1.0)

    difficulty = pathway.get("difficulty", "Medium")
    difficulty_gain_map = {"Low": 0.05, "Medium": 0.10, "High": 0.16}
    engineering_gain = mutation_intensity * difficulty_gain_map.get(difficulty, 0.10)

    bottleneck = get_bottleneck_step(pathway)
    bottleneck_eff = float((bottleneck or {}).get("efficiency", 0.8))
    bottleneck_headroom = max(0.0, 1.0 - bottleneck_eff)
    bottleneck_relief = mutation_intensity * bottleneck_headroom * 0.20

    process_multiplier = temp_penalty * ph_penalty * oxygen_factor * feedstock_factor
    adjusted_yield = _clip01(base["yield"] * process_multiplier + engineering_gain + bottleneck_relief)

    # Higher process stress and heavier engineering increase uncertainty and risk.
    process_stress = (1.0 - temp_penalty) + (1.0 - ph_penalty) + abs(1.0 - oxygen_factor)
    uncertainty = float(np.clip(base["std"] + process_stress * 0.03 + mutation_intensity * 0.04, 0.01, 0.35))
    risk_score = float(np.clip(process_stress * 0.30 + mutation_intensity * 0.60, 0.0, 1.0))

    return {
        "baseline_yield": float(base["yield"]),
        "predicted_yield": adjusted_yield,
        "uncertainty": uncertainty,
        "risk_score": risk_score,
        "delta_vs_baseline": float(adjusted_yield - base["yield"]),
        "drivers": {
            "temperature_penalty": float(temp_penalty),
            "ph_penalty": float(ph_penalty),
            "oxygen_factor": float(oxygen_factor),
            "feedstock_factor": float(feedstock_factor),
            "engineering_gain": float(engineering_gain),
            "bottleneck_relief": float(bottleneck_relief),
        },
    }


def build_intervention_plans(pathway: dict, scenario: dict, top_k: int = 3) -> list:
    """Return ranked intervention plans with expected gain and risk."""
    sim = simulate_pathway(pathway, scenario)
    base_yield = sim["predicted_yield"]
    bottleneck = get_bottleneck_step(pathway)
    bottleneck_enzyme = (bottleneck or {}).get("enzyme", "rate-limiting enzyme")
    bottleneck_gene = (bottleneck or {}).get("gene", "target gene")

    mutation_intensity = float(np.clip(float(scenario.get("mutation_intensity", 0.4)), 0.0, 1.0))
    oxygen_mode = str(scenario.get("oxygen_mode", "Microaerobic"))

    options = [
        {
            "action": f"Overexpress {bottleneck_gene}",
            "rationale": f"Directly relieves bottleneck at {bottleneck_enzyme}.",
            "gain": 0.05 + 0.06 * mutation_intensity,
            "risk": 0.28 + 0.25 * mutation_intensity,
        },
        {
            "action": "Delete competing byproduct pathway genes",
            "rationale": "Redirects carbon flux toward target product.",
            "gain": 0.04 + 0.05 * mutation_intensity,
            "risk": 0.24 + 0.20 * mutation_intensity,
        },
        {
            "action": "Engineer cofactor regeneration (NADH/NADPH)",
            "rationale": "Improves redox balance and pathway throughput.",
            "gain": 0.03 + 0.04 * mutation_intensity,
            "risk": 0.20 + 0.18 * mutation_intensity,
        },
        {
            "action": "Adaptive lab evolution under selective pressure",
            "rationale": "Improves tolerance and productivity under process stress.",
            "gain": 0.02 + 0.03 * mutation_intensity,
            "risk": 0.14 + 0.15 * mutation_intensity,
        },
    ]

    # Oxygen-aware hinting for fermentation pathways.
    if oxygen_mode == "Anaerobic":
        options.append({
            "action": "Tune anaerobic ATP maintenance and redox sink",
            "rationale": "Prevents energy limitation in strict anaerobic operation.",
            "gain": 0.03,
            "risk": 0.22,
        })

    plans = []
    for i, opt in enumerate(options, start=1):
        projected_yield = _clip01(base_yield + opt["gain"])
        projected_risk = float(np.clip((sim["risk_score"] + opt["risk"]) / 2.0, 0.0, 1.0))
        score = projected_yield - 0.35 * projected_risk
        plans.append({
            "id": f"plan_{i:02d}",
            "action": opt["action"],
            "rationale": opt["rationale"],
            "expected_gain": float(opt["gain"]),
            "projected_yield": projected_yield,
            "risk": projected_risk,
            "priority_score": float(score),
        })

    plans = sorted(plans, key=lambda x: x["priority_score"], reverse=True)
    return plans[:max(1, top_k)]


def counterfactual_sensitivity(pathway: dict, scenario: dict) -> list:
    """Generate simple counterfactual statements for key knobs."""
    baseline = simulate_pathway(pathway, scenario)
    rows = []

    probes = [
        ("temperature_c", +4.0, "Increase temperature by 4°C"),
        ("temperature_c", -4.0, "Decrease temperature by 4°C"),
        ("ph", +0.5, "Increase pH by 0.5"),
        ("ph", -0.5, "Decrease pH by 0.5"),
        ("mutation_intensity", +0.2, "Increase mutation intensity by 0.2"),
        ("mutation_intensity", -0.2, "Decrease mutation intensity by 0.2"),
    ]

    for key, delta, label in probes:
        alt = dict(scenario)
        current = float(alt.get(key, 0.0))
        if key == "mutation_intensity":
            alt[key] = float(np.clip(current + delta, 0.0, 1.0))
        elif key == "ph":
            alt[key] = float(np.clip(current + delta, 4.5, 9.0))
        elif key == "temperature_c":
            alt[key] = float(np.clip(current + delta, 20.0, 50.0))
        alt_sim = simulate_pathway(pathway, alt)
        rows.append({
            "change": label,
            "delta_yield": float(alt_sim["predicted_yield"] - baseline["predicted_yield"]),
            "new_yield": float(alt_sim["predicted_yield"]),
            "new_risk": float(alt_sim["risk_score"]),
        })

    return sorted(rows, key=lambda x: abs(x["delta_yield"]), reverse=True)


# ─── Flux Balance Analysis ───────────────────────────────────────────────────

def run_fba(pathway: dict, scenario: dict | None = None) -> dict:
    """Maximize product flux via LP (scipy.optimize.linprog / HiGHS).

    Variables  : v[0..n-1] = pathway step fluxes
                 v[n]      = substrate exchange (input, fixed ≤ 1.0)
                 v[n+1]    = product exchange (output, maximized)
    Constraints: S·v = 0  (steady-state mass balance at each metabolite node)
    Bounds     : 0 ≤ v_i ≤ efficiency_i  (capacity limit per step)
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return {"status": "scipy_missing", "fluxes": {}, "optimal_yield": 0.0, "reactions": []}

    sc = scenario or {}
    mutation_intensity = float(np.clip(float(sc.get("mutation_intensity", 0.4)), 0.0, 1.0))
    oxygen_mode        = str(sc.get("oxygen_mode", "Microaerobic"))
    temperature_c      = float(sc.get("temperature_c", 37.0))

    steps = pathway.get("steps", [])
    if not steps:
        return {"status": "empty", "fluxes": {}, "optimal_yield": 0.0, "reactions": []}

    bottleneck = get_bottleneck_step(pathway)
    bottleneck_enzyme = (bottleneck or {}).get("enzyme", "")

    # ── Metabolite ordering ────────────────────────────────────────────────────
    metabolites: list[str] = []
    for step in steps:
        for key in ("from", "to"):
            m = step[key]
            if m not in metabolites:
                metabolites.append(m)

    substrate = metabolites[0]
    product   = metabolites[-1]
    n_steps   = len(steps)
    n_mets    = len(metabolites)
    n_vars    = n_steps + 2      # steps + substrate_in + product_out

    # ── Stoichiometric matrix [n_mets × n_vars] ───────────────────────────────
    S_mat = np.zeros((n_mets, n_vars))
    for j, step in enumerate(steps):
        frm, to_ = step["from"], step["to"]
        if frm in metabolites:
            S_mat[metabolites.index(frm), j] = -1.0
        if to_ in metabolites:
            S_mat[metabolites.index(to_),  j] = +1.0

    # substrate exchange produces substrate (index n_steps)
    S_mat[metabolites.index(substrate), n_steps]     = +1.0
    # product exchange consumes product (index n_steps+1)
    S_mat[metabolites.index(product),   n_steps + 1] = -1.0

    # ── Bounds ────────────────────────────────────────────────────────────────
    temp_factor = float(max(0.60, 1.0 - abs(temperature_c - 37.0) * 0.012))
    bounds: list[tuple[float, float | None]] = []
    for j, step in enumerate(steps):
        eff = float(step.get("efficiency", 0.85)) * temp_factor
        # Oxygen penalty for aerobic enzymes in anaerobic mode
        if oxygen_mode == "Anaerobic":
            enzyme_lc = step.get("enzyme", "").lower()
            if any(k in enzyme_lc for k in ("oxidase", "oxygenase", "synthase (fas)", "carboxylase")):
                eff *= 0.80
        # Mutation-driven improvement of the bottleneck step
        if bottleneck_enzyme and step.get("enzyme", "") == bottleneck_enzyme:
            eff = min(0.99, eff + mutation_intensity * 0.18)
        bounds.append((0.0, float(np.clip(eff, 0.05, 0.99))))

    bounds.append((0.0, 1.0))   # substrate_in ∈ [0, 1]
    bounds.append((0.0, None))  # product_out  ∈ [0, ∞)

    # ── Objective: maximise product_out ────────────────────────────────────────
    c = np.zeros(n_vars)
    c[n_steps + 1] = -1.0  # linprog minimises → negate

    result = linprog(c, A_eq=S_mat, b_eq=np.zeros(n_mets), bounds=bounds, method="highs")

    rxn_labels = [
        f"{s['from'][:22]} → {s['to'][:22]}"
        for s in steps
    ]

    if result.status == 0:
        fluxes = {rxn_labels[j]: float(result.x[j]) for j in range(n_steps)}
        eff_bounds = [float(b[1]) for b in bounds[:n_steps] if b[1] is not None]
        # Identify the step carrying lowest flux (= flux-limiting)
        limiting_idx = int(np.argmin(result.x[:n_steps]))
        return {
            "status": "optimal",
            "fluxes": fluxes,
            "optimal_yield":  float(result.x[n_steps + 1]),
            "substrate_in":   float(result.x[n_steps]),
            "reactions":      rxn_labels,
            "metabolites":    metabolites,
            "limiting_step":  rxn_labels[limiting_idx],
            "efficiency_bounds": eff_bounds,
        }

    # Fallback: proportional approximation
    effs = [float(s.get("efficiency", 0.85)) for s in steps]
    fluxes_fb = {rxn_labels[j]: effs[j] for j in range(n_steps)}
    limiting_idx_fb = int(np.argmin(effs))
    return {
        "status": "approximate",
        "fluxes": fluxes_fb,
        "optimal_yield":  float(min(effs)),
        "substrate_in":   1.0,
        "reactions":      rxn_labels,
        "metabolites":    metabolites,
        "limiting_step":  rxn_labels[limiting_idx_fb],
        "efficiency_bounds": effs,
    }


def plot_fba_fluxes(fba_result: dict) -> go.Figure:
    """Horizontal bar chart of FBA-optimised flux distribution."""
    fluxes   = fba_result.get("fluxes", {})
    limiting = fba_result.get("limiting_step", "")
    eff_bnd  = fba_result.get("efficiency_bounds", [])

    if not fluxes:
        fig = go.Figure()
        fig.update_layout(
            title="No FBA results available",
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font=dict(color="#FAFAFA"), height=220,
        )
        return fig

    reactions = list(fluxes.keys())
    values    = [fluxes[r] for r in reactions]
    n         = len(reactions)
    colors    = ["#FF6B6B" if r == limiting else "#00D4FF" for r in reactions]
    capacity  = eff_bnd if len(eff_bnd) == n else [1.0] * n

    fig = go.Figure()

    # Capacity bars (background, faint)
    fig.add_trace(go.Bar(
        x=capacity, y=reactions,
        orientation="h",
        marker=dict(color="rgba(255,255,255,0.08)"),
        name="Step capacity",
        hovertemplate="%{y}<br>Capacity: %{x:.3f}<extra></extra>",
    ))

    # Actual flux bars
    fig.add_trace(go.Bar(
        x=values, y=reactions,
        orientation="h",
        marker_color=colors,
        name="Optimised flux",
        text=[f"{v:.3f}" for v in values],
        textposition="auto",
        textfont=dict(size=11),
        hovertemplate="%{y}<br>Flux: %{x:.3f}<extra></extra>",
    ))

    opt_yield = fba_result.get("optimal_yield", 0.0)
    status    = fba_result.get("status", "")
    title_str = (
        f"FBA Flux Distribution  |  Optimal yield: {opt_yield:.3f}"
        + ("  (approx.)" if status == "approximate" else "")
    )

    fig.update_layout(
        barmode="overlay",
        title=title_str,
        xaxis_title="Normalised flux",
        xaxis=dict(range=[0, max(capacity + [0.1]) * 1.1]),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#333"),
        height=max(280, n * 52 + 80),
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


# ─── Summary table ────────────────────────────────────────────────────────────

def pathway_summary_df(pathways: list) -> pd.DataFrame:
    rows = []
    predictor = get_bio_predictor()
    for p in pathways:
        pred = predictor.predict(p)
        rows.append({
            "Name": p["name"],
            "Input": p["input"],
            "Output": p["output"],
            "Organism": p["organism"],
            "Yield (g/g)": f"{p.get('yield_g_per_g', 0):.2f}",
            "AI Pred. Yield": f"{pred['yield']:.2f} ± {pred['std']:.2f}",
            "Difficulty": p.get("difficulty", "—"),
            "Category": p.get("category", "—"),
            "Steps": len(p.get("steps", [])),
        })
    return pd.DataFrame(rows)


# ─── Comparative bar chart ────────────────────────────────────────────────────

def plot_yield_comparison(pathways: list) -> go.Figure:
    predictor = get_bio_predictor()
    names, actual_yields, pred_yields, errors = [], [], [], []
    for p in pathways:
        pred = predictor.predict(p)
        names.append(p["name"][:30])
        actual_yields.append(p.get("yield_g_per_g", 0))
        pred_yields.append(pred["yield"])
        errors.append(pred["std"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Reported Yield",
        x=names, y=actual_yields,
        marker_color="#A8FF78",
    ))
    fig.add_trace(go.Bar(
        name="AI Predicted",
        x=names, y=pred_yields,
        error_y=dict(type="data", array=errors, visible=True),
        marker_color="#00D4FF",
    ))
    fig.update_layout(
        barmode="group",
        title="Pathway Yield: Reported vs AI Prediction",
        yaxis_title="Yield (g product / g substrate)",
        xaxis_tickangle=-30,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E"),
        height=420,
    )
    return fig
