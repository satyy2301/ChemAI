import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# ─── Element property table ───────────────────────────────────────────────────
ELEMENT_PROPS = {
    "Cu": {"Z": 29, "EN": 1.90, "R": 128, "group": 11, "period": 4},
    "Pt": {"Z": 78, "EN": 2.28, "R": 136, "group": 10, "period": 6},
    "Pd": {"Z": 46, "EN": 2.20, "R": 137, "group": 10, "period": 5},
    "Fe": {"Z": 26, "EN": 1.83, "R": 126, "group": 8,  "period": 4},
    "Ni": {"Z": 28, "EN": 1.91, "R": 124, "group": 10, "period": 4},
    "Co": {"Z": 27, "EN": 1.88, "R": 125, "group": 9,  "period": 4},
    "Ru": {"Z": 44, "EN": 2.20, "R": 134, "group": 8,  "period": 5},
    "Rh": {"Z": 45, "EN": 2.28, "R": 134, "group": 9,  "period": 5},
    "Au": {"Z": 79, "EN": 2.54, "R": 144, "group": 11, "period": 6},
    "Ag": {"Z": 47, "EN": 1.93, "R": 144, "group": 11, "period": 5},
    "Ir": {"Z": 77, "EN": 2.20, "R": 136, "group": 9,  "period": 6},
    "Mo": {"Z": 42, "EN": 2.16, "R": 139, "group": 6,  "period": 5},
    "W":  {"Z": 74, "EN": 2.36, "R": 139, "group": 6,  "period": 6},
    "Zn": {"Z": 30, "EN": 1.65, "R": 122, "group": 12, "period": 4},
    "In": {"Z": 49, "EN": 1.78, "R": 167, "group": 13, "period": 5},
    "Sn": {"Z": 50, "EN": 1.96, "R": 141, "group": 14, "period": 5},
    "Al": {"Z": 13, "EN": 1.61, "R": 143, "group": 13, "period": 3},
    "Ti": {"Z": 22, "EN": 1.54, "R": 147, "group": 4,  "period": 4},
    "Mn": {"Z": 25, "EN": 1.55, "R": 127, "group": 7,  "period": 4},
    "Ga": {"Z": 31, "EN": 1.81, "R": 122, "group": 13, "period": 4},
    "Ce": {"Z": 58, "EN": 1.12, "R": 182, "group": 3,  "period": 6},
    "Zr": {"Z": 40, "EN": 1.33, "R": 155, "group": 4,  "period": 5},
    "P":  {"Z": 15, "EN": 2.19, "R": 107, "group": 15, "period": 3},
    "S":  {"Z": 16, "EN": 2.58, "R": 105, "group": 16, "period": 3},
    "C":  {"Z": 6,  "EN": 2.55, "R": 77,  "group": 14, "period": 2},
    "Si": {"Z": 14, "EN": 1.90, "R": 111, "group": 14, "period": 3},
}

DOPANT_CANDIDATES = ["Cu", "Pt", "Pd", "Fe", "Ni", "Co", "Ru", "Rh", "Mo", "W",
                     "Zn", "In", "Ga", "Mn", "Ag", "Ir", "Ti", "Sn"]

SURFACE_FACETS = ["(111)", "(100)", "(110)", "(211)", "(210)", "(321)"]

REACTION_LABELS = {
    "CO2_to_Methanol": "CO₂ → Methanol",
    "HER":             "Water Splitting (HER)",
    "N2_Fixation":     "N₂ Fixation (NH₃)",
    "RWGS":            "RWGS (CO₂ → CO)",
    "Methanation":     "CO₂ Methanation",
    "Fischer_Tropsch": "Fischer-Tropsch (CO → Fuel)",
    "CO_Oxidation":    "CO Oxidation",
    "OER":             "Water Splitting (OER)",
}

# ─── Feature engineering ──────────────────────────────────────────────────────

def composition_to_features(composition: dict) -> np.ndarray:
    """Weighted-average element properties → fixed-length feature vector."""
    total = sum(composition.values()) or 1.0
    feat = {"Z": 0, "EN": 0, "R": 0, "group": 0, "period": 0}
    known = {e: v for e, v in composition.items() if e in ELEMENT_PROPS}
    for el, frac in known.items():
        w = frac / total
        for k in feat:
            feat[k] += w * ELEMENT_PROPS[el][k]

    Z_vals  = [ELEMENT_PROPS[e]["Z"]  for e in known]
    EN_vals = [ELEMENT_PROPS[e]["EN"] for e in known]
    return np.array([
        feat["Z"], feat["EN"], feat["R"], feat["group"], feat["period"],
        np.std(Z_vals)  if len(Z_vals)  > 1 else 0.0,
        np.std(EN_vals) if len(EN_vals) > 1 else 0.0,
        len(composition),
    ])

# ─── ML model (trained at import on the catalogue + synthetic noise) ──────────

class CatalystPredictor:
    def __init__(self, catalysts: list):
        self._scaler = StandardScaler()
        self._models = {}
        self._fit(catalysts)

    def _fit(self, catalysts: list):
        targets = ["adsorption_energy", "stability_score", "activity_score"]
        X, ys = [], {t: [] for t in targets}

        for c in catalysts:
            feat = composition_to_features(c["composition"])
            X.append(feat)
            for t in targets:
                ys[t].append(c[t])

        # Augment with Gaussian noise copies
        rng = np.random.default_rng(42)
        X_aug, y_aug = list(X), {t: list(ys[t]) for t in targets}
        for _ in range(6):
            for i, feat in enumerate(X):
                noisy = feat + rng.normal(0, 0.03 * np.abs(feat).clip(0.01), feat.shape)
                X_aug.append(noisy)
                for t in targets:
                    y_aug[t].append(ys[t][i] + rng.normal(0, 0.03))

        X_np = np.array(X_aug)
        X_sc = self._scaler.fit_transform(X_np)

        for t in targets:
            rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42)
            rf.fit(X_sc, np.array(y_aug[t]))
            self._models[t] = rf

    def predict(self, composition: dict) -> dict:
        feat = composition_to_features(composition).reshape(1, -1)
        feat_sc = self._scaler.transform(feat)
        result = {}
        for t, model in self._models.items():
            preds = np.array([tree.predict(feat_sc)[0] for tree in model.estimators_])
            result[t] = float(np.mean(preds))
            result[f"{t}_std"] = float(np.std(preds))
        return result

    def uncertainty(self, composition: dict) -> float:
        feat = composition_to_features(composition).reshape(1, -1)
        feat_sc = self._scaler.transform(feat)
        stds = []
        for model in self._models.values():
            preds = np.array([tree.predict(feat_sc)[0] for tree in model.estimators_])
            stds.append(np.std(preds))
        return float(np.mean(stds))

    def retrain(self, extra_catalysts: list, base_catalysts: list):
        self._fit(base_catalysts + extra_catalysts)


# ─── Catalyst database helpers ───────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "catalysts_db.json"
_PREDICTOR: CatalystPredictor | None = None


def _load_raw() -> list:
    with open(_DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_predictor() -> CatalystPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = CatalystPredictor(_load_raw())
    return _PREDICTOR


def load_catalysts(reaction_filter: str | None = None) -> list:
    data = _load_raw()
    if reaction_filter:
        data = [c for c in data if c["reaction"] == reaction_filter]
    return data


def get_reactions() -> list:
    all_cats = _load_raw()
    seen = {}
    for c in all_cats:
        r = c["reaction"]
        if r not in seen:
            seen[r] = REACTION_LABELS.get(r, r)
    return [(k, v) for k, v in seen.items()]


# ─── Candidate generation ────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_variations(base: dict, strategy: str = "mixed", n: int = 5) -> list:
    """Generate n catalyst variants from a base catalyst."""
    predictor = get_predictor()
    results = []
    rng = random.Random(42)

    for i in range(n):
        comp = dict(base["composition"])
        name_parts = [base["name"]]
        facet = base["surface_facet"]

        if strategy in ("doping", "mixed") and i % 3 != 2:
            # Add a dopant at 5-15%
            dopant = rng.choice([d for d in DOPANT_CANDIDATES if d not in comp])
            dose = round(rng.uniform(0.05, 0.15), 2)
            # Dilute existing elements proportionally
            scale = 1.0 - dose
            comp = {k: round(v * scale, 3) for k, v in comp.items()}
            comp[dopant] = dose
            name_parts.append(f"+{dopant}{int(dose*100)}%")

        if strategy in ("surface", "mixed") and i % 3 == 2:
            facet = rng.choice([f for f in SURFACE_FACETS if f != base["surface_facet"]])
            name_parts.append(f"@{facet}")

        # Predict with ML
        pred = predictor.predict(comp)
        uncertainty = predictor.uncertainty(comp)

        candidate = {
            "id": f"gen_{base['id']}_{i+1:02d}",
            "name": " ".join(name_parts[1:]) or base["name"] + " (var.)",
            "formula": "/".join(f"{k}{int(v*100)}" for k, v in comp.items()),
            "composition": comp,
            "support": base.get("support", "None"),
            "surface_facet": facet,
            "reaction": base["reaction"],
            "adsorption_energy": round(pred["adsorption_energy"], 3),
            "stability_score":   round(_clamp(pred["stability_score"], 0, 1), 3),
            "activity_score":    round(_clamp(pred["activity_score"], 0, 1), 3),
            "selectivity_score": round(_clamp(base.get("selectivity_score", 0.85)
                                              + rng.uniform(-0.08, 0.08), 0, 1), 3),
            "uncertainty":       round(uncertainty, 4),
            "source": "AI-generated",
            "description": f"AI variant of {base['name']}. Strategy: {strategy}.",
        }
        results.append(candidate)

    return results


# ─── Ranking ─────────────────────────────────────────────────────────────────

def rank_catalysts(catalysts: list, w_activity: float = 0.45,
                   w_stability: float = 0.35, w_selectivity: float = 0.20) -> list:
    """Return catalysts sorted by weighted score (highest first)."""
    scored = []
    for c in catalysts:
        score = (
            w_activity    * c.get("activity_score", 0.5) +
            w_stability   * c.get("stability_score", 0.5) +
            w_selectivity * c.get("selectivity_score", 0.5)
        )
        scored.append({**c, "composite_score": round(score, 4)})
    return sorted(scored, key=lambda x: x["composite_score"], reverse=True)


# ─── Visualisations ──────────────────────────────────────────────────────────

def plot_tradeoff(catalysts: list, highlight_id: str | None = None) -> go.Figure:
    """Activity vs Stability scatter with uncertainty bubbles."""
    def _safe_float(value, fallback: float = 0.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(v):
            return fallback
        return v

    df = pd.DataFrame(catalysts)

    colors = []
    for _, row in df.iterrows():
        if row.get("source") == "AI-generated":
            colors.append("#00D4FF")
        elif row.get("id") == highlight_id:
            colors.append("#FF6B6B")
        else:
            colors.append("#A8FF78")

    size_col = "uncertainty" if "uncertainty" in df.columns else None
    if size_col:
        uncertainties = pd.to_numeric(df["uncertainty"], errors="coerce")
        uncertainties = uncertainties.where(np.isfinite(uncertainties), np.nan)
        uncertainties = uncertainties.fillna(0.0).clip(lower=0.0)
        sizes = (uncertainties * 800 + 12).tolist()
    else:
        sizes = [14.0] * len(df)

    fig = go.Figure()

    for point_idx, (_, row) in enumerate(df.iterrows()):
        is_ai = row.get("source") == "AI-generated"
        marker_size = _safe_float(sizes[point_idx], fallback=14.0)
        if marker_size < 0:
            marker_size = 14.0
        activity = _safe_float(row.get("activity_score"), fallback=0.0)
        stability = _safe_float(row.get("stability_score"), fallback=0.0)
        selectivity = _safe_float(row.get("selectivity_score"), fallback=0.0)
        composite = _safe_float(row.get("composite_score"), fallback=0.0)

        fig.add_trace(go.Scatter(
            x=[activity],
            y=[stability],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=colors[point_idx],
                opacity=0.85,
                line=dict(width=1.5, color="#ffffff"),
            ),
            text=[row.get("name", "")[:18]],
            textposition="top center",
            textfont=dict(size=9),
            name=row.get("name", ""),
            customdata=[[
                row.get("name", ""),
                row.get("adsorption_energy", "N/A"),
                selectivity,
                composite,
                "AI" if is_ai else "Known",
            ]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Activity: %{x:.3f}<br>"
                "Stability: %{y:.3f}<br>"
                "Adsorption E: %{customdata[1]} eV<br>"
                "Selectivity: %{customdata[2]:.3f}<br>"
                "Score: %{customdata[3]:.3f}<br>"
                "Source: %{customdata[4]}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Add dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color="#A8FF78", size=10),
                             name="Known catalyst"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color="#00D4FF", size=10),
                             name="AI-generated"))

    fig.update_layout(
        title="Activity vs Stability Trade-off",
        xaxis_title="Activity Score →",
        yaxis_title="Stability Score →",
        xaxis=dict(range=[0, 1.05], gridcolor="#2a2a2a"),
        yaxis=dict(range=[0, 1.05], gridcolor="#2a2a2a"),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E", bordercolor="#333"),
        height=480,
    )
    return fig


def plot_radar(catalyst: dict) -> go.Figure:
    categories = ["Activity", "Stability", "Selectivity", "Composite Score"]
    values = [
        catalyst.get("activity_score", 0),
        catalyst.get("stability_score", 0),
        catalyst.get("selectivity_score", 0),
        catalyst.get("composite_score", 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.25)",
        line=dict(color="#00D4FF", width=2),
        name=catalyst.get("name", ""),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1A1A2E",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
            angularaxis=dict(gridcolor="#333"),
        ),
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        showlegend=False,
        height=350,
    )
    return fig


def plot_composition_bar(catalyst: dict) -> go.Figure:
    comp = catalyst.get("composition", {})
    elements = list(comp.keys())
    fractions = [v * 100 for v in comp.values()]
    colors = px.colors.qualitative.Plotly[:len(elements)]

    fig = go.Figure(go.Bar(
        x=elements,
        y=fractions,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in fractions],
        textposition="outside",
    ))
    fig.update_layout(
        title="Elemental Composition",
        yaxis_title="Atomic %",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        height=300,
    )
    return fig


# ─── Reaction Energy Profiles (BEP-scaled, catalyst-specific) ────────────────

# Reference adsorption energies (eV) for a representative catalyst per reaction
_REACTION_REFS: dict[str, float] = {
    "CO2_to_Methanol": -0.72,
    "Fischer_Tropsch": -0.85,
    "HER":             -0.30,
    "N2_Fixation":     -0.65,
    "RWGS":            -0.58,
    "Methanation":     -0.80,
    "CO_Oxidation":    -0.45,
    "OER":             -0.38,
}

# Base energy profiles per reaction (eV, relative to reactants = 0.00).
# Each entry: list of (label, energy) for intermediates; ts_energies[i] is
# the transition-state energy between intermediate i and i+1.
# Constraint: ts_energies[i] >= max(E_i, E_{i+1}).
_BASE_PROFILES: dict[str, dict] = {
    "CO2_to_Methanol": {
        "equation": "CO₂ + 3H₂ → CH₃OH + H₂O",
        "intermediates": [
            ("CO₂ + 3H₂",     0.00),
            ("CO₂* + 3H*",    0.18),
            ("HCOO* + 2H*",  -0.45),
            ("H₂COO* + H*",  -0.80),
            ("H₂CO* + H₂O*", -1.22),
            ("H₃CO* + H₂O",  -1.60),
            ("CH₃OH + H₂O",  -0.88),
        ],
        "ts_energies": [0.42, 0.55, -0.08, -0.42, -0.88, -0.53],
    },
    "Fischer_Tropsch": {
        "equation": "CO + 2H₂ → −CH₂− + H₂O",
        "intermediates": [
            ("CO + 2H₂",      0.00),
            ("CO* + 2H*",     0.12),
            ("CHO* + H*",    -0.35),
            ("CH* + H₂O*",   -0.82),
            ("CH₂* (chain)", -1.28),
            ("−CH₂− + H₂O",  -0.95),
        ],
        "ts_energies": [0.45, 0.52, -0.10, -0.48, -0.92],
    },
    "HER": {
        "equation": "2H⁺ + 2e⁻ → H₂",
        "intermediates": [
            ("2H⁺ + 2e⁻",       0.00),
            ("H* + H⁺ + e⁻",   -0.30),
            ("2H* (Tafel)",     -0.60),
            ("H₂(g)",           -0.45),
        ],
        "ts_energies": [0.18, -0.08, 0.28],
    },
    "N2_Fixation": {
        "equation": "N₂ + 3H₂ → 2NH₃",
        "intermediates": [
            ("N₂ + 3H₂",    0.00),
            ("N₂* + 3H*",   0.22),
            ("2N* + 3H*",  -0.18),
            ("N*+NH*+2H*", -0.62),
            ("2NH* + 2H*", -1.05),
            ("NH₂* + NH₃", -1.48),
            ("2NH₃",       -0.92),
        ],
        "ts_energies": [0.65, 1.12, 0.35, -0.18, -0.62, -0.65],
    },
    "RWGS": {
        "equation": "CO₂ + H₂ → CO + H₂O",
        "intermediates": [
            ("CO₂ + H₂",   0.00),
            ("CO₂* + H*",  0.15),
            ("HCOO*",     -0.38),
            ("CO* + OH*",  -0.72),
            ("CO + H₂O",  -0.42),
        ],
        "ts_energies": [0.38, 0.48, -0.05, -0.20],
    },
    "Methanation": {
        "equation": "CO₂ + 4H₂ → CH₄ + 2H₂O",
        "intermediates": [
            ("CO₂ + 4H₂",        0.00),
            ("CO* + H₂O + 3H*",  -0.15),
            ("C* + H₂O + 3H*",   -0.55),
            ("CH* + H₂O + 2H*",  -0.98),
            ("CH₂* + H₂O + H*",  -1.35),
            ("CH₃* + H₂O",       -1.68),
            ("CH₄ + 2H₂O",       -1.28),
        ],
        "ts_energies": [0.32, 0.25, -0.08, -0.45, -0.88, -1.25],
    },
    "CO_Oxidation": {
        "equation": "CO + ½O₂ → CO₂",
        "intermediates": [
            ("CO + ½O₂",  0.00),
            ("CO* + O*", -0.42),
            ("CO₂*",     -0.18),
            ("CO₂(g)",   -1.45),
        ],
        "ts_energies": [0.18, 0.42, 0.05],
    },
    "OER": {
        "equation": "2H₂O → O₂ + 4H⁺ + 4e⁻",
        "intermediates": [
            ("2H₂O",            0.00),
            ("OH* + H⁺ + e⁻",  0.45),
            ("O* + H⁺ + e⁻",   0.98),
            ("OOH* + H⁺ + e⁻", 1.52),
            ("O₂ + H⁺ + e⁻",   1.23),
        ],
        "ts_energies": [0.72, 1.25, 1.68, 1.80],
    },
}


def _scale_profile_for_catalyst(profile: dict, catalyst: dict,
                                 reaction_key: str) -> dict:
    """BEP-linear scaling: stronger-binding catalyst stabilises intermediates."""
    ref_ads = _REACTION_REFS.get(reaction_key, -0.60)
    cat_ads = float(catalyst.get("adsorption_energy", ref_ads))
    delta = cat_ads - ref_ads   # negative → stronger binding than reference
    alpha = 0.55                # BEP slope

    ints = profile["intermediates"]
    ts   = profile["ts_energies"]
    n    = len(ints)

    # Keep reactant (index 0) at 0.00 and product (index -1) at fixed ΔG_rxn.
    scaled_ints = [ints[0]]
    for label, E in ints[1:-1]:
        scaled_ints.append((label, round(E + alpha * delta, 3)))
    scaled_ints.append(ints[-1])

    # Scale TS; clamp so each TS stays above both adjacent intermediates.
    scaled_ts = []
    for i, E_ts in enumerate(ts):
        E_left  = scaled_ints[i][1]
        E_right = scaled_ints[i + 1][1]
        E_ts_s  = round(E_ts + alpha * delta, 3)
        E_ts_s  = max(E_ts_s, max(E_left, E_right) + 0.05)
        scaled_ts.append(E_ts_s)

    return {**profile, "intermediates": scaled_ints, "ts_energies": scaled_ts}


def get_energy_profile_data(reaction_key: str, catalyst: dict) -> dict | None:
    """Return the BEP-scaled energy profile for a given catalyst and reaction."""
    base = _BASE_PROFILES.get(reaction_key)
    if base is None:
        return None
    return _scale_profile_for_catalyst(base, catalyst, reaction_key)


def plot_reaction_energy_profile(reaction_key: str, catalyst: dict) -> go.Figure:
    """Smooth reaction-coordinate diagram with activation-energy annotation."""
    from scipy.interpolate import CubicSpline

    data = get_energy_profile_data(reaction_key, catalyst)
    if data is None:
        fig = go.Figure()
        fig.update_layout(
            title="Energy profile not available for this reaction",
            paper_bgcolor="#0E1117", font=dict(color="#FAFAFA"),
        )
        return fig

    ints = data["intermediates"]
    ts_e = data["ts_energies"]
    n    = len(ints)

    # Build x/y arrays: intermediate at integer x, TS at half-integer x.
    x_pts, y_pts = [], []
    for i, (_, E) in enumerate(ints):
        x_pts.append(float(i))
        y_pts.append(E)
        if i < n - 1:
            x_pts.append(float(i) + 0.5)
            y_pts.append(ts_e[i])

    cs     = CubicSpline(np.array(x_pts), np.array(y_pts))
    x_fine = np.linspace(0, n - 1, 600)
    y_fine = cs(x_fine)

    # Activation barriers and rate-limiting step (highest Ea from left side).
    barriers = [ts_e[i] - ints[i][1] for i in range(n - 1)]
    rls      = int(np.argmax(barriers))
    Ea_rls   = barriers[rls]
    delta_G  = ints[-1][1] - ints[0][1]
    dg_color = "#30D158" if delta_G < 0 else "#FF453A"

    fig = go.Figure()

    # Shaded area below the curve.
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fine, x_fine[::-1]]),
        y=np.concatenate([y_fine, np.full(len(y_fine), min(y_fine) - 0.3)]),
        fill="toself",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Smooth energy curve.
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_fine,
        mode="lines",
        line=dict(color="#00D4FF", width=2.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Zero-energy reference line.
    fig.add_shape(type="line", x0=-0.3, x1=n - 0.7, y0=0, y1=0,
                  line=dict(color="#333", width=1, dash="dot"))

    # Horizontal plateau bars at each intermediate.
    hw = 0.20
    for i, (_, E) in enumerate(ints):
        fig.add_shape(
            type="line",
            x0=i - hw, x1=i + hw, y0=E, y1=E,
            line=dict(color="#FFFFFF", width=3),
        )

    # Intermediate scatter (markers + energy labels).
    for i, (label, E) in enumerate(ints):
        color = "#FF6B6B" if i in (rls, rls + 1) else "#A8FF78"
        tpos  = "top center" if i % 2 == 0 else "bottom center"
        fig.add_trace(go.Scatter(
            x=[float(i)], y=[E],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(width=2, color="#fff")),
            text=[f"{E:+.2f}"],
            textposition=tpos,
            textfont=dict(size=9, color=color),
            hovertemplate=f"<b>{label}</b><br>E = {E:+.3f} eV<extra></extra>",
            showlegend=False,
        ))

    # Transition-state markers.
    for i, E_ts in enumerate(ts_e):
        mc  = "#FF453A" if i == rls else "#FF9F0A"
        fig.add_trace(go.Scatter(
            x=[i + 0.5], y=[E_ts],
            mode="markers",
            marker=dict(size=11, color=mc, symbol="diamond",
                        line=dict(width=2, color="#fff")),
            hovertemplate=(
                f"<b>TS {i + 1}</b><br>"
                f"E_ts = {E_ts:+.3f} eV<br>"
                f"Eₐ = {barriers[i]:.3f} eV"
                + (" ← rate-limiting" if i == rls else "")
                + "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Ea bracket for rate-limiting step.
    x_rls  = float(rls)
    x_ts   = rls + 0.5
    E_left = ints[rls][1]
    E_ts   = ts_e[rls]
    fig.add_shape(type="line",
                  x0=x_rls, x1=x_rls, y0=E_left, y1=E_ts,
                  line=dict(color="#FF453A", width=1.5, dash="dash"))
    fig.add_shape(type="line",
                  x0=x_rls, x1=x_ts, y0=E_ts, y1=E_ts,
                  line=dict(color="#FF453A", width=1, dash="dot"))
    fig.add_annotation(
        x=x_rls + 0.08, y=(E_left + E_ts) / 2,
        text=f"Eₐ = {Ea_rls:.2f} eV<br>(rate-limiting)",
        showarrow=False,
        font=dict(size=10, color="#FF453A"),
        bgcolor="rgba(255,69,58,0.15)",
        bordercolor="#FF453A",
        borderwidth=1,
        borderpad=4,
        xanchor="left",
    )

    # ΔG annotation near the product.
    fig.add_annotation(
        x=n - 1, y=ints[-1][1],
        text=f"ΔG = {delta_G:+.2f} eV",
        showarrow=True,
        arrowhead=2,
        arrowcolor=dg_color,
        ax=40, ay=0,
        font=dict(size=10, color=dg_color),
        bgcolor=f"rgba({'48,209,88' if delta_G < 0 else '255,69,58'},0.15)",
        bordercolor=dg_color,
        borderwidth=1,
        borderpad=4,
    )

    # Legend dummy traces.
    for color, sym, lbl in [
        ("#A8FF78", "circle",  "Intermediate"),
        ("#FF6B6B", "circle",  "Rate-limiting intermediate"),
        ("#FF453A", "diamond", "Rate-limiting TS"),
        ("#FF9F0A", "diamond", "Transition state"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, symbol=sym, size=9),
            name=lbl,
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Reaction Energy Profile</b> — {data['equation']}<br>"
                f"<sup>Catalyst: {catalyst.get('name', '')} · "
                f"ΔG = {delta_G:+.2f} eV · "
                f"Eₐ (RLS) = {Ea_rls:.2f} eV</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=[lbl for lbl, _ in ints],
            tickangle=-35,
            tickfont=dict(size=9),
            gridcolor="#1A1A1A",
            range=[-0.4, n - 0.6],
        ),
        yaxis=dict(
            title="Relative Energy (eV)",
            gridcolor="#1A1A1A",
            zeroline=True,
            zerolinecolor="#444",
            zerolinewidth=1,
        ),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA", family="Inter"),
        legend=dict(
            bgcolor="rgba(20,20,30,0.85)",
            bordercolor="#333",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        height=520,
        margin=dict(l=60, r=80, t=100, b=130),
    )
    return fig
