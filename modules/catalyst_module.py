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

    for i, row in df.iterrows():
        is_ai = row.get("source") == "AI-generated"
        marker_size = _safe_float(sizes[i], fallback=14.0)
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
                color=colors[i],
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
