"""
feedback.py — Experiment logging, active learning, and model improvement loop.
Backed by SQLAlchemy repository (SQLite default, Postgres via DATABASE_URL).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.db.repository import get_repo

DB_PATH = Path(__file__).parent.parent / "data" / "chemai.db"


def init_db():
    get_repo().init_db()


def log_experiment(exp_type: str, name: str, pred_value: float,
                   actual_value: float, metric: str,
                   notes: str = "", composition: dict | None = None,
                   user: str = "anonymous", version_tag: str = "",
                   data_quality: str = "good",
                   source_provenance: str = "internal_experiment",
                   reaction_run_id: int | None = None) -> int:
    return get_repo().log_experiment(
        exp_type=exp_type, name=name, pred_value=pred_value,
        actual_value=actual_value, metric=metric, notes=notes,
        composition=composition, user=user, version_tag=version_tag,
        data_quality=data_quality, source_provenance=source_provenance,
        reaction_run_id=reaction_run_id,
    )


def log_scenario_run(pathway_name: str, scenario: dict, predicted_yield: float,
                     uncertainty: float, risk_score: float,
                     chosen_plan: str = "", expected_gain: float = 0.0):
    get_repo().log_scenario_run(
        pathway_name, scenario, predicted_yield, uncertainty,
        risk_score, chosen_plan, expected_gain,
    )


def queue_experiment(exp_type: str, candidate_name: str, plan_text: str,
                     predicted_value: float, risk_score: float,
                     payload: dict | None = None):
    get_repo().queue_experiment(
        exp_type, candidate_name, plan_text,
        predicted_value, risk_score, payload,
    )


def get_experiment_queue(status: str | None = None) -> pd.DataFrame:
    return get_repo().get_experiment_queue(status)


def complete_queued_experiment(queue_id: int, actual_value: float, notes: str = ""):
    get_repo().complete_queued_experiment(queue_id, actual_value, notes)


def get_scenario_runs(limit: int = 50) -> pd.DataFrame:
    return get_repo().get_scenario_runs(limit)


def leaderboard_by_impact(limit: int = 10) -> pd.DataFrame:
    runs = get_scenario_runs(limit=500)
    if runs.empty:
        return pd.DataFrame(columns=[
            "Pathway", "Runs", "Avg Pred Yield", "Avg Gain", "Avg Risk", "Impact Score"
        ])
    grouped = runs.groupby("pathway_name", as_index=False).agg(
        runs=("id", "count"),
        avg_pred_yield=("predicted_yield", "mean"),
        avg_gain=("expected_gain", "mean"),
        avg_risk=("risk_score", "mean"),
    )
    grouped["impact_score"] = (
        grouped["avg_pred_yield"] + 0.8 * grouped["avg_gain"] - 0.4 * grouped["avg_risk"]
    )
    grouped = grouped.sort_values("impact_score", ascending=False).head(limit)
    return grouped.rename(columns={
        "pathway_name": "Pathway", "runs": "Runs",
        "avg_pred_yield": "Avg Pred Yield", "avg_gain": "Avg Gain",
        "avg_risk": "Avg Risk", "impact_score": "Impact Score",
    })


def get_experiments(exp_type: str | None = None) -> pd.DataFrame:
    return get_repo().get_experiments(exp_type)


def get_model_versions(exp_type: str | None = None) -> pd.DataFrame:
    return get_repo().get_model_versions(exp_type)


def record_retrain(exp_type: str, mae: float, rmse: float, n_samples: int,
                   model_name: str = "default", task: str = "yield",
                   promoted: bool = False, metrics: dict | None = None):
    get_repo().record_retrain(
        exp_type, mae, rmse, n_samples,
        model_name=model_name, task=task, promoted=promoted, metrics=metrics,
    )


def compute_metrics(exp_type: str) -> dict:
    df = get_experiments(exp_type)
    if df.empty or "pred_value" not in df.columns:
        return {"mae": None, "rmse": None, "n": 0}
    err = df["actual_value"] - df["pred_value"]
    return {
        "mae": round(float(err.abs().mean()), 4),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
        "n": len(df),
    }


def get_al_suggestions(candidates: list, top_k: int = 3) -> list:
    scored = [c for c in candidates if "uncertainty" in c]
    if not scored:
        return candidates[:top_k]
    return sorted(scored, key=lambda c: c["uncertainty"], reverse=True)[:top_k]


def add_annotation(user: str, text: str, exp_id: int | None = None,
                   exp_type: str = "", target_name: str = "",
                   reaction_run_id: int | None = None):
    get_repo().add_annotation(
        user, text, exp_id, exp_type, target_name, reaction_run_id,
    )


def get_annotations(exp_id: int | None = None, exp_type: str | None = None) -> pd.DataFrame:
    return get_repo().get_annotations(exp_id, exp_type)


def get_user_activity() -> pd.DataFrame:
    return get_repo().get_user_activity()


def get_provenance_summary() -> pd.DataFrame:
    return get_repo().get_provenance_summary()


def get_experiment_history(name: str, exp_type: str) -> pd.DataFrame:
    return get_repo().get_experiment_history(name, exp_type)


def plot_predicted_vs_actual(exp_type: str | None = None) -> go.Figure:
    df = get_experiments(exp_type)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No experiments yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig
    fig = go.Figure()
    color_map = {
        "activity": "#00D4FF", "yield": "#A8FF78",
        "stability": "#FFD700", "reaction_yield": "#BF5AF2",
    }
    for metric in df["metric"].dropna().unique():
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["pred_value"], y=sub["actual_value"],
            mode="markers+text",
            marker=dict(size=11, color=color_map.get(metric, "#888"),
                        line=dict(width=1.5, color="#fff")),
            text=sub["name"].str[:15], textposition="top center",
            textfont=dict(size=8), name=str(metric).capitalize(),
            hovertemplate="<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
        ))
    all_vals = pd.concat([df["pred_value"], df["actual_value"]]).dropna()
    lo, hi = float(all_vals.min()) - 0.05, float(all_vals.max()) + 0.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#555", dash="dash"), name="Perfect prediction",
    ))
    fig.update_layout(
        title="Predicted vs Actual (All Experiments)",
        xaxis_title="Predicted", yaxis_title="Actual",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1A2E"), height=420,
    )
    return fig


def plot_model_improvement(exp_type: str) -> go.Figure:
    df = get_model_versions(exp_type)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No model history yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["mae"], mode="lines+markers", name="MAE",
        line=dict(color="#00D4FF", width=2), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rmse"], mode="lines+markers", name="RMSE",
        line=dict(color="#FF6B6B", width=2), marker=dict(size=8),
    ))
    fig.update_layout(
        title=f"Model Accuracy Over Time ({exp_type.capitalize()})",
        yaxis_title="Error", xaxis_title="Retrain date",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1A2E"), height=350,
    )
    return fig


def plot_experiment_timeline(exp_type: str | None = None) -> go.Figure:
    df = get_experiments(exp_type)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No experiments yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig
    df = df.copy()
    df["error"] = (df["actual_value"] - df["pred_value"]).abs()
    fig = px.scatter(
        df, x="timestamp", y="error", color="exp_type", symbol="metric",
        hover_data=["name", "pred_value", "actual_value"],
        title="Absolute Prediction Error Over Time",
        labels={"error": "|Actual − Predicted|", "timestamp": "Date"},
        color_discrete_map={"catalyst": "#00D4FF", "bio": "#A8FF78", "reaction": "#BF5AF2"},
    )
    fig.update_layout(
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1A2E"), height=380,
    )
    return fig


def plot_provenance_chart() -> go.Figure:
    df = get_provenance_summary()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No provenance data yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig
    label_map = {
        "internal_experiment": "Internal Experiment",
        "published_paper": "Published Paper",
        "screening": "Screening Campaign",
        "db_retrieved": "DB Retrieved",
        "ai_simulation": "AI Simulation",
    }
    quality_colors = {"good": "#30D158", "uncertain": "#FF9F0A", "outlier": "#FF453A"}
    fig = go.Figure()
    for quality in ["good", "uncertain", "outlier"]:
        sub = df[df["data_quality"] == quality]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["source_provenance"].map(lambda x: label_map.get(str(x), str(x))),
            y=sub["count"], name=quality.capitalize(),
            marker_color=quality_colors.get(quality, "#888"),
        ))
    fig.update_layout(
        title="Experiments by Source & Data Quality", barmode="stack",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"), legend=dict(bgcolor="#1A1A2E"),
        height=340, xaxis_tickangle=-15,
    )
    return fig


_HYPO_RULES: dict[tuple, str] = {
    ("catalyst", "over", "multi"): (
        "Multi-element dopant may segregate or form inactive phases under reaction conditions."
    ),
    ("catalyst", "over", "binary"): (
        "Strong intermediate binding or surface reconstruction not captured by descriptors."
    ),
    ("catalyst", "under", "multi"): (
        "Multi-element synergy not encoded in single-element descriptors."
    ),
    ("catalyst", "under", "binary"): (
        "In-situ surface reconstruction may create additional active sites."
    ),
    ("bio", "over", None): (
        "Product inhibition or competing metabolic flux may reduce actual yield."
    ),
    ("bio", "under", None): (
        "Unexpected enzyme upregulation may boost actual yield."
    ),
    ("reaction", "over", None): (
        "Side reactions or decomposition not captured by the template may reduce yield."
    ),
    ("reaction", "under", None): (
        "Favorable conditions or catalyst activation may exceed template prediction."
    ),
}


def _generate_hypothesis(row: pd.Series) -> str:
    exp_type = str(row.get("exp_type", ""))
    actual = float(row.get("actual_value", 0) or 0)
    pred = float(row.get("pred_value", 0) or 0)
    direction = "under" if actual > pred else "over"
    abs_err = abs(actual - pred)
    comp: dict = {}
    try:
        comp = json.loads(row.get("composition", "{}") or "{}")
    except Exception:
        pass
    n_el = len(comp)
    el_class = "multi" if n_el >= 3 else "binary"
    dominant = max(comp, key=lambda k: comp.get(k) or 0.0) if comp else "unknown"
    key = (exp_type, direction, el_class if exp_type == "catalyst" else None)
    fallback = (exp_type, direction, None)
    body = _HYPO_RULES.get(key) or _HYPO_RULES.get(fallback, "Investigate potential model bias.")
    return f"|Error| = {abs_err:.3f} ({direction}-predicted). Dominant: {dominant}. {body}"


def flag_discrepancies(exp_type: str | None = None, threshold: float = 0.02) -> pd.DataFrame:
    df = get_experiments(exp_type)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["error"] = (
        pd.to_numeric(df["actual_value"], errors="coerce")
        - pd.to_numeric(df["pred_value"], errors="coerce")
    )
    df["abs_error"] = df["error"].abs()
    flagged = df[df["abs_error"] > threshold].copy()
    if flagged.empty:
        return flagged
    flagged["flag"] = flagged["error"].apply(
        lambda e: "UNDER-PREDICTED" if e > 0 else "OVER-PREDICTED"
    )
    flagged["hypothesis"] = flagged.apply(_generate_hypothesis, axis=1)
    return flagged.sort_values("abs_error", ascending=False).reset_index(drop=True)


def export_ranked_csv(candidates: list) -> bytes:
    rows = []
    for i, c in enumerate(candidates):
        unc = c.get("uncertainty", "")
        rows.append({
            "Rank": i + 1, "Name": c.get("name", ""),
            "Formula": c.get("formula", ""),
            "Composite Score": round(c.get("composite_score", 0), 4),
            "Activity Score": round(c.get("activity_score", 0), 4),
            "Uncertainty": round(unc, 4) if isinstance(unc, float) else unc,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def export_ranked_json(candidates: list) -> bytes:
    safe = []
    for i, c in enumerate(candidates):
        row = {k: v for k, v in c.items() if k != "composition"}
        row["rank"] = i + 1
        safe.append(row)
    return json.dumps(safe, indent=2).encode("utf-8")


def generate_candidates_sdf(candidates: list) -> str:
    blocks = []
    for i, c in enumerate(candidates):
        props = [
            ("RANK", str(i + 1)),
            ("FORMULA", str(c.get("formula", ""))),
            ("COMPOSITE_SCORE", str(round(c.get("composite_score", 0), 4))),
        ]
        lines = [
            c.get("name", f"candidate_{i+1}"),
            f"  ChemAI  {datetime.now().strftime('%m%d%y%H%M')}",
            "", "  0  0  0  0  0  0  0  0  0  0999 V2000", "M  END",
        ]
        for tag, val in props:
            lines += [f">  <{tag}>", val, ""]
        lines.append("$$$$")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def generate_lab_report(candidates: list, reaction: str, top_n: int = 5) -> str:
    sep = "=" * 64
    lines = [
        sep, "ChemAI — Catalyst Candidate Lab Brief",
        f"Reaction  : {reaction}",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Top-{top_n} candidates recommended for experimental testing", sep, "",
    ]
    for i, c in enumerate(candidates[:top_n]):
        lines += [
            f"#{i+1}  {c.get('name', '—')}",
            f"    Formula         : {c.get('formula', '—')}",
            f"    Composite Score : {c.get('composite_score', 0):.4f}", "",
        ]
    lines += ["-" * 64, "Generated by ChemAI — Open Chemistry Reaction Library"]
    return "\n".join(lines)


def export_reaction_rxn(run: dict) -> str:
    """Export a reaction run as RXN format."""
    cond = run.get("conditions", {})
    pred = run.get("predicted", {})
    reactants = pred.get("reactants", [])
    products = pred.get("products", [])
    lines = [
        "$RXN", "", f"ChemAI Run #{run.get('id', '')}", "",
        f"  {len(reactants):3d}  {len(products):3d}",
    ]
    for _ in reactants:
        lines.append("$MOL")
        lines.append("Reactant")
    for _ in products:
        lines.append("$MOL")
        lines.append("Product")
    lines.append(f"# Conditions: T={cond.get('temperature_c', 'RT')}C")
    return "\n".join(lines)
