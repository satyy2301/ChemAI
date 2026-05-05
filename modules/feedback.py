"""
feedback.py — Experiment logging, active learning, and model improvement loop.
Backed by SQLite (no server needed).
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DB_PATH = Path(__file__).parent.parent / "data" / "experiments.db"


# ─── DB initialisation ────────────────────────────────────────────────────────

def _migrate_db():
    """Add provenance/collaboration columns to experiments table for backward compat."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    for col, defn in [
        ("user",              "TEXT DEFAULT 'anonymous'"),
        ("version_tag",       "TEXT DEFAULT 'v1'"),
        ("data_quality",      "TEXT DEFAULT 'good'"),
        ("source_provenance", "TEXT DEFAULT 'internal_experiment'"),
    ]:
        try:
            cur.execute(f"ALTER TABLE experiments ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    con.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            exp_type         TEXT    NOT NULL,
            name             TEXT    NOT NULL,
            pred_value       REAL,
            actual_value     REAL,
            metric           TEXT,
            notes            TEXT,
            composition      TEXT,
            user             TEXT    DEFAULT 'anonymous',
            version_tag      TEXT    DEFAULT 'v1',
            data_quality     TEXT    DEFAULT 'good',
            source_provenance TEXT   DEFAULT 'internal_experiment'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            exp_type    TEXT    NOT NULL,
            mae         REAL,
            rmse        REAL,
            n_samples   INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scenario_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            pathway_name    TEXT    NOT NULL,
            scenario_json   TEXT    NOT NULL,
            predicted_yield REAL,
            uncertainty     REAL,
            risk_score      REAL,
            chosen_plan     TEXT,
            expected_gain   REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            exp_type        TEXT    NOT NULL,
            candidate_name  TEXT    NOT NULL,
            plan_text       TEXT,
            predicted_value REAL,
            risk_score      REAL,
            payload_json    TEXT,
            status          TEXT    NOT NULL DEFAULT 'queued',
            actual_value    REAL,
            notes           TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            user        TEXT    NOT NULL,
            exp_id      INTEGER,
            exp_type    TEXT,
            target_name TEXT,
            text        TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()
    _migrate_db()
    _seed_demo_data()


def _seed_demo_data():
    """Insert a few starter rows so the dashboard has something to show."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments")
    if cur.fetchone()[0] > 0:
        con.close()
        return  # Already seeded

    seed_rows = [
        # (ts, exp_type, name, pred, actual, metric, notes, comp, user, vtag, quality, provenance)
        ("2026-04-10 09:00", "catalyst", "Cu/ZnO/Al₂O₃",        0.78, 0.76, "activity", "Standard run, baseline.",   "{}", "alice", "v1", "good",      "internal_experiment"),
        ("2026-04-11 11:30", "catalyst", "Pd/In₂O₃",             0.85, 0.83, "activity", "Good agreement.",           "{}", "bob",   "v1", "good",      "published_paper"),
        ("2026-04-12 14:00", "catalyst", "In₂O₃/ZrO₂",           0.71, 0.74, "activity", "Slightly under-predicted.", "{}", "alice", "v1", "good",      "internal_experiment"),
        ("2026-04-13 10:00", "catalyst", "Cu-Zn-Ga/Al₂O₃",       0.80, 0.77, "activity", "Ga doping confirmed.",      "{}", "carol", "v1", "good",      "screening"),
        ("2026-04-14 15:00", "bio",      "Glucose → Ethanol",     0.51, 0.49, "yield",    "Close to theoretical.",     "{}", "bob",   "v1", "good",      "published_paper"),
        ("2026-04-15 09:30", "bio",      "Glucose → Isobutanol",  0.41, 0.37, "yield",    "Lower than predicted.",     "{}", "carol", "v1", "uncertain", "internal_experiment"),
        ("2026-04-16 11:00", "bio",      "Fatty Acids → Biodiesel",0.90,0.88, "yield",    "Good result.",              "{}", "alice", "v1", "good",      "db_retrieved"),
        ("2026-04-17 14:30", "catalyst", "CoP/Carbon",            0.83, 0.80, "activity", "HER test.",                 "{}", "bob",   "v1", "good",      "screening"),
        ("2026-04-18 10:00", "catalyst", "Ni-Fe/CeO₂",            0.84, 0.86, "activity", "Slightly over-performed.",  "{}", "carol", "v1", "uncertain", "internal_experiment"),
        ("2026-04-20 16:00", "bio",      "Glucose → Lactic Acid", 0.88, 0.91, "yield",    "Exceeded prediction.",      "{}", "alice", "v2", "good",      "published_paper"),
    ]
    seed_model_rows = [
        ("2026-04-12 00:00", "catalyst", 0.042, 0.058, 4),
        ("2026-04-16 00:00", "catalyst", 0.031, 0.044, 6),
        ("2026-04-16 00:00", "bio",      0.038, 0.051, 3),
        ("2026-04-20 00:00", "catalyst", 0.025, 0.035, 8),
        ("2026-04-20 00:00", "bio",      0.028, 0.040, 4),
    ]
    cur.executemany(
        "INSERT INTO experiments (timestamp, exp_type, name, pred_value, actual_value,"
        " metric, notes, composition, user, version_tag, data_quality, source_provenance)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", seed_rows
    )
    cur.executemany(
        "INSERT INTO model_versions (timestamp, exp_type, mae, rmse, n_samples)"
        " VALUES (?,?,?,?,?)", seed_model_rows
    )
    seed_annotations = [
        ("2026-04-10 10:00", "alice", 1, "catalyst", "Cu/ZnO/Al₂O₃",        "Baseline confirmed. Good starting point for Ga/In doping studies."),
        ("2026-04-11 12:00", "bob",   2, "catalyst", "Pd/In₂O₃",             "Matches Liu et al. 2024 within 2.5%. Paper data source verified."),
        ("2026-04-13 11:00", "carol", 4, "catalyst", "Cu-Zn-Ga/Al₂O₃",      "Ga at 5 mol% optimal. Higher concentrations reduce long-term stability."),
        ("2026-04-15 10:00", "alice", 6, "bio",      "Glucose → Isobutanol", "Suspected product inhibition at high titers. Flagged as uncertain."),
        ("2026-04-18 11:00", "bob",   9, "catalyst", "Ni-Fe/CeO₂",           "Over-performance likely due to Fe surface segregation. Revisit prep."),
        ("2026-04-20 17:00", "carol",10, "bio",      "Glucose → Lactic Acid","Exceeded prediction after pH opt. Promoting to v2 for replication."),
    ]
    cur.executemany(
        "INSERT INTO annotations (timestamp, user, exp_id, exp_type, target_name, text)"
        " VALUES (?,?,?,?,?,?)", seed_annotations
    )
    con.commit()
    con.close()


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def log_experiment(exp_type: str, name: str, pred_value: float,
                   actual_value: float, metric: str,
                   notes: str = "", composition: dict | None = None,
                   user: str = "anonymous", version_tag: str = "",
                   data_quality: str = "good",
                   source_provenance: str = "internal_experiment"):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if not version_tag:
        cur.execute(
            "SELECT COUNT(*) FROM experiments WHERE name=? AND exp_type=?",
            (name, exp_type),
        )
        n = cur.fetchone()[0]
        version_tag = f"v{n + 1}"
    cur.execute(
        "INSERT INTO experiments (timestamp, exp_type, name, pred_value, actual_value,"
        " metric, notes, composition, user, version_tag, data_quality, source_provenance)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            exp_type, name, pred_value, actual_value, metric,
            notes, json.dumps(composition or {}),
            user, version_tag, data_quality, source_provenance,
        )
    )
    con.commit()
    con.close()


def log_scenario_run(pathway_name: str, scenario: dict, predicted_yield: float,
                     uncertainty: float, risk_score: float,
                     chosen_plan: str = "", expected_gain: float = 0.0):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO scenario_runs (timestamp, pathway_name, scenario_json, predicted_yield,"
        " uncertainty, risk_score, chosen_plan, expected_gain) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            pathway_name,
            json.dumps(scenario),
            predicted_yield,
            uncertainty,
            risk_score,
            chosen_plan,
            expected_gain,
        )
    )
    con.commit()
    con.close()


def queue_experiment(exp_type: str, candidate_name: str, plan_text: str,
                     predicted_value: float, risk_score: float,
                     payload: dict | None = None):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO experiment_queue (timestamp, exp_type, candidate_name, plan_text,"
        " predicted_value, risk_score, payload_json, status) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            exp_type,
            candidate_name,
            plan_text,
            predicted_value,
            risk_score,
            json.dumps(payload or {}),
            "queued",
        )
    )
    con.commit()
    con.close()


def get_experiment_queue(status: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM experiment_queue"
    params: tuple = ()
    if status:
        q += " WHERE status = ?"
        params = (status,)
    q += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df


def complete_queued_experiment(queue_id: int, actual_value: float, notes: str = ""):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "UPDATE experiment_queue SET status = ?, actual_value = ?, notes = ? WHERE id = ?",
        ("completed", actual_value, notes, queue_id)
    )
    con.commit()
    con.close()


def get_scenario_runs(limit: int = 50) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM scenario_runs ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql_query(q, con, params=(limit,))
    con.close()
    return df


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
    grouped["impact_score"] = grouped["avg_pred_yield"] + 0.8 * grouped["avg_gain"] - 0.4 * grouped["avg_risk"]
    grouped = grouped.sort_values("impact_score", ascending=False).head(limit)
    grouped = grouped.rename(columns={
        "pathway_name": "Pathway",
        "runs": "Runs",
        "avg_pred_yield": "Avg Pred Yield",
        "avg_gain": "Avg Gain",
        "avg_risk": "Avg Risk",
        "impact_score": "Impact Score",
    })
    return grouped


def get_experiments(exp_type: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM experiments"
    params: tuple = ()
    if exp_type:
        q += " WHERE exp_type = ?"
        params = (exp_type,)
    q += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df


def get_model_versions(exp_type: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM model_versions"
    params: tuple = ()
    if exp_type:
        q += " WHERE exp_type = ?"
        params = (exp_type,)
    q += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df


# ─── Model update trigger ─────────────────────────────────────────────────────

def record_retrain(exp_type: str, mae: float, rmse: float, n_samples: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO model_versions (timestamp, exp_type, mae, rmse, n_samples)"
        " VALUES (?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), exp_type, mae, rmse, n_samples)
    )
    con.commit()
    con.close()


def compute_metrics(exp_type: str) -> dict:
    df = get_experiments(exp_type)
    if df.empty or "pred_value" not in df.columns:
        return {"mae": None, "rmse": None, "n": 0}
    err = df["actual_value"] - df["pred_value"]
    mae  = float(err.abs().mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "n": len(df)}


# ─── Active learning: uncertainty sampling ────────────────────────────────────

def get_al_suggestions(candidates: list, top_k: int = 3) -> list:
    """Return the top_k candidates with the highest uncertainty (most informative to test)."""
    scored = [c for c in candidates if "uncertainty" in c]
    if not scored:
        return candidates[:top_k]
    return sorted(scored, key=lambda c: c["uncertainty"], reverse=True)[:top_k]


# ─── Dashboard plots ─────────────────────────────────────────────────────────

def plot_predicted_vs_actual(exp_type: str | None = None) -> go.Figure:
    df = get_experiments(exp_type)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No experiments yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig

    fig = go.Figure()
    for metric, color in [("activity", "#00D4FF"), ("yield", "#A8FF78"),
                           ("stability", "#FFD700")]:
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["pred_value"],
            y=sub["actual_value"],
            mode="markers+text",
            marker=dict(size=11, color=color,
                        line=dict(width=1.5, color="#fff")),
            text=sub["name"].str[:15],
            textposition="top center",
            textfont=dict(size=8),
            name=metric.capitalize(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Predicted: %{x:.3f}<br>"
                "Actual: %{y:.3f}<extra></extra>"
            )
        ))

    # Perfect prediction line
    all_vals = pd.concat([df["pred_value"], df["actual_value"]]).dropna()
    lo, hi = float(all_vals.min()) - 0.05, float(all_vals.max()) + 0.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="#555", dash="dash"),
        name="Perfect prediction",
        showlegend=True,
    ))
    fig.update_layout(
        title="Predicted vs Actual (All Experiments)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E"),
        height=420,
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
        x=df["timestamp"], y=df["mae"],
        mode="lines+markers", name="MAE",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["rmse"],
        mode="lines+markers", name="RMSE",
        line=dict(color="#FF6B6B", width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title=f"Model Accuracy Over Time ({exp_type.capitalize()})",
        yaxis_title="Error",
        xaxis_title="Retrain date",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E"),
        height=350,
    )
    return fig


def plot_experiment_timeline(exp_type: str | None = None) -> go.Figure:
    df = get_experiments(exp_type)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No experiments yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig

    df["error"] = (df["actual_value"] - df["pred_value"]).abs()
    fig = px.scatter(
        df, x="timestamp", y="error",
        color="exp_type", symbol="metric",
        hover_data=["name", "pred_value", "actual_value"],
        title="Absolute Prediction Error Over Time",
        labels={"error": "|Actual − Predicted|", "timestamp": "Date"},
        color_discrete_map={"catalyst": "#00D4FF", "bio": "#A8FF78"},
    )
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E"),
        height=380,
    )
    return fig


# ─── Collaboration & provenance ───────────────────────────────────────────────

def add_annotation(user: str, text: str, exp_id: int | None = None,
                   exp_type: str = "", target_name: str = ""):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO annotations (timestamp, user, exp_id, exp_type, target_name, text)"
        " VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), user, exp_id, exp_type, target_name, text),
    )
    con.commit()
    con.close()


def get_annotations(exp_id: int | None = None, exp_type: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM annotations"
    conditions, params = [], []
    if exp_id is not None:
        conditions.append("exp_id = ?")
        params.append(exp_id)
    if exp_type:
        conditions.append("exp_type = ?")
        params.append(exp_type)
    if conditions:
        q += " WHERE " + " AND ".join(conditions)
    q += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df


def get_user_activity() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT user, COUNT(*) as experiments, MAX(timestamp) as last_active"
        " FROM experiments GROUP BY user ORDER BY experiments DESC",
        con,
    )
    con.close()
    return df


def get_provenance_summary() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT source_provenance, data_quality, COUNT(*) as count"
        " FROM experiments GROUP BY source_provenance, data_quality",
        con,
    )
    con.close()
    return df


def get_experiment_history(name: str, exp_type: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT id, timestamp, user, version_tag, pred_value, actual_value,"
        " data_quality, source_provenance, notes"
        " FROM experiments WHERE name=? AND exp_type=? ORDER BY timestamp ASC",
        con, params=(name, exp_type),
    )
    con.close()
    return df


def plot_provenance_chart() -> go.Figure:
    df = get_provenance_summary()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No provenance data yet", paper_bgcolor="#0E1117",
                          font=dict(color="#FAFAFA"))
        return fig
    label_map = {
        "internal_experiment": "Internal Experiment",
        "published_paper":     "Published Paper",
        "screening":           "Screening Campaign",
        "db_retrieved":        "DB Retrieved",
        "ai_simulation":       "AI Simulation",
    }
    quality_colors = {"good": "#30D158", "uncertain": "#FF9F0A", "outlier": "#FF453A"}
    fig = go.Figure()
    for quality in ["good", "uncertain", "outlier"]:
        sub = df[df["data_quality"] == quality]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["source_provenance"].map(lambda x: label_map.get(str(x), str(x))),
            y=sub["count"],
            name=quality.capitalize(),
            marker_color=quality_colors.get(quality, "#888"),
        ))
    fig.update_layout(
        title="Experiments by Source & Data Quality",
        barmode="stack",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        legend=dict(bgcolor="#1A1A2E"),
        height=340,
        xaxis_tickangle=-15,
    )
    return fig


# ─── Discrepancy flagging & automated hypothesis generation ───────────────────

_HYPO_RULES: dict[tuple, str] = {
    ("catalyst", "over",  "multi"):  (
        "Multi-element dopant may segregate or form inactive phases under reaction conditions, "
        "reducing active-site density below what static composition features predict."
    ),
    ("catalyst", "over",  "binary"): (
        "Strong intermediate binding (catalyst poisoning) or surface reconstruction not captured "
        "by weighted-average descriptors likely explains the over-prediction."
    ),
    ("catalyst", "under", "multi"):  (
        "Multi-element synergy (ensemble/ligand effects among ≥3 components) not encoded in "
        "single-element descriptors. Adding pairwise interaction features may close the gap."
    ),
    ("catalyst", "under", "binary"): (
        "In-situ surface reconstruction or a reactive phase change may create additional active "
        "sites beyond what the as-prepared composition suggests."
    ),
    ("bio", "over",  None): (
        "Product inhibition or competing metabolic flux (overflow metabolism) may reduce actual "
        "yield. Revisit kinetic constants and inhibition terms for these conditions."
    ),
    ("bio", "under", None): (
        "Unexpected enzyme upregulation or cofactor availability under these conditions may boost "
        "actual yield. Run flux-balance analysis at this operating point to confirm."
    ),
}


def _generate_hypothesis(row: pd.Series) -> str:
    exp_type  = str(row.get("exp_type", ""))
    actual    = float(row.get("actual_value", 0) or 0)
    pred      = float(row.get("pred_value",   0) or 0)
    direction = "under" if actual > pred else "over"
    abs_err   = abs(actual - pred)

    comp: dict = {}
    try:
        comp = json.loads(row.get("composition", "{}") or "{}")
    except Exception:
        pass

    n_el     = len(comp)
    el_class = "multi" if n_el >= 3 else "binary"
    dominant = max(comp, key=lambda k: comp.get(k) or 0.0) if comp else "unknown"

    key      = (exp_type, direction, el_class if exp_type == "catalyst" else None)
    fallback = (exp_type, direction, None)
    body     = _HYPO_RULES.get(key) or _HYPO_RULES.get(fallback, "Investigate potential model bias for this composition.")

    return (
        f"|Error| = {abs_err:.3f}  ({direction}-predicted).  "
        f"Dominant element: {dominant}.  {body}"
    )


def flag_discrepancies(exp_type: str | None = None,
                       threshold: float = 0.02) -> pd.DataFrame:
    """
    Return experiments where |actual − pred| > threshold, ordered worst-first.
    Each row gets a 'flag' (OVER/UNDER-PREDICTED) and an AI-generated 'hypothesis'.
    """
    df = get_experiments(exp_type)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["error"]     = (pd.to_numeric(df["actual_value"], errors="coerce")
                       - pd.to_numeric(df["pred_value"],  errors="coerce"))
    df["abs_error"] = df["error"].abs()

    flagged = df[df["abs_error"] > threshold].copy()
    if flagged.empty:
        return flagged

    flagged["flag"]       = flagged["error"].apply(
        lambda e: "UNDER-PREDICTED" if e > 0 else "OVER-PREDICTED"
    )
    flagged["hypothesis"] = flagged.apply(_generate_hypothesis, axis=1)
    return flagged.sort_values("abs_error", ascending=False).reset_index(drop=True)


# ─── Export helpers ───────────────────────────────────────────────────────────

def export_ranked_csv(candidates: list) -> bytes:
    """Return ranked candidates as UTF-8 CSV bytes for st.download_button."""
    rows = []
    for i, c in enumerate(candidates):
        unc = c.get("uncertainty", "")
        rows.append({
            "Rank":              i + 1,
            "Name":              c.get("name", ""),
            "Formula":           c.get("formula", ""),
            "Composite Score":   round(c.get("composite_score",   0), 4),
            "Activity Score":    round(c.get("activity_score",    0), 4),
            "Stability Score":   round(c.get("stability_score",   0), 4),
            "Selectivity Score": round(c.get("selectivity_score", 0), 4),
            "Adsorption E (eV)": round(c.get("adsorption_energy", 0), 4),
            "Uncertainty":       round(unc, 4) if isinstance(unc, float) else unc,
            "Surface Facet":     c.get("surface_facet", ""),
            "Source":            c.get("source", ""),
            "Reaction":          c.get("reaction", ""),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def export_ranked_json(candidates: list) -> bytes:
    """Return ranked candidates as UTF-8 JSON bytes."""
    safe = []
    for i, c in enumerate(candidates):
        row = {k: v for k, v in c.items() if k != "composition"}
        row["rank"] = i + 1
        safe.append(row)
    return json.dumps(safe, indent=2).encode("utf-8")


def generate_candidates_sdf(candidates: list) -> str:
    """
    Generate a property-only SDF (V2000, 0 atoms).
    Valid for property tagging; parseable by RDKit / OpenBabel / ChemDraw.
    """
    blocks = []
    for i, c in enumerate(candidates):
        unc = c.get("uncertainty", 0)
        unc_val = round(unc, 4) if isinstance(unc, float) else 0.0
        props = [
            ("RANK",             str(i + 1)),
            ("FORMULA",          str(c.get("formula", ""))),
            ("COMPOSITE_SCORE",  str(round(c.get("composite_score",   0), 4))),
            ("ACTIVITY_SCORE",   str(round(c.get("activity_score",    0), 4))),
            ("STABILITY_SCORE",  str(round(c.get("stability_score",   0), 4))),
            ("SELECTIVITY_SCORE",str(round(c.get("selectivity_score", 0), 4))),
            ("ADSORPTION_E_EV",  str(round(c.get("adsorption_energy", 0), 4))),
            ("UNCERTAINTY",      str(unc_val)),
            ("SURFACE_FACET",    str(c.get("surface_facet", ""))),
            ("SOURCE",           str(c.get("source", ""))),
            ("REACTION",         str(c.get("reaction", ""))),
        ]
        lines = [
            c.get("name", f"candidate_{i+1}"),
            f"  ChemAI  {datetime.now().strftime('%m%d%y%H%M')}",
            "",
            "  0  0  0  0  0  0  0  0  0  0999 V2000",
            "M  END",
        ]
        for tag, val in props:
            lines += [f">  <{tag}>", val, ""]
        lines.append("$$$$")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def generate_lab_report(candidates: list, reaction: str, top_n: int = 5) -> str:
    """Plain-text lab brief for the top-n candidates."""
    sep   = "=" * 64
    lines = [
        sep,
        "ChemAI — Catalyst Candidate Lab Brief",
        f"Reaction  : {reaction}",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Top-{top_n} candidates recommended for experimental testing",
        sep, "",
    ]
    for i, c in enumerate(candidates[:top_n]):
        unc = c.get("uncertainty", "—")
        lines += [
            f"#{i+1}  {c.get('name', '—')}",
            f"    Formula         : {c.get('formula', '—')}",
            f"    Composite Score : {c.get('composite_score',   0):.4f}",
            f"    Activity        : {c.get('activity_score',    0):.4f}",
            f"    Stability       : {c.get('stability_score',   0):.4f}",
            f"    Selectivity     : {c.get('selectivity_score', 0):.4f}",
            f"    Adsorption E    : {c.get('adsorption_energy', 0):.3f} eV",
            f"    Surface Facet   : {c.get('surface_facet', '—')}",
            f"    Uncertainty     : {round(unc, 4) if isinstance(unc, float) else unc}",
            f"    Source          : {c.get('source', '—')}",
            "",
        ]
    lines += [
        "-" * 64,
        "Generated by ChemAI — Unified AI Lab for Fuel Discovery",
        "Stack: Streamlit · scikit-learn · Plotly · SQLite",
    ]
    return "\n".join(lines)
