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

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            exp_type    TEXT    NOT NULL,   -- 'catalyst' | 'bio'
            name        TEXT    NOT NULL,
            pred_value  REAL,               -- model prediction
            actual_value REAL,              -- what the lab measured
            metric      TEXT,               -- 'activity' | 'yield' | 'stability'
            notes       TEXT,
            composition TEXT                -- JSON string (optional)
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
    con.commit()
    con.close()
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
        # (timestamp, exp_type, name, pred, actual, metric, notes, comp)
        ("2026-04-10 09:00", "catalyst", "Cu/ZnO/Al₂O₃",    0.78, 0.76, "activity",  "Standard run, baseline.", "{}"),
        ("2026-04-11 11:30", "catalyst", "Pd/In₂O₃",         0.85, 0.83, "activity",  "Good agreement.", "{}"),
        ("2026-04-12 14:00", "catalyst", "In₂O₃/ZrO₂",       0.71, 0.74, "activity",  "Slightly under-predicted.", "{}"),
        ("2026-04-13 10:00", "catalyst", "Cu-Zn-Ga/Al₂O₃",   0.80, 0.77, "activity",  "Ga doping confirmed.", "{}"),
        ("2026-04-14 15:00", "bio",      "Glucose → Ethanol", 0.51, 0.49, "yield",     "Close to theoretical.", "{}"),
        ("2026-04-15 09:30", "bio",      "Glucose → Isobutanol", 0.41, 0.37, "yield",  "Lower than predicted.", "{}"),
        ("2026-04-16 11:00", "bio",      "Fatty Acids → Biodiesel", 0.90, 0.88, "yield", "Good result.", "{}"),
        ("2026-04-17 14:30", "catalyst", "CoP/Carbon",        0.83, 0.80, "activity",  "HER test.", "{}"),
        ("2026-04-18 10:00", "catalyst", "Ni-Fe/CeO₂",        0.84, 0.86, "activity",  "Slightly over-performed.", "{}"),
        ("2026-04-20 16:00", "bio",      "Glucose → Lactic Acid", 0.88, 0.91, "yield", "Exceeded prediction.", "{}"),
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
        " metric, notes, composition) VALUES (?,?,?,?,?,?,?,?)", seed_rows
    )
    cur.executemany(
        "INSERT INTO model_versions (timestamp, exp_type, mae, rmse, n_samples)"
        " VALUES (?,?,?,?,?)", seed_model_rows
    )
    con.commit()
    con.close()


# ─── CRUD helpers ─────────────────────────────────────────────────────────────

def log_experiment(exp_type: str, name: str, pred_value: float,
                   actual_value: float, metric: str,
                   notes: str = "", composition: dict | None = None):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO experiments (timestamp, exp_type, name, pred_value, actual_value,"
        " metric, notes, composition) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            exp_type, name, pred_value, actual_value, metric,
            notes, json.dumps(composition or {}),
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
