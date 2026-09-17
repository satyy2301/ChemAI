"""Automated model retraining with promote-on-improve logic."""
import json

from modules.db.repository import get_repo
from modules import feedback as fb
from modules.ml.yield_model import get_yield_model

MIN_SAMPLES = 3
PROMOTE_THRESHOLD = 10  # new eligible runs since last promote


def _evaluate_holdout(runs: list[dict]) -> tuple[float, float]:
    """Time-ordered 80/20 holdout MAE/RMSE on yield."""
    import numpy as np
    if len(runs) < MIN_SAMPLES:
        return 1.0, 1.0
    split = max(1, int(len(runs) * 0.8))
    train, test = runs[:split], runs[split:]
    if not test:
        test = train[-1:]
        train = train[:-1]
    model = get_yield_model()
    result = model.retrain(train)
    if result.get("status") != "ok":
        return 1.0, 1.0
    errors = []
    for run in test:
        pred = run.get("predicted", {})
        actual = run.get("actual", {})
        if actual.get("yield") is None:
            continue
        p = model.predict(
            pred.get("reactants", []),
            pred.get("products", []),
            run.get("conditions", {}),
        )
        errors.append(abs(p["yield"] - actual["yield"]))
    if not errors:
        return result.get("mae", 1.0), result.get("rmse", 1.0)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    return mae, rmse


def run_retrain_job(force: bool = False) -> dict:
    """Retrain yield model on quality-gated runs; promote only if improved."""
    repo = get_repo()
    eligible = repo.get_training_eligible_runs(min_quality=0.6)

    if len(eligible) < MIN_SAMPLES and not force:
        return {"status": "insufficient_data", "n_samples": len(eligible), "promoted": False}

    mae, rmse = _evaluate_holdout(eligible)
    latest = repo.get_latest_promoted_model("reaction", "yield")
    prev_mae = latest.get("mae", 1.0) if latest else 1.0

    promoted = mae < prev_mae or latest is None
    if promoted:
        model = get_yield_model()
        model.retrain(eligible)
        fb.record_retrain(
            "reaction", mae=mae, rmse=rmse, n_samples=len(eligible),
            model_name="yield_rf", task="yield", promoted=True,
            metrics={"holdout_mae": mae, "holdout_rmse": rmse},
        )
        status = "promoted"
    else:
        fb.record_retrain(
            "reaction", mae=mae, rmse=rmse, n_samples=len(eligible),
            model_name="yield_rf", task="yield", promoted=False,
            metrics={"holdout_mae": mae, "holdout_rmse": rmse, "reason": "no_improvement"},
        )
        status = "no_improvement"

    # Also retrain catalyst/bio if enough experiments
    cat_exps = fb.get_experiments("catalyst")
    if len(cat_exps) >= MIN_SAMPLES:
        from modules import catalyst_module as cm
        extra = []
        for _, row in cat_exps.iterrows():
            comp = {}
            try:
                comp = json.loads(row.get("composition", "{}"))
            except Exception:
                comp = {"Cu": 0.6}
            extra.append({
                "composition": comp,
                "adsorption_energy": row["actual_value"] * -1,
                "stability_score": row["actual_value"],
                "activity_score": row["actual_value"],
            })
        cm.get_predictor().retrain(extra, cm.load_catalysts())

    bio_exps = fb.get_experiments("bio")
    if len(bio_exps) >= MIN_SAMPLES:
        from modules import bio_module as bm
        extra_paths = []
        for _, row in bio_exps.iterrows():
            extra_paths.append({
                "steps": [{"efficiency": row["actual_value"]}] * 4,
                "difficulty": "Medium",
                "yield_g_per_g": row["actual_value"],
            })
        bm.get_bio_predictor().retrain(extra_paths, bm.load_pathways())

    return {
        "status": status,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "promoted": promoted,
        "n_samples": len(eligible),
    }


def process_job_queue():
    """Process pending jobs from job_queue table."""
    repo = get_repo()
    jobs = repo.get_pending_jobs(limit=5)
    for job in jobs:
        if job["job_type"] == "retrain_yield":
            result = run_retrain_job(force=True)
            repo.complete_job(job["id"], result)
        elif job["job_type"] == "benchmark":
            from modules.benchmarks.runner import run_benchmark
            suite = job["payload"].get("suite_name", "organic_smarts")
            result = run_benchmark(suite)
            repo.complete_job(job["id"], result)
        elif job["job_type"] == "external_sim":
            from modules.integrations.external_sim import run_external_simulation
            p = job["payload"]
            result = run_external_simulation(
                p.get("reactants", []), p.get("products", []),
                p.get("conditions", {}),
            )
            repo.complete_job(job["id"], result)
        else:
            repo.complete_job(job["id"], {"status": "unknown_job_type"}, status="failed")
