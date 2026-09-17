"""Reaction run API routes."""
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_db, verify_api_key
from api.schemas import SubmitResultsRequest
from modules import reaction_library as rl
from modules import feedback as fb
from modules.quality import compute_quality_score, is_training_eligible, score_and_gate_run

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}")
def get_run(run_id: int, _=Depends(verify_api_key)):
    run = rl.get_reaction_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@router.post("/{run_id}/results")
def submit_results(run_id: int, req: SubmitResultsRequest,
                   _=Depends(verify_api_key)):
    run = rl.get_reaction_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    actual = {"yield": req.yield_value}
    if req.selectivity is not None:
        actual["selectivity"] = req.selectivity

    run_for_score = {**run, "actual": actual}
    gated = score_and_gate_run(run_for_score, req.data_quality, "api")
    qs = gated["quality_score"]

    rl.update_reaction_run_actual(run_id, actual, qs)

    pred_yield = run.get("predicted", {}).get("yield", 0)
    fb.log_experiment(
        exp_type="reaction",
        name=f"Run #{run_id}",
        pred_value=pred_yield,
        actual_value=req.yield_value,
        metric="reaction_yield",
        notes=req.notes,
        user="api",
        data_quality=req.data_quality,
        source_provenance=req.provenance,
        reaction_run_id=run_id,
    )

    if gated["training_eligible"]:
        get_db().increment_reputation("api", 0.1)

    return {"run_id": run_id, "quality_score": qs, "training_eligible": gated["training_eligible"]}


@router.post("/{run_id}/flag")
def flag_run(run_id: int, reason: str = "", _=Depends(verify_api_key)):
    get_db().add_flag("reaction_run", run_id, "api", reason)
    return {"flagged": True, "run_id": run_id}
