"""Model prediction and retraining API routes."""
from fastapi import APIRouter, Depends

from api.deps import verify_api_key
from api.schemas import RetrainResponse
from infra.retrain_jobs.retrain_worker import run_retrain_job
from modules.ml.yield_model import get_yield_model

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/predict")
def predict_yield(
    reactants: str,
    products: str = "",
    temperature_c: float = 25,
    _=Depends(verify_api_key),
):
    r_list = [s.strip() for s in reactants.split(",") if s.strip()]
    p_list = [s.strip() for s in products.split(",") if s.strip()]
    model = get_yield_model()
    result = model.predict(r_list, p_list, {"temperature_c": temperature_c})
    return result


@router.post("/retrain", response_model=RetrainResponse)
def retrain_models(_=Depends(verify_api_key)):
    result = run_retrain_job(force=True)
    return RetrainResponse(
        status=result.get("status", "unknown"),
        mae=result.get("mae"),
        rmse=result.get("rmse"),
        promoted=result.get("promoted", False),
        n_samples=result.get("n_samples", 0),
    )
