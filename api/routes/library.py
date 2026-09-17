"""Library browse API routes."""
from fastapi import APIRouter, Depends

from api.deps import verify_api_key
from modules import reaction_library as rl

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/")
def list_library(domain: str | None = None, _=Depends(verify_api_key)):
    rl.seed_library()
    reactions = rl.list_reactions(domain=domain)
    runs = rl.list_reaction_runs(limit=50)
    return {"reactions": reactions, "recent_runs": runs, "count": len(reactions)}
