"""Reaction API routes."""
from fastapi import APIRouter, Depends, Query

from api.deps import get_db, verify_api_key
from api.schemas import CreateReactionRequest, RunReactionRequest, RunReactionResponse, SearchResponse
from modules import reaction_engine as re
from modules import reaction_library as rl

router = APIRouter(prefix="/reactions", tags=["reactions"])


@router.post("/run", response_model=RunReactionResponse)
def run_reaction(req: RunReactionRequest, _=Depends(verify_api_key)):
    rl.seed_library()
    rxn_smarts = req.reaction_smarts
    base_yield = req.base_yield
    if req.reaction_id:
        rxn = rl.get_reaction(req.reaction_id)
        if rxn:
            rxn_smarts = rxn["rxn_smarts"]
            base_yield = rxn.get("base_yield", base_yield)

    result = re.run_reaction(
        reactants=req.reactants,
        reaction_id=req.reaction_id,
        reaction_smarts=rxn_smarts,
        conditions=req.conditions,
        base_yield=base_yield,
    )

    run_id = None
    if result.validity:
        run_id = rl.create_reaction_run(
            reaction_id=req.reaction_id,
            conditions=req.conditions,
            predicted=result.predicted_outcome,
            user_id="api",
        )

    return RunReactionResponse(
        products=result.products,
        byproducts=result.byproducts,
        validity=result.validity,
        confidence=result.confidence,
        engine_tier=result.engine_tier,
        predicted_outcome=result.predicted_outcome,
        warnings=result.warnings,
        run_id=run_id,
    )


@router.get("/search", response_model=SearchResponse)
def search_reactions(
    q: str = Query("", alias="q"),
    domain: str | None = None,
    tag: str | None = None,
    _=Depends(verify_api_key),
):
    rl.seed_library()
    results = rl.search_reactions(q, domain, tag)
    return SearchResponse(reactions=results, count=len(results))


@router.post("/")
def create_reaction(req: CreateReactionRequest, _=Depends(verify_api_key)):
    rid = rl.create_reaction(
        req.name, req.rxn_smarts, req.reactants, req.products,
        req.domain, req.tags, created_by="api", base_yield=req.base_yield,
    )
    return {"id": rid, "name": req.name}


@router.post("/{reaction_id}/fork")
def fork_reaction(reaction_id: int, _=Depends(verify_api_key)):
    new_id = rl.fork_reaction(reaction_id, "api")
    if not new_id:
        return {"error": "Reaction not found"}
    return {"id": new_id, "forked_from": reaction_id}
