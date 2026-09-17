"""Pydantic schemas for ChemAI API."""
from typing import Any

from pydantic import BaseModel, Field


class RunReactionRequest(BaseModel):
    reactants: list[str]
    reaction_id: int | None = None
    reaction_smarts: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    base_yield: float = 0.75


class RunReactionResponse(BaseModel):
    products: list[str]
    byproducts: list[str]
    validity: bool
    confidence: float
    engine_tier: int
    predicted_outcome: dict[str, Any]
    warnings: list[str]
    run_id: int | None = None


class SubmitResultsRequest(BaseModel):
    yield_value: float
    selectivity: float | None = None
    notes: str = ""
    data_quality: str = "good"
    provenance: str = "internal_experiment"


class CreateReactionRequest(BaseModel):
    name: str
    rxn_smarts: str
    reactants: list[str] = []
    products: list[str] = []
    domain: str = "organic"
    tags: list[str] = []
    base_yield: float = 0.75


class SearchResponse(BaseModel):
    reactions: list[dict[str, Any]]
    count: int


class RetrainResponse(BaseModel):
    status: str
    mae: float | None = None
    rmse: float | None = None
    promoted: bool = False
    n_samples: int = 0
