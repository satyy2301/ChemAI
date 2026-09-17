"""Unified outcome schema for catalyst, bio, and reaction domains."""
from typing import TypedDict


class OutcomeSchema(TypedDict, total=False):
    yield_value: float
    selectivity: float
    conversion: float
    ee_percent: float | None
    side_products: list[str]
    rate: float | None
    energy_barrier_ev: float | None
    toxicity_flags: list[str]
    cost_estimate: float | None
    uncertainty: float
    reactants: list[str]
    products: list[str]
    byproducts: list[str]
    engine_tier: int
    confidence: float


def empty_outcome() -> dict:
    return {
        "yield": 0.0,  # canonical key used in JSON storage
        "yield_value": 0.0,
        "selectivity": 0.0, "conversion": 0.0,
        "ee_percent": None, "side_products": [], "rate": None,
        "energy_barrier_ev": None, "toxicity_flags": [],
        "cost_estimate": None, "uncertainty": 0.0,
        "reactants": [], "products": [], "byproducts": [],
        "engine_tier": 1, "confidence": 0.0,
    }


def from_catalyst(activity: float, selectivity: float = 0.0,
                  uncertainty: float = 0.0) -> dict:
    o = empty_outcome()
    o["conversion"] = activity
    o["selectivity"] = selectivity
    o["yield"] = activity * selectivity if selectivity else activity
    o["uncertainty"] = uncertainty
    return o


def from_bio(yield_val: float, uncertainty: float = 0.0) -> dict:
    o = empty_outcome()
    o["yield"] = yield_val
    o["conversion"] = yield_val
    o["uncertainty"] = uncertainty
    return o
