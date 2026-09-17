"""
reaction_engine.py — RDKit SMARTS execution, PubChem resolver, tiered yield prediction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

from modules.reactant_input import normalize_reactant_line, parse_reactant_input
from modules.schemas.outcome import empty_outcome
from modules.safety import SafetyResult, screen_molecules

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

_PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
_TIMEOUT = 10


@dataclass
class ReactionResult:
    products: list[str] = field(default_factory=list)
    byproducts: list[str] = field(default_factory=list)
    validity: bool = False
    confidence: float = 0.0
    engine_tier: int = 1
    predicted_outcome: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    safety: SafetyResult | None = None
    reactants_canonical: list[str] = field(default_factory=list)


def resolve_smiles(input_str: str) -> str:
    """Resolve SMILES from common name, formula, or PubChem lookup."""
    s = (input_str or "").strip()
    if not s:
        return ""
    normalized, _ = normalize_reactant_line(s)
    if normalized != s:
        s = normalized
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    # Try PubChem name lookup
    try:
        url = _PUBCHEM_URL.format(name=requests.utils.quote(s))
        r = requests.get(url, timeout=_TIMEOUT)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            if props and props[0].get("CanonicalSMILES"):
                return props[0]["CanonicalSMILES"]
    except Exception:
        pass
    return s


def canonicalize_smiles_list(smiles_list: list[str]) -> tuple[list[str], list[str]]:
    """Return (canonical_list, warnings)."""
    canonical = []
    warnings = []
    for smi in smiles_list:
        if not smi or not smi.strip():
            continue
        norm, note = normalize_reactant_line(smi.strip())
        if note:
            warnings.append(note)
        resolved = resolve_smiles(norm or smi.strip())
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(resolved)
            if mol is None:
                warnings.append(f"Could not parse: {smi[:40]}")
                canonical.append(resolved)
            else:
                canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        else:
            canonical.append(resolved)
    return canonical, warnings


def validate_stoichiometry(reactants: list[str], products: list[str]) -> bool:
    if not RDKIT_AVAILABLE:
        return bool(reactants and products)
    def _atom_count(smiles_list):
        counts = {}
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                counts[sym] = counts.get(sym, 0) + 1
        return counts
    r_counts = _atom_count(reactants)
    p_counts = _atom_count(products)
    if r_counts is None or p_counts is None:
        return False
    # Loose check: heavy atoms present in products should appear in reactants
    for sym, cnt in p_counts.items():
        if sym == "H":
            continue
        if r_counts.get(sym, 0) < cnt:
            return False
    return True


def _run_smarts(reactants: list[str], rxn_smarts: str) -> tuple[list[str], list[str]]:
    """Execute SMARTS reaction; return (products, warnings)."""
    warnings = []
    if not RDKIT_AVAILABLE:
        return [], ["RDKit not installed"]
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    except Exception as e:
        return [], [f"Invalid reaction SMARTS: {e}"]
    if rxn is None:
        return [], [f"Invalid reaction SMARTS: {rxn_smarts[:60]}"]
    mols = []
    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.append(f"Invalid reactant: {smi[:40]}")
            return [], warnings
        mols.append(mol)
    try:
        outcomes = rxn.RunReactants(tuple(mols))
    except Exception as e:
        return [], [f"Reaction execution failed: {e}"]
    if not outcomes:
        return [], ["No products generated from SMARTS"]
    products = []
    for outcome in outcomes[0]:
        try:
            Chem.SanitizeMol(outcome)
            products.append(Chem.MolToSmiles(outcome, isomericSmiles=True))
        except Exception:
            warnings.append("Product sanitization failed for one outcome")
    return list(set(products)), warnings


def _heuristic_yield(base_yield: float, conditions: dict) -> tuple[float, float]:
    """Tier-2 yield estimate from conditions. Returns (yield, uncertainty)."""
    yield_val = base_yield
    t = float(conditions.get("temperature_c", 25))
    if t > 150:
        yield_val *= 0.85
    elif t < 0:
        yield_val *= 0.70
    ph = float(conditions.get("ph", 7.0))
    if ph < 3 or ph > 11:
        yield_val *= 0.90
    solvent = str(conditions.get("solvent", "")).lower()
    if "polar aprotic" in solvent or "dmf" in solvent or "dmso" in solvent:
        yield_val *= 1.05
    if conditions.get("catalyst"):
        yield_val *= 1.08
    pressure = float(conditions.get("pressure_bar", 1))
    if pressure > 10:
        yield_val *= 1.03
    yield_val = min(0.99, max(0.05, yield_val))
    uncertainty = 0.08 + abs(base_yield - yield_val)
    return round(yield_val, 4), round(uncertainty, 4)


def _try_ml_yield(reactants: list[str], products: list[str],
                  conditions: dict, reaction_id: int | None) -> tuple[float, float, int] | None:
    """Tier-3 ML yield if enough training data. Returns (yield, uncertainty, tier) or None."""
    try:
        from modules.ml.yield_model import get_yield_model
        from modules.db.repository import get_repo
        eligible = get_repo().get_training_eligible_runs(min_quality=0.6)
        if len(eligible) < 10:
            return None
        model = get_yield_model()
        pred = model.predict(reactants, products, conditions)
        return pred["yield"], pred["uncertainty"], 3
    except Exception:
        return None


def _try_external_sim(reactants: list[str], products: list[str],
                      conditions: dict) -> tuple[dict, int] | None:
    """Tier-4 external simulator stub."""
    try:
        from modules.integrations.external_sim import run_external_simulation
        result = run_external_simulation(reactants, products, conditions)
        if result.get("status") == "ok":
            return result, 4
    except Exception:
        pass
    return None


def run_reaction(
    reactants: list[str],
    reaction_id: int | None = None,
    reaction_smarts: str | None = None,
    conditions: dict | None = None,
    base_yield: float = 0.75,
    skip_safety: bool = False,
) -> ReactionResult:
    """
    Execute a reaction with tiered prediction routing.

    Tier 1: RDKit SMARTS
    Tier 2: Heuristic yield
    Tier 3: ML yield model (if ≥10 training runs)
    Tier 4: External simulator (if configured)
    """
    conditions = conditions or {}
    warnings: list[str] = []

    # Resolve reactants
    reactants_canonical, parse_warnings = canonicalize_smiles_list(reactants)
    warnings.extend(parse_warnings)

    if not reactants_canonical:
        return ReactionResult(
            validity=False, warnings=["No valid reactants provided"],
            predicted_outcome=empty_outcome(),
        )

    # Safety screen
    safety = screen_molecules(reactants_canonical)
    if not skip_safety and safety.blocked:
        outcome = empty_outcome()
        outcome["toxicity_flags"] = safety.alerts
        return ReactionResult(
            validity=False, warnings=safety.alerts,
            safety=safety, reactants_canonical=reactants_canonical,
            predicted_outcome=outcome,
        )

    # Tier 1: SMARTS execution
    if not reaction_smarts:
        return ReactionResult(
            validity=False,
            warnings=["No reaction SMARTS provided"],
            reactants_canonical=reactants_canonical,
            predicted_outcome=empty_outcome(),
            safety=safety,
        )

    products, rxn_warnings = _run_smarts(reactants_canonical, reaction_smarts)
    warnings.extend(rxn_warnings)

    if not products:
        return ReactionResult(
            validity=False, confidence=0.1, engine_tier=1,
            warnings=warnings, reactants_canonical=reactants_canonical,
            predicted_outcome=empty_outcome(), safety=safety,
        )

    stoich_ok = validate_stoichiometry(reactants_canonical, products)
    if not stoich_ok:
        warnings.append("Stoichiometry check failed (atom balance)")

    confidence = 0.85 if stoich_ok else 0.55
    engine_tier = 1

    # Build outcome
    outcome = empty_outcome()
    outcome["reactants"] = reactants_canonical
    outcome["products"] = products
    outcome["engine_tier"] = engine_tier
    outcome["confidence"] = confidence
    if safety and safety.alerts:
        outcome["toxicity_flags"] = safety.alerts

    # Tier 2: Heuristic yield
    yield_val, uncertainty = _heuristic_yield(base_yield, conditions)
    engine_tier = 2
    outcome["engine_tier"] = engine_tier
    outcome["yield"] = yield_val
    outcome["selectivity"] = round(min(0.99, yield_val + 0.05), 4)
    outcome["conversion"] = yield_val
    outcome["uncertainty"] = uncertainty
    confidence = min(confidence, 0.80)

    # Tier 3: ML
    ml_result = _try_ml_yield(reactants_canonical, products, conditions, reaction_id)
    if ml_result:
        yield_val, uncertainty, engine_tier = ml_result
        outcome["yield"] = yield_val
        outcome["selectivity"] = round(min(0.99, yield_val + 0.03), 4)
        outcome["conversion"] = yield_val
        outcome["uncertainty"] = uncertainty
        outcome["engine_tier"] = engine_tier
        confidence = 0.75

    # Tier 4: External
    ext = _try_external_sim(reactants_canonical, products, conditions)
    if ext:
        ext_data, engine_tier = ext
        outcome["yield"] = ext_data.get("yield", yield_val)
        outcome["energy_barrier_ev"] = ext_data.get("energy_barrier_ev")
        outcome["engine_tier"] = engine_tier
        outcome["uncertainty"] = ext_data.get("uncertainty", uncertainty)
        confidence = ext_data.get("confidence", 0.70)

    outcome["confidence"] = confidence

    return ReactionResult(
        products=products,
        byproducts=[],
        validity=True,
        confidence=confidence,
        engine_tier=engine_tier,
        predicted_outcome=outcome,
        warnings=warnings,
        safety=safety,
        reactants_canonical=reactants_canonical,
    )
