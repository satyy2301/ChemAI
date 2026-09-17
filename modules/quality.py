"""Data quality scoring and training eligibility gating."""
import json

from modules.db.repository import get_repo

PROVENANCE_WEIGHTS = {
    "published_paper": 0.30,
    "internal_experiment": 0.25,
    "screening": 0.20,
    "db_retrieved": 0.22,
    "ai_simulation": 0.10,
}

QUALITY_WEIGHTS = {"good": 1.0, "uncertain": 0.6, "outlier": 0.0}


def validate_run_payload(reactants: list, conditions: dict,
                         actual: dict | None = None) -> tuple[bool, list[str]]:
    """Validate minimum required fields for a reaction run."""
    errors = []
    if not reactants:
        errors.append("At least one reactant SMILES required")
    if not conditions:
        errors.append("Conditions dict required")
    if actual is not None and "yield" not in actual:
        errors.append("Actual outcome must include yield")
    return len(errors) == 0, errors


def validate_smiles_list(smiles_list: list) -> tuple[bool, list[str]]:
    """RDKit sanitize check for all SMILES."""
    try:
        from rdkit import Chem
    except ImportError:
        return True, []
    errors = []
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            errors.append(f"Invalid SMILES: {smi[:50]}")
        else:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                errors.append(f"Sanitize failed for {smi[:30]}: {e}")
    return len(errors) == 0, errors


def compute_quality_score(run: dict, data_quality: str = "good",
                          user: str = "anonymous") -> float:
    """
    Score 0-1 from completeness, provenance, replicate bonus, user reputation.
    """
    score = 0.0
    cond = run.get("conditions", {})
    pred = run.get("predicted", {})
    actual = run.get("actual", {})

    # Completeness (0-0.4)
    fields = ["temperature_c", "solvent", "catalyst"]
    filled = sum(1 for f in fields if cond.get(f))
    score += min(0.4, filled * 0.13)
    if pred.get("products"):
        score += 0.05
    if actual.get("yield") is not None:
        score += 0.05

    # Provenance (0-0.3)
    prov = run.get("provenance", "internal_experiment")
    score += PROVENANCE_WEIGHTS.get(prov, 0.15) * QUALITY_WEIGHTS.get(data_quality, 0.5)

    # Replicate bonus (0-0.2) — same reaction_id with prior runs
    reaction_id = run.get("reaction_id")
    if reaction_id:
        prior = get_repo().list_reaction_runs(reaction_id=reaction_id, limit=5)
        if len(prior) > 1:
            score += min(0.2, 0.05 * len(prior))

    # User reputation (0-0.1)
    rep = get_repo().get_user_reputation(user)
    score += min(0.1, rep * 0.02)

    return round(min(1.0, score), 3)


def is_training_eligible(quality_score: float, threshold: float = 0.6,
                         data_quality: str = "good",
                         run_id: int | None = None) -> bool:
    if data_quality == "outlier":
        return False
    if quality_score < threshold:
        return False
    if run_id and get_repo().is_flagged("reaction_run", run_id):
        return False
    return True


def score_and_gate_run(run: dict, data_quality: str = "good",
                       user: str = "anonymous") -> dict:
    """Compute quality score and return eligibility info."""
    qs = compute_quality_score(run, data_quality, user)
    eligible = is_training_eligible(qs, data_quality=data_quality,
                                    run_id=run.get("id"))
    return {"quality_score": qs, "training_eligible": eligible}
