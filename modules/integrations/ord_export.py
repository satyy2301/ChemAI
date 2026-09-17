"""ORD (Open Reaction Database) JSON export stub."""
import json
from datetime import datetime


def export_run_to_ord(run: dict, reaction: dict | None = None) -> str:
    """Export a reaction run as minimal ORD-compatible JSON."""
    pred = run.get("predicted", {})
    actual = run.get("actual", {})
    cond = run.get("conditions", {})

    record = {
        "reaction_id": f"chemai_run_{run.get('id', 0)}",
        "inputs": {
            "reactants": [{"smiles": s} for s in pred.get("reactants", [])],
        },
        "conditions": {
            "temperature": {"setpoint": {"value": cond.get("temperature_c", 25), "units": "CELSIUS"}},
            "stirring": {"type": "CUSTOM", "details": cond.get("solvent", "")},
        },
        "outcomes": [{
            "products": [{"smiles": s} for s in pred.get("products", [])],
            "analysis": {
                "yield": {"value": actual.get("yield", pred.get("yield", 0))},
            },
        }],
        "provenance": {
            "experiment_id": str(run.get("id", "")),
            "source": "ChemAI",
            "created_at": run.get("created_at", datetime.utcnow().isoformat()),
        },
    }
    if reaction:
        record["reaction_name"] = reaction.get("name", "")
        record["reaction_smarts"] = reaction.get("rxn_smarts", "")

    return json.dumps(record, indent=2)
