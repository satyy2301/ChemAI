"""Library-wide active learning suggestions."""
from collections import defaultdict

from modules.db.repository import get_repo


def get_library_suggestions(top_k: int = 5) -> list[dict]:
    """
    Suggest reaction families / conditions where the community needs more data.
    Ranks by uncertainty × data scarcity.
    """
    runs = get_repo().list_reaction_runs(public_only=True, limit=500)
    if not runs:
        return [{
            "reaction_family": "General organic",
            "suggestion": "Log your first reaction run to seed the library.",
            "priority": 1.0, "data_points": 0,
        }]

    family_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "uncertainties": [], "conditions": []}
    )

    for run in runs:
        pred = run.get("predicted", {})
        rid = run.get("reaction_id")
        family = f"reaction_{rid}" if rid else "custom"
        if rid:
            rxn = get_repo().get_reaction(rid)
            if rxn:
                family = rxn.get("name", family)
        family_stats[family]["count"] += 1
        unc = pred.get("uncertainty", 0.1)
        family_stats[family]["uncertainties"].append(unc)
        family_stats[family]["conditions"].append(run.get("conditions", {}))

    suggestions = []
    max_count = max(s["count"] for s in family_stats.values()) or 1
    for family, stats in family_stats.items():
        avg_unc = sum(stats["uncertainties"]) / len(stats["uncertainties"])
        scarcity = 1.0 - (stats["count"] / max_count)
        priority = avg_unc * (0.5 + scarcity)
        cond = stats["conditions"][-1] if stats["conditions"] else {}
        solvent = cond.get("solvent", "unspecified")
        temp = cond.get("temperature_c", 25)
        suggestions.append({
            "reaction_family": family,
            "suggestion": (
                f"Community needs data on: {family}, "
                f"solvent={solvent}, T={temp}°C"
            ),
            "priority": round(priority, 4),
            "data_points": stats["count"],
            "avg_uncertainty": round(avg_unc, 4),
        })

    return sorted(suggestions, key=lambda x: x["priority"], reverse=True)[:top_k]


def get_high_uncertainty_runs(top_k: int = 5) -> list[dict]:
    """Return completed runs with highest prediction uncertainty."""
    runs = get_repo().list_reaction_runs(limit=200)
    scored = []
    for run in runs:
        pred = run.get("predicted", {})
        unc = pred.get("uncertainty", 0)
        if unc > 0:
            scored.append({**run, "uncertainty": unc})
    return sorted(scored, key=lambda x: x["uncertainty"], reverse=True)[:top_k]
