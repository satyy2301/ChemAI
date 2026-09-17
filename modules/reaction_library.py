"""
reaction_library.py — CRUD, search, fork, and seed loading for reaction templates.
"""
import json
from pathlib import Path

from modules.db.repository import get_repo

_SEED_PATH = Path(__file__).parent.parent / "data" / "seed_reactions.json"
_TEMPLATES_DIR = Path(__file__).parent.parent / "data" / "reaction_templates"

_seeded = False


def _load_json_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def seed_library(force: bool = False):
    """Load seed reactions into DB if not already present."""
    global _seeded
    repo = get_repo()
    if _seeded and not force:
        return repo.count_reactions()

    all_templates: list[dict] = []
    all_templates.extend(_load_json_file(_SEED_PATH))
    if _TEMPLATES_DIR.exists():
        for fp in _TEMPLATES_DIR.glob("*.json"):
            all_templates.extend(_load_json_file(fp))

    for tpl in all_templates:
        existing = repo.get_reaction_by_name(tpl["name"])
        if existing and not force:
            if existing.get("rxn_smarts") != tpl.get("rxn_smarts"):
                from modules.db.base import get_session
                from modules.db.models import Reaction
                session = get_session()
                try:
                    row = session.query(Reaction).filter(Reaction.name == tpl["name"]).first()
                    if row:
                        row.rxn_smarts = tpl["rxn_smarts"]
                        row.reactants_json = json.dumps(tpl.get("reactants", []))
                        row.products_json = json.dumps(tpl.get("products", []))
                        row.base_yield = tpl.get("base_yield", 0.75)
                        session.commit()
                finally:
                    session.close()
            continue
        if existing and force:
            continue
        repo.create_reaction(
            name=tpl["name"],
            rxn_smarts=tpl["rxn_smarts"],
            reactants=tpl.get("reactants", []),
            products=tpl.get("products", []),
            domain=tpl.get("domain", "organic"),
            tags=tpl.get("tags", []),
            created_by=tpl.get("created_by", "system"),
            is_public=tpl.get("is_public", True),
            base_yield=tpl.get("base_yield", 0.75),
        )
    _seeded = True
    return repo.count_reactions()


def create_reaction(name: str, rxn_smarts: str, reactants: list,
                    products: list, domain: str = "organic",
                    tags: list | None = None, created_by: str = "anonymous",
                    is_public: bool = True, base_yield: float = 0.75) -> int:
    return get_repo().create_reaction(
        name, rxn_smarts, reactants, products, domain, tags,
        created_by, is_public, base_yield=base_yield,
    )


def get_reaction(reaction_id: int) -> dict | None:
    return get_repo().get_reaction(reaction_id)


def search_reactions(query: str = "", domain: str | None = None,
                     tag: str | None = None,
                     substructure_smiles: str | None = None) -> list[dict]:
    results = get_repo().search_reactions(query, domain, tag)
    if substructure_smiles:
        try:
            from rdkit import Chem
            pattern = Chem.MolFromSmiles(substructure_smiles)
            if pattern:
                filtered = []
                for r in results:
                    for smi in r.get("reactants", []):
                        mol = Chem.MolFromSmiles(smi)
                        if mol and mol.HasSubstructMatch(pattern):
                            filtered.append(r)
                            break
                results = filtered
        except ImportError:
            pass
    return results


def list_reactions(domain: str | None = None, public_only: bool = True) -> list[dict]:
    return get_repo().list_reactions(domain, public_only)


def fork_reaction(reaction_id: int, user: str) -> int | None:
    return get_repo().fork_reaction(reaction_id, user)


def queue_unsupported_reaction(user_input: str, user_id: str = "anonymous") -> int:
    return get_repo().queue_unsupported_reaction(user_input, user_id)


def list_reaction_requests(status: str = "pending") -> list[dict]:
    return get_repo().list_reaction_requests(status)


def create_reaction_run(reaction_id: int | None, conditions: dict,
                        predicted: dict, user_id: str = "anonymous",
                        provenance: str = "internal_experiment",
                        quality_score: float = 0.0,
                        visibility: str = "public") -> int:
    return get_repo().create_reaction_run(
        reaction_id, conditions, predicted, user_id,
        provenance, quality_score, visibility,
    )


def update_reaction_run_actual(run_id: int, actual: dict,
                               quality_score: float | None = None):
    get_repo().update_reaction_run_actual(run_id, actual, quality_score)


def get_reaction_run(run_id: int) -> dict | None:
    return get_repo().get_reaction_run(run_id)


def list_reaction_runs(reaction_id: int | None = None,
                       public_only: bool = True, limit: int = 100) -> list[dict]:
    return get_repo().list_reaction_runs(reaction_id, public_only, limit)


def get_reaction_options() -> list[dict]:
    """Return reactions formatted for UI selectbox."""
    seed_library()
    reactions = list_reactions()
    return [{"id": r["id"], "label": f"{r['name']} ({r['domain']})", **r}
            for r in reactions]


def get_reaction_by_name(name: str) -> dict | None:
    return get_repo().get_reaction_by_name(name)


def apply_template_to_session(rxn: dict, options: list[dict]) -> str:
    """Return selectbox label for a reaction and the rxn_smarts string."""
    label = next((o["label"] for o in options if o["id"] == rxn["id"]), rxn["name"])
    return label, rxn.get("rxn_smarts", "")


def get_lab_template_patch(rxn_name: str) -> dict | None:
    """Session-state patch for Reaction Lab when selecting a template by name."""
    for o in get_reaction_options():
        if o["name"] == rxn_name:
            return {
                "lab_rxn_select": o["label"],
                "lab_custom_smarts": o.get("rxn_smarts", ""),
                "lab_preset_reaction": rxn_name,
            }
    return None


def smarts_reactant_count(rxn_smarts: str) -> int:
    if not rxn_smarts or ">>" not in rxn_smarts:
        return 0
    return len(rxn_smarts.split(">>", 1)[0].split("."))


# Backwards-compatible alias (older app.py referenced the private name)
_smarts_reactant_count = smarts_reactant_count


def _reactant_set_key(reactants: list[str]) -> frozenset[str]:
    from modules.reaction_engine import canonicalize_smiles_list
    canonical, _ = canonicalize_smiles_list(reactants)
    return frozenset(canonical)


def _match_entry(rxn: dict, confidence: float) -> dict:
    return {
        "reaction": rxn,
        "n_products": len(rxn.get("products") or []),
        "products_preview": (rxn.get("products") or [])[:3],
        "confidence": confidence,
    }


def suggest_reactions_fast(reactants: list[str], top_k: int = 5) -> list[dict]:
    """Metadata-only template match — no RDKit execution."""
    seed_library()
    if not reactants:
        return []
    target_key = _reactant_set_key(reactants)
    n_reactants = len(reactants)
    matches: list[dict] = []
    for rxn in list_reactions():
        lib_key = frozenset(rxn.get("reactants") or [])
        if lib_key and lib_key == target_key:
            matches.append(_match_entry(rxn, 0.99))
            continue
        if smarts_reactant_count(rxn.get("rxn_smarts", "")) != n_reactants:
            continue
        if lib_key and lib_key.issubset(target_key):
            matches.append(_match_entry(rxn, 0.85))
    matches.sort(key=lambda m: (-m["confidence"], -m["n_products"]))
    return matches[:top_k]


def suggest_reactions_deep(reactants: list[str], top_k: int = 5) -> list[dict]:
    """Full SMARTS scan — slower; use on explicit user request or run fallback."""
    from modules.reaction_engine import run_reaction

    fast = suggest_reactions_fast(reactants, top_k=top_k)
    if fast:
        return fast

    seed_library()
    matches: list[dict] = []
    n_reactants = len(reactants)
    for rxn in list_reactions():
        if smarts_reactant_count(rxn.get("rxn_smarts", "")) != n_reactants:
            continue
        try:
            result = run_reaction(
                reactants=reactants,
                reaction_smarts=rxn.get("rxn_smarts"),
                base_yield=rxn.get("base_yield", 0.75),
                skip_safety=True,
            )
        except Exception:
            continue
        if result.validity and result.products:
            matches.append({
                "reaction": rxn,
                "n_products": len(result.products),
                "products_preview": result.products[:3],
                "confidence": result.confidence,
            })
    matches.sort(key=lambda m: (-m["confidence"], -m["n_products"]))
    return matches[:top_k]


def suggest_reactions_for_reactants(reactants: list[str], top_k: int = 5,
                                    deep: bool = False) -> list[dict]:
    """Find matching templates. Default is fast metadata match only."""
    if deep:
        return suggest_reactions_deep(reactants, top_k=top_k)
    return suggest_reactions_fast(reactants, top_k=top_k)


def run_with_best_template(reactants: list[str], conditions: dict,
                           preferred_rxn: dict | None = None) -> tuple:
    """
    Run reaction, auto-falling back to best matching template if preferred fails.
    Returns (ReactionResult, chosen_reaction_dict).
    """
    from modules.reaction_engine import ReactionResult, run_reaction

    preferred_smarts = (preferred_rxn or {}).get("rxn_smarts", "")
    preferred_count_ok = (
        preferred_rxn
        and smarts_reactant_count(preferred_smarts) == len(reactants)
    )

    if preferred_rxn and preferred_count_ok:
        result = run_reaction(
            reactants=reactants,
            reaction_id=preferred_rxn.get("id"),
            reaction_smarts=preferred_smarts,
            conditions=conditions,
            base_yield=preferred_rxn.get("base_yield", 0.75),
        )
        if result.validity:
            return result, preferred_rxn

    candidates: list[dict] = []
    seen: set[int] = set()
    for sug in suggest_reactions_fast(reactants, top_k=5):
        rxn = sug["reaction"]
        if rxn["id"] not in seen:
            candidates.append(rxn)
            seen.add(rxn["id"])

    if not candidates:
        for sug in suggest_reactions_deep(reactants, top_k=5):
            rxn = sug["reaction"]
            if rxn["id"] not in seen:
                candidates.append(rxn)
                seen.add(rxn["id"])

    for rxn in candidates:
        if preferred_rxn and rxn.get("id") == preferred_rxn.get("id"):
            continue
        result = run_reaction(
            reactants=reactants,
            reaction_id=rxn.get("id"),
            reaction_smarts=rxn.get("rxn_smarts"),
            conditions=conditions,
            base_yield=rxn.get("base_yield", 0.75),
        )
        if result.validity:
            result.warnings.insert(0, f"Auto-selected template: {rxn['name']}")
            return result, rxn

    if not candidates:
        if preferred_rxn and not preferred_count_ok:
            return ReactionResult(
                validity=False,
                warnings=[
                    f"No template in the library matches these reactants. "
                    f"Selected template '{preferred_rxn.get('name', '—')}' expects "
                    f"{smarts_reactant_count(preferred_smarts)} reactant(s), "
                    f"but you entered {len(reactants)}. "
                    "Click **Suggest template**, **Request new template**, or browse the Reaction Library."
                ],
            ), preferred_rxn
        return ReactionResult(
            validity=False,
            warnings=[
                "No matching template found in the library for these reactants. "
                "Request a new template or browse the Reaction Library."
            ],
        ), preferred_rxn or {}

    if preferred_rxn and preferred_count_ok:
        return run_reaction(
            reactants=reactants,
            reaction_id=preferred_rxn.get("id"),
            reaction_smarts=preferred_smarts,
            conditions=conditions,
            base_yield=preferred_rxn.get("base_yield", 0.75),
        ), preferred_rxn

    return ReactionResult(
        validity=False,
        warnings=["No matching template found in the library for these reactants."],
    ), preferred_rxn or {}
