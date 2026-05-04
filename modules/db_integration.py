"""
Scientific database integration for ChemAI.
Sources: Catalysis Hub (GraphQL), Materials Project (REST), BRENDA (curated local).
"""

from __future__ import annotations
import json
import time
from typing import Optional

import requests

_CH_URL = "https://api.catalysis-hub.org/graphql"
_MP_URL = "https://api.materialsproject.org/materials/summary/"
_TIMEOUT = 10

# ── species keywords for client-side reaction matching ────────────────────────
_REACTION_SPECIES: dict[str, list[str]] = {
    "HER":             ["H2gas", "Hstar", "H2"],
    "CO_Oxidation":    ["COgas", "COstar", "CO2gas", "Ostar"],
    "CO2_to_Methanol": ["CO2gas", "CH3OH", "CH2OH", "HCOOstar"],
    "Fischer_Tropsch": ["COgas", "CH4gas", "CHstar", "CHOstar"],
    "RWGS":            ["CO2gas", "COgas", "H2gas", "H2Ogas"],
    "Methanation":     ["CO2gas", "CH4gas", "H2gas"],
    "N2_Fixation":     ["N2gas", "NHstar", "NH3gas", "Nstar"],
    "OER":             ["H2Ogas", "O2gas", "OHstar", "Ostar"],
}

# ── BRENDA-curated enzyme kinetics (covers ChemAI bio pathways) ───────────────
# Structure: reaction_key → list of enzyme records
_BRENDA_LOCAL: dict[str, list[dict]] = {
    "HER": [
        {
            "enzyme": "Hydrogenase [FeFe]",
            "ec_number": "1.12.7.2",
            "substrate": "H⁺ + 2e⁻",
            "product": "H₂",
            "km_mm": 0.18,
            "kcat_s": 9000,
            "organism": "Clostridium pasteurianum",
            "ph_optimum": 7.0,
            "temp_optimum_c": 37,
            "source": "BRENDA",
        },
        {
            "enzyme": "Hydrogenase [NiFe]",
            "ec_number": "1.12.2.1",
            "substrate": "H₂",
            "product": "H⁺ + 2e⁻",
            "km_mm": 0.065,
            "kcat_s": 700,
            "organism": "Desulfovibrio gigas",
            "ph_optimum": 7.5,
            "temp_optimum_c": 30,
            "source": "BRENDA",
        },
    ],
    "N2_Fixation": [
        {
            "enzyme": "Nitrogenase (MoFe)",
            "ec_number": "1.18.6.1",
            "substrate": "N₂ + 8H⁺ + 8e⁻",
            "product": "2NH₃ + H₂",
            "km_mm": 0.1,
            "kcat_s": 5,
            "organism": "Azotobacter vinelandii",
            "ph_optimum": 7.4,
            "temp_optimum_c": 30,
            "source": "BRENDA",
        },
    ],
    "CO2_to_Methanol": [
        {
            "enzyme": "Formate dehydrogenase",
            "ec_number": "1.2.1.2",
            "substrate": "CO₂ + NADH",
            "product": "Formate + NAD⁺",
            "km_mm": 0.5,
            "kcat_s": 12,
            "organism": "Candida boidinii",
            "ph_optimum": 7.5,
            "temp_optimum_c": 25,
            "source": "BRENDA",
        },
        {
            "enzyme": "Alcohol dehydrogenase",
            "ec_number": "1.1.1.1",
            "substrate": "Formaldehyde + NADH",
            "product": "Methanol + NAD⁺",
            "km_mm": 1.0,
            "kcat_s": 250,
            "organism": "Saccharomyces cerevisiae",
            "ph_optimum": 7.0,
            "temp_optimum_c": 30,
            "source": "BRENDA",
        },
    ],
    "Fischer_Tropsch": [
        {
            "enzyme": "CO dehydrogenase",
            "ec_number": "1.2.7.4",
            "substrate": "CO + H₂O",
            "product": "CO₂ + 2H⁺ + 2e⁻",
            "km_mm": 0.02,
            "kcat_s": 39000,
            "organism": "Carboxydothermus hydrogenoformans",
            "ph_optimum": 7.0,
            "temp_optimum_c": 70,
            "source": "BRENDA",
        },
    ],
    "OER": [
        {
            "enzyme": "Photosystem II (PSII)",
            "ec_number": "1.10.3.9",
            "substrate": "2H₂O",
            "product": "O₂ + 4H⁺ + 4e⁻",
            "km_mm": None,
            "kcat_s": 400,
            "organism": "Spinacia oleracea",
            "ph_optimum": 6.5,
            "temp_optimum_c": 25,
            "source": "BRENDA",
        },
    ],
    "RWGS": [
        {
            "enzyme": "Carbon monoxide dehydrogenase",
            "ec_number": "1.2.99.2",
            "substrate": "CO₂ + H₂",
            "product": "CO + H₂O",
            "km_mm": 0.15,
            "kcat_s": 150,
            "organism": "Moorella thermoacetica",
            "ph_optimum": 7.0,
            "temp_optimum_c": 55,
            "source": "BRENDA",
        },
    ],
    "Methanation": [
        {
            "enzyme": "Methyl-coenzyme M reductase",
            "ec_number": "2.8.4.1",
            "substrate": "Methyl-CoM + CoB",
            "product": "CH₄ + CoM-S-S-CoB",
            "km_mm": 0.3,
            "kcat_s": 40,
            "organism": "Methanobacterium thermoautotrophicum",
            "ph_optimum": 7.0,
            "temp_optimum_c": 65,
            "source": "BRENDA",
        },
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _dominant_element(composition: dict) -> str:
    """Return the element with the highest fraction."""
    return max(composition, key=lambda e: composition[e])


def _gql(query: str) -> Optional[dict]:
    """Execute a GraphQL query against Catalysis Hub; return data dict or None."""
    try:
        r = requests.post(_CH_URL, json={"query": query}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("data")
    except Exception:
        return None


def _matches_reaction(reactants_json: str, products_json: str, reaction_key: str) -> bool:
    """Check if a Catalysis Hub reaction is relevant to the given reaction key."""
    keywords = _REACTION_SPECIES.get(reaction_key, [])
    combined = (reactants_json or "") + (products_json or "")
    return any(kw in combined for kw in keywords)


def _parse_energy_str(v) -> Optional[float]:
    """Safely parse a possibly-None float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── public API ────────────────────────────────────────────────────────────────

def fetch_catalysis_hub(
    composition: dict,
    reaction_key: str,
    max_results: int = 10,
) -> dict:
    """
    Query Catalysis Hub for DFT adsorption/reaction data for a catalyst surface.

    Returns a dict with keys:
      status  : "ok" | "no_data" | "error"
      source  : "Catalysis Hub"
      element : dominant surface element queried
      rows    : list of record dicts
      error   : error message (only on "error" status)
    """
    element = _dominant_element(composition)
    query = (
        f'{{ reactions(surfaceComposition: "{element}", first: 100) {{'
        f'  edges {{ node {{ surfaceComposition facet reactants products '
        f'           reactionEnergy activationEnergy dftCode dftFunctional pubId }} }}'
        f'}} }}'
    )
    data = _gql(query)
    if data is None:
        return {"status": "error", "source": "Catalysis Hub", "element": element,
                "rows": [], "error": "Network or API error"}

    edges = data.get("reactions", {}).get("edges", [])
    rows = []
    for e in edges:
        n = e["node"]
        r_str = n.get("reactants") or ""
        p_str = n.get("products") or ""
        if not _matches_reaction(r_str, p_str, reaction_key):
            continue
        rows.append({
            "surface": n.get("surfaceComposition", element),
            "facet": n.get("facet", "?"),
            "reactants": r_str,
            "products": p_str,
            "reaction_energy_ev": _parse_energy_str(n.get("reactionEnergy")),
            "activation_energy_ev": _parse_energy_str(n.get("activationEnergy")),
            "dft_functional": n.get("dftFunctional") or "N/A",
            "dft_code": n.get("dftCode") or "N/A",
            "pub_id": n.get("pubId") or "",
            "source": "Catalysis Hub",
        })
        if len(rows) >= max_results:
            break

    status = "ok" if rows else "no_data"
    return {"status": status, "source": "Catalysis Hub", "element": element, "rows": rows}


def fetch_materials_project(
    composition: dict,
    api_key: str,
    max_results: int = 5,
) -> dict:
    """
    Query Materials Project REST API for stability and electronic structure.

    Returns a dict with keys:
      status  : "ok" | "no_key" | "no_data" | "error"
      source  : "Materials Project"
      rows    : list of record dicts
      error   : error message (only on "error" status)
    """
    if not api_key or api_key.strip() == "":
        return {"status": "no_key", "source": "Materials Project",
                "rows": [], "error": "No API key provided"}

    # Build formula from dominant elements (top 2 by fraction)
    sorted_elems = sorted(composition.items(), key=lambda x: -x[1])
    formula_elems = [e for e, _ in sorted_elems[:2]]
    formula = "-".join(formula_elems)

    params = {
        "formula": formula,
        "fields": "material_id,formula_pretty,formation_energy_per_atom,energy_above_hull,band_gap,nsites",
        "_limit": max_results,
    }
    headers = {"X-API-KEY": api_key.strip()}

    try:
        r = requests.get(_MP_URL, params=params, headers=headers, timeout=_TIMEOUT)
        if r.status_code == 401:
            return {"status": "error", "source": "Materials Project",
                    "rows": [], "error": "Invalid API key (401 Unauthorized)"}
        r.raise_for_status()
        items = r.json().get("data", [])
    except requests.HTTPError as exc:
        return {"status": "error", "source": "Materials Project",
                "rows": [], "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "source": "Materials Project",
                "rows": [], "error": f"Network error: {exc}"}

    rows = []
    for it in items:
        rows.append({
            "material_id": it.get("material_id", "?"),
            "formula": it.get("formula_pretty", formula),
            "formation_energy_ev_atom": _parse_energy_str(it.get("formation_energy_per_atom")),
            "energy_above_hull_ev": _parse_energy_str(it.get("energy_above_hull")),
            "band_gap_ev": _parse_energy_str(it.get("band_gap")),
            "n_sites": it.get("nsites"),
            "source": "Materials Project",
        })

    status = "ok" if rows else "no_data"
    return {"status": status, "source": "Materials Project", "rows": rows}


def fetch_brenda_local(reaction_key: str) -> dict:
    """
    Return curated BRENDA-style enzyme kinetics data for a reaction.

    Returns a dict with keys:
      status : "ok" | "no_data"
      source : "BRENDA (curated)"
      rows   : list of enzyme record dicts
    """
    rows = _BRENDA_LOCAL.get(reaction_key, [])
    return {
        "status": "ok" if rows else "no_data",
        "source": "BRENDA (curated)",
        "rows": rows,
    }


def fetch_all(
    catalyst: dict,
    reaction_key: str,
    mp_api_key: str = "",
) -> dict:
    """
    Orchestrate all three database fetches for a given catalyst + reaction.

    catalyst  : dict with at least a 'composition' key (element → fraction map)
    reaction_key : one of the 8 ChemAI reaction keys

    Returns {
        'catalysis_hub': {...},
        'materials_project': {...},
        'brenda': {...},
    }
    """
    composition = catalyst.get("composition", {"X": 1.0})
    return {
        "catalysis_hub": fetch_catalysis_hub(composition, reaction_key),
        "materials_project": fetch_materials_project(composition, mp_api_key),
        "brenda": fetch_brenda_local(reaction_key),
    }
