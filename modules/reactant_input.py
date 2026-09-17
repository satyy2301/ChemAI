"""
User-friendly reactant input: common names, formulas, and SMILES normalization.
"""

# Lower-case keys → SMILES (checked before PubChem / RDKit)
COMMON_ALIASES: dict[str, str] = {
    # gases
    "o2": "O=O",
    "o₂": "O=O",
    "oxygen": "O=O",
    "dioxygen": "O=O",
    "o": "O=O",          # in multi-reactant context users usually mean O₂
    "h2": "[HH]",
    "h₂": "[HH]",
    "hydrogen": "[HH]",
    "n2": "N#N",
    "n₂": "N#N",
    "nitrogen": "N#N",
    "co": "[C-]#[O+]",
    "carbon monoxide": "[C-]#[O+]",
    "co2": "O=C=O",
    "co₂": "O=C=O",
    "carbon dioxide": "O=C=O",
    "ch4": "C",
    "methane": "C",
    "nh3": "N",
    "ammonia": "N",
    "h2o": "O",
    "water": "O",
    # common organics
    "ethanol": "CCO",
    "methanol": "CO",
    "acetic acid": "CC(=O)O",
    "benzene": "c1ccccc1",
    "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}

# Quick-start presets: label → (reactants per line, reaction name to auto-select)
QUICK_PRESETS: dict[str, dict] = {
    "CO + O₂ → CO₂ (oxidation)": {
        "reactants": "CO\nO2",
        "reaction_name": "CO Oxidation",
    },
    "CH₄ + O₂ → CO₂ (combustion)": {
        "reactants": "CH4\nO2",
        "reaction_name": "Methane Combustion",
    },
    "H₂ + O₂ → H₂O (combustion)": {
        "reactants": "H2\nO2",
        "reaction_name": "Hydrogen Combustion",
    },
    "Esterification (acid + alcohol)": {
        "reactants": "CC(=O)O\nCO",
        "reaction_name": "Esterification",
    },
    "Hydrogenation (ethene)": {
        "reactants": "C=C",
        "reaction_name": "Hydrogenation Alkene",
    },
}


def normalize_reactant_line(line: str) -> tuple[str, str | None]:
    """
    Normalize one reactant line. Returns (smiles, note_if_interpreted).
    """
    raw = (line or "").strip()
    if not raw:
        return "", None

    key = raw.lower().strip()
    if key in COMMON_ALIASES:
        smi = COMMON_ALIASES[key]
        if key == "o":
            return smi, "Interpreted 'O' as O₂ (oxygen gas). Use 'water' or H2O for water."
        return smi, f"Resolved '{raw}' → {smi}"

    # Formula without subscripts: Co2 → co2 alias won't match "Co2" - handle case
    if key.replace(" ", "") in COMMON_ALIASES:
        smi = COMMON_ALIASES[key.replace(" ", "")]
        return smi, f"Resolved '{raw}' → {smi}"

    return raw, None


def parse_reactant_input(text: str) -> tuple[list[str], list[str]]:
    """
    Parse multi-line reactant input (names, formulas, or SMILES).
    Returns (normalized_smiles_list, user_notes).
    """
    notes: list[str] = []
    smiles_list: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Allow "CO + O2" on one line
        parts = [p.strip() for p in line.replace(",", "+").split("+") if p.strip()]
        for part in parts:
            smi, note = normalize_reactant_line(part)
            if smi:
                smiles_list.append(smi)
            if note:
                notes.append(note)
    return smiles_list, notes


def nomenclature_help() -> str:
    return """
**How to enter reactants** (one per line, or separated with `+`):

| You type | Meaning | SMILES used |
|----------|---------|-------------|
| `O2` or `oxygen` | Oxygen gas | `O=O` |
| `O` | Oxygen gas (not water) | `O=O` |
| `water` or `H2O` | Water | `O` |
| `CO` | Carbon monoxide | `[C-]#[O+]` |
| `CO2` | Carbon dioxide | `O=C=O` |
| `H2` | Hydrogen | `[HH]` |
| `C=C` | Ethene | `C=C` |

You can also use **chemical names** (e.g. `ethanol`, `benzene`) or full **SMILES** directly.
Click **Resolve names via PubChem** to look up IUPAC names.
"""


def template_help() -> str:
    return """
**What is a reaction template?**

A template is the **rule** that tells the computer how your reactants combine to form products.
It is written in **SMARTS** (a pattern language for reactions).

- **Step 1** = what you have (reactants)
- **Step 2** = what kind of reaction happens (template)
- The engine applies the template to your reactants to **predict products**

Example: for **CO + O₂ → CO₂**, pick the template **"CO Oxidation"**.
If you pick the wrong template (e.g. Water Splitting OER), you will get an error even with correct reactants.

Use **Suggest template** after entering reactants — the app will find matching templates automatically.
"""


def discovery_help() -> str:
    return """
**How to discover reactions**

1. **Suggest template** — searches the library for templates that match your reactants
2. **Reaction Library** — browse or search by name, tag (`fuel`, `combustion`), or domain
3. **Use in Lab** — load a library template directly into Reaction Lab
4. **Request new template** — queue reactants for community review when nothing matches
5. **Fork** — copy an existing template in the library and edit its SMARTS

ChemAI is **template-based**: it predicts products by applying library reaction rules (SMARTS), not by inventing new chemistry from scratch.
"""
