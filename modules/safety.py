"""Safety screening for reactants and products using RDKit FilterCatalog."""
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import FilterCatalog
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Custom SMARTS for hazardous functional groups
_HAZARD_SMARTS = [
    ("Azide", "[N-]=[N+]=[N-]"),
    ("Peroxide", "[O][O]"),
    ("Nitro", "[N+](=O)[O-]"),
    ("Acyl azide", "C(=O)N=[N+]=[N-]"),
    ("Triazine explosive proxy", "c1ncncn1"),
]


@dataclass
class SafetyResult:
    blocked: bool
    alerts: list[str]
    severity: str  # "none", "warning", "critical"


def _get_catalog():
    if not RDKIT_AVAILABLE:
        return None
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


_CATALOG = None


def _catalog():
    global _CATALOG
    if _CATALOG is None and RDKIT_AVAILABLE:
        _CATALOG = _get_catalog()
    return _CATALOG


def screen_smiles(smiles: str) -> list[str]:
    """Return list of safety alerts for a single SMILES string."""
    alerts = []
    if not RDKIT_AVAILABLE:
        return alerts
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        alerts.append(f"Invalid SMILES: {smiles[:40]}")
        return alerts
    cat = _catalog()
    if cat:
        try:
            matches = cat.GetMatches(mol)
            for entry in matches:
                alerts.append(f"FilterCatalog: {entry.GetDescription()}")
        except AttributeError:
            entry = cat.GetFirstMatch(mol)
            while entry is not None:
                alerts.append(f"FilterCatalog: {entry.GetDescription()}")
                try:
                    entry = cat.GetNextMatch(mol)
                except AttributeError:
                    break
    for name, smarts in _HAZARD_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(f"Hazard: {name} detected")
    return alerts


def screen_molecules(smiles_list: list[str]) -> SafetyResult:
    """Screen a list of SMILES; block if critical hazards found."""
    all_alerts = []
    critical_keywords = {"Azide", "Peroxide", "Acyl azide", "explosive"}
    for smi in smiles_list:
        if not smi or not smi.strip():
            continue
        all_alerts.extend(screen_smiles(smi.strip()))
    critical = [a for a in all_alerts if any(k in a for k in critical_keywords)]
    if critical:
        return SafetyResult(blocked=True, alerts=all_alerts, severity="critical")
    if all_alerts:
        return SafetyResult(blocked=False, alerts=all_alerts, severity="warning")
    return SafetyResult(blocked=False, alerts=[], severity="none")
