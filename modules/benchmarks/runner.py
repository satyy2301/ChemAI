"""Benchmark suite runner for ChemAI."""
import json
from pathlib import Path

from modules import reaction_engine as re
from modules import reaction_library as rl
from modules.db.repository import get_repo

_SUITES_DIR = Path(__file__).parent / "suites"


def _load_suite(name: str) -> dict:
    path = _SUITES_DIR / f"{name}.json"
    if not path.exists():
        return {"name": name, "tests": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(suite_name: str) -> dict:
    """Run a benchmark suite and persist results."""
    suite = _load_suite(suite_name)
    tests = suite.get("tests", [])
    passed = 0
    failed = 0
    details = []

    for test in tests:
        test_type = test.get("type", "smarts")
        ok = False
        msg = ""

        if test_type == "smarts":
            result = re.run_reaction(
                reactants=test.get("reactants", []),
                reaction_smarts=test.get("rxn_smarts"),
                conditions=test.get("conditions", {}),
                base_yield=test.get("base_yield", 0.75),
                skip_safety=test.get("skip_safety", False),
            )
            ok = result.validity and len(result.products) > 0
            if test.get("min_yield"):
                y = result.predicted_outcome.get("yield", 0)
                ok = ok and y >= test["min_yield"]
            msg = f"products={len(result.products)}, tier={result.engine_tier}"

        elif test_type == "safety_block":
            from modules.safety import screen_molecules
            sr = screen_molecules(test.get("smiles", []))
            ok = sr.blocked == test.get("expect_blocked", False)
            msg = f"blocked={sr.blocked}, alerts={len(sr.alerts)}"

        elif test_type == "library_count":
            rl.seed_library()
            count = get_repo().count_reactions()
            ok = count >= test.get("min_count", 50)
            msg = f"reactions={count}"

        if ok:
            passed += 1
        else:
            failed += 1
        details.append({"name": test.get("name", "?"), "passed": ok, "message": msg})

    metrics = {
        "suite": suite_name,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "pass_rate": round(passed / max(1, passed + failed), 3),
        "details": details,
    }
    get_repo().save_benchmark_run(suite_name, metrics)
    return metrics
