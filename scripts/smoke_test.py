"""End-to-end smoke test for ChemAI Reaction Library."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import feedback as fb
from modules import reaction_engine as re
from modules import reaction_library as rl
from modules import reactant_input as ri
from modules.benchmarks import run_benchmark
from modules.safety import screen_molecules


def main():
    print("1. Init DB...")
    fb.init_db()
    n = rl.seed_library()
    print(f"   Seeded {n} reactions")

    print("2. Run hydrogenation...")
    rxns = rl.search_reactions("Hydrogenation Alkene")
    assert rxns, "Hydrogenation Alkene not found"
    rxn = rxns[0]
    reactants = rxn.get("reactants", ["C=C"])
    result = re.run_reaction(
        reactants=reactants,
        reaction_smarts=rxn["rxn_smarts"],
        conditions={"temperature_c": 80, "solvent": "toluene", "catalyst": "H2SO4"},
        base_yield=rxn.get("base_yield", 0.75),
    )
    assert result.validity, f"Reaction failed: {result.warnings}"
    print(f"   Products: {result.products[:2]}, yield={result.predicted_outcome.get('yield')}")

    print("3. Log experiment...")
    run_id = rl.create_reaction_run(
        reaction_id=rxn["id"],
        conditions={"temperature_c": 80},
        predicted=result.predicted_outcome,
        user_id="smoke_test",
    )
    rl.update_reaction_run_actual(run_id, {"yield": 0.79}, quality_score=0.7)
    exp_id = fb.log_experiment(
        "reaction", rxn["name"], result.predicted_outcome["yield"], 0.79,
        "reaction_yield", user="smoke_test", reaction_run_id=run_id,
    )
    print(f"   Run #{run_id}, experiment #{exp_id}")

    print("4. Safety screen...")
    sr = screen_molecules(["CC(=O)O"])
    print(f"   blocked={sr.blocked}, alerts={len(sr.alerts)}")

    print("5. Benchmarks...")
    bm = run_benchmark("organic_smarts")
    print(f"   Pass rate: {bm.get('pass_rate', 0):.0%}")

    print("6. Fork reaction...")
    fork_id = rl.fork_reaction(rxn["id"], "smoke_test")
    assert fork_id, "Fork failed"
    print(f"   Forked as #{fork_id}")

    print("7. Gas combustion templates...")
    for label, reactant_text, expected_name in [
        ("CH4 + O2", "CH4\nO2", "Methane Combustion"),
        ("H2 + O2", "H2\nO2", "Hydrogen Combustion"),
    ]:
        parsed, _ = ri.parse_reactant_input(reactant_text)
        sugs = rl.suggest_reactions_fast(parsed, top_k=3)
        assert sugs, f"No fast match for {label}"
        assert sugs[0]["reaction"]["name"] == expected_name, (
            f"{label}: expected {expected_name}, got {sugs[0]['reaction']['name']}"
        )
        result, used = rl.run_with_best_template(parsed, {"temperature_c": 25})
        assert result.validity, f"{label} run failed: {result.warnings}"
        print(f"   {label} -> {used['name']}, products={result.products[:2]}")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
