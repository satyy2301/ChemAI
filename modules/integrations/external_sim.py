"""External simulation stub (IBM RXN, ASKCOS, xTB)."""
import os


def run_external_simulation(reactants: list, products: list,
                            conditions: dict) -> dict:
    """
    Call external simulator if API keys configured.
    Returns status dict.
    """
    rxn_key = os.environ.get("IBM_RXN_API_KEY", "")
    askcos_key = os.environ.get("ASKCOS_API_KEY", "")

    if not rxn_key and not askcos_key:
        return {"status": "not_configured", "message": "No external API keys set"}

    # Stub: simulate a response when keys are present
    base_yield = 0.72
    t = float(conditions.get("temperature_c", 25))
    if t > 100:
        base_yield *= 0.95
    return {
        "status": "ok",
        "yield": round(base_yield, 4),
        "uncertainty": 0.06,
        "confidence": 0.70,
        "energy_barrier_ev": 1.2,
        "source": "external_stub",
        "message": "External simulator stub response (configure real API for production)",
    }
