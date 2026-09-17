"""Morgan fingerprint + RandomForest yield predictor."""
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

_FP_SIZE = 1024
_FP_RADIUS = 2


def _morgan_fp(smiles: str) -> np.ndarray:
    if not RDKIT_AVAILABLE:
        return np.zeros(_FP_SIZE)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(_FP_SIZE)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, _FP_RADIUS, nBits=_FP_SIZE)
    arr = np.zeros(_FP_SIZE)
    for i in range(_FP_SIZE):
        arr[i] = fp[i]
    return arr


def _condition_vector(conditions: dict) -> np.ndarray:
    return np.array([
        float(conditions.get("temperature_c", 25)),
        float(conditions.get("pressure_bar", 1)),
        float(conditions.get("ph", 7.0)),
        1.0 if conditions.get("catalyst") else 0.0,
        hash(str(conditions.get("solvent", ""))) % 100 / 100.0,
    ])


def _featurize(reactants: list, products: list, conditions: dict) -> np.ndarray:
    fps = []
    for smi in reactants + products:
        fps.append(_morgan_fp(smi))
    if not fps:
        fps = [np.zeros(_FP_SIZE)]
    mol_fp = np.mean(fps, axis=0)
    cond = _condition_vector(conditions)
    return np.concatenate([mol_fp, cond])


class YieldModel:
    def __init__(self):
        self._scaler = StandardScaler()
        self._model = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42,
        )
        self._fitted = False

    def _build_training_data(self, runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for run in runs:
            pred = run.get("predicted", {})
            actual = run.get("actual", {})
            if actual.get("yield") is None:
                continue
            reactants = pred.get("reactants", [])
            products = pred.get("products", [])
            cond = run.get("conditions", {})
            X.append(_featurize(reactants, products, cond))
            y.append(float(actual["yield"]))
        return np.array(X), np.array(y)

    def retrain(self, runs: list[dict]) -> dict:
        X, y = self._build_training_data(runs)
        if len(X) < 3:
            return {"status": "insufficient_data", "n": len(X)}
        X_sc = self._scaler.fit_transform(X)
        self._model.fit(X_sc, y)
        self._fitted = True
        preds = self._model.predict(X_sc)
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        return {"status": "ok", "n": len(X), "mae": mae, "rmse": rmse}

    def predict(self, reactants: list, products: list,
                conditions: dict) -> dict:
        feat = _featurize(reactants, products, conditions).reshape(1, -1)
        if not self._fitted:
            # Bootstrap with synthetic prior
            rng = np.random.default_rng(42)
            X_boot = rng.normal(0, 1, (20, feat.shape[1]))
            y_boot = rng.uniform(0.3, 0.9, 20)
            self._scaler.fit(X_boot)
            self._model.fit(self._scaler.transform(X_boot), y_boot)
            self._fitted = True
        feat_sc = self._scaler.transform(feat)
        preds = [t.predict(feat_sc)[0] for t in self._model.estimators_]
        yield_val = float(np.mean(preds))
        uncertainty = float(np.std(preds))
        return {
            "yield": round(min(0.99, max(0.01, yield_val)), 4),
            "uncertainty": round(uncertainty, 4),
            "selectivity": round(min(0.99, yield_val + 0.02), 4),
        }


_model: YieldModel | None = None


def get_yield_model() -> YieldModel:
    global _model
    if _model is None:
        _model = YieldModel()
    return _model
