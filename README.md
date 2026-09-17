# ChemAI — Open Chemistry Reaction Library

Unified AI lab and community reaction library for catalyst discovery, bio pathways, and general organic/catalysis reactions.

## Features

- **Reaction Lab** — paste SMILES, pick from 50+ templates, predict products, log measured yields
- **Reaction Library** — browse, search, fork, and flag community reactions
- **Catalyst Co-Pilot** — AI catalyst engineering with active learning
- **Bio Pathway Designer** — metabolic pathway simulation and intervention planning
- **Active Learning Lab** — uncertainty sampling and automated model retraining
- **Experiment Dashboard** — predicted vs actual, benchmarks, collaboration
- **REST API** — FastAPI at `/docs` for notebooks and integrations

## Tech Stack

- Streamlit (UI) · FastAPI (API) · RDKit (chemistry) · SQLAlchemy (SQLite/Postgres)
- scikit-learn (yield/catalyst ML) · Plotly · 3Dmol.js

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

API (optional, separate terminal):

```bash
uvicorn api.main:app --port 8000 --reload
```

Open http://localhost:8501 (Streamlit) or http://localhost:8000/docs (API).

The app is open access — no login required.

## Reaction Lab Workflow

1. Enter reactant SMILES (one per line) or resolve names via PubChem
2. Select a reaction template or paste custom SMARTS
3. Set conditions (T, pH, solvent, catalyst)
4. Run — RDKit applies SMARTS, heuristic/ML predicts yield
5. Log measured yield — data feeds community model retraining

## Database

- **Default:** SQLite at `data/chemai.db`
- **Postgres/Neon:** set `DATABASE_URL=postgresql://...`
- Legacy `data/experiments.db` is auto-migrated on first run

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Postgres connection string (optional) |
| `CHEMAI_API_KEY` | API key for REST endpoints (optional) |
| `IBM_RXN_API_KEY` | External simulator stub (optional) |
| `ASKCOS_API_KEY` | External simulator stub (optional) |

## API Endpoints

```
POST /reactions/run          Run a reaction
GET  /reactions/search?q=    Search library
POST /reactions/{id}/fork    Fork a reaction
POST /runs/{id}/results      Submit measured results
GET  /models/predict         ML yield prediction
POST /models/retrain         Trigger retrain
GET  /library/               Browse library
```

## Automated Retraining

- Weekly cron (Sunday 02:00) via APScheduler in API lifespan
- Manual trigger from Active Learning Lab or `POST /models/retrain`
- Only promotes model when holdout MAE improves
- Quality-gated: runs with `quality_score >= 0.6` enter training

## Benchmarks

Run from Dashboard → Benchmarks tab or:

```bash
python -c "from modules.benchmarks import run_benchmark; print(run_benchmark('organic_smarts'))"
```

## Smoke Test

```bash
python scripts/smoke_test.py
```

## Data License

Public reaction runs are intended for CC-BY community sharing. Add a LICENSE file for distribution.

## Project Structure

```
chemAI/
├── app.py
├── api/
├── modules/
│   ├── db/              SQLAlchemy persistence
│   ├── reaction_engine.py
│   ├── reaction_library.py
│   ├── ml/              Yield model + active learning
│   ├── benchmarks/
│   └── integrations/
├── data/
│   ├── seed_reactions.json
└── infra/
    ├── migrations/
    └── retrain_jobs/
```
