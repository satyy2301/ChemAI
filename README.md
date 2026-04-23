# ChemAI

Unified AI Lab for Fuel Discovery.

ChemAI is a hackathon-ready, closed-loop discovery platform for:
- Catalyst discovery (materials and reaction performance ranking)
- Synthetic biology pathway design (organism, enzymes, bottlenecks)
- Active learning from experimental feedback

The app is built for fast setup, easy demoing, and simple deployment.

## What It Does

### 1) Catalyst Co-Pilot
- Select a target reaction
- Browse known catalysts
- Generate AI variants (doping and surface strategies)
- Rank candidates by activity, stability, and selectivity
- Visualize trade-offs and log experiment outcomes

### 2) Bio Pathway Designer
- Select a target biochemical pathway
- Visualize reaction steps as a graph
- Identify bottlenecks
- Get mutation/improvement suggestions
- Compare reported vs predicted yield

### 3) Active Learning Lab
- Suggest which candidates to test next using model uncertainty
- Prioritize high-information experiments
- Retrain models from new experiment logs

### 4) Experiment Dashboard
- Predicted vs actual comparisons
- Error timeline and model metric history
- Persistent experiment tracking with SQLite

## Tech Stack

- Streamlit (frontend and app runtime)
- scikit-learn (RandomForest-based prediction)
- Plotly (interactive charts)
- NetworkX (pathway graph structure)
- SQLite (experiment and retraining history)
- pandas / NumPy (data and feature processing)

## Project Structure

```text
chemAI/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── catalysts_db.json
│   ├── bio_db.json
│   └── experiments.db (auto-created/updated)
└── modules/
    ├── __init__.py
    ├── catalyst_module.py
    ├── bio_module.py
    └── feedback.py
```

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
pip install -r requirements.txt
```

### Run

Use one of the commands below:

```bash
python -m streamlit run app.py
```

or

```bash
streamlit run app.py
```

App URL (default):
- http://localhost:8501

## Demo Flow (5 minutes)

1. Open Overview and explain the closed-loop workflow.
2. Go to Catalyst Co-Pilot:
   - Choose a reaction
   - Generate AI candidates
   - Show ranking and trade-off chart
3. Go to Bio Pathway Designer:
   - Pick a pathway
   - Highlight bottleneck + mutation suggestions
4. Log one experiment result.
5. Open Active Learning Lab and show suggested next experiments.
6. Open Dashboard and show predicted vs actual trend.

## Deployment

### Streamlit Community Cloud (easiest)
1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from this repo.
4. Set main file path to `app.py`.
5. Deploy.

### Optional: Container deployment
- Add Docker support for hosting on cloud VMs or container platforms.

## Notes

- The current models are lightweight and hackathon-friendly (fast inference).
- Data is pre-seeded for strong demos and can be extended.
- SQLite provides a no-infra persistence layer for feedback loops.

## Future Improvements

- Replace synthetic features with pretrained chemistry/biology encoders
- Add user authentication and team workspaces
- Integrate real lab APIs and experiment ingestion
- Add automated benchmarking for candidate quality

## License

This project is intended for hackathon and educational use.
Add a LICENSE file if you plan public/open-source distribution.
