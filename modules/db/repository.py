"""Data access layer for ChemAI — used by feedback.py, reaction_library, API."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import desc, func, or_

from .base import get_session, init_engine, legacy_db_path
from .models import (
    Annotation, BenchmarkRun, Experiment, ExperimentQueue, Flag, JobQueue,
    ModelVersion, Reaction, ReactionRequest, ReactionRun, ScenarioRun, User,
)


class Repository:
    def __init__(self):
        init_engine()

    def _session(self):
        return get_session()

    # ─── init & migration ───────────────────────────────────────────────────

    def init_db(self):
        init_engine()
        self._migrate_legacy_experiments_db()
        if self.count_experiments() == 0:
            self._seed_demo_data()

    def _migrate_legacy_experiments_db(self):
        legacy = legacy_db_path()
        if not legacy.exists():
            return
        if self.count_experiments() > 0:
            return
        con = sqlite3.connect(legacy)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        session = self._session()
        try:
            if "experiments" in tables:
                for row in cur.execute("SELECT * FROM experiments"):
                    d = dict(row)
                    session.add(Experiment(
                        timestamp=d.get("timestamp", ""),
                        exp_type=d.get("exp_type", ""),
                        name=d.get("name", ""),
                        pred_value=d.get("pred_value"),
                        actual_value=d.get("actual_value"),
                        metric=d.get("metric"),
                        notes=d.get("notes", ""),
                        composition=d.get("composition", "{}"),
                        user=d.get("user", "anonymous"),
                        version_tag=d.get("version_tag", "v1"),
                        data_quality=d.get("data_quality", "good"),
                        source_provenance=d.get("source_provenance", "internal_experiment"),
                    ))
            if "model_versions" in tables:
                for row in cur.execute("SELECT * FROM model_versions"):
                    d = dict(row)
                    session.add(ModelVersion(
                        timestamp=d.get("timestamp", ""),
                        exp_type=d.get("exp_type", ""),
                        mae=d.get("mae"),
                        rmse=d.get("rmse"),
                        n_samples=d.get("n_samples"),
                    ))
            if "scenario_runs" in tables:
                for row in cur.execute("SELECT * FROM scenario_runs"):
                    d = dict(row)
                    session.add(ScenarioRun(
                        timestamp=d.get("timestamp", ""),
                        pathway_name=d.get("pathway_name", ""),
                        scenario_json=d.get("scenario_json", "{}"),
                        predicted_yield=d.get("predicted_yield"),
                        uncertainty=d.get("uncertainty"),
                        risk_score=d.get("risk_score"),
                        chosen_plan=d.get("chosen_plan", ""),
                        expected_gain=d.get("expected_gain", 0.0),
                    ))
            if "experiment_queue" in tables:
                for row in cur.execute("SELECT * FROM experiment_queue"):
                    d = dict(row)
                    session.add(ExperimentQueue(
                        timestamp=d.get("timestamp", ""),
                        exp_type=d.get("exp_type", ""),
                        candidate_name=d.get("candidate_name", ""),
                        plan_text=d.get("plan_text", ""),
                        predicted_value=d.get("predicted_value"),
                        risk_score=d.get("risk_score"),
                        payload_json=d.get("payload_json", "{}"),
                        status=d.get("status", "queued"),
                        actual_value=d.get("actual_value"),
                        notes=d.get("notes", ""),
                    ))
            if "annotations" in tables:
                for row in cur.execute("SELECT * FROM annotations"):
                    d = dict(row)
                    session.add(Annotation(
                        timestamp=d.get("timestamp", ""),
                        user=d.get("user", ""),
                        exp_id=d.get("exp_id"),
                        exp_type=d.get("exp_type", ""),
                        target_name=d.get("target_name", ""),
                        text=d.get("text", ""),
                    ))
            session.commit()
        finally:
            session.close()
        con.close()

    def _seed_demo_data(self):
        session = self._session()
        try:
            seed_rows = [
                ("2026-04-10 09:00", "catalyst", "Cu/ZnO/Al₂O₃", 0.78, 0.76, "activity",
                 "Standard run, baseline.", "{}", "alice", "v1", "good", "internal_experiment"),
                ("2026-04-11 11:30", "catalyst", "Pd/In₂O₃", 0.85, 0.83, "activity",
                 "Good agreement.", "{}", "bob", "v1", "good", "published_paper"),
                ("2026-04-12 14:00", "catalyst", "In₂O₃/ZrO₂", 0.71, 0.74, "activity",
                 "Slightly under-predicted.", "{}", "alice", "v1", "good", "internal_experiment"),
                ("2026-04-13 10:00", "catalyst", "Cu-Zn-Ga/Al₂O₃", 0.80, 0.77, "activity",
                 "Ga doping confirmed.", "{}", "carol", "v1", "good", "screening"),
                ("2026-04-14 15:00", "bio", "Glucose → Ethanol", 0.51, 0.49, "yield",
                 "Close to theoretical.", "{}", "bob", "v1", "good", "published_paper"),
                ("2026-04-15 09:30", "bio", "Glucose → Isobutanol", 0.41, 0.37, "yield",
                 "Lower than predicted.", "{}", "carol", "v1", "uncertain", "internal_experiment"),
                ("2026-04-16 11:00", "bio", "Fatty Acids → Biodiesel", 0.90, 0.88, "yield",
                 "Good result.", "{}", "alice", "v1", "good", "db_retrieved"),
                ("2026-04-17 14:30", "catalyst", "CoP/Carbon", 0.83, 0.80, "activity",
                 "HER test.", "{}", "bob", "v1", "good", "screening"),
                ("2026-04-18 10:00", "catalyst", "Ni-Fe/CeO₂", 0.84, 0.86, "activity",
                 "Slightly over-performed.", "{}", "carol", "v1", "uncertain", "internal_experiment"),
                ("2026-04-20 16:00", "bio", "Glucose → Lactic Acid", 0.88, 0.91, "yield",
                 "Exceeded prediction.", "{}", "alice", "v2", "good", "published_paper"),
            ]
            for row in seed_rows:
                session.add(Experiment(
                    timestamp=row[0], exp_type=row[1], name=row[2],
                    pred_value=row[3], actual_value=row[4], metric=row[5],
                    notes=row[6], composition=row[7], user=row[8],
                    version_tag=row[9], data_quality=row[10], source_provenance=row[11],
                ))
            for row in [
                ("2026-04-12 00:00", "catalyst", 0.042, 0.058, 4),
                ("2026-04-16 00:00", "catalyst", 0.031, 0.044, 6),
                ("2026-04-16 00:00", "bio", 0.038, 0.051, 3),
                ("2026-04-20 00:00", "catalyst", 0.025, 0.035, 8),
                ("2026-04-20 00:00", "bio", 0.028, 0.040, 4),
            ]:
                session.add(ModelVersion(
                    timestamp=row[0], exp_type=row[1], mae=row[2],
                    rmse=row[3], n_samples=row[4], promoted=True,
                ))
            for row in [
                ("2026-04-10 10:00", "alice", 1, "catalyst", "Cu/ZnO/Al₂O₃",
                 "Baseline confirmed. Good starting point for Ga/In doping studies."),
                ("2026-04-11 12:00", "bob", 2, "catalyst", "Pd/In₂O₃",
                 "Matches Liu et al. 2024 within 2.5%. Paper data source verified."),
                ("2026-04-13 11:00", "carol", 4, "catalyst", "Cu-Zn-Ga/Al₂O₃",
                 "Ga at 5 mol% optimal. Higher concentrations reduce long-term stability."),
                ("2026-04-15 10:00", "alice", 6, "bio", "Glucose → Isobutanol",
                 "Suspected product inhibition at high titers. Flagged as uncertain."),
                ("2026-04-18 11:00", "bob", 9, "catalyst", "Ni-Fe/CeO₂",
                 "Over-performance likely due to Fe surface segregation. Revisit prep."),
                ("2026-04-20 17:00", "carol", 10, "bio", "Glucose → Lactic Acid",
                 "Exceeded prediction after pH opt. Promoting to v2 for replication."),
            ]:
                session.add(Annotation(
                    timestamp=row[0], user=row[1], exp_id=row[2],
                    exp_type=row[3], target_name=row[4], text=row[5],
                ))
            session.commit()
        finally:
            session.close()

    def count_experiments(self) -> int:
        session = self._session()
        try:
            return session.query(func.count(Experiment.id)).scalar() or 0
        finally:
            session.close()

    # ─── experiments ────────────────────────────────────────────────────────

    def log_experiment(self, exp_type: str, name: str, pred_value: float,
                       actual_value: float, metric: str, notes: str = "",
                       composition: dict | None = None, user: str = "anonymous",
                       version_tag: str = "", data_quality: str = "good",
                       source_provenance: str = "internal_experiment",
                       reaction_run_id: int | None = None) -> int:
        session = self._session()
        try:
            if not version_tag:
                n = session.query(func.count(Experiment.id)).filter(
                    Experiment.name == name, Experiment.exp_type == exp_type
                ).scalar() or 0
                version_tag = f"v{n + 1}"
            exp = Experiment(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                exp_type=exp_type, name=name, pred_value=pred_value,
                actual_value=actual_value, metric=metric, notes=notes,
                composition=json.dumps(composition or {}),
                user=user, version_tag=version_tag,
                data_quality=data_quality, source_provenance=source_provenance,
                reaction_run_id=reaction_run_id,
            )
            session.add(exp)
            session.commit()
            return exp.id
        finally:
            session.close()

    def get_experiments(self, exp_type: str | None = None) -> pd.DataFrame:
        session = self._session()
        try:
            q = session.query(Experiment).order_by(Experiment.timestamp.asc())
            if exp_type:
                q = q.filter(Experiment.exp_type == exp_type)
            rows = [self._exp_to_dict(e) for e in q.all()]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            session.close()

    @staticmethod
    def _exp_to_dict(e: Experiment) -> dict:
        return {
            "id": e.id, "timestamp": e.timestamp, "exp_type": e.exp_type,
            "name": e.name, "pred_value": e.pred_value,
            "actual_value": e.actual_value, "metric": e.metric,
            "notes": e.notes, "composition": e.composition,
            "user": e.user, "version_tag": e.version_tag,
            "data_quality": e.data_quality,
            "source_provenance": e.source_provenance,
            "reaction_run_id": e.reaction_run_id,
        }

    def get_experiment_history(self, name: str, exp_type: str) -> pd.DataFrame:
        session = self._session()
        try:
            rows = session.query(Experiment).filter(
                Experiment.name == name, Experiment.exp_type == exp_type
            ).order_by(Experiment.timestamp.asc()).all()
            return pd.DataFrame([self._exp_to_dict(r) for r in rows])
        finally:
            session.close()

    # ─── model versions ─────────────────────────────────────────────────────

    def record_retrain(self, exp_type: str, mae: float, rmse: float,
                       n_samples: int, model_name: str = "default",
                       task: str = "yield", promoted: bool = False,
                       metrics: dict | None = None):
        session = self._session()
        try:
            session.add(ModelVersion(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                exp_type=exp_type, mae=mae, rmse=rmse, n_samples=n_samples,
                model_name=model_name, task=task,
                metrics_json=json.dumps(metrics or {}), promoted=promoted,
            ))
            session.commit()
        finally:
            session.close()

    def get_model_versions(self, exp_type: str | None = None) -> pd.DataFrame:
        session = self._session()
        try:
            q = session.query(ModelVersion).order_by(ModelVersion.timestamp.asc())
            if exp_type:
                q = q.filter(ModelVersion.exp_type == exp_type)
            rows = [{
                "id": m.id, "timestamp": m.timestamp, "exp_type": m.exp_type,
                "mae": m.mae, "rmse": m.rmse, "n_samples": m.n_samples,
                "model_name": m.model_name, "task": m.task,
                "promoted": m.promoted,
            } for m in q.all()]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            session.close()

    def get_latest_promoted_model(self, exp_type: str, task: str = "yield") -> dict | None:
        session = self._session()
        try:
            m = session.query(ModelVersion).filter(
                ModelVersion.exp_type == exp_type,
                ModelVersion.task == task,
                ModelVersion.promoted == True,
            ).order_by(desc(ModelVersion.id)).first()
            if not m:
                return None
            return {"mae": m.mae, "rmse": m.rmse, "n_samples": m.n_samples,
                    "timestamp": m.timestamp}
        finally:
            session.close()

    # ─── scenario runs ──────────────────────────────────────────────────────

    def log_scenario_run(self, pathway_name: str, scenario: dict,
                         predicted_yield: float, uncertainty: float,
                         risk_score: float, chosen_plan: str = "",
                         expected_gain: float = 0.0):
        session = self._session()
        try:
            session.add(ScenarioRun(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                pathway_name=pathway_name, scenario_json=json.dumps(scenario),
                predicted_yield=predicted_yield, uncertainty=uncertainty,
                risk_score=risk_score, chosen_plan=chosen_plan,
                expected_gain=expected_gain,
            ))
            session.commit()
        finally:
            session.close()

    def get_scenario_runs(self, limit: int = 50) -> pd.DataFrame:
        session = self._session()
        try:
            rows = session.query(ScenarioRun).order_by(
                desc(ScenarioRun.id)
            ).limit(limit).all()
            return pd.DataFrame([{
                "id": r.id, "timestamp": r.timestamp,
                "pathway_name": r.pathway_name, "scenario_json": r.scenario_json,
                "predicted_yield": r.predicted_yield, "uncertainty": r.uncertainty,
                "risk_score": r.risk_score, "chosen_plan": r.chosen_plan,
                "expected_gain": r.expected_gain,
            } for r in rows])
        finally:
            session.close()

    # ─── experiment queue ───────────────────────────────────────────────────

    def queue_experiment(self, exp_type: str, candidate_name: str, plan_text: str,
                         predicted_value: float, risk_score: float,
                         payload: dict | None = None):
        session = self._session()
        try:
            session.add(ExperimentQueue(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                exp_type=exp_type, candidate_name=candidate_name,
                plan_text=plan_text, predicted_value=predicted_value,
                risk_score=risk_score, payload_json=json.dumps(payload or {}),
                status="queued",
            ))
            session.commit()
        finally:
            session.close()

    def get_experiment_queue(self, status: str | None = None) -> pd.DataFrame:
        session = self._session()
        try:
            q = session.query(ExperimentQueue).order_by(desc(ExperimentQueue.id))
            if status:
                q = q.filter(ExperimentQueue.status == status)
            rows = [{
                "id": e.id, "timestamp": e.timestamp, "exp_type": e.exp_type,
                "candidate_name": e.candidate_name, "plan_text": e.plan_text,
                "predicted_value": e.predicted_value, "risk_score": e.risk_score,
                "payload_json": e.payload_json, "status": e.status,
                "actual_value": e.actual_value, "notes": e.notes,
            } for e in q.all()]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            session.close()

    def complete_queued_experiment(self, queue_id: int, actual_value: float,
                                   notes: str = ""):
        session = self._session()
        try:
            item = session.query(ExperimentQueue).get(queue_id)
            if item:
                item.status = "completed"
                item.actual_value = actual_value
                item.notes = notes
                session.commit()
        finally:
            session.close()

    # ─── annotations ────────────────────────────────────────────────────────

    def add_annotation(self, user: str, text: str, exp_id: int | None = None,
                       exp_type: str = "", target_name: str = "",
                       reaction_run_id: int | None = None):
        session = self._session()
        try:
            session.add(Annotation(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                user=user, exp_id=exp_id, exp_type=exp_type,
                target_name=target_name, text=text,
                reaction_run_id=reaction_run_id,
            ))
            session.commit()
        finally:
            session.close()

    def get_annotations(self, exp_id: int | None = None,
                        exp_type: str | None = None) -> pd.DataFrame:
        session = self._session()
        try:
            q = session.query(Annotation).order_by(desc(Annotation.id))
            if exp_id is not None:
                q = q.filter(Annotation.exp_id == exp_id)
            if exp_type:
                q = q.filter(Annotation.exp_type == exp_type)
            rows = [{
                "id": a.id, "timestamp": a.timestamp, "user": a.user,
                "exp_id": a.exp_id, "exp_type": a.exp_type,
                "target_name": a.target_name, "text": a.text,
                "reaction_run_id": a.reaction_run_id,
            } for a in q.all()]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            session.close()

    def get_user_activity(self) -> pd.DataFrame:
        session = self._session()
        try:
            rows = session.query(
                Experiment.user,
                func.count(Experiment.id).label("experiments"),
                func.max(Experiment.timestamp).label("last_active"),
            ).group_by(Experiment.user).order_by(desc("experiments")).all()
            return pd.DataFrame([{
                "user": r.user, "experiments": r.experiments,
                "last_active": r.last_active,
            } for r in rows])
        finally:
            session.close()

    def get_provenance_summary(self) -> pd.DataFrame:
        session = self._session()
        try:
            rows = session.query(
                Experiment.source_provenance, Experiment.data_quality,
                func.count(Experiment.id).label("count"),
            ).group_by(
                Experiment.source_provenance, Experiment.data_quality
            ).all()
            return pd.DataFrame([{
                "source_provenance": r.source_provenance,
                "data_quality": r.data_quality, "count": r.count,
            } for r in rows])
        finally:
            session.close()

    # ─── reactions ──────────────────────────────────────────────────────────

    def create_reaction(self, name: str, rxn_smarts: str,
                        reactants: list, products: list,
                        domain: str = "organic", tags: list | None = None,
                        created_by: str = "system", is_public: bool = True,
                        forked_from: int | None = None,
                        base_yield: float = 0.75) -> int:
        session = self._session()
        try:
            r = Reaction(
                name=name, rxn_smarts=rxn_smarts,
                reactants_json=json.dumps(reactants),
                products_json=json.dumps(products),
                domain=domain, tags_json=json.dumps(tags or []),
                created_by=created_by, is_public=is_public,
                forked_from=forked_from, base_yield=base_yield,
            )
            session.add(r)
            session.commit()
            return r.id
        finally:
            session.close()

    def get_reaction(self, reaction_id: int) -> dict | None:
        session = self._session()
        try:
            r = session.query(Reaction).get(reaction_id)
            return self._reaction_to_dict(r) if r else None
        finally:
            session.close()

    def get_reaction_by_name(self, name: str) -> dict | None:
        session = self._session()
        try:
            r = session.query(Reaction).filter(Reaction.name == name).first()
            return self._reaction_to_dict(r) if r else None
        finally:
            session.close()

    def list_reactions(self, domain: str | None = None,
                       public_only: bool = True) -> list[dict]:
        session = self._session()
        try:
            q = session.query(Reaction).order_by(Reaction.name)
            if public_only:
                q = q.filter(Reaction.is_public == True)
            if domain:
                q = q.filter(Reaction.domain == domain)
            return [self._reaction_to_dict(r) for r in q.all()]
        finally:
            session.close()

    def search_reactions(self, query: str = "", domain: str | None = None,
                         tag: str | None = None) -> list[dict]:
        session = self._session()
        try:
            q = session.query(Reaction).filter(Reaction.is_public == True)
            if domain:
                q = q.filter(Reaction.domain == domain)
            if query:
                q = q.filter(or_(
                    Reaction.name.ilike(f"%{query}%"),
                    Reaction.rxn_smarts.ilike(f"%{query}%"),
                ))
            results = [self._reaction_to_dict(r) for r in q.all()]
            if tag:
                results = [r for r in results if tag in r.get("tags", [])]
            return results
        finally:
            session.close()

    def count_reactions(self) -> int:
        session = self._session()
        try:
            return session.query(func.count(Reaction.id)).scalar() or 0
        finally:
            session.close()

    @staticmethod
    def _reaction_to_dict(r: Reaction) -> dict:
        return {
            "id": r.id, "name": r.name, "rxn_smarts": r.rxn_smarts,
            "reactants": json.loads(r.reactants_json or "[]"),
            "products": json.loads(r.products_json or "[]"),
            "domain": r.domain,
            "tags": json.loads(r.tags_json or "[]"),
            "created_by": r.created_by, "is_public": r.is_public,
            "forked_from": r.forked_from, "base_yield": r.base_yield,
            "created_at": str(r.created_at) if r.created_at else "",
        }

    def fork_reaction(self, reaction_id: int, user: str) -> int | None:
        src = self.get_reaction(reaction_id)
        if not src:
            return None
        return self.create_reaction(
            name=f"{src['name']} (fork)",
            rxn_smarts=src["rxn_smarts"],
            reactants=src["reactants"], products=src["products"],
            domain=src["domain"], tags=src["tags"],
            created_by=user, forked_from=reaction_id,
            base_yield=src.get("base_yield", 0.75),
        )

    # ─── reaction runs ──────────────────────────────────────────────────────

    def create_reaction_run(self, reaction_id: int | None, conditions: dict,
                            predicted: dict, user_id: str = "anonymous",
                            provenance: str = "internal_experiment",
                            quality_score: float = 0.0,
                            visibility: str = "public") -> int:
        session = self._session()
        try:
            run = ReactionRun(
                reaction_id=reaction_id,
                conditions_json=json.dumps(conditions),
                predicted_json=json.dumps(predicted),
                status="predicted", user_id=user_id,
                provenance=provenance, quality_score=quality_score,
                visibility=visibility,
            )
            session.add(run)
            session.commit()
            return run.id
        finally:
            session.close()

    def update_reaction_run_actual(self, run_id: int, actual: dict,
                                   quality_score: float | None = None):
        session = self._session()
        try:
            run = session.query(ReactionRun).get(run_id)
            if run:
                run.actual_json = json.dumps(actual)
                run.status = "completed"
                if quality_score is not None:
                    run.quality_score = quality_score
                session.commit()
        finally:
            session.close()

    def get_reaction_run(self, run_id: int) -> dict | None:
        session = self._session()
        try:
            run = session.query(ReactionRun).get(run_id)
            return self._run_to_dict(run) if run else None
        finally:
            session.close()

    def list_reaction_runs(self, reaction_id: int | None = None,
                           public_only: bool = True,
                           limit: int = 100) -> list[dict]:
        session = self._session()
        try:
            q = session.query(ReactionRun).order_by(desc(ReactionRun.id))
            if reaction_id:
                q = q.filter(ReactionRun.reaction_id == reaction_id)
            if public_only:
                q = q.filter(ReactionRun.visibility == "public")
            return [self._run_to_dict(r) for r in q.limit(limit).all()]
        finally:
            session.close()

    def get_training_eligible_runs(self, min_quality: float = 0.6) -> list[dict]:
        session = self._session()
        try:
            flagged_ids = {f.target_id for f in session.query(Flag).filter(
                Flag.target_type == "reaction_run"
            ).all()}
            runs = session.query(ReactionRun).filter(
                ReactionRun.status == "completed",
                ReactionRun.quality_score >= min_quality,
            ).all()
            return [
                self._run_to_dict(r) for r in runs
                if r.id not in flagged_ids
            ]
        finally:
            session.close()

    def count_eligible_runs_since(self, since_id: int = 0) -> int:
        session = self._session()
        try:
            return session.query(func.count(ReactionRun.id)).filter(
                ReactionRun.id > since_id,
                ReactionRun.status == "completed",
                ReactionRun.quality_score >= 0.6,
            ).scalar() or 0
        finally:
            session.close()

    @staticmethod
    def _run_to_dict(run: ReactionRun) -> dict:
        return {
            "id": run.id, "reaction_id": run.reaction_id,
            "conditions": json.loads(run.conditions_json or "{}"),
            "predicted": json.loads(run.predicted_json or "{}"),
            "actual": json.loads(run.actual_json or "{}"),
            "status": run.status, "user_id": run.user_id,
            "provenance": run.provenance,
            "quality_score": run.quality_score,
            "visibility": run.visibility,
            "created_at": str(run.created_at) if run.created_at else "",
        }

    # ─── reaction requests ────────────────────────────────────────────────────

    def queue_unsupported_reaction(self, user_input: str,
                                   user_id: str = "anonymous") -> int:
        session = self._session()
        try:
            req = ReactionRequest(user_input=user_input, user_id=user_id)
            session.add(req)
            session.commit()
            return req.id
        finally:
            session.close()

    def list_reaction_requests(self, status: str = "pending") -> list[dict]:
        session = self._session()
        try:
            rows = session.query(ReactionRequest).filter(
                ReactionRequest.status == status
            ).order_by(desc(ReactionRequest.id)).all()
            return [{
                "id": r.id, "user_input": r.user_input,
                "status": r.status, "user_id": r.user_id,
                "created_at": str(r.created_at),
            } for r in rows]
        finally:
            session.close()

    # ─── users ──────────────────────────────────────────────────────────────

    def get_or_create_user(self, username: str, email: str = "",
                           password_hash: str = "") -> int:
        session = self._session()
        try:
            u = session.query(User).filter(User.username == username).first()
            if u:
                return u.id
            u = User(username=username, email=email, password_hash=password_hash)
            session.add(u)
            session.commit()
            return u.id
        finally:
            session.close()

    def increment_reputation(self, username: str, amount: float = 0.1):
        session = self._session()
        try:
            u = session.query(User).filter(User.username == username).first()
            if u:
                u.reputation = (u.reputation or 0) + amount
                session.commit()
        finally:
            session.close()

    def get_user_reputation(self, username: str) -> float:
        session = self._session()
        try:
            u = session.query(User).filter(User.username == username).first()
            return u.reputation if u else 0.0
        finally:
            session.close()

    def top_contributors(self, limit: int = 10) -> pd.DataFrame:
        session = self._session()
        try:
            rows = session.query(
                ReactionRun.user_id,
                func.count(ReactionRun.id).label("runs"),
            ).filter(
                ReactionRun.status == "completed",
                ReactionRun.quality_score >= 0.6,
            ).group_by(ReactionRun.user_id).order_by(
                desc("runs")
            ).limit(limit).all()
            return pd.DataFrame([{"user": r.user_id, "runs": r.runs} for r in rows])
        finally:
            session.close()

    # ─── flags ──────────────────────────────────────────────────────────────

    def add_flag(self, target_type: str, target_id: int,
                 user_id: str, reason: str):
        session = self._session()
        try:
            session.add(Flag(
                target_type=target_type, target_id=target_id,
                user_id=user_id, reason=reason,
            ))
            session.commit()
        finally:
            session.close()

    def is_flagged(self, target_type: str, target_id: int) -> bool:
        session = self._session()
        try:
            return session.query(Flag).filter(
                Flag.target_type == target_type,
                Flag.target_id == target_id,
            ).count() > 0
        finally:
            session.close()

    # ─── job queue ──────────────────────────────────────────────────────────

    def enqueue_job(self, job_type: str, payload: dict) -> int:
        session = self._session()
        try:
            job = JobQueue(job_type=job_type, payload_json=json.dumps(payload))
            session.add(job)
            session.commit()
            return job.id
        finally:
            session.close()

    def get_pending_jobs(self, limit: int = 10) -> list[dict]:
        session = self._session()
        try:
            rows = session.query(JobQueue).filter(
                JobQueue.status == "queued"
            ).order_by(JobQueue.scheduled_at).limit(limit).all()
            return [{
                "id": j.id, "job_type": j.job_type,
                "payload": json.loads(j.payload_json or "{}"),
                "status": j.status,
            } for j in rows]
        finally:
            session.close()

    def complete_job(self, job_id: int, result: dict, status: str = "completed"):
        session = self._session()
        try:
            job = session.query(JobQueue).get(job_id)
            if job:
                job.status = status
                job.result_json = json.dumps(result)
                job.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    def get_job(self, job_id: int) -> dict | None:
        session = self._session()
        try:
            j = session.query(JobQueue).get(job_id)
            if not j:
                return None
            return {
                "id": j.id, "job_type": j.job_type,
                "payload": json.loads(j.payload_json or "{}"),
                "status": j.status,
                "result": json.loads(j.result_json or "{}"),
                "scheduled_at": str(j.scheduled_at),
                "completed_at": str(j.completed_at) if j.completed_at else None,
            }
        finally:
            session.close()

    # ─── benchmarks ───────────────────────────────────────────────────────

    def save_benchmark_run(self, suite_name: str, metrics: dict) -> int:
        session = self._session()
        try:
            b = BenchmarkRun(suite_name=suite_name, metrics_json=json.dumps(metrics))
            session.add(b)
            session.commit()
            return b.id
        finally:
            session.close()

    def get_benchmark_runs(self, suite_name: str | None = None,
                           limit: int = 20) -> pd.DataFrame:
        session = self._session()
        try:
            q = session.query(BenchmarkRun).order_by(desc(BenchmarkRun.id))
            if suite_name:
                q = q.filter(BenchmarkRun.suite_name == suite_name)
            rows = [{
                "id": b.id, "suite_name": b.suite_name,
                "metrics": json.loads(b.metrics_json or "{}"),
                "created_at": str(b.created_at),
            } for b in q.limit(limit).all()]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            session.close()


_repo: Repository | None = None


def get_repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = Repository()
    return _repo
