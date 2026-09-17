"""SQLAlchemy ORM models for ChemAI reaction library."""
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(128), default="")
    password_hash = Column(String(256), default="")
    reputation = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Reaction(Base):
    __tablename__ = "reactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    rxn_smarts = Column(Text, nullable=False)
    reactants_json = Column(Text, default="[]")
    products_json = Column(Text, default="[]")
    domain = Column(String(64), default="organic")
    tags_json = Column(Text, default="[]")
    created_by = Column(String(64), default="system")
    is_public = Column(Boolean, default=True)
    forked_from = Column(Integer, ForeignKey("reactions.id"), nullable=True)
    base_yield = Column(Float, default=0.75)
    created_at = Column(DateTime, default=datetime.utcnow)

    runs = relationship("ReactionRun", back_populates="reaction")


class ReactionRun(Base):
    __tablename__ = "reaction_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reaction_id = Column(Integer, ForeignKey("reactions.id"), nullable=True)
    conditions_json = Column(Text, default="{}")
    predicted_json = Column(Text, default="{}")
    actual_json = Column(Text, default="{}")
    status = Column(String(32), default="predicted")
    user_id = Column(String(64), default="anonymous")
    provenance = Column(String(64), default="internal_experiment")
    quality_score = Column(Float, default=0.0)
    visibility = Column(String(16), default="public")
    created_at = Column(DateTime, default=datetime.utcnow)

    reaction = relationship("Reaction", back_populates="runs")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(32), nullable=False)
    exp_type = Column(String(32), nullable=False)
    name = Column(String(256), nullable=False)
    pred_value = Column(Float)
    actual_value = Column(Float)
    metric = Column(String(32))
    notes = Column(Text, default="")
    composition = Column(Text, default="{}")
    user = Column(String(64), default="anonymous")
    version_tag = Column(String(16), default="v1")
    data_quality = Column(String(32), default="good")
    source_provenance = Column(String(64), default="internal_experiment")
    reaction_run_id = Column(Integer, ForeignKey("reaction_runs.id"), nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(32), nullable=False)
    exp_type = Column(String(32), nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    n_samples = Column(Integer)
    model_name = Column(String(64), default="default")
    task = Column(String(64), default="yield")
    metrics_json = Column(Text, default="{}")
    promoted = Column(Boolean, default=False)


class ScenarioRun(Base):
    __tablename__ = "scenario_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(32), nullable=False)
    pathway_name = Column(String(256), nullable=False)
    scenario_json = Column(Text, nullable=False)
    predicted_yield = Column(Float)
    uncertainty = Column(Float)
    risk_score = Column(Float)
    chosen_plan = Column(Text, default="")
    expected_gain = Column(Float, default=0.0)


class ExperimentQueue(Base):
    __tablename__ = "experiment_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(32), nullable=False)
    exp_type = Column(String(32), nullable=False)
    candidate_name = Column(String(256), nullable=False)
    plan_text = Column(Text, default="")
    predicted_value = Column(Float)
    risk_score = Column(Float)
    payload_json = Column(Text, default="{}")
    status = Column(String(32), default="queued")
    actual_value = Column(Float)
    notes = Column(Text, default="")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(32), nullable=False)
    user = Column(String(64), nullable=False)
    exp_id = Column(Integer, nullable=True)
    exp_type = Column(String(32), default="")
    target_name = Column(String(256), default="")
    text = Column(Text, nullable=False)
    reaction_run_id = Column(Integer, ForeignKey("reaction_runs.id"), nullable=True)


class ReactionRequest(Base):
    __tablename__ = "reaction_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_input = Column(Text, nullable=False)
    status = Column(String(32), default="pending")
    user_id = Column(String(64), default="anonymous")
    created_at = Column(DateTime, default=datetime.utcnow)


class JobQueue(Base):
    __tablename__ = "job_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(64), nullable=False)
    payload_json = Column(Text, default="{}")
    status = Column(String(32), default="queued")
    scheduled_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result_json = Column(Text, default="{}")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    suite_name = Column(String(128), nullable=False)
    metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class Flag(Base):
    __tablename__ = "flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_type = Column(String(32), nullable=False)
    target_id = Column(Integer, nullable=False)
    user_id = Column(String(64), default="anonymous")
    reason = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
