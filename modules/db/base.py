"""SQLAlchemy engine factory — SQLite default, Postgres when DATABASE_URL is set."""
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_DB = _DATA_DIR / "chemai.db"
_LEGACY_DB = _DATA_DIR / "experiments.db"

_engine = None
_SessionLocal = None


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_DB.as_posix()}"


def init_engine():
    global _engine, _SessionLocal
    url = get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)

    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_engine():
    if _engine is None:
        init_engine()
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal()


def legacy_db_path() -> Path:
    return _LEGACY_DB
