"""ChemAI FastAPI application."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import library, models, reactions, runs
from infra.retrain_jobs.retrain_worker import process_job_queue, run_retrain_job
from modules import feedback as fb
from modules import reaction_library as rl

_scheduler = None


def _start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(run_retrain_job, "cron", day_of_week="sun", hour=2, id="weekly_retrain")
        _scheduler.add_job(process_job_queue, "interval", seconds=60, id="job_poll")
        _scheduler.start()
    except ImportError:
        pass


def _stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    fb.init_db()
    rl.seed_library()
    _start_scheduler()
    yield
    _stop_scheduler()


app = FastAPI(
    title="ChemAI Reaction Library API",
    description="Open Chemistry Reaction Library REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CHEMAI_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reactions.router)
app.include_router(runs.router)
app.include_router(models.router)
app.include_router(library.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "chemai-api"}
