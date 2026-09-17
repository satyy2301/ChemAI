"""FastAPI dependencies."""
import os

from fastapi import Header, HTTPException

from modules.db.repository import get_repo


def get_db():
    return get_repo()


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    expected = os.environ.get("CHEMAI_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
