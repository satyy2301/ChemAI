"""ChemAI access — open to all users (no login required)."""

DEFAULT_USER = "guest"


def get_current_user() -> str:
    return DEFAULT_USER
