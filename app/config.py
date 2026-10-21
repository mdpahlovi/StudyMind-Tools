"""Shared environment loading for the tools service."""

from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_environment() -> None:
    """Loads Tools/.env; variables already set in the environment win."""
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
