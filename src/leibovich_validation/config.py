from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings read from the environment or the repository's `.env` file."""

    model_config = SettingsConfigDict(env_prefix="LP_", env_file=REPO_ROOT / ".env", extra="ignore")

    data_dir: Path

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()


def get_settings() -> Settings:
    return Settings()
