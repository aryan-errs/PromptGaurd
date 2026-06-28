"""Server configuration — loaded from environment variables and optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RiskTier = Literal["low", "medium", "high"]


class Settings(BaseSettings):
    """All configuration for the PromptGuard server.

    Variables are read from the environment with the ``PROMPTGUARD_`` prefix,
    and from a ``.env`` file in the current working directory if present.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROMPTGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Default AppProfile ─────────────────────────────────────────────────
    profile_name: str = Field("default", description="Profile name tag")
    risk_tier: RiskTier = Field("medium", description="Detection sensitivity tier")
    allow_security_discussion: bool = Field(
        False,
        description="Raise semantic thresholds for security-chatbot deployments",
    )
    template_delimiters: list[str] = Field(
        default_factory=list,
        description="App template delimiters (comma-separated in env)",
    )
    tools_enabled: bool = Field(False, description="Mark deployment as tools-enabled (higher risk)")

    # ── Auth ───────────────────────────────────────────────────────────────
    api_key: str | None = Field(
        None,
        description="Shared secret sent in X-API-Key header; None = auth disabled",
    )

    # ── Rate limiting ──────────────────────────────────────────────────────
    rate_limit_rpm: Annotated[int, Field(ge=0)] = Field(
        0, description="Max requests per minute per IP; 0 = disabled"
    )

    # ── Request limits ─────────────────────────────────────────────────────
    max_request_bytes: Annotated[int, Field(ge=64)] = Field(
        65_536, description="Maximum request body size in bytes (default 64 KB)"
    )

    # ── ML classifier ─────────────────────────────────────────────────────
    classifier_model_path: str | None = Field(
        None,
        description=(
            "Path to a trained Backend A .pkl artifact. "
            "When set, readyz waits until the model is warm. "
            "When absent, the heuristic fallback is used and readyz is immediately ready."
        ),
    )

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0", description="Bind host")
    port: int = Field(8000, description="Bind port")
    workers: int = Field(1, description="Uvicorn worker count")
    log_level: str = Field("info", description="Uvicorn log level")

    @field_validator("template_delimiters", mode="before")
    @classmethod
    def _parse_delimiters(cls, v: object) -> list[str]:
        """Allow comma-separated string from env var."""
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        return list(v) if v else []

    def to_app_profile(self) -> AppProfile:  # noqa: F821
        from promptguard import AppProfile

        return AppProfile(
            name=self.profile_name,
            allow_security_discussion=self.allow_security_discussion,
            risk_tier=self.risk_tier,
            template_delimiters=self.template_delimiters,
            tools_enabled=self.tools_enabled,
        )

    @property
    def classifier_model_path_obj(self) -> Path | None:
        return Path(self.classifier_model_path) if self.classifier_model_path else None


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
