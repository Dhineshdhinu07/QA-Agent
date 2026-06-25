"""
Central configuration for QA Agent.

Reads all environment variables from .env and parses capability_map.yaml.
Every other file in the project imports the `settings` singleton from here
instead of calling os.getenv() directly.

Why this matters: if a required variable is missing, this module raises a
clear error at startup — not halfway through a test run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root of the repo — used to locate capability_map.yaml regardless of
# which directory the CLI is invoked from.
REPO_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    All environment variables the system needs, declared as typed fields.

    pydantic-settings reads these automatically from the .env file.
    If a field has no default and is missing from .env, the app exits
    immediately with a clear message about which variable is missing.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # ANTHROPIC_API_KEY and anthropic_api_key are the same
        extra="ignore",         # ignore unknown vars in .env instead of erroring
    )

    # ── AI ────────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(description="Your Anthropic API key")

    # ── Sandbox ───────────────────────────────────────────────────────────────
    sandbox_url: str = Field(description="Base URL for the Friendbuy sandbox")
    sandbox_username: str = Field(description="Login email for sandbox")
    sandbox_password: str = Field(description="Login password for sandbox")
    sandbox_merchant_name: str = Field(
        default="Queen's Consolidated",
        description="Merchant to select after login — always this value unless overridden",
    )

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str = Field(description="GitHub personal access token for posting PR comments")
    github_repo: str = Field(description="Target repo in org/repo format, e.g. friendbuy/platform")

    # ── Jira (Phase 2 — not required in Phase 1) ──────────────────────────────
    jira_base_url: str = Field(default="", description="e.g. https://friendbuy.atlassian.net")
    jira_email: str = Field(default="")
    jira_api_token: str = Field(default="")

    # ── Langfuse (Phase 2 — tracing) ──────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    @field_validator("github_repo")
    @classmethod
    def validate_github_repo_format(cls, v: str) -> str:
        """Ensure github_repo is in org/repo format before anything tries to use it."""
        if v and "/" not in v:
            raise ValueError(
                f"GITHUB_REPO must be in 'org/repo' format, got: '{v}'"
            )
        return v


class CapabilityMap:
    """
    Wraps the capability_map.yaml file.

    Provides a single method: get_model(capability) → model ID string.
    This is how every node looks up which Claude model to use without
    hardcoding model names in business logic.
    """

    def __init__(self, path: Path) -> None:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # The YAML has a top-level "capability_map" key
        self._map: dict[str, str] = raw.get("capability_map", {})

    def get_model(self, capability: str) -> str:
        """
        Look up the model ID for a capability.

        Raises KeyError with a helpful message if the capability isn't
        in the YAML — so misconfiguration is caught early.
        """
        if capability not in self._map:
            available = ", ".join(self._map.keys())
            raise KeyError(
                f"Unknown capability '{capability}'. "
                f"Available capabilities: {available}"
            )
        return self._map[capability]

    def all(self) -> dict[str, str]:
        return dict(self._map)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.

    @lru_cache ensures .env is only read once per process, not on every
    function call. Call get_settings() anywhere — it's always the same object.
    """
    return Settings()


@lru_cache(maxsize=1)
def get_capability_map() -> CapabilityMap:
    """Returns the singleton CapabilityMap parsed from capability_map.yaml."""
    yaml_path = REPO_ROOT / "capability_map.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"capability_map.yaml not found at {yaml_path}. "
            "Make sure you are running from the repo root."
        )
    return CapabilityMap(yaml_path)


# Do NOT create module-level singletons here.
# Importing this module must never fail even when .env is absent
# (e.g. during testing or capability_map-only lookups).
#
# Other files should call get_settings() / get_capability_map() explicitly:
#
#   from qa_agent.config import get_settings, get_capability_map
#   settings = get_settings()          # call once at the top of a node file
#   capability_map = get_capability_map()
#
# The @lru_cache on both functions guarantees they are only computed once
# per process regardless of how many times they are called.
