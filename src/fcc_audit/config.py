"""Configuration loading and shared dataclasses.

Loads ``config/pipeline.yaml``, expands ``${ENV_VAR}`` references (used for
Redshift secrets), and resolves paths relative to the project root.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} references using the environment."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass(frozen=True)
class Provider:
    id: int
    name: str


@dataclass
class Config:
    """Parsed pipeline configuration with convenience accessors."""

    raw: dict[str, Any]
    project_root: Path

    # ---- source ----
    @property
    def backend(self) -> str:
        return self.raw["source"]["backend"]

    @property
    def fcc(self) -> dict[str, Any]:
        return self.raw["source"]["fcc"]

    @property
    def redshift(self) -> dict[str, Any]:
        return self.raw["source"]["redshift"]

    @property
    def fixture(self) -> dict[str, Any]:
        return self.raw["source"]["fixture"]

    # ---- analysis ----
    @property
    def providers_all(self) -> bool:
        """True when providers should be auto-discovered from the catalog."""
        p = self.raw["analysis"]["providers"]
        return isinstance(p, str) and p.lower() == "all"

    @property
    def providers(self) -> list[Provider]:
        """Explicit provider list, or [] when set to 'all' (use discovery)."""
        p = self.raw["analysis"]["providers"]
        if self.providers_all:
            return []
        return [Provider(**x) for x in p]

    @property
    def known_providers(self) -> list[Provider]:
        return [Provider(**x) for x in self.raw["analysis"].get("known_providers", [])]

    @property
    def technologies(self) -> dict:
        """Map technology name -> {code, tiers: {label: {mindown, minup}}}."""
        return self.raw["analysis"]["technologies"]

    def analysis_units(self):
        """Yield (technology, tier_label, tier_spec) for every tech/tier to run."""
        for tech, spec in self.technologies.items():
            for tier_label, tier_spec in spec.get("tiers", {}).items():
                yield tech, tier_label, tier_spec

    @property
    def combine_environments(self) -> bool:
        return bool(self.raw["analysis"].get("combine_environments", True))

    @property
    def environments(self) -> list[str]:
        return self.raw["analysis"].get("environments", [])

    @property
    def environment_codes(self) -> dict:
        return self.raw["analysis"].get("environment_codes", {})

    def environment_groups(self) -> list[tuple[str, list[int] | None]]:
        """Return (label, codes) groups to iterate. When combining, a single
        group with codes=None (no environment filtering)."""
        if self.combine_environments:
            return [("all", None)]
        return [(env, self.environment_codes.get(env)) for env in self.environments]

    @property
    def vintage_current(self) -> str | None:
        return self.raw["analysis"]["vintages"]["current"]

    @property
    def vintage_prior(self) -> str | None:
        return self.raw["analysis"]["vintages"]["prior"]

    # ---- geography ----
    @property
    def geography(self) -> dict[str, Any]:
        return self.raw["geography"]

    @property
    def towers(self) -> dict[str, Any]:
        return self.raw["towers"]

    @property
    def reconcile(self) -> dict[str, Any]:
        return self.raw["reconcile"]

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw["scoring"]

    # ---- paths (resolved absolute) ----
    def path(self, key: str) -> Path:
        p = self.project_root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def provider_by_id(self, provider_id: int) -> Provider | None:
        for p in (*self.providers, *self.known_providers):
            if p.id == provider_id:
                return p
        return None


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a directory containing config/ is found."""
    start = (start or Path(__file__)).resolve()
    for parent in [start, *start.parents]:
        if (parent / "config" / "pipeline.yaml").exists():
            return parent
    # Fallback: two levels up from this file (src/fcc_audit/ -> project root)
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> Config:
    """Load and parse the pipeline configuration."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / "config" / "pipeline.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _expand_env(raw)
    return Config(raw=raw, project_root=root)
