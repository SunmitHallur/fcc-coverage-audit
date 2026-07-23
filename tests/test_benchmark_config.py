"""Benchmark section must be a top-level config key (not nested under scoring)."""
from __future__ import annotations

from pathlib import Path

from fcc_audit.config import load_config


def test_benchmark_section_exists_with_sixteen_counties():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config" / "pipeline.yaml")
    bench = cfg.raw.get("benchmark")
    assert isinstance(bench, dict), "config must have a top-level 'benchmark' section"
    counties = bench.get("counties") or []
    assert len(counties) == 16, f"expected 16 labeled counties, got {len(counties)}"
    assert bench.get("service_label") == "5G-NR 7/1"
    assert "vintages" in bench
    # Must not accidentally live under scoring (previous YAML indent bug).
    scoring = cfg.raw.get("scoring") or {}
    assert "counties" not in scoring
    assert "service_label" not in scoring
