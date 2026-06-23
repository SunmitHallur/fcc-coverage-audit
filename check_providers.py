"""One-off: what technologies do these provider_ids actually file in a vintage?
Catalog access needs no auth token (only downloads do), so this is read-only.
"""
import sys
from collections import defaultdict

from fcc_audit.config import load_config
from fcc_audit.acquire import FccDownloadSource, _RAW_COVERAGE_TYPE

VINTAGE = "December 31, 2025"
TARGETS = {13001, 130018, 130067}

cfg = load_config("config/pipeline.yaml")
src = FccDownloadSource(cfg)

catalog = src._catalog_for_vintage(VINTAGE)
techs = defaultdict(set)
for r in catalog:
    if r.get("data_type") != _RAW_COVERAGE_TYPE:
        continue
    try:
        pid = int(r.get("provider_id") or -1)
    except (TypeError, ValueError):
        continue
    if pid in TARGETS:
        techs[pid].add(str(r.get("technology_code_desc")))

print(f"vintage: {VINTAGE}\n")
for pid in sorted(TARGETS):
    if pid in techs:
        print(f"provider {pid}: files for -> {sorted(techs[pid])}")
    else:
        print(f"provider {pid}: NOT in catalog as raw mobile coverage at all")
