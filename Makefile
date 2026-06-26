.PHONY: test lint typecheck verify clean

PYTHON ?= python
PYTEST  = $(PYTHON) -m pytest
RUFF    = $(PYTHON) -m ruff
MYPY    = $(PYTHON) -m mypy

# ── Quality gates ─────────────────────────────────────────────────────────────

test:
	PYTHONPATH=src $(PYTEST) tests/ -v --tb=short -x --timeout=120

# Offline fixture-only tests (fast, no network required)
test-fixtures:
	PYTHONPATH=src $(PYTEST) tests/test_pipeline.py tests/test_validate.py \
	    tests/test_attribute.py tests/test_web_bundle.py tests/test_property.py \
	    -v --tb=short -x --timeout=120

lint:
	$(RUFF) check src/ tests/ --fix

typecheck:
	$(MYPY) src/fcc_audit/ --ignore-missing-imports --no-strict-optional \
	    --exclude 'map_render|reconcile'

# Full verification: lint + typecheck + fixture tests
verify: lint typecheck test-fixtures
	@echo "✓ All quality gates passed."

clean:
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '.hypothesis' -exec rm -rf {} + 2>/dev/null || true
