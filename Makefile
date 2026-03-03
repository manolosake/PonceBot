.PHONY: verify test lint security coverage validate-s02-trace

PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

verify: lint security test coverage

lint:
	@if [ -z "$(PYTHON)" ]; then echo "ERROR: python3/python not found"; exit 127; fi
	$(PYTHON) -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	$(PYTHON) tools/security_check.py --strict

test:
	$(PYTHON) -m unittest -q

coverage:
	$(PYTHON) tools/coverage_gate.py --min 0.65

validate-s02-trace:
	@ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	$(PYTHON) tools/s02_trace_checker.py --artifacts-dir "$$ART"
