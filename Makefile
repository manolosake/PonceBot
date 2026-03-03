.PHONY: verify test lint security coverage

PYTHON ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

verify: lint security test coverage

lint:
	$(PYTHON) -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	$(PYTHON) tools/security_check.py --strict

test:
	$(PYTHON) -m unittest -q

coverage:
	$(PYTHON) tools/coverage_gate.py --min 0.65
