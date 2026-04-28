.PHONY: verify test pytest lint security coverage

PYTHON ?= python3

verify: lint security test coverage

lint:
	$(PYTHON) -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	$(PYTHON) tools/security_check.py --strict

test:
	$(PYTHON) -m unittest -q

pytest:
	./scripts/bootstrap_pytest_python3.sh -m pytest

coverage:
	$(PYTHON) tools/coverage_gate.py --min 0.70
