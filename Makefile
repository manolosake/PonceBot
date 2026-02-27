.PHONY: verify test lint security coverage

verify: lint security test coverage

lint:
	python -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	python tools/security_check.py --strict

test:
	python -m unittest -q

coverage:
	python tools/coverage_gate.py --min 0.65
