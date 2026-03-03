.PHONY: verify test lint security coverage delivery-evidence-gate

verify: lint security test coverage

lint:
	python -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	python tools/security_check.py --strict

test:
	python -m unittest -q

coverage:
	python tools/coverage_gate.py --min 0.65

delivery-evidence-gate:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	python3 tools/delivery_evidence_gate.py --artifacts-dir "$(ARTIFACTS_DIR)" --workspace-dir "."
