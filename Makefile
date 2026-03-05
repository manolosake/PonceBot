.PHONY: verify test lint security coverage backend-traceability-runtime-export

PYTHON := $(strip $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null))

ifeq ($(PYTHON),)
$(error No Python runtime found. Install python3 (preferred) or python)
endif

verify: lint security test coverage

lint:
	$(PYTHON) -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	$(PYTHON) tools/security_check.py --strict

test:
	$(PYTHON) -m unittest -q

coverage:
	$(PYTHON) tools/coverage_gate.py --min 0.65

backend-traceability-runtime-export:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	@if [ -z "$(FRONTEND_JOB_ID)" ]; then echo "FRONTEND_JOB_ID is required"; exit 2; fi
	python3 tools/backend_traceability_runtime_export.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--frontend-job-id "$(FRONTEND_JOB_ID)" \
		--target-artifact-dir "$(TARGET_ARTIFACT_DIR)"
