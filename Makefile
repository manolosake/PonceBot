.PHONY: verify test lint security coverage delivery-evidence-gate wormhole-trace-export patch-status-reconcile crosslane-validate

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

wormhole-trace-export:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	python3 tools/wormhole_trace_export.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--reported-branch-mode "expected"

patch-status-reconcile:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	python3 tools/patch_status_reconciler.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--out "$(ARTIFACTS_DIR)/patch_status_reconciler_report.json"

crosslane-validate:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	python3 tools/crosslane_traceability_validator.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--out "$(ARTIFACTS_DIR)/crosslane_validator_report.json"
