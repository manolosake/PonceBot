.PHONY: verify test lint security coverage backend-traceability-runtime-export manifest-drift-check atomic-root-publish-from-smoke close-time-postfinal-guard close-time-terminal-enforce close-regression-harness delivery-evidence-gate wormhole-trace-export backend-provenance-export crosslane-validate wormhole-contract-validate wormhole-contract-export publish-atomic-guard visual-preview-audit visual-preview-smoke validate-s02-trace bundle-s02-atomic patch-status-reconcile

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
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@if [ -z "$(FRONTEND_JOB_ID)" ]; then echo "FRONTEND_JOB_ID is required"; exit 2; fi
	@$(PYTHON) tools/backend_traceability_runtime_export.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--frontend-job-id "$(FRONTEND_JOB_ID)" \
		--target-artifact-dir "$(TARGET_ARTIFACT_DIR)"

manifest-drift-check:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(CRITICAL_FILES)" ]; then echo "CRITICAL_FILES is required (comma-separated)"; exit 2; fi
	@$(PYTHON) tools/manifest_drift_checker.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--critical-files "$(CRITICAL_FILES)" \
		--sleep-seconds "$${SLEEP_SECONDS:-2}" \
		--manifest-t0 "$(ARTIFACTS_DIR)/bundle_manifest_t0.json" \
		--manifest-tn "$(ARTIFACTS_DIR)/bundle_manifest_t_plus_n.json" \
		--report "$(ARTIFACTS_DIR)/manifest_drift_report.json"

close-time-postfinal-guard:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ ! -d "$(ARTIFACTS_DIR)/smoke_stable" ]; then echo "smoke_stable bundle is required"; exit 2; fi
	@$(PYTHON) tools/postfinal_root_smoke_guard.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--smoke-stable-dir "$(ARTIFACTS_DIR)/smoke_stable" \
		--files "changes.patch,git_status.txt" \
		--summary "$(ARTIFACTS_DIR)/sre_close_summary.json" \
		--report "$(ARTIFACTS_DIR)/postfinal_close_guard_report.json" \
		--sleep-seconds "$${SLEEP_SECONDS:-5.0}"

atomic-root-publish-from-smoke:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ ! -d "$(ARTIFACTS_DIR)/smoke_stable" ]; then echo "smoke_stable bundle is required"; exit 2; fi
	@$(PYTHON) tools/postfinal_root_smoke_guard.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--smoke-stable-dir "$(ARTIFACTS_DIR)/smoke_stable" \
		--files "changes.patch,git_status.txt" \
		--summary "$(ARTIFACTS_DIR)/sre_close_summary.json" \
		--report "$(ARTIFACTS_DIR)/atomic_root_publish_report.json" \
		--sleep-seconds "$${SLEEP_SECONDS:-0.0}" \
		--sync-root-from-smoke \
		--allow-missing-summary

close-time-terminal-enforce:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(MAKE) close-time-postfinal-guard ARTIFACTS_DIR="$(ARTIFACTS_DIR)" SLEEP_SECONDS="$${SLEEP_SECONDS:-5.0}"
	@$(PYTHON) tools/terminal_live_probe.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--files "changes.patch,git_status.txt" \
		--samples "$${PROBE_SAMPLES:-2}" \
		--interval-seconds "$${PROBE_INTERVAL_SECONDS:-0.8}" \
		--report "$(ARTIFACTS_DIR)/terminal_live_probe_report.json"

close-regression-harness:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(PYTHON) tools/r1_close_regression_harness.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--probe-interval-seconds "$${SLEEP_SECONDS:-0.2}"

# Compatibility aliases for delivery/traceability lanes after post-merge reconciliation.
# branchlock: d65b5f6f backend_fix_makefile_branchlock
# fix-pass: d65b5f6f backend_publish_makefile_fix_r1
delivery-evidence-gate:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(PYTHON) tools/delivery_evidence_gate.py --artifacts-dir "$(ARTIFACTS_DIR)" --workspace-dir "."

wormhole-trace-export:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	@$(PYTHON) tools/wormhole_trace_export.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--reported-branch-mode "expected"

backend-provenance-export: wormhole-trace-export

crosslane-validate:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@$(PYTHON) tools/crosslane_traceability_validator.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--out "$(ARTIFACTS_DIR)/crosslane_validator_report.json"

wormhole-contract-export:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(ORDER_BRANCH)" ]; then echo "ORDER_BRANCH is required"; exit 2; fi
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	@$(PYTHON) tools/wormhole_atomic_packager.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)"

wormhole-contract-validate:
	@$(PYTHON) tools/wormhole_scene_contract.py validate --contract docs/contracts/wormhole_scene_contract.v1.json

publish-atomic-guard:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(PYTHON) tools/publish_atomic_guard.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--out "$(ARTIFACTS_DIR)/publish_atomic_guard_report.json" \
		--log "$(ARTIFACTS_DIR)/publish_atomic_guard.log"

visual-preview-audit:
	@if [ -z "$(PREVIEW_HTML)" ]; then echo "PREVIEW_HTML is required"; exit 2; fi
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ -z "$(TICKET_ID)" ]; then echo "TICKET_ID is required"; exit 2; fi
	@$(PYTHON) tools/visual_preview_audit.py \
		--preview-html "$(PREVIEW_HTML)" \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--order-branch "$(ORDER_BRANCH)"

visual-preview-smoke:
	@$(PYTHON) tools/visual_preview_audit.py \
		--preview-html tools/fixtures/preview_fixture.html \
		--artifacts-dir .codexbot_tmp/visual-preview-smoke \
		--ticket-id local-smoke \
		--order-branch sre_visual_pipeline \
		--capture-mode synthetic \
		--force-capture

validate-s02-trace:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(PYTHON) tools/s02_trace_checker.py --artifacts-dir "$(ARTIFACTS_DIR)"

bundle-s02-atomic:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(PYTHON) tools/s02_bundle_atomic.py --artifacts-dir "$(ARTIFACTS_DIR)"

patch-status-reconcile:
	@echo "patch-status-reconcile replaced by crosslane-validate + manifest-drift-check (no-op alias)."
