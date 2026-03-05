.PHONY: verify test lint security coverage backend-traceability-runtime-export close-time-postfinal-guard close-time-terminal-enforce

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
	@mkdir -p "$(ARTIFACTS_DIR)"
	@$(PYTHON) tools/backend_traceability_runtime_export.py \
		--repo-root "." \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)" \
		--frontend-job-id "$(FRONTEND_JOB_ID)" \
		--target-artifact-dir "$(TARGET_ARTIFACT_DIR)"

close-time-postfinal-guard:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@if [ ! -d "$(ARTIFACTS_DIR)/smoke_stable" ]; then echo "smoke_stable bundle is required"; exit 2; fi
	@$(PYTHON) tools/postfinal_root_smoke_guard.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--smoke-stable-dir "$(ARTIFACTS_DIR)/smoke_stable" \
		--files "changes.patch,git_status.txt" \
		--summary "$(ARTIFACTS_DIR)/sre_close_summary.json" \
		--report "$(ARTIFACTS_DIR)/postfinal_close_guard_report.json" \
		--sleep-seconds "$${SLEEP_SECONDS:-1.0}"

close-time-terminal-enforce:
	@if [ -z "$(ARTIFACTS_DIR)" ]; then echo "ARTIFACTS_DIR is required"; exit 2; fi
	@$(MAKE) close-time-postfinal-guard ARTIFACTS_DIR="$(ARTIFACTS_DIR)" SLEEP_SECONDS="$${SLEEP_SECONDS:-1.0}"
	@$(PYTHON) tools/terminal_live_probe.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--files "changes.patch,git_status.txt" \
		--samples "$${PROBE_SAMPLES:-2}" \
		--interval-seconds "$${PROBE_INTERVAL_SECONDS:-0.5}" \
		--report "$(ARTIFACTS_DIR)/terminal_live_probe_report.json"
	@sleep "$${REPROBE_GAP_SECONDS:-0.5}"
	@$(PYTHON) tools/terminal_live_probe.py \
		--root-dir "$(ARTIFACTS_DIR)" \
		--files "changes.patch,git_status.txt" \
		--samples "$${REPROBE_SAMPLES:-2}" \
		--interval-seconds "$${REPROBE_INTERVAL_SECONDS:-0.5}" \
		--report "$(ARTIFACTS_DIR)/terminal_live_probe_recheck_report.json"
	@$(PYTHON) tools/terminal_probe_compare.py \
		--probe-a "$(ARTIFACTS_DIR)/terminal_live_probe_report.json" \
		--probe-b "$(ARTIFACTS_DIR)/terminal_live_probe_recheck_report.json" \
		--report "$(ARTIFACTS_DIR)/terminal_live_probe_compare_report.json"
	@$(PYTHON) tools/terminal_done_decider.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--files "changes.patch,git_status.txt" \
		--guard-report "$(ARTIFACTS_DIR)/postfinal_close_guard_report.json" \
		--probe2-report "$(ARTIFACTS_DIR)/terminal_live_probe_recheck_report.json" \
		--report "$(ARTIFACTS_DIR)/terminal_done_decider_report.json"
