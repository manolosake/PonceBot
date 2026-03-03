.PHONY: verify test lint security coverage validate-s02-trace validate-s02-independent validate-s02-sanity validate-s02-close-gate bundle-s02-atomic perf-harness-v3 preview-integrity-gate local-smoke-pack

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

validate-s02-independent:
	@ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	$(PYTHON) tools/s02_independent_check.py --artifacts-dir "$$ART"

validate-s02-sanity:
	@ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	$(PYTHON) tools/s02_sanity_gate.py --artifacts-dir "$$ART"

validate-s02-close-gate:
	@ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	ORDER=$${ORDER_BRANCH:?set ORDER_BRANCH}; \
	REPORTED=$${REPORTED_ORDER_BRANCH:-}; \
	$(PYTHON) tools/s02_close_gate.py --artifacts-dir "$$ART" --expected-order-branch "$$ORDER" --reported-order-branch "$$REPORTED"

bundle-s02-atomic:
	@ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	$(PYTHON) tools/s02_bundle_atomic.py --artifacts-dir "$$ART"

perf-harness-v3:
	@OUT=$${OUT_DIR:?set OUT_DIR}; \
	BASE=$${BASELINE_WORKSPACE:?set BASELINE_WORKSPACE}; \
	CAND=$${CANDIDATE_WORKSPACE:?set CANDIDATE_WORKSPACE}; \
	DUR=$${DURATION_SECONDS:-8}; \
	$(PYTHON) tools/perf_harness_v3.py --baseline-workspace "$$BASE" --candidate-workspace "$$CAND" --out-dir "$$OUT" --duration-seconds "$$DUR"

preview-integrity-gate:
	@WS=$${WORKSPACE:?set WORKSPACE}; \
	OUT=$${OUT:?set OUT}; \
	ORDER=$${ORDER_BRANCH:-}; \
	EXP_SHA=$${EXPECTED_PREVIEW_SHA:-}; \
	MANIFEST=$${MANIFEST_PATH:-}; \
	$(PYTHON) tools/preview_integrity_gate.py --workspace "$$WS" --out "$$OUT" --expected-branch "$$ORDER" --expected-preview-sha "$$EXP_SHA" --manifest-path "$$MANIFEST"

local-smoke-pack:
	@WS=$${WORKSPACE:?set WORKSPACE}; \
	ART=$${ARTIFACTS_DIR:?set ARTIFACTS_DIR}; \
	ORDER=$${ORDER_BRANCH:?set ORDER_BRANCH}; \
	$(PYTHON) tools/local_smoke_pack.py --workspace "$$WS" --artifacts-dir "$$ART" --order-branch "$$ORDER"
