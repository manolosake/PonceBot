.PHONY: verify test lint security coverage wormhole-contract-validate wormhole-contract-export wormhole-contract-integrity-gate publish-atomic-guard publish-postwrite-verify wormhole-contract-publish visual-preview-audit visual-preview-smoke

verify: lint security test coverage

lint:
	python -m py_compile bot.py state_store.py orchestrator/*.py tools/*.py test_*.py

security:
	python tools/security_check.py --strict

test:
	python -m unittest -q

coverage:
	python tools/coverage_gate.py --min 0.65

wormhole-contract-validate:
	python3 tools/wormhole_scene_contract.py validate --contract docs/contracts/wormhole_scene_contract.v1.json

wormhole-contract-export:
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	@test -n "$(ORDER_BRANCH)" || (echo "ORDER_BRANCH is required"; exit 2)
	@test -n "$(TICKET_ID)" || (echo "TICKET_ID is required"; exit 2)
	python3 tools/wormhole_atomic_packager.py \
		--repo-root . \
		--contract docs/contracts/wormhole_scene_contract.v1.json \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--expected-branch "$(ORDER_BRANCH)"

wormhole-contract-integrity-gate:
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	@test -n "$(ORDER_BRANCH)" || (echo "ORDER_BRANCH is required"; exit 2)
	@test -n "$(TICKET_ID)" || (echo "TICKET_ID is required"; exit 2)
	python3 tools/wormhole_scene_integrity_gate.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--contract-source docs/contracts/wormhole_scene_contract.v1.json \
		--expected-branch "$(ORDER_BRANCH)" \
		--expected-ticket-id "$(TICKET_ID)"

publish-atomic-guard:
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	python3 tools/publish_atomic_guard.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--out "$(ARTIFACTS_DIR)/publish_atomic_guard_report.json" \
		--log "$(ARTIFACTS_DIR)/publish_atomic_guard.log"

publish-postwrite-verify:
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	python3 tools/bundle_postwrite_verify.py \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--manifest "$(ARTIFACTS_DIR)/bundle_immutability_manifest.json" \
		--out "$(ARTIFACTS_DIR)/bundle_postwrite_verify_report.json"

wormhole-contract-publish:
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	@test -n "$(ORDER_BRANCH)" || (echo "ORDER_BRANCH is required"; exit 2)
	@test -n "$(TICKET_ID)" || (echo "TICKET_ID is required"; exit 2)
	$(MAKE) wormhole-contract-validate
	$(MAKE) wormhole-contract-export ARTIFACTS_DIR="$(ARTIFACTS_DIR)" ORDER_BRANCH="$(ORDER_BRANCH)" TICKET_ID="$(TICKET_ID)"
	$(MAKE) wormhole-contract-integrity-gate ARTIFACTS_DIR="$(ARTIFACTS_DIR)" ORDER_BRANCH="$(ORDER_BRANCH)" TICKET_ID="$(TICKET_ID)"
	$(MAKE) publish-postwrite-verify ARTIFACTS_DIR="$(ARTIFACTS_DIR)"

visual-preview-audit:
	@test -n "$(PREVIEW_HTML)" || (echo "PREVIEW_HTML is required"; exit 2)
	@test -n "$(ARTIFACTS_DIR)" || (echo "ARTIFACTS_DIR is required"; exit 2)
	@test -n "$(TICKET_ID)" || (echo "TICKET_ID is required"; exit 2)
	python3 tools/visual_preview_audit.py \
		--preview-html "$(PREVIEW_HTML)" \
		--artifacts-dir "$(ARTIFACTS_DIR)" \
		--ticket-id "$(TICKET_ID)" \
		--order-branch "$(ORDER_BRANCH)"

visual-preview-smoke:
	python3 tools/visual_preview_audit.py \
		--preview-html tools/fixtures/preview_fixture.html \
		--artifacts-dir .codexbot_tmp/visual-preview-smoke \
		--ticket-id local-smoke \
		--order-branch sre_visual_pipeline \
		--capture-mode synthetic \
		--force-capture
