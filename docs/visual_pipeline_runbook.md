# Visual Pipeline Runbook (preview.html -> audit bundle)

This runbook validates or captures desktop/tablet/mobile screenshots from `preview.html`,
retries failed captures with exponential backoff, writes a metadata manifest, and packages
all evidence for audit.

## Inputs

- `preview.html` source: usually workspace `.codexbot_preview/preview.html`
- `ticket_id`: traceability id from orchestration/ticketing
- `order_branch` (optional but recommended): expected order branch key

## Standard command

```bash
make visual-preview-audit \
  PREVIEW_HTML=/abs/path/to/preview.html \
  ARTIFACTS_DIR=/abs/path/to/artifacts/visual \
  TICKET_ID=a24101c9-f0ce-43f5-9298-79562357cc3f \
  ORDER_BRANCH=sre_visual_pipeline
```

Equivalent direct CLI:

```bash
python3 tools/visual_preview_audit.py \
  --preview-html /abs/path/to/preview.html \
  --artifacts-dir /abs/path/to/artifacts/visual \
  --ticket-id a24101c9-f0ce-43f5-9298-79562357cc3f \
  --order-branch sre_visual_pipeline
```

## Retry/backoff behavior

- Capture targets: `desktop`, `tablet`, `mobile`
- Retries per viewport: `--max-attempts` (default `3`)
- Backoff: doubles from `--backoff-initial-seconds` up to `--backoff-max-seconds`
- Existing valid PNGs are reused unless `--force-capture` is set

## Artifacts produced

- `preview.html` (copied input)
- `preview-desktop.png`
- `preview-tablet.png`
- `preview-mobile.png`
- `visual_preview_manifest.json` (metadata + per-viewport status/attempts/hash)
- `visual_preview_audit_report.json` (compact summary)
- `visual_preview_audit_bundle.tar.gz` (packaged audit evidence)

## Smoke command

Use synthetic mode when Playwright/Chromium is unavailable in local environments:

```bash
make visual-preview-smoke
```

Output path:

- `.codexbot_tmp/visual-preview-smoke`

