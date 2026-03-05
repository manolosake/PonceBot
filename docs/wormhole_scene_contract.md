# Wormhole Scene Contract v1

This contract defines backend-controlled scene parameters consumed by frontend for reproducible wormhole rendering.

## Canonical source
- `docs/contracts/wormhole_scene_contract.v1.json`

## Required reproducibility fields
- `contract_name`
- `contract_version`
- `scene_version`
- `seed.strategy`
- `seed.value`
- `quality_presets`
- `viewport_defaults`
- `traceability.signature_fields`

## Controlled parameters
- `particle_density`
- `ring_layers`
- `meridian_lines`
- `volumetric_intensity`
- `distortion_strength`
- `parallax_depth`

## Presets
- `cinematic`
- `balanced`
- `performance`

## Frontend consumption
Frontend should read `wormhole_scene_contract.json` from exported artifacts and apply `viewport_defaults[desktop|tablet|mobile]` as base preset, then override only inside `parameter_ranges`.

## QA traceability export
Run:

```bash
make wormhole-contract-export \
  ARTIFACTS_DIR=/path/to/artifacts \
  ORDER_BRANCH=feature/order-a24101c9-proactive-sprint-poncebot-reliability- \
  TICKET_ID=a24101c9-f0ce-43f5-9298-79562357cc3f
```

Expected artifacts:
- `wormhole_scene_contract.json`
- `wormhole_scene_contract.sha256`
- `wormhole_scene_trace.json`
- `wormhole_scene_contract_export_report.json`

## PASS criteria
- Contract validates with `status=PASS`.
- Export report exists and includes stable `scene_signature_sha256`.
- `branch_matches_expected=true` in trace report for the target run.
