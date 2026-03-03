# S-02 Close Gate

Executable gate: `tools/s02_close_gate.py`

Purpose:
- Enforce immutable close checks over critical bundle artifacts.
- Enforce branch provenance (`reported_order_branch == expected_order_branch`).
- Fail hard (`exit_code != 0`) on drift or branch mismatch.

## Command

```bash
ORDER_BRANCH="<expected-branch>" \
ARTIFACTS_DIR="<bundle-artifacts-dir>" \
make validate-s02-close-gate
```

Optional override (for controlled probes/tests):

```bash
REPORTED_ORDER_BRANCH="<override-value>" \
ORDER_BRANCH="<expected-branch>" \
ARTIFACTS_DIR="<bundle-artifacts-dir>" \
make validate-s02-close-gate
```

## Output

- `<ARTIFACTS_DIR>/s02_close_gate_report.json`
- `<ARTIFACTS_DIR>/s02_close_gate.log`

Required report fields:
- `artifacts_dir`
- `expected_order_branch`
- `reported_order_branch`
- `drift_detected`
- `mismatches`
- `status`
- `exit_code`

## Failure Conditions

- Any missing or empty critical file.
- Any drift between current filesystem values and `s02_trace_validation` / `post_publish` / manifest for `git_status.txt` or `changes.patch`.
- Any branch mismatch:
  - `reported_order_branch != expected_order_branch`

Gate returns non-zero on failure.
