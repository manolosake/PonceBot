# Degraded Fallback Mode (Temporary)

Purpose: keep backend flow operable while external dependencies are blocked.

## Feature Flag Model
This mode is activated operationally via script and traced in DB:
- `trace.degraded_mode_active = true`
- `job_events`: `degraded_fallback_enabled` / `degraded_fallback_disabled`
- `audit_log`: `degraded_fallback_mode_enabled` / `degraded_fallback_mode_disabled`

## Enable (activate degraded path)
```bash
python3 tools/degraded_fallback_mode.py enable \
  --db /home/aponce/codexbot/data/jobs.sqlite \
  --role backend \
  --ticket 89020d31-6653-411c-aae4-9468ad944308 \
  --reason external_dependency_blocked
```

Behavior:
- splits `waiting_deps` backend jobs into:
  - runnable path (`queued` or `waiting_deps` with internal deps only)
  - blocked external node (`waiting_deps` with external deps only)
- logs every rewire action.

## Status
```bash
python3 tools/degraded_fallback_mode.py status --db /home/aponce/codexbot/data/jobs.sqlite
```

## Disable (rollback to normal path markers)
```bash
python3 tools/degraded_fallback_mode.py disable --db /home/aponce/codexbot/data/jobs.sqlite
```

Rollback policy:
- disables degraded markers and records audit event.
- normal orchestration policies resume.
