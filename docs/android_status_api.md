# ExecutiveDashboard Status API (Android Contract)

Contrato de datos para Android (Live View, Snapshot, Alertas, Riesgos y Decisiones) compatible con web.

## Transporte y versionado

- Snapshot: `GET` JSON.
- Live View: `SSE` (`text/event-stream`) con eventos `snapshot`.
- Paths versionados (preferidos):
  - `GET /api/v1/status/snapshot`
  - `GET /api/v1/status/stream`
  - `GET /api/v1/status/alerts`
  - `GET /api/v1/status/risks`
  - `GET /api/v1/status/decisions`
- Back-compat aliases: `/api/status/*`.
- Header: `X-ED-API-Version: v1`.
- Payload:
  - `api_version: "v1"`
  - `schema_version: 1`

Regla de compatibilidad:
- Cambios aditivos: mismo `schema_version`.
- Breaking changes: nuevo `/api/vN/*` y `schema_version`.

## Auth ligera (token)

Si `BOT_STATUS_HTTP_TOKEN` está definido:
- Preferido: `Authorization: Bearer <token>`
- Fallback: `?token=<token>`

Errores:
- `401 {"error":"unauthorized"}`

## Live View (SSE)

`GET /api/v1/status/stream[?chat_id=<int>]`

Frames:
- `event: snapshot`
- `id: <generated_at_ms>`
- `data: <json snapshot>`
- keepalive cuando no hay cambios: `: keep-alive`

Objetivo de latencia (staging): `<= 5s`
- Recomendado:
  - `BOT_STATUS_STREAM_INTERVAL_SECONDS=0.5..1.5`
  - `BOT_STATUS_CACHE_TTL_SECONDS=1..2`

## Snapshot estable

`GET /api/v1/status/snapshot[?chat_id=<int>]`

Campos top-level (estables):
- `api_version`, `schema_version`
- `generated_at`, `chat_id`, `snapshot_hash`
- `source_newest_updated_at`, `staleness_seconds`
- `live_view`: `{ transport, event, target_latency_seconds, staleness_seconds }`
- `orders_active[]`
- `workers[]`
- `alerts[]`
- `risks[]`
- `decisions_pending[]`

`workers[]`:
- `worker_id`, `role`, `slot`
- `current` y `next` (`task_status|null`)

`task_status`:
- `job_id`, `job_id_short`, `role`, `state`, `priority`, `request_type`, `mode_hint`
- `requires_approval`, `approved`
- `owner`, `chat_id`, `user_id`, `parent_job_id`
- `created_at`, `updated_at`, `title`
- `live_phase`, `live_at`, `live_pid`, `live_workdir`, `live_workspace_slot`
- `result_summary`, `result_next_action`

## Alertas, Riesgos y Decisiones

### Alertas

`GET /api/v1/status/alerts[?chat_id=<int>]`

Respuesta:
- `items[]` con `{ kind, severity, count, summary }`
- `kind` actuales: `approval_blocked`, `failed_jobs`, `stalled_tasks`, `queue_pressure`, `snapshot_stale`

### Riesgos

`GET /api/v1/status/risks[?chat_id=<int>]`

Respuesta:
- `items[]` con `{ risk_id, level, source, summary, impact, count }`

### Decisiones

`GET /api/v1/status/decisions[?chat_id=<int>]`

Respuesta:
- `items[]` con decisiones pendientes de:
  - jobs bloqueados por aprobación (`job_approval`)
  - decision logs con `next_action` (`order_decision`)

## Ejemplos

### Snapshot

```json
{
  "api_version": "v1",
  "schema_version": 1,
  "generated_at": 1760000000.25,
  "chat_id": 123,
  "snapshot_hash": "abc123",
  "source_newest_updated_at": 1760000000.1,
  "staleness_seconds": 0.15,
  "live_view": {
    "transport": "sse",
    "event": "snapshot",
    "target_latency_seconds": 4,
    "staleness_seconds": 0.15
  },
  "alerts": [
    {
      "kind": "approval_blocked",
      "severity": "warning",
      "count": 1,
      "summary": "1 job(s) esperando aprobación"
    }
  ],
  "risks": [
    {
      "risk_id": "approval_dependency",
      "level": "medium",
      "source": "orchestrator",
      "summary": "Dependencia de aprobación humana puede frenar entregas.",
      "impact": "throughput",
      "count": 1
    }
  ],
  "decisions_pending": [
    {
      "kind": "order_decision",
      "order_id": "11111111-1111-1111-1111-111111111111",
      "job_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      "state": "blocked",
      "summary": "Falta evidencia QA",
      "next_action": "Aprobar evidencia o pedir corrección",
      "updated_at": 1760000000.1
    }
  ],
  "orders_active": [],
  "workers": []
}
```

### Alerts

```json
{
  "api_version": "v1",
  "schema_version": 1,
  "generated_at": 1760000000.25,
  "chat_id": 123,
  "items": [
    {
      "kind": "failed_jobs",
      "severity": "critical",
      "count": 2,
      "summary": "2 job(s) en estado failed"
    }
  ]
}
```

### Risks

```json
{
  "api_version": "v1",
  "schema_version": 1,
  "generated_at": 1760000000.25,
  "chat_id": 123,
  "items": [
    {
      "risk_id": "queue_backlog",
      "level": "medium",
      "source": "scheduler",
      "summary": "Backlog alto puede degradar SLA.",
      "impact": "sla",
      "count": 13
    }
  ]
}
```

### Decisions

```json
{
  "api_version": "v1",
  "schema_version": 1,
  "generated_at": 1760000000.25,
  "chat_id": 123,
  "items": [
    {
      "kind": "job_approval",
      "priority": 2,
      "state": "blocked_approval",
      "order_id": "11111111-1111-1111-1111-111111111111",
      "job_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
      "job_id_short": "bbbbbbbb",
      "title": "Validar release",
      "next_action": "Aprobar o rechazar ejecución",
      "updated_at": 1760000000.15
    }
  ]
}
```
