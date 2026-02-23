# Mobile Status API (v1)

Contrato móvil estable (Android) para snapshot + live stream + alertas/riesgos/decisiones.

## Endpoints v1

- `GET /api/v1/status/snapshot`
- `GET /api/v1/status/stream` (SSE)
- `GET /api/v1/status/alerts`
- `GET /api/v1/status/risks`
- `GET /api/v1/status/decisions`

Back-compat aliases:
- `/api/status/snapshot|stream|alerts|risks|decisions`

## Auth

Con `BOT_STATUS_HTTP_TOKEN` activo:
- `Authorization: Bearer <token>` (preferido)
- `?token=<token>` (fallback)

Errores:
- `401 unauthorized`
- `429 rate_limited|too_many_streams`

## Payload base

Todos los responses incluyen:
- `api_version: "v1"`
- `schema_version: 1`
- `generated_at`
- `chat_id`

Snapshot añade:
- `workers`, `orders_active`
- `alerts`, `risks`, `decisions_pending`
- `live_view`, `staleness_seconds`, `snapshot_hash`

## SSE

El stream emite:
- `event: snapshot`
- `id: <generated_at_ms>`
- `data: <snapshot-json>`

Keep-alive:
- `: keep-alive`

## Latencia objetivo

Para Live View <= 5s:
- `BOT_STATUS_STREAM_INTERVAL_SECONDS=0.5..1.5`
- `BOT_STATUS_CACHE_TTL_SECONDS=1..2`

## Compatibilidad

- Cambios aditivos: permitidos en v1.
- Cliente móvil debe ignorar campos desconocidos.
- Cambios breaking: `/api/v2/*` + `schema_version` nuevo.

## Referencia detallada

Ver `docs/android_status_api.md` para contrato y ejemplos completos JSON.
