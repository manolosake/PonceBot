# ExecutiveDashboard Status API (Android Contract)

This API is intended for a lightweight Android client to read the current system status (workers + current/next tasks), sourced from Orders/Autopilot.

**Transport**
- Snapshot: HTTP `GET` returning JSON
- Stream: **SSE** (`text/event-stream`) streaming JSON snapshots (recommended for Android; simple + works well over Tailscale)

**Versioning**
- URL version: `/api/v1/...`
- Response header: `X-ED-API-Version: v1`
- Payload fields:
  - `api_version`: `"v1"`
  - `schema_version`: `1` (integer; increment on breaking payload changes)

Compatibility rule:
- Clients should ignore unknown fields.
- Breaking changes: new `schema_version`, and new `/api/v{N}` paths.

## Auth (Token)

Use a single shared token (lightweight) suitable for consumption from Android + Tailscale.

Supported:
1. Header (preferred): `Authorization: Bearer <token>`
2. Query param (fallback): `?token=<token>`

If `BOT_STATUS_HTTP_TOKEN` is empty, auth is disabled (not recommended unless the listener is strictly localhost-only).

## Endpoints

### Snapshot

`GET /api/v1/status/snapshot`

Optional query:
- `chat_id=<int>`: scope snapshot to a single Telegram chat. If omitted, returns global view.

Responses:
- `200` JSON snapshot
- `401` if token missing/invalid (when enabled)
- `429` rate limited, with `Retry-After` seconds header

Example:
```bash
curl -sS \
  -H "Authorization: Bearer $BOT_STATUS_HTTP_TOKEN" \
  "http://127.0.0.1:8090/api/v1/status/snapshot?chat_id=123"
```

### Stream (SSE)

`GET /api/v1/status/stream`

Optional query:
- `chat_id=<int>`

Event format:
- `event: snapshot`
- `id: <ms_timestamp>` (monotonic-ish; useful for debugging/reconnect)
- `data: <JSON snapshot>`

Heartbeats:
- Periodic comment frames `: keep-alive` when no data change is detected (prevents idle disconnects).

Example:
```bash
curl -N \
  -H "Authorization: Bearer $BOT_STATUS_HTTP_TOKEN" \
  "http://127.0.0.1:8090/api/v1/status/stream"
```

## Snapshot Payload (Stable)

Top-level:
- `api_version`: string (`"v1"`)
- `schema_version`: int (`1`)
- `generated_at`: float (unix seconds)
- `snapshot_hash`: string (sha256 of canonicalized snapshot; used to detect changes)
- `chat_id`: int|null (scope)
- `source_newest_updated_at`: float|null (newest task `updated_at` observed)
- `staleness_seconds`: float|null (`generated_at - source_newest_updated_at`)
- `orders_active`: array of active orders
- `workers`: array of worker slots across roles

`orders_active[]`:
- `order_id`: string (UUID)
- `order_id_short`: string (first 8 chars)
- `chat_id`: int
- `status`: string (`active|paused|done`)
- `priority`: int (1..3)
- `title`: string
- `updated_at`: float (unix seconds)
- `children_counts`: object `{ queued: int, running: int, blocked: int, done: int, failed: int, cancelled: int }` (keys optional)

`workers[]` (one entry per `role:slot`):
- `worker_id`: string (`"<role>:<slot>"`)
- `role`: string (`jarvis|frontend|backend|qa|sre|product_ops|security|research|release_mgr`)
- `slot`: int (1..N per role)
- `current`: task_status|null
- `next`: task_status|null

`task_status` (for `current` and `next`):
- `job_id`: string (UUID)
- `job_id_short`: string (first 8 chars)
- `role`: string
- `state`: string (`queued|running|blocked|done|failed|cancelled`)
- `priority`: int (1..3)
- `request_type`: string (`status|query|review|maintenance|task`)
- `owner`: string|null
- `chat_id`: int
- `user_id`: int|null
- `parent_job_id`: string|null
- `created_at`: float
- `updated_at`: float
- `title`: string (trimmed single-line summary)
- `live_phase`: string|null
- `live_workspace_slot`: int|null

## Rate Limits / Limits

Defaults (configurable):
- Snapshot: token-bucket per source IP, default `2 rps` burst `4`
- SSE: max concurrent streams per source IP, default `2`

On limit:
- `429` with `Retry-After` header.

## Network Notes (Tailscale)

Recommended deployment for Android:
- Bind to tailnet interface/IP only, or to `0.0.0.0:<port>` and restrict via firewall + token.
- Consume from Android via the machine tailnet IP/hostname, e.g. `http://100.x.y.z:8090/api/v1/status/snapshot`.

