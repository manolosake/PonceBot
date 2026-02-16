# Mobile Status API (v1)

This document formalizes a stable contract for the mobile client (Android).

## Transport

- Snapshot: HTTP GET, JSON response.
- Stream: HTTP GET, Server-Sent Events (SSE) with `event: snapshot` and JSON payload in `data:`.
- Versioned paths exist and should be preferred by mobile:
  - `GET /api/v1/status/snapshot`
  - `GET /api/v1/status/stream`
- Back-compat aliases (same payload):
  - `GET /api/status/snapshot`
  - `GET /api/status/stream`

## Auth (lightweight token)

If `BOT_STATUS_HTTP_TOKEN` is set on the server:

- Required header:
  - `Authorization: Bearer <token>`
- Fallback (supported, not recommended):
  - `?token=<token>`

If the token is missing/invalid: `401 {"error":"unauthorized"}`.

## Rate Limits

Snapshot endpoint is rate-limited per client IP with a token bucket:

- `BOT_STATUS_HTTP_SNAPSHOT_RPS` (default `2.0`) and `BOT_STATUS_HTTP_SNAPSHOT_BURST` (default `4.0`).
- On limit: `429 {"error":"rate_limited"}` with `Retry-After: 1`.

SSE is rate-limited per client IP by concurrent connections:

- `BOT_STATUS_HTTP_MAX_SSE_PER_IP` (default `2`).
- On limit: `429 {"error":"too_many_streams"}` with `Retry-After: 5`.

## Network (Tailscale)

Recommended deployment for Android:

- Run the status HTTP API bound to a reachable interface:
  - `BOT_STATUS_HTTP_LISTEN=0.0.0.0:8090` or `BOT_STATUS_HTTP_LISTEN=<tailscale_ip>:8090`
- Access from Android using the Tailscale IP (typically `100.x.y.z`):
  - `http://100.x.y.z:8090/api/v1/status/snapshot`
  - `http://100.x.y.z:8090/api/v1/status/stream`

Tailscale provides the secure network; the token is still recommended as a second layer.

## Query Params

Optional filters:

- `chat_id` (int): scopes the snapshot/stream to a single Telegram chat:
  - `GET /api/v1/status/snapshot?chat_id=123`

## Response Headers

- `X-ED-API-Version: v1`
- `Cache-Control: no-store` (snapshot)
- `Content-Type: application/json` (snapshot) / `text/event-stream` (stream)

## Payload (stable)

Top-level keys:

- `api_version`: `"v1"`
- `schema_version`: `1`
- `generated_at`: float seconds since epoch
- `chat_id`: int or null
- `orders_active`: list of active orders (Autopilot source of truth)
- `workers`: list of workers for all roles (role+slot)
- `source_newest_updated_at`: float or null (newest `updated_at` across observed tasks)
- `staleness_seconds`: float or null (`now - source_newest_updated_at`)
- `snapshot_hash`: hex string (stable hash of the snapshot body without `snapshot_hash`)

### Worker object

- `worker_id`: string, stable (`"<role>:<slot>"`)
- `role`: string (e.g. `backend`)
- `slot`: int (1-based)
- `current`: task_status or null
- `next`: task_status or null

### task_status object

- `job_id`: string (UUID)
- `job_id_short`: first 8 chars
- `role`: string
- `state`: string (`queued|running|blocked|done|failed|cancelled`)
- `priority`: int (1-3)
- `request_type`: string
- `owner`: string or null
- `chat_id`: int
- `user_id`: int or null
- `parent_job_id`: string or null
- `created_at`: float
- `updated_at`: float
- `title`: string (short, <= ~120 chars)
- `live_phase`: string or null
- `live_workspace_slot`: int or null

### orders_active object

- `order_id`, `order_id_short`
- `chat_id`
- `status` (e.g. `active`)
- `priority`
- `title`
- `updated_at`
- `children_counts`: map of job state -> count

## SSE Stream

The stream emits:

- `event: snapshot`
- `id: <generated_at_ms>`
- `data: <json snapshot>`

If no changes, the server sends keep-alives:

- `: keep-alive`

