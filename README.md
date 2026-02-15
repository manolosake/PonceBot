# codexbot (Telegram -> Codex)

Un bot de Telegram que ejecuta `codex exec` en tu servidor y te devuelve la salida por Telegram.

## Requisitos

- Python 3 (sin dependencias extra; usa solo stdlib).
- `codex` en `PATH`.
- Para modo local sin OpenAI API: Ollama corriendo en `http://127.0.0.1:11434` (por defecto).
- Salida a Internet desde el servidor hacia `api.telegram.org:443` (polling).

## Setup

1. Crea un bot en Telegram con `@BotFather` y guarda el token.
2. Crea el archivo de config:
   - `cp codexbot.env.example codexbot.env`
   - Edita `codexbot.env` y setea `TELEGRAM_BOT_TOKEN=...`
   - Para secretos, usa `touch .env.local` y agrega solo valores sensibles (`OPENAI_API_KEY`, claves SSH, etc.); `run.sh` lo carga automáticamente y no se versiona.
3. Arranca el bot:
   - `./run.sh`
4. En Telegram, escribe cualquier mensaje al bot. Como no esta configurada la allow-list todavia, el bot contestara `Unauthorized` incluyendo `chat_id` y `user_id`.
5. En `codexbot.env`, setea al menos uno:
   - `TELEGRAM_ALLOWED_USER_IDS=123456789`
   - `TELEGRAM_ALLOWED_CHAT_IDS=123456789`
6. Reinicia el bot.

## Uso

- `/help`
- `/whoami`
- `/status`
- `/synccommands` (fuerza re-sincronizacion de sugerencias `/...` en Telegram)
- `/cancel` (cancela el job en curso y descarta los jobs en cola de ese chat)
- `/brief` (resumen ejecutivo corto del estado del orquestador)
- `/new` (inicia un hilo nuevo de Codex para ese chat)
- `/thread` (muestra el `thread_id` actual de Codex para ese chat)
- `/snapshot <url|texto>` (solicita tarea `frontend` de captura visual)
- `/skills` (lista skills instaladas, deshabilitadas y `.system`)
- `/skills catalog [filter]` (lista skills instalables desde `openai/skills`)
- `/skills install <skill>` (instala una skill curated en `~/.codex/skills/<skill>`)
- `/skills enable <skill>` (reactiva una skill deshabilitada)
- `/skills disable <skill>` (deshabilita una skill local moviendola a `~/.codex/skills/.disabled/`)
- Texto normal: corre en modo default (`CODEX_DEFAULT_MODE`, recomendado `ro`)
- `/ro <prompt>`: sandbox read-only
- `/rw <prompt>`: sandbox workspace-write (puede editar archivos en `CODEX_WORKDIR`)
- `/full <prompt>`: sandbox danger-full-access (sin sandbox; unsafe)
- Passthrough a Codex: solo `/exec ...`, `/review ...`, `/codex ...` (otros slash commands muestran ayuda)

## Modo "sin limitaciones" (MUY PELIGROSO)

Si quieres que el cliente se conecte "casi directo" a Codex y sin restricciones del bot, habilita:

- `BOT_UNSAFE_DIRECT_CODEX=1`: desactiva la validacion de argv y trata la mayoria de `/<algo>` como `codex <algo> ...`.
- `CODEX_DEFAULT_MODE=full` (o `CODEX_FORCE_FULL_ACCESS=1`): fuerza `--sandbox danger-full-access` por defecto.

Advertencia: con esto, cualquier usuario/chat autorizado puede disparar ejecuciones con acceso total al host
(lectura/escritura fuera del workdir, acceso a archivos locales, etc.). Idealmente corre esto dentro de una VM/container
aislado.

## OSS (Ollama) vs "Codex normal"

- Default: usar tu login/config actual de Codex (OpenAI) via CLI: `CODEX_USE_OSS=0`
- Local (sin OpenAI API): `CODEX_USE_OSS=1` y `CODEX_LOCAL_PROVIDER=ollama`
  - Define `CODEX_OSS_MODEL` (por defecto: `qwen2.5-coder:7b`)

## 24/7

Si tu servidor usa systemd, puedes instalar un servicio con los units en `systemd/`.
Si no, la alternativa simple es correrlo dentro de `tmux`/`screen` o con `nohup`.

### Hardening recomendado (evita "se quedo trabado")

- Usa los units actualizados: incluyen `ExecStartPre` (valida env), `StartLimit*` (evita restart loop infinito) y `OnFailure` (manda alerta fuera del proceso del bot).
- Configura un chat de alertas:
  - en Telegram corre `/setnotify` (guarda `notify_chat_id`), o
  - fija `TELEGRAM_NOTIFY_CHAT_ID=<tu_chat_id>` en `codexbot.env`.
- Prueba la ruta de alerta:
  - `systemctl --user start codexbot-alert-user@codexbot.service`
  - Debe llegarte un mensaje al chat de notificacion.
- Si arrancas manualmente con `./run.sh`, valida antes con:
  - `./run.sh --check-env`

## Notas

- Por default, el bot drena (descarta) updates pendientes al iniciar para evitar re-ejecutar backlog despues de un restart: `BOT_DRAIN_UPDATES_ON_START=1`.
- Por default, el bot mantiene un "hilo" por chat usando sesiones nativas de Codex CLI (`codex exec resume`). Para reiniciar el hilo: `/new`.
- Si queres que el bot sea casi transparente (sin parsing de comandos del bot) habilita `BOT_STRICT_PROXY=1`. En ese modo, casi todo el texto se reenvia directo a Codex, y solo quedan `/new`, `/thread`, `/cancel`.
- Si quieres evitar que un flood llene la memoria, puedes limitar la cola: `BOT_QUEUE_MAXSIZE=100` (0 = sin limite).
- Puedes limitar la cola por chat con `BOT_MAX_QUEUED_PER_CHAT=1` y habilitar heartbeat con `BOT_HEARTBEAT_SECONDS=60`.
- Si la respuesta final es muy larga, puede enviarse como archivo `.txt` (controlado por `BOT_SEND_AS_FILE_THRESHOLD_CHARS`).
- Formato en Telegram: por defecto usa `BOT_TELEGRAM_PARSE_MODE=HTML` para renderizar `code` y fences ```...``` correctamente.
- Puedes mandar imagenes al bot (foto o documento de tipo imagen). El bot las descarga y las pasa a Codex como `--image`.
  - Prompt: usa el caption si lo mandas; si no, usa un prompt default.
  - Limite de descarga: `BOT_MAX_DOWNLOAD_BYTES` (0 = sin limite).
- Puedes mandar notas de voz / audio y transcribirlas a texto:
  - Habilita `BOT_TRANSCRIBE_AUDIO=1`.
  - Recomendado (gratis/local): `BOT_TRANSCRIBE_BACKEND=whispercpp` (requiere ffmpeg + whisper.cpp + modelo ggml).
  - Alternativa (API): `BOT_TRANSCRIBE_BACKEND=openai` y configura `OPENAI_API_KEY`.
  - El texto transcrito se procesa igual que si lo hubieras escrito (incluye /ro /rw /exec, etc.).

## Plan de implementación (v1)

### 1) Certezas del estado actual

Lo siguiente está implementado en este repo:

- Orquestación multirol:
  - `orchestrator/schemas/task.py`, `orchestrator/schemas/result.py`
  - `orchestrator/storage.py` (SQLite con `jobs` y eventos)
  - `orchestrator/queue.py`
  - `orchestrator/agents.yaml`
  - `orchestrator/dispatcher.py`
  - `orchestrator/runner.py`
- Flujo desde Telegram:
  - Enrutado a cola de orquestador para comandos de ejecución textuales.
  - `/agents`, `/job <id>`, `/daily`, `/approve`, `/pause`, `/resume`, `/cancel <id>`.
  - `/status` muestra métricas de cola legacy + cola de orquestador.
- Arranque:
  - `main()` intenta inicializar `orchestrator_queue`, levanta workers de orquestador y scheduler opcional.
  - Se reintentan trabajos `running` previos: se regresan a `queued`.
- Seguridad/operación base:
  - Bloqueo por imagen/adjunto para entrar al orquestador (los mensajes con archivos se mantienen en ejecución legacy).
  - `run.sh` usa `codexbot.env` y ahora carga también `.env.local`.
  - `.gitignore` ignora `codexbot.env`, `.env.local`, y `data/jobs.sqlite`.

### 2) Supuestos (a validar en entorno real)

- Se asume acceso estable al host de operación (`100.93.21.71` por Tailscale) y acceso SSH ya resuelto.
- Se asume disponibilidad de `gpt-5.2` (o alias estable configurable).
- Se asume capacidad de correr múltiples procesos `codex` en paralelo sin saturar recursos.

### 3) Fases y alcance inmediato (próximos pasos)

- **Fase 1 — Entrega actual (v1 base):**
  - Orquestador persistente (SQLite), roles y cola de trabajos.
  - Comandos operativos para CEO y control por rol.
  - Scheduler de digest cada `BOT_ORCHESTRATOR_DAILY_DIGEST_SECONDS`.
- **Fase 2 — Operación autónoma real (a completar):**
  - Auto-revisión de tareas por rol.
  - Controles de cuota/tiempo por rol.
  - Mejorar evidencias visuales y pipeline de screenshot real.
  - Kill-switch (`/emergency_stop`) y reglas de gobernanza más finas.

### 4) Configuración clave del orquestador

- `BOT_ORCHESTRATOR_ENABLED`
- `BOT_ORCHESTRATOR_DB_PATH`
- `BOT_ORCHESTRATOR_DAILY_DIGEST_SECONDS`
- `BOT_ORCHESTRATOR_DEFAULT_ROLE`
- `BOT_ORCHESTRATOR_DEFAULT_PRIORITY`
- `BOT_ORCHESTRATOR_DEFAULT_MAX_COST_WINDOW_USD`
- `BOT_ORCHESTRATOR_AGENT_PROFILES`

### 5) Puesta en marcha recomendada

1. Copia secretos a `.env.local` (no se versiona): `OPENAI_API_KEY`, tokens, etc.
2. Arranca y valida:
   - `./run.sh --check-env`
   - `./run.sh`
3. Verifica:
   - `/status`
   - `/agents`
   - `/job <id>` tras enviar un comando de texto.
