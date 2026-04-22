#!/usr/bin/env python3
import json
import os
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path('/home/aponce/codexbot/data/jobs.sqlite')
DEFAULT_STATE_FILE = Path('/home/aponce/codexbot/state.json')
DEFAULT_OUT_DIR = Path('/home/aponce/codexbot/data/artifacts/proactive_health')
DB = Path(os.environ.get('CODEXBOT_ORCH_DB', str(DEFAULT_DB)))
STATE_FILE = Path(os.environ.get('CODEXBOT_STATE_FILE', str(DEFAULT_STATE_FILE)))
OUT_DIR = Path(os.environ.get('CODEXBOT_PROACTIVE_HEALTH_OUT_DIR', str(DEFAULT_OUT_DIR)))
ACTIVE_STATES = ('queued', 'running', 'waiting_deps', 'blocked', 'blocked_approval')
LOCAL_ROLES = {'architect_local', 'implementer_local', 'reviewer_local'}
CLI_ROLES = {'backend', 'frontend', 'qa', 'sre', 'security', 'research', 'product_ops', 'release_mgr'}
DELIVERY_ROLES = {'backend', 'frontend', 'implementer_local', 'sre', 'security', 'research', 'product_ops', 'release_mgr'}
VALIDATION_ROLES = {'qa', 'reviewer_local'}
CONTROLLER_ROLES = {'skynet', 'jarvis'}
IGNORE_ARTIFACT_TOKENS = (
    'local_ollama_response',
    'local_ollama_meta',
    'local_ollama_live',
    'local_ollama_stream',
    'local_ollama_prompt',
)
STALE_LOCAL_S = 15 * 60
LOOKBACK_S = 24 * 3600


def worst_status(*values: str) -> str:
    rank = {'OK': 0, 'WARN': 1, 'CRITICAL': 2}
    best = 'OK'
    best_rank = -1
    for raw in values:
        status = str(raw or 'OK').strip().upper() or 'OK'
        value = rank.get(status, -1)
        if value > best_rank:
            best_rank = value
            best = status
    return best


def load_json(raw: str):
    try:
        return json.loads(raw or '{}')
    except Exception:
        return {}


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def is_proactive(title: str, body: str) -> bool:
    blob = f"{title or ''}\n{body or ''}".lower()
    return ('proactive sprint' in blob) or ('[proactive:' in blob)


def trace_has_nontrivial_artifacts(trace: dict) -> bool:
    artifacts = trace.get('result_artifacts') or []
    if not isinstance(artifacts, list):
        return False
    for raw in artifacts:
        item = str(raw or '').strip().lower()
        if not item:
            continue
        if not any(tok in item for tok in IGNORE_ARTIFACT_TOKENS):
            return True
    return False


def summary_has_progress_signal(text: str) -> bool:
    blob = str(text or '').strip().lower()
    if len(blob) < 60:
        return False
    tokens = (
        'implemented', 'fixed', 'patched', 'updated', 'added', 'removed', 'refactored',
        'validated', 'verified', 'tested', 'passing', 'pass', 'coverage', 'screenshot',
        'artifact', 'evidence', 'log', 'diff', 'branch', 'commit', 'pull request',
        'healthz', 'latency', 'alert', 'migration', 'deploy', 'build', 'smoke',
    )
    return any(tok in blob for tok in tokens)


def summary_has_validation_signal(text: str) -> bool:
    blob = str(text or '').strip().lower()
    if len(blob) < 24:
        return False
    tokens = (
        'pass', 'passed', 'fail', 'failed', 'ready', 'needs_rework', 'no-go',
        'go/no-go', 'validated', 'verified', 'regression', 'smoke', 'tests',
        'qa', 'evidence', 'artifact',
    )
    return any(tok in blob for tok in tokens)


def summary_has_ready_signal(text: str) -> bool:
    blob = str(text or '').strip().lower()
    if not blob:
        return False
    if any(tok in blob for tok in ('no-go', 'no go', 'nogo', 'needs_rework', 'needs rework', 'rework required', 'failed')):
        return False
    return any(tok in blob for tok in ('ready', 'approved', 'pass', 'passed', 'verified'))


def sanitize_slice_token(value: str, fallback: str = 'slice') -> str:
    token = ''.join(ch if (ch.isalnum() or ch in '_-') else '_' for ch in str(value or '').strip().lower()).strip('_')
    if not token:
        token = fallback
    if len(token) > 64:
        token = token[:64].rstrip('_')
    return token or fallback


def slice_id_from_key(key: str, fallback: str = 'slice') -> str:
    raw = str(key or '').strip().lower()
    if not raw:
        return sanitize_slice_token(fallback, fallback='slice')
    m = re.match(r'local_(?:arch|impl|review)_(?:guard|blocker|ground)_(.+)$', raw)
    if m:
        return sanitize_slice_token(m.group(1), fallback=fallback)
    return sanitize_slice_token(raw, fallback=fallback)


def trace_patch_info(trace: dict) -> dict:
    if not isinstance(trace, dict):
        return {}
    raw = trace.get('local_patch_info')
    if isinstance(raw, dict):
        return raw
    raw = trace.get('patch_info')
    if isinstance(raw, dict):
        return raw
    return {}


def trace_local_no_change(trace: dict) -> bool:
    if not isinstance(trace, dict):
        return False
    if bool(trace.get('slice_no_code_change', False)):
        return True
    info = trace_patch_info(trace)
    return bool(info.get('no_code_change_required', False))


def trace_local_patch_applied(trace: dict) -> bool:
    if not isinstance(trace, dict):
        return False
    if trace_local_no_change(trace):
        return False
    if bool(trace.get('slice_patch_applied', False)):
        return True
    info = trace_patch_info(trace)
    files = info.get('changed_files')
    return bool(isinstance(files, list) and any(str(x or '').strip() for x in files))


def trace_local_patch_validated(trace: dict) -> bool:
    if not isinstance(trace, dict):
        return False
    if bool(trace.get('slice_validation_ok', False)):
        return True
    info = trace_patch_info(trace)
    return bool(info.get('validation_ok', False))


def order_autonomy_funnel(children, *, order_id: str, now: float, since: float) -> dict:
    slices = {}
    implementer_attempts = 0
    implementer_failures = 0
    loop_breaker_count = 0
    latest_gate_ts = 0.0
    controller_verifications_no_slice = []

    for child in children:
        role = str(child.get('role') or '').strip().lower()
        state = str(child.get('state') or '').strip().lower()
        updated_at = float(child.get('updated_at') or child.get('created_at') or 0.0)
        if updated_at <= 0 or updated_at < since:
            continue
        trace = load_json(child.get('trace') or '{}')
        labels = load_json(child.get('labels') or '{}')
        explicit_slice_id = sanitize_slice_token(str(trace.get('slice_id') or ''), fallback='')
        if bool(trace.get('loop_breaker_triggered', False)):
            loop_breaker_count += 1
        if role == 'implementer_local':
            implementer_attempts += 1
            if state in ('failed', 'cancelled', 'blocked', 'blocked_approval'):
                implementer_failures += 1
        if role not in LOCAL_ROLES and role not in CONTROLLER_ROLES:
            continue
        slice_id = str(trace.get('slice_id') or '').strip()
        if not slice_id:
            slice_id = slice_id_from_key(str(labels.get('key') or ''), fallback=f"{order_id[:8]}_slice")
        bucket = slices.setdefault(
            slice_id,
            {
                'started': False,
                'applied': False,
                'validated': False,
                'review_ready_candidate': False,
                'reviewed_ready': False,
                'controller_verified': False,
                'failed': False,
                'started_at': 0.0,
                'closed_at': 0.0,
            },
        )
        created_at = float(child.get('created_at') or updated_at or 0.0)
        if created_at > 0 and (bucket['started_at'] <= 0 or created_at < bucket['started_at']):
            bucket['started_at'] = created_at
        if state in ('failed', 'cancelled', 'blocked', 'blocked_approval'):
            bucket['failed'] = True
        summary = str(trace.get('result_summary') or '')

        if role == 'implementer_local':
            bucket['started'] = True
            if state == 'done' and trace_local_patch_applied(trace):
                bucket['applied'] = True
                latest_gate_ts = max(latest_gate_ts, updated_at)
            if state == 'done' and trace_local_patch_applied(trace) and trace_local_patch_validated(trace):
                bucket['validated'] = True
                latest_gate_ts = max(latest_gate_ts, updated_at)
            continue

        if role == 'reviewer_local' and state == 'done':
            if summary_has_ready_signal(summary):
                bucket['review_ready_candidate'] = True
                latest_gate_ts = max(latest_gate_ts, updated_at)
            continue

        if role in CONTROLLER_ROLES and state == 'done':
            if ('verified_improvement' in summary.lower()) or bool(trace.get('improvement_verified', False)):
                if explicit_slice_id and explicit_slice_id in slices:
                    target = slices.get(explicit_slice_id) or bucket
                    target['controller_verified'] = True
                    target['closed_at'] = max(float(target.get('closed_at') or 0.0), updated_at)
                    latest_gate_ts = max(latest_gate_ts, updated_at)
                else:
                    controller_verifications_no_slice.append(updated_at if updated_at > 0 else now)

    for bucket in slices.values():
        bucket['reviewed_ready'] = bool(bucket.get('review_ready_candidate') and bucket.get('validated'))

    if controller_verifications_no_slice:
        for event_ts in sorted(controller_verifications_no_slice):
            chosen_slice = ''
            chosen_started = 0.0
            for sid, bucket in slices.items():
                if bool(bucket.get('controller_verified')):
                    continue
                if not (bool(bucket.get('validated')) and bool(bucket.get('reviewed_ready'))):
                    continue
                started_at = float(bucket.get('started_at') or 0.0)
                if started_at > (event_ts + 5.0):
                    continue
                if started_at >= chosen_started:
                    chosen_started = started_at
                    chosen_slice = sid
            if not chosen_slice:
                continue
            picked = slices.get(chosen_slice) or {}
            picked['controller_verified'] = True
            picked['closed_at'] = max(float(picked.get('closed_at') or 0.0), float(event_ts or now))
            latest_gate_ts = max(latest_gate_ts, float(event_ts or now))

    started = 0
    applied = 0
    validated = 0
    closed = 0
    mtv_samples = []
    has_failed = False
    for bucket in slices.values():
        is_started = bool(bucket.get('started') or bucket.get('applied') or bucket.get('validated') or bucket.get('reviewed_ready'))
        if is_started:
            started += 1
        if bool(bucket.get('applied')):
            applied += 1
        if bool(bucket.get('validated')):
            validated += 1
        is_closed = bool(bucket.get('validated') and bucket.get('reviewed_ready') and bucket.get('controller_verified'))
        if is_closed:
            closed += 1
            start_ts = float(bucket.get('started_at') or 0.0)
            end_ts = float(bucket.get('closed_at') or 0.0) or now
            if start_ts > 0 and end_ts >= start_ts:
                mtv_samples.append(end_ts - start_ts)
        if bool(bucket.get('failed')):
            has_failed = True

    quality_gate_status = 'planned'
    if closed > 0:
        quality_gate_status = 'closed'
    elif any(bool(x.get('reviewed_ready')) for x in slices.values()):
        quality_gate_status = 'reviewed_ready'
    elif validated > 0:
        quality_gate_status = 'validated'
    elif applied > 0:
        quality_gate_status = 'applied'
    elif has_failed:
        quality_gate_status = 'failed'
    elif started > 0:
        quality_gate_status = 'implementing'

    fail_rate = 0.0
    if implementer_attempts > 0:
        fail_rate = float(implementer_failures) / float(implementer_attempts)
    mtv = None
    if mtv_samples:
        mtv = float(sum(mtv_samples) / max(1, len(mtv_samples)))
    last_gate_transition_age_s = None
    if latest_gate_ts > 0:
        last_gate_transition_age_s = max(0, int(now - latest_gate_ts))

    return {
        'slices_started': int(started),
        'slices_applied': int(applied),
        'slices_validated': int(validated),
        'slices_closed': int(closed),
        'implementer_fail_rate': round(float(fail_rate), 4),
        'mean_time_to_validated_improvement': mtv,
        'loop_breaker_count': int(loop_breaker_count),
        'quality_gate_status': quality_gate_status,
        'improvement_verified': bool(closed > 0),
        'implementer_attempts': int(implementer_attempts),
        'implementer_failures': int(implementer_failures),
        'last_gate_transition_age_s': last_gate_transition_age_s,
    }


def order_evidence_state(children):
    delivery_ok = False
    validation_ok = False
    latest_delivery_ts = 0.0
    latest_validation_ts = 0.0
    for child in children:
        if str(child['state'] or '').strip().lower() != 'done':
            continue
        role = str(child['role'] or '').strip().lower()
        trace = load_json(child['trace'])
        result_status = str(trace.get('result_status') or child['state'] or '').strip().lower()
        summary = str(trace.get('result_summary') or '').strip()
        updated_at = float(child.get('updated_at') or child.get('created_at') or 0.0)
        if result_status in ('error', 'failed', 'blocked', 'blocked_approval', 'cancelled'):
            continue
        if role in DELIVERY_ROLES and (trace_has_nontrivial_artifacts(trace) or summary_has_progress_signal(summary)):
            delivery_ok = True
            latest_delivery_ts = max(latest_delivery_ts, updated_at)
        if role in VALIDATION_ROLES:
            validation_verdict = ('no-go' in f'{result_status}\n{summary}'.lower()) or summary_has_validation_signal(summary)
            validation_evidence = trace_has_nontrivial_artifacts(trace) or len(summary) >= 80
            if validation_verdict and validation_evidence:
                validation_ok = True
                latest_validation_ts = max(latest_validation_ts, updated_at)
        if delivery_ok and validation_ok and latest_validation_ts >= max(0.0, latest_delivery_ts - 30.0):
            break
    improvement_ok = bool(delivery_ok and validation_ok and latest_validation_ts >= max(0.0, latest_delivery_ts - 30.0))
    return delivery_ok, validation_ok, improvement_ok, latest_delivery_ts, latest_validation_ts


def fetch_rows(cur, query: str, params=()):
    rows = cur.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _write_error_report(*, now: float, stamp: str, reason: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'generated_at': datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
        'operational_status': 'CRITICAL',
        'error': reason,
        'db_path': str(DB),
        'state_file': str(STATE_FILE),
        'order_reports': [],
        'anomalies': [],
        'trend_flags': [],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    markdown = '\n'.join(
        [
            '# Proactive Health Report',
            f'- status=CRITICAL reason={reason}',
            f'- db_path={DB}',
            f'- state_file={STATE_FILE}',
        ]
    ) + '\n'
    latest_json = OUT_DIR / 'latest.json'
    latest_md = OUT_DIR / 'latest.md'
    stamped_json = OUT_DIR / f'report_{stamp}.json'
    stamped_md = OUT_DIR / f'report_{stamp}.md'
    for path, content in ((latest_json, payload), (latest_md, markdown), (stamped_json, payload), (stamped_md, markdown)):
        path.write_text(content, encoding='utf-8')
    return 2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    state = load_state()
    factory_hard_stop = bool(state.get('factory_hard_stop', False) or state.get('proactive_lane_manual_pause', False))
    try:
        soft_pause_until = float(state.get('factory_soft_pause_until') or 0.0)
    except Exception:
        soft_pause_until = 0.0
    factory_soft_pause_active = bool((not factory_hard_stop) and soft_pause_until > now)
    factory_pause_reason = str(state.get('factory_pause_reason') or state.get('proactive_lane_paused_reason') or '').strip()

    if not DB.exists():
        return _write_error_report(now=now, stamp=stamp, reason='db_missing')

    if not DB.exists():
        return _write_error_report(now=now, stamp=stamp, reason='db_missing')

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    order_rows = fetch_rows(cur, "SELECT * FROM ceo_orders WHERE status IN ('active','paused') ORDER BY updated_at DESC")
    proactive_orders = [row for row in order_rows if is_proactive(row.get('title', ''), row.get('body', ''))]
    try:
        repo_rows = fetch_rows(cur, 'SELECT * FROM repo_registry ORDER BY priority ASC, updated_at DESC')
    except Exception:
        repo_rows = []
    try:
        heartbeat_rows = fetch_rows(cur, 'SELECT * FROM agent_runtime_state ORDER BY updated_at DESC')
    except Exception:
        heartbeat_rows = []

    since = now - LOOKBACK_S
    order_reports = []
    anomalies = []
    funnel_totals = {
        'slices_started': 0,
        'slices_applied': 0,
        'slices_validated': 0,
        'slices_closed': 0,
        'implementer_attempts': 0,
        'implementer_failures': 0,
        'loop_breaker_count': 0,
    }
    mtv_samples = []
    for order in proactive_orders:
        order_id = str(order.get('order_id') or '').strip()
        if not order_id:
            continue
        children = fetch_rows(
            cur,
            'SELECT job_id, state, role, labels, trace, updated_at, created_at FROM jobs WHERE parent_job_id = ? ORDER BY updated_at DESC',
            (order_id,),
        )
        open_children = [c for c in children if str(c.get('state') or '').strip().lower() in ACTIVE_STATES]
        activity_ts = [float(order.get('updated_at') or 0.0)]
        activity_ts.extend(float(c.get('updated_at') or c.get('created_at') or 0.0) for c in children)
        last_activity_at = max((ts for ts in activity_ts if ts > 0), default=0.0)
        last_activity_age_s = max(0, int(now - last_activity_at)) if last_activity_at > 0 else None
        counts_by_role = {}
        stale_local = []
        for child in open_children:
            role = str(child.get('role') or '').strip().lower()
            counts_by_role[role] = counts_by_role.get(role, 0) + 1
            updated_at = float(child.get('updated_at') or child.get('created_at') or now)
            age_s = max(0, int(now - updated_at))
            if role in LOCAL_ROLES and age_s >= STALE_LOCAL_S:
                label_data = load_json(child.get('labels') or '{}')
                stale_local.append({
                    'job_id': str(child.get('job_id') or ''),
                    'role': role,
                    'state': str(child.get('state') or ''),
                    'age_s': age_s,
                    'key': str(label_data.get('key') or ''),
                })
        mode = 'idle'
        has_local = any(role in LOCAL_ROLES for role in counts_by_role)
        has_cli = any(role in CLI_ROLES for role in counts_by_role)
        if has_local and has_cli:
            mode = 'mixed'
        elif has_cli:
            mode = 'cli'
        elif has_local:
            mode = 'local'
        autonomy_funnel = order_autonomy_funnel(children, order_id=order_id, now=now, since=since)
        delivery_ok = bool(int(autonomy_funnel.get('slices_applied', 0) or 0) > 0)
        validation_ok = bool(int(autonomy_funnel.get('slices_validated', 0) or 0) > 0)
        improvement = bool(autonomy_funnel.get('improvement_verified', False))
        latest_delivery_ts = 0.0
        latest_validation_ts = 0.0
        if validation_ok and delivery_ok:
            latest_delivery_ts = float(order.get('updated_at') or 0.0)
            latest_validation_ts = latest_delivery_ts
        phase = str(order.get('phase') or '').strip().lower()
        ready_with_open = phase == 'ready_for_merge' and bool(open_children)
        if ready_with_open:
            anomalies.append({
                'type': 'ready_with_open_work',
                'order_id': order_id,
                'phase': order.get('phase'),
                'open_jobs': len(open_children),
            })
        validation_covers_latest_delivery = bool(validation_ok and int(autonomy_funnel.get('slices_validated', 0) or 0) >= int(autonomy_funnel.get('slices_applied', 0) or 0))
        if phase == 'ready_for_merge' and not improvement:
            anomalies.append({
                'type': 'ready_without_quality_gate',
                'order_id': order_id,
                'delivery_evidence_seen': delivery_ok,
                'independent_validation_seen': validation_ok,
                'validation_covers_latest_delivery': validation_covers_latest_delivery,
            })
        if phase == 'review' and bool(open_children):
            review_age = autonomy_funnel.get('last_gate_transition_age_s')
            if review_age is not None and int(review_age) > 45 * 60 and not improvement:
                anomalies.append({
                    'type': 'review_stalled_over_45m',
                    'order_id': order_id,
                    'phase': order.get('phase'),
                    'last_gate_transition_age_s': int(review_age),
                })
        if stale_local:
            anomalies.append({
                'type': 'stale_local_open',
                'order_id': order_id,
                'count': len(stale_local),
            })
        if (not open_children) and (not improvement) and (last_activity_age_s is not None) and last_activity_age_s >= 900:
            anomalies.append({
                'type': 'idle_without_improvement',
                'order_id': order_id,
                'phase': order.get('phase'),
                'last_activity_age_s': last_activity_age_s,
            })
        funnel_totals['slices_started'] += int(autonomy_funnel.get('slices_started', 0) or 0)
        funnel_totals['slices_applied'] += int(autonomy_funnel.get('slices_applied', 0) or 0)
        funnel_totals['slices_validated'] += int(autonomy_funnel.get('slices_validated', 0) or 0)
        funnel_totals['slices_closed'] += int(autonomy_funnel.get('slices_closed', 0) or 0)
        funnel_totals['implementer_attempts'] += int(autonomy_funnel.get('implementer_attempts', 0) or 0)
        funnel_totals['implementer_failures'] += int(autonomy_funnel.get('implementer_failures', 0) or 0)
        funnel_totals['loop_breaker_count'] += int(autonomy_funnel.get('loop_breaker_count', 0) or 0)
        mtv = autonomy_funnel.get('mean_time_to_validated_improvement')
        if isinstance(mtv, (int, float)) and float(mtv) >= 0:
            mtv_samples.append(float(mtv))
        order_reports.append({
            'order_id': order_id,
            'title': str(order.get('title') or ''),
            'phase': str(order.get('phase') or ''),
            'mode': mode,
            'open_jobs': len(open_children),
            'counts_by_role': counts_by_role,
            'stale_local': stale_local[:8],
            'delivery_evidence_seen': delivery_ok,
            'independent_validation_seen': validation_ok,
            'validation_covers_latest_delivery': validation_covers_latest_delivery,
            'meaningful_improvement_seen': improvement,
            'last_activity_age_s': last_activity_age_s,
            'latest_delivery_ts': latest_delivery_ts,
            'latest_validation_ts': latest_validation_ts,
            'quality_gate_status': str(autonomy_funnel.get('quality_gate_status') or 'planned'),
            'autonomy_funnel': autonomy_funnel,
        })

    empty_response_failures = cur.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE updated_at >= ? AND trace LIKE ?",
        (since, '%empty response from local ollama model%'),
    ).fetchone()['n']
    enabled_repos = [row for row in repo_rows if str(row.get('status') or '').strip().lower() == 'active' and bool(int(row.get('autonomy_enabled') or 0))]
    stale_heartbeats = 0
    for hb in heartbeat_rows:
        try:
            heartbeat_at = float(hb.get('heartbeat_at') or 0.0)
        except Exception:
            heartbeat_at = 0.0
        if heartbeat_at <= 0 or (now - heartbeat_at) >= STALE_LOCAL_S:
            stale_heartbeats += 1

    def audit_count(event_type: str) -> int:
        return int(
            cur.execute(
                'SELECT COUNT(*) AS n FROM audit_log WHERE ts >= ? AND event_type = ?',
                (since, event_type),
            ).fetchone()['n']
        )

    implementer_fail_rate = 0.0
    if int(funnel_totals['implementer_attempts']) > 0:
        implementer_fail_rate = float(funnel_totals['implementer_failures']) / float(funnel_totals['implementer_attempts'])
    mean_time_to_validated_improvement = None
    if mtv_samples:
        mean_time_to_validated_improvement = float(sum(mtv_samples) / max(1, len(mtv_samples)))

    metrics = {
        'proactive_active_orders': len(order_reports),
        'slices_started': int(funnel_totals['slices_started']),
        'slices_applied': int(funnel_totals['slices_applied']),
        'slices_validated': int(funnel_totals['slices_validated']),
        'slices_closed': int(funnel_totals['slices_closed']),
        'implementer_fail_rate': round(float(implementer_fail_rate), 4),
        'mean_time_to_validated_improvement': mean_time_to_validated_improvement,
        'loop_breaker_count': int(funnel_totals['loop_breaker_count']),
        'empty_response_failures_24h': int(empty_response_failures),
        'proactive_cli_takeovers_24h': audit_count('delegation.proactive_cli_takeover'),
        'proactive_cli_seeded_24h': audit_count('delegation.proactive_cli_seeded'),
        'proactive_cli_reseeded_24h': audit_count('delegation.proactive_cli_reseeded'),
        'stale_local_cancels_24h': audit_count('task.proactive_local_stale_cancelled'),
        'final_sweeps_24h': audit_count('order.final_sweep_enqueued'),
        'factory_model_fallback_24h': audit_count('factory.model_fallback'),
        'factory_repo_orders_24h': audit_count('factory.repo_order.created'),
        'factory_enabled_repos': len(enabled_repos),
        'factory_registered_repos': len(repo_rows),
        'factory_stale_heartbeats': int(stale_heartbeats),
    }

    operational_status = 'OK'
    if factory_hard_stop:
        operational_status = 'CRITICAL'
        anomalies.append({
            'type': 'factory_hard_stop',
            'reason': factory_pause_reason or 'unspecified',
        })
    elif factory_soft_pause_active:
        operational_status = 'WARN'
        anomalies.append({
            'type': 'factory_soft_pause',
            'reason': factory_pause_reason or 'unspecified',
            'soft_pause_until': soft_pause_until,
        })
    elif any(item['type'] in ('ready_with_open_work', 'ready_without_quality_gate') for item in anomalies):
        operational_status = 'CRITICAL'
    elif anomalies:
        operational_status = 'WARN'
    if metrics['factory_enabled_repos'] > 0 and not proactive_orders:
        anomalies.append({
            'type': 'factory_without_active_orders',
            'enabled_repos': metrics['factory_enabled_repos'],
        })
        operational_status = worst_status(operational_status, 'WARN')
    if metrics['factory_stale_heartbeats'] > 0:
        anomalies.append({
            'type': 'factory_stale_heartbeats',
            'count': metrics['factory_stale_heartbeats'],
        })
        operational_status = worst_status(operational_status, 'WARN')

    trend_flags = []
    if metrics['empty_response_failures_24h'] >= 10:
        trend_flags.append({
            'type': 'high_empty_local_responses',
            'count': metrics['empty_response_failures_24h'],
        })
    if metrics['stale_local_cancels_24h'] >= 6:
        trend_flags.append({
            'type': 'high_stale_local_cancels',
            'count': metrics['stale_local_cancels_24h'],
        })
    if metrics['final_sweeps_24h'] >= 8:
        trend_flags.append({
            'type': 'high_final_sweeps',
            'count': metrics['final_sweeps_24h'],
        })
    if metrics['factory_model_fallback_24h'] >= 5:
        trend_flags.append({
            'type': 'high_model_fallbacks',
            'count': metrics['factory_model_fallback_24h'],
        })
    trend_status = 'WARN' if trend_flags else 'OK'
    status = worst_status(operational_status, trend_status)

    report = {
        'ts': now,
        'stamp': stamp,
        'status': status,
        'operational_status': operational_status,
        'trend_status': trend_status,
        'factory': {
            'hard_stop': factory_hard_stop,
            'soft_pause_active': factory_soft_pause_active,
            'pause_reason': factory_pause_reason,
            'soft_pause_until': soft_pause_until if factory_soft_pause_active else None,
            'registered_repos': len(repo_rows),
            'enabled_repos': len(enabled_repos),
            'stale_heartbeats': int(stale_heartbeats),
        },
        'metrics': metrics,
        'autonomy_funnel': {
            'slices_started': metrics['slices_started'],
            'slices_applied': metrics['slices_applied'],
            'slices_validated': metrics['slices_validated'],
            'slices_closed': metrics['slices_closed'],
            'implementer_fail_rate': metrics['implementer_fail_rate'],
            'mean_time_to_validated_improvement': metrics['mean_time_to_validated_improvement'],
            'loop_breaker_count': metrics['loop_breaker_count'],
            'autonomy_minimum_slo': {
                'implementer_fail_rate_lte': 0.35,
                'closed_over_started_gte': 0.50,
            },
        },
        'orders': order_reports,
        'anomalies': anomalies,
        'trend_flags': trend_flags,
    }

    lines = [
        f'# Proactive Health {stamp}',
        '',
        f'- Status: {status}',
        f'- Operational status: {operational_status}',
        f'- Trend status: {trend_status}',
        f"- Factory state: {'hard_stop' if factory_hard_stop else ('soft_pause' if factory_soft_pause_active else 'active')}",
        f"- Factory repos: registered={metrics['factory_registered_repos']} enabled={metrics['factory_enabled_repos']}",
        f"- Factory stale heartbeats: {metrics['factory_stale_heartbeats']}",
        f"- Active proactive orders: {metrics['proactive_active_orders']}",
        f"- Autonomy funnel (24h): started={metrics['slices_started']} applied={metrics['slices_applied']} validated={metrics['slices_validated']} closed={metrics['slices_closed']}",
        f"- Implementer fail rate (24h): {metrics['implementer_fail_rate']:.2%}",
        f"- Mean time to validated improvement: {('n/a' if metrics['mean_time_to_validated_improvement'] is None else str(int(metrics['mean_time_to_validated_improvement'])) + 's')}",
        f"- Loop breaker count (24h): {metrics['loop_breaker_count']}",
        f"- Empty local responses (24h): {metrics['empty_response_failures_24h']}",
        f"- CLI takeovers (24h): {metrics['proactive_cli_takeovers_24h']}",
        f"- CLI seeds/reseeds (24h): {metrics['proactive_cli_seeded_24h']}/{metrics['proactive_cli_reseeded_24h']}",
        f"- Stale local cancels (24h): {metrics['stale_local_cancels_24h']}",
        f"- Final sweeps (24h): {metrics['final_sweeps_24h']}",
        f"- Model fallbacks (24h): {metrics['factory_model_fallback_24h']}",
        '',
        '## Orders',
    ]
    if not order_reports:
        lines.append('- No active proactive orders.')
    for item in order_reports:
        lines.append(
            f"- {item['order_id'][:8]} phase={item['phase']} mode={item['mode']} open={item['open_jobs']} gate={item['quality_gate_status']} validated_improvement={item['meaningful_improvement_seen']} activity_age_s={item['last_activity_age_s']}"
        )
        funnel = item.get('autonomy_funnel') or {}
        lines.append(
            "  "
            f"funnel: started={int(funnel.get('slices_started', 0) or 0)} "
            f"applied={int(funnel.get('slices_applied', 0) or 0)} "
            f"validated={int(funnel.get('slices_validated', 0) or 0)} "
            f"closed={int(funnel.get('slices_closed', 0) or 0)} "
            f"fail_rate={float(funnel.get('implementer_fail_rate', 0.0) or 0.0):.2%}"
        )
        if item['counts_by_role']:
            role_line = ', '.join(f"{role}:{count}" for role, count in sorted(item['counts_by_role'].items()))
            lines.append(f"  roles: {role_line}")
        if item['stale_local']:
            for stale in item['stale_local'][:4]:
                lines.append(
                    f"  stale: {stale['job_id'][:8]} role={stale['role']} state={stale['state']} age_s={stale['age_s']} key={stale['key']}"
                )
    lines.append('')
    lines.append('## Anomalies')
    if not anomalies:
        lines.append('- None')
    else:
        for item in anomalies:
            lines.append(f"- {item['type']}: {json.dumps(item, ensure_ascii=False)}")
    lines.append('')
    lines.append('## Trend Flags')
    if not trend_flags:
        lines.append('- None')
    else:
        for item in trend_flags:
            lines.append(f"- {item['type']}: {json.dumps(item, ensure_ascii=False)}")

    latest_json = OUT_DIR / 'latest.json'
    latest_md = OUT_DIR / 'latest.md'
    stamped_json = OUT_DIR / f'report_{stamp}.json'
    stamped_md = OUT_DIR / f'report_{stamp}.md'
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    markdown = '\n'.join(lines) + '\n'
    for path, content in ((latest_json, payload), (latest_md, markdown), (stamped_json, payload), (stamped_md, markdown)):
        path.write_text(content, encoding='utf-8')

    print(markdown)
    return 1 if operational_status == 'CRITICAL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
