# PonceBot Autonomous Studio

PonceBot turns r530 into a Telegram-first, Codex-powered software factory.

It has two isolated operating lanes:

- **CEO on-demand lane:** Alejandro messages PonceBot through Telegram. Jarvis answers directly for simple questions or delegates bounded work to delivery roles.
- **Autonomous Studio lane:** Skynet periodically evaluates the workspace, chooses high-value software bets, delegates implementation, validates, ships, and learns.

The system is intentionally conservative with Codex usage. It should prefer fewer valuable outcomes over many low-signal jobs.

## Current Production Shape

The r530 deployment runs from:

- repo: `/home/aponce/codexbot`
- branch: `main`
- runtime worktree: `/home/aponce/codexbot/data/runtime_worktrees/current`
- queue DB: `/home/aponce/codexbot/data/jobs.sqlite`
- service: `codexbot.service` as a **user** systemd service
- isolated CEO worker: `poncebot-ceo.service`
- deploy monitor: `codexbot-main-deploy-monitor.service`

Source-of-truth files:

- `bot.py`: Telegram bot, scheduler, orchestration, Studio memory, gates, and outcome logic.
- `alexa_gateway.py`: Alexa Custom Skill gateway for Echo Pop voice access.
- `orchestrator/agents.yaml`: agent roster, models, effort, prompts, and role limits.
- `orchestrator/runbooks.yaml`: scheduled runbooks and diagnostics.
- `codexbot.env.example`: documented environment knobs.
- `systemd/`: user/system service definitions.
- `tools/`: deploy, diagnostics, reports, security, and production helpers.

Echo Pop voice setup is documented in `docs/alexa_echo_pop.md`.

Production-specific secrets and overrides live outside git or in the private runtime env file. Do not commit real tokens, passwords, IDs, or private env files.

## Operating Model

### CEO On-Demand Lane

Jarvis is the front door for Telegram.

- Pure questions should be answered directly and briefly.
- Real work should become a traceable ticket/order.
- Jarvis delegates to delivery roles only when useful.
- The CEO gets a concise status card and an executive wrap-up, not a stream of noisy logs.
- Manual CEO work is isolated from the autonomous factory lane through the CEO command plane.

Useful Telegram commands:

- `/help`: command list
- `/whoami`: show Telegram ids
- `/agents`: agent queue and live status
- `/ticket <id>`: ticket tree
- `/job <id>`: job details
- `/orders`: active orders
- `/order show|pause|done <id>`: manage an order
- `/watch` / `/unwatch`: single live status message
- `/pause <role>` / `/resume <role>`: role control
- `/emergency_stop` / `/emergency_resume`: global stop/resume

### Autonomous Studio Lane

Skynet owns proactive work. It should behave like a technical/product director, not like a cron job that creates work for its own sake.

Each Studio cycle follows this mental model:

1. **Sense:** inspect repos, GitHub state, jobs, deploys, logs, branches, runbooks, portfolio assets, and memory.
2. **Understand:** infer what hurts, what has failed before, what is stale, and what has real opportunity.
3. **Imagine:** generate possible bets: bug fix, product feature, PonceBot improvement, ExecutiveDashboard improvement, or new monetizable project.
4. **Debate:** reject weak, repetitive, no-delta, cosmetic, non-shippable, or non-monetizable work before spending more credits.
5. **Choose:** pick one bounded bet or do no work if all options are weak.
6. **Build:** delegate implementation to the right delivery role. Skynet must not edit repository files directly.
7. **Judge:** require QA/reviewer/release evidence: tests, diffs, logs, screenshots, deploy checks, or clear residual risks.
8. **Ship:** merge/push/deploy to `main`, or publish a new private GitHub repo when the work is a new project.
9. **Learn:** record the outcome and use it to improve future selection.

Completion is outcome-based. A job being `done` is not enough.

Allowed final outcomes:

- `shipped_to_main`: merged, pushed, and deployed when the repo has a deploy path.
- `published_project`: new private GitHub repo exists with README/demo/validation evidence.
- `blocked_need_operator`: one specific human decision is required.
- `rejected_low_value`: the idea was killed before wasting more execution.
- `failed_root_caused`: the work failed and the cause is recorded.

## Cost Discipline

Codex credits are spent only when the system invokes Codex-backed agents.

Typical cost profile by cycle stage:

- Low or no Codex: `Sense`, local DB reads, git status, logs, health checks, deploy checks, local gates, and simple persistence.
- Medium Codex: `Understand`, `Imagine`, `Debate`, and `Choose` when Skynet or Critic reason over candidate bets.
- Highest Codex: `Build` and `Judge`, because delivery and QA agents inspect code, write changes, and review results.
- Usually low Codex: `Ship` and `Learn`, unless a merge conflict, release note, or root-cause summary needs agentic reasoning.

The budget-first production profile is intentionally slow:

- one orchestrator worker
- one proactive order at a time
- one active autonomous subtask at a time
- one proactive iteration round
- proactive lane interval measured in hours, not minutes

This is expected behavior. If no high-value work is available, PonceBot should stay idle or reject the bet instead of burning credits.

## Agent Roster

The authoritative roster is `orchestrator/agents.yaml`.

The current budget-controlled profile uses `gpt-5.4` with `medium` effort for the configured roles, including Jarvis and Skynet. Raise model or effort only when the value of the work justifies higher credit usage.

Current role model:

- `jarvis`: CEO front door and command routing.
- `skynet`: Autonomous Studio controller, planning, review, and final judgement.
- `critic`: adversarial product/engineering review before execution.
- `product_ops`: acceptance criteria, scope, and product framing.
- `research`: repo/opportunity investigation.
- `backend`: services, data, APIs, tests, and most code changes.
- `frontend`: UI changes with visual evidence.
- `qa`: regression and acceptance validation.
- `sre`: systemd, runtime, deploy, logs, health, and production reliability.
- `security`: hardening and risk review.
- `release_mgr`: merge, push, release, cleanup, and deploy readiness.
- local specialist roles: advisory/recovery roles when explicitly routed.

Skynet should not use every profile by default. It should delegate only the roles needed for the selected slice.

## Branch, Merge, and Deploy Rules

Normal operating rule:

- `main` is the base branch and production truth for PonceBot.
- Proactive branches should be short-lived and merge back to `main` only after validation.
- Do not count branch-only work as success.
- Runtime `current` should be deployed from `main`.
- The deploy monitor keeps the runtime aligned with `origin/main`.

Release evidence should include:

- commit hash
- branch merged to `main`
- push confirmation
- deploy/current runtime hash
- relevant test or health output
- cleanup notes for stale worktrees/branches when applicable

## Memory and Portfolio

Autonomous Studio uses persistent state beyond terminal logs:

- `jobs`: work queue and role-level execution.
- `ceo_orders`: operator and proactive orders.
- `studio_cycles`: candidate bets, selected thesis, debate summary, and prompt packets.
- `studio_memory`: durable facts and lessons.
- `studio_portfolio_projects`: new or incubated products.
- `repo_registry` / `projects_registry`: known repos and projects.
- `audit_log`, `job_events`, and related tables: traceability and recovery evidence.

The system should remember:

- what repos exist and why they matter
- what features already exist
- repeated failures or no-delta patterns
- what projects were created
- what was rejected and why
- what reached `main`
- what deployed
- what Alejandro prefers: visible, useful, creative, shippable, and low-noise work

## New Projects

New projects are valid only when they can become useful or monetizable.

A new project should have:

- folder under `/home/aponce/<project-name>`
- clean git repo
- private GitHub remote under the operator account
- README with buyer/user, problem, offer, validation, and next milestone
- runnable or demoable artifact when practical
- registration in the dashboard/project registry
- outcome `published_project`, not just local files

The factory should not create many shallow MVPs if compounding an existing product is more valuable.

## Self-Healing and Failure Handling

PonceBot should repair or close operationally terminal states without waiting forever.

Important recovery behavior:

- stale or terminal blocked children should not keep a root proactive order alive forever
- write-policy violations should be root-caused and closed or recovered through delegated implementation
- no-delta work should become `rejected_low_value`, not success
- weak incubator work can be preempted by the economic discipline gate
- idle diagnostics should verify there are no hidden workers before declaring the factory idle

This does not mean every failure auto-fixes itself. Destructive work, public launch decisions, external spending, credential changes, and data-loss risks still require the operator.

## Dashboard Expectations

ExecutiveDashboard should show operator signal, not raw noise.

Agents view should make clear:

- whether PonceBot is active or paused
- current proactive thesis and objective
- repo, branch, worktree, phase, and agents involved
- why the work matters
- evidence still missing
- last shipped/published/rejected item
- agent diagnostics with terminals still available for inspection

Projects view should make clear:

- GitHub remote
- branch count
- open PR/issues when available
- latest `main` commits
- last push/deploy status
- next milestone
- stale/blocked/active state

The dashboard is not the source of truth. It should render the DB, git, runtime, and telemetry truth clearly.

## Safety

Default posture:

- allow only configured Telegram users/chats
- keep secrets out of git
- prefer read-only or bounded write modes
- use breakglass only with explicit reason and short TTL
- require concrete evidence before claiming success
- ask for approval before public publishing, destructive operations, external spending, billing, credentials, or data-loss risks

Breakglass notes:

- `CODEX_DANGEROUS_BYPASS_SANDBOX=1` is not normal mode.
- `BOT_BREAKGLASS_REASON` and `BOT_BREAKGLASS_TTL_SECONDS` should be explicit.
- Admin allowlists should be configured before relying on Telegram breakglass controls.

## Verification

Before shipping code changes:

```bash
make verify
```

Common targeted checks:

```bash
python3 -m py_compile bot.py test_bot.py
python3 -m unittest test_bot.TestStudioOutcomeMemory
python3 -m unittest test_bot.TestSkynetLocalOnlyProactivePolicy
python3 -m unittest -q
```

If `./scripts/jest_sharded_agg.sh` exists in the relevant workspace, run it for frontend/dashboard changes. If it does not exist, report that explicitly instead of pretending it ran.

## Operations

Service checks on r530:

```bash
systemctl --user status codexbot.service --no-pager -l
systemctl --user status poncebot-ceo.service --no-pager -l
systemctl --user status codexbot-main-deploy-monitor.service --no-pager -l
journalctl --user -u codexbot.service -n 120 --no-pager
```

Queue checks:

```bash
cd /home/aponce/codexbot
.venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect("data/jobs.sqlite")
con.row_factory = sqlite3.Row
for row in con.execute("select state, role, count(*) c from jobs group by state, role order by c desc"):
    print(dict(row))
con.close()
PY
```

Deploy alignment check:

```bash
cd /home/aponce/codexbot
git rev-parse --short HEAD
git rev-parse --short origin/main
git -C data/runtime_worktrees/current rev-parse --short HEAD
```

## What Good Looks Like

In a healthy 48 to 72 hour window:

- PonceBot stays active under systemd.
- Proactive cycles either ship, publish, reject low-value work, or fail with a clear root cause.
- There are no zombie orders.
- Valuable work reaches `main`, deploy, or a private GitHub repo.
- Local-only projects do not count as success.
- The dashboard explains what Skynet is doing and why.
- Credit usage remains bounded by low WIP, slow cadence, and aggressive rejection of weak work.

PonceBot is successful only when it produces durable value: better products, better factory capability, fewer failures, deployable code, published private assets, and clearer decisions.
