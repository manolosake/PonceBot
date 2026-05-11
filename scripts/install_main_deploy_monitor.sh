#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"

install -d "$UNIT_DIR" "${HOME}/production-sites/current" "${HOME}/production-sites/releases"
install -m 0644 "$REPO_ROOT/systemd/poncebot-static-prod.service" "$UNIT_DIR/poncebot-static-prod.service"
install -m 0644 "$REPO_ROOT/systemd/codexbot-main-deploy-monitor.service" "$UNIT_DIR/codexbot-main-deploy-monitor.service"
install -m 0644 "$REPO_ROOT/systemd/codexbot-main-deploy-monitor.timer" "$UNIT_DIR/codexbot-main-deploy-monitor.timer"

systemctl --user daemon-reload
systemctl --user enable --now poncebot-static-prod.service
systemctl --user enable --now codexbot-main-deploy-monitor.timer
systemctl --user start codexbot-main-deploy-monitor.service

systemctl --user --no-pager --plain is-active poncebot-static-prod.service
systemctl --user --no-pager --plain is-active codexbot-main-deploy-monitor.timer
