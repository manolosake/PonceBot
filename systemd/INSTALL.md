# systemd Installation

This repo includes two service setups:
- A system-wide service (root-owned)
- A user service (recommended for a single user like `aponce`)

## Option A: System Service (root)

1. Copy units:
```bash
sudo cp /home/aponce/codexbot/systemd/codexbot.service /etc/systemd/system/codexbot.service
sudo cp /home/aponce/codexbot/systemd/codexbot-alert@.service /etc/systemd/system/codexbot-alert@.service
```

2. Reload + enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now codexbot
```

3. Logs:
```bash
sudo journalctl -u codexbot -f
```

## Option B: User Service (recommended)

1. Install units:
```bash
mkdir -p ~/.config/systemd/user
cp /home/aponce/codexbot/systemd/codexbot-user.service ~/.config/systemd/user/codexbot.service
cp /home/aponce/codexbot/systemd/codexbot-alert-user@.service ~/.config/systemd/user/codexbot-alert-user@.service
```

Optional (Ollama as a user service):
```bash
cp /home/aponce/codexbot/systemd/ollama-user.service ~/.config/systemd/user/ollama.service
```

2. (Optional) Keep it running 24/7 even without an active login session:
```bash
loginctl enable-linger aponce
```

3. Enable:
```bash
systemctl --user daemon-reload
systemctl --user enable --now codexbot
```

If you installed Ollama as a service:
```bash
systemctl --user enable --now ollama
```

4. Logs:
```bash
journalctl --user -u codexbot -f
journalctl --user -u codexbot-alert-user@codexbot.service -n 50 --no-pager
journalctl --user -u ollama -f
```

## Notes

- Put secrets in a separate file outside the repo (recommended):
  - `~/.config/codexbot/secrets.env`
  - pass it via `ENV_LOCAL_FILE=...` or systemd `Environment=ENV_LOCAL_FILE=...`
