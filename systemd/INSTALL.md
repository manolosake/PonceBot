# systemd install

## Opcion A: Servicio del sistema (recomendado)

1. Copia el unit:
   - `sudo cp /home/aponce/codexbot/systemd/codexbot.service /etc/systemd/system/codexbot.service`
   - `sudo cp /home/aponce/codexbot/systemd/codexbot-alert@.service /etc/systemd/system/codexbot-alert@.service`
2. Recarga y habilita:
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now codexbot`
3. Logs:
   - `sudo journalctl -u codexbot -f`

## Opcion B: Servicio de usuario

1. Instala el unit:
   - `mkdir -p ~/.config/systemd/user`
   - `cp /home/aponce/codexbot/systemd/codexbot-user.service ~/.config/systemd/user/codexbot.service`
   - `cp /home/aponce/codexbot/systemd/codexbot-alert-user@.service ~/.config/systemd/user/codexbot-alert-user@.service`
   - (Opcional) para Ollama:
     - `cp /home/aponce/codexbot/systemd/ollama-user.service ~/.config/systemd/user/ollama.service`
2. (Opcional) Para que arranque sin login, habilita linger:
   - `loginctl enable-linger aponce`
3. Habilita:
   - `systemctl --user daemon-reload`
   - (si usas Ollama) `systemctl --user enable --now ollama`
   - `systemctl --user enable --now codexbot`
4. Logs:
   - `journalctl --user -u codexbot -f`
   - `journalctl --user -u codexbot-alert-user@codexbot.service -n 50 --no-pager`
   - `journalctl --user -u ollama -f`
