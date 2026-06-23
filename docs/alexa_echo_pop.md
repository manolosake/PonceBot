# Echo Pop como voz de PonceBot

Este flujo usa el Echo Pop sin hardware adicional:

```text
Echo Pop mic -> Alexa Cloud -> Alexa Custom Skill -> PonceBot Alexa gateway -> PonceBot queue -> Echo Pop speaker
```

## Lo que si hace

- Usa el microfono y bocina del Echo Pop.
- Permite hablar con PonceBot usando una Alexa Custom Skill.
- Responde rapido a Alexa. Si Jarvis termina en pocos segundos, Alexa lee la respuesta. Si tarda mas, Alexa confirma que dejo un ticket en PonceBot.

## Lo que no hace

- No cambia el wake word a "Hola PonceBot".
- No obtiene audio crudo del microfono del Echo.
- No evita Alexa Cloud.

## Variables

Agregar al env del runtime, por ejemplo en `/home/aponce/codexbot/codexbot.env`:

```bash
PONCEBOT_ALEXA_LISTEN=127.0.0.1:8095
PONCEBOT_ALEXA_ENDPOINT_PATH=/alexa
PONCEBOT_ALEXA_PATH_SECRET=pon-un-secreto-largo-aqui
PONCEBOT_ALEXA_SKILL_ID=amzn1.ask.skill.xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PONCEBOT_ALEXA_VERIFY_SIGNATURE=1
PONCEBOT_ALEXA_CHAT_ID=8355547734
PONCEBOT_ALEXA_USER_ID=8355547734
PONCEBOT_ALEXA_WAIT_SECONDS=5.5
```

Para pruebas locales sin headers firmados de Alexa:

```bash
PONCEBOT_ALEXA_VERIFY_SIGNATURE=0
```

## Servicio local

Opcion recomendada: instalar como user service, igual que el runtime principal:

```bash
mkdir -p ~/.config/systemd/user
cp /home/aponce/codexbot/systemd/user/poncebot-alexa.service ~/.config/systemd/user/poncebot-alexa.service
systemctl --user daemon-reload
systemctl --user enable --now poncebot-alexa.service
```

Alternativa como system service, solo si quieres administrarlo con root:

```bash
sudo cp /home/aponce/codexbot/systemd/poncebot-alexa.service /etc/systemd/system/poncebot-alexa.service
sudo systemctl daemon-reload
sudo systemctl enable --now poncebot-alexa.service
```

Probar salud:

```bash
curl http://127.0.0.1:8095/health
```

Logs:

```bash
sudo journalctl -u poncebot-alexa -f
```

## Exponer a Alexa

Alexa necesita HTTPS publico. Opciones:

- Cloudflare Tunnel hacia `http://127.0.0.1:8095` (recomendado)
- ngrok hacia `http://127.0.0.1:8095` (util para pruebas)
- reverse proxy propio con TLS

La URL final del endpoint debe incluir el secreto de ruta:

```text
https://tu-dominio.example/alexa/pon-un-secreto-largo-aqui
```

## Alexa Developer Console

1. Crear una Custom Skill.
2. Locale recomendado: `es-MX` o `es-US`, segun el idioma configurado en el Echo.
3. Invocation name: `ponce bot`.
4. Importar o copiar el skill package desde `alexa/skill-package`.
5. Si se configura manualmente, copiar el interaction model desde:
   - `alexa/skill-package/interactionModels/custom/es-MX.json`
   - `alexa/skill-package/interactionModels/custom/es-US.json`
6. Endpoint: HTTPS, usando la URL publica del gateway.
7. Copiar el Skill ID a `PONCEBOT_ALEXA_SKILL_ID`.

## Paso que requiere al operador

Amazon no permite crear/configurar la skill sin login del owner. La cuenta
`manolosake@gmail.com` debe iniciar sesion en Amazon Developer y completar MFA si
Amazon lo pide. Despues de eso, PonceBot puede operar con el endpoint ya
configurado.

## Frases de uso

```text
Alexa, abre Ponce Bot.
Alexa, pregunta a Ponce Bot que tengo pendiente hoy.
Alexa, dile a Ponce Bot revisa el estado de los agentes.
```
