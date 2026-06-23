# PonceBot Alexa Skill

This package is for a private Alexa Custom Skill that routes Echo Pop voice
requests into the isolated PonceBot on-demand command plane.

## Files

- `skill-package/skill.json`: Alexa skill manifest. Replace the endpoint URI
  after the HTTPS tunnel URL and route secret are known.
- `skill-package/interactionModels/custom/es-MX.json`: Spanish Mexico model.
- `skill-package/interactionModels/custom/es-US.json`: Spanish US model.

## Manual Amazon step

Amazon authentication, account creation, and MFA must be completed by the owner
of `manolosake@gmail.com`. After login, the package can be imported in the Alexa
Developer Console or deployed with ASK CLI.
