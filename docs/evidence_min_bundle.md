# Evidence Bundle Minimum (Backend/QA)

Objetivo: evitar entregas vacías estandarizando nombres y validación mínima.

## Archivos mínimos esperados
- `changes.patch` o `repo_patch.diff`
- `git_status.txt`
- log de pruebas: `tests.log` (o `test.log`, `verify.log`, `test_orchestrator.log`)
- salida estándar: `evidence_summary.json`

## Comando canónico
```bash
python3 tools/evidence_min_bundle.py --artifacts-dir <ARTIFACTS_DIR>/execution
```

## Resultado
- `EXIT_CODE=0` => bundle mínimo válido (`status=PASS`)
- `EXIT_CODE!=0` => faltan evidencias críticas (`status=FAIL`)

## Flujo recomendado
1. Generar `changes.patch` y `git_status.txt`.
2. Ejecutar pruebas y guardar `tests.log` con marcador (`EXIT_CODE=` o PASS/FAIL).
3. Ejecutar validador mínimo.
4. Entregar `evidence_summary.json` + `evidence_validation.log`.
