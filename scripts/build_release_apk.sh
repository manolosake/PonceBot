#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${MOBILE_APP_PROJECT_DIR:-/home/aponce/OmniCrewApp.android}"
ARTIFACTS_DIR="${MOBILE_APP_ARTIFACTS_DIR:-/home/aponce/codexbot/data/artifacts/mobile}"
KEYSTORE_PATH="${MOBILE_APP_KEYSTORE_PATH:-/home/aponce/.config/omnicrew-app/keystore.jks}"
KEY_ALIAS="${MOBILE_APP_KEY_ALIAS:-omnicrew_release}"
KEYSTORE_PASS="${MOBILE_APP_KEYSTORE_PASS:-}"
KEY_PASS="${MOBILE_APP_KEY_PASS:-}"
ANDROID_DOCKER_IMAGE="${ANDROID_DOCKER_IMAGE:-ghcr.io/cirruslabs/android-sdk:34}"

if [[ -z "$KEYSTORE_PASS" || -z "$KEY_PASS" ]]; then
  echo "missing keystore passwords: MOBILE_APP_KEYSTORE_PASS/MOBILE_APP_KEY_PASS" >&2
  exit 2
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "project directory not found: $PROJECT_DIR" >&2
  exit 2
fi

mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$(dirname "$KEYSTORE_PATH")"

VERSION="$(grep -E "^[[:space:]]*versionName[[:space:]]*=" "$PROJECT_DIR/app/build.gradle.kts" | head -n1 | sed -E "s/.*=\s*\"([^\"]+)\".*/\1/")"
if [[ -z "$VERSION" ]]; then
  VERSION="1.0.0"
fi

TS_UTC="$(date -u +%Y%m%d-%H%M%S)"
BUILD_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'nogit')"

echo "[apk] queued"
echo "[apk] building image=$ANDROID_DOCKER_IMAGE"

docker run --rm \
  -e MOBILE_APP_KEYSTORE_PATH="/keystore/$(basename "$KEYSTORE_PATH")" \
  -e MOBILE_APP_KEYSTORE_PASS="$KEYSTORE_PASS" \
  -e MOBILE_APP_KEY_ALIAS="$KEY_ALIAS" \
  -e MOBILE_APP_KEY_PASS="$KEY_PASS" \
  -v "$PROJECT_DIR:/workspace" \
  -v "$(dirname "$KEYSTORE_PATH"):/keystore" \
  -w /workspace \
  "$ANDROID_DOCKER_IMAGE" \
  bash -lc '
    set -euo pipefail
    if [[ ! -f "$MOBILE_APP_KEYSTORE_PATH" ]]; then
      echo "[apk] generating keystore"
      keytool -genkeypair -v \
        -keystore "$MOBILE_APP_KEYSTORE_PATH" \
        -alias "$MOBILE_APP_KEY_ALIAS" \
        -storepass "$MOBILE_APP_KEYSTORE_PASS" \
        -keypass "$MOBILE_APP_KEY_PASS" \
        -keyalg RSA -keysize 4096 -validity 3650 \
        -dname "CN=OmniCrew, OU=Engineering, O=OmniCrew, L=CDMX, ST=CDMX, C=MX"
    fi

    chmod +x ./gradlew
    export GRADLE_USER_HOME="/tmp/gradle-home-$RANDOM-$RANDOM"
    mkdir -p "$GRADLE_USER_HOME"
    trap "rm -rf \"$GRADLE_USER_HOME\"" EXIT
    ./gradlew --no-daemon clean testDebugUnitTest assembleRelease
  '

APK_SRC="$PROJECT_DIR/app/build/outputs/apk/release/app-release.apk"
if [[ ! -f "$APK_SRC" ]]; then
  echo "release apk not found at $APK_SRC" >&2
  exit 3
fi

APK_OUT="$ARTIFACTS_DIR/omnicrewapp-${VERSION}-${TS_UTC}-universal-release.apk"
cp -f "$APK_SRC" "$APK_OUT"

SHA_FILE="$APK_OUT.sha256"
SHA256="$(sha256sum "$APK_OUT" | awk '{print $1}')"
printf "%s  %s\n" "$SHA256" "$(basename "$APK_OUT")" > "$SHA_FILE"

REPORT_PATH="$ARTIFACTS_DIR/build_report.json"
python3 - <<PY2
import json
from pathlib import Path
report = {
    "ok": True,
    "version": "${VERSION}",
    "git_sha": "${GIT_SHA}",
    "built_at": "${BUILD_ISO}",
    "artifact_path": "${APK_OUT}",
    "artifact_name": Path("${APK_OUT}").name,
    "sha256": "${SHA256}",
    "sha256_file": "${SHA_FILE}",
}
Path("${REPORT_PATH}").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY2

echo "[apk] done"
echo "APK_PATH=${APK_OUT}"
echo "SHA256=${SHA256}"
echo "REPORT_PATH=${REPORT_PATH}"
