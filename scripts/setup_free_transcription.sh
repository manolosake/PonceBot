#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"

BIN_DIR="$ROOT/bin"
VENDOR_DIR="$ROOT/vendor"
MODELS_DIR="$ROOT/models"

mkdir -p "$BIN_DIR" "$VENDOR_DIR" "$MODELS_DIR"

echo "[1/3] Descargando ffmpeg (static)..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
curl -fsSL "$FFMPEG_URL" -o "$TMP/ffmpeg.tar.xz"
tar -C "$TMP" -xf "$TMP/ffmpeg.tar.xz"

FFMPEG_DIR="$(find "$TMP" -maxdepth 1 -type d -name 'ffmpeg-*-amd64-static' | head -n 1)"
if [[ -z "${FFMPEG_DIR:-}" ]]; then
  echo "No pude encontrar el directorio extraido de ffmpeg." >&2
  exit 1
fi

cp -f "$FFMPEG_DIR/ffmpeg" "$BIN_DIR/ffmpeg"
cp -f "$FFMPEG_DIR/ffprobe" "$BIN_DIR/ffprobe" || true
chmod +x "$BIN_DIR/ffmpeg" "$BIN_DIR/ffprobe" || true

echo "[2/3] Clonando/build de whisper.cpp..."
WHISPER_DIR="$VENDOR_DIR/whisper.cpp"
if [[ ! -d "$WHISPER_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
else
  (cd "$WHISPER_DIR" && git fetch --depth 1 origin && git reset --hard origin/master) >/dev/null 2>&1 || true
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake no encontrado; instalando en un venv local..."
  VENV_DIR="$VENDOR_DIR/.venv-cmake"
  if [[ ! -x "$VENV_DIR/bin/cmake" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q --upgrade pip
    "$VENV_DIR/bin/pip" install -q cmake
  fi
  export PATH="$VENV_DIR/bin:$PATH"
fi

(cd "$WHISPER_DIR" && make -j"$(nproc)")

if [[ -x "$WHISPER_DIR/build/bin/whisper-cli" ]]; then
  cp -f "$WHISPER_DIR/build/bin/whisper-cli" "$BIN_DIR/whisper-cli"
elif [[ -x "$WHISPER_DIR/build/bin/main" ]]; then
  cp -f "$WHISPER_DIR/build/bin/main" "$BIN_DIR/main"
else
  echo "No encontre whisper-cli ni main en whisper.cpp" >&2
  exit 1
fi

chmod +x "$BIN_DIR/whisper-cli" "$BIN_DIR/main" 2>/dev/null || true

echo "[3/3] Descargando modelo ggml (medium)..."
(cd "$WHISPER_DIR" && bash ./models/download-ggml-model.sh medium)
cp -f "$WHISPER_DIR/models/ggml-medium.bin" "$MODELS_DIR/ggml-medium.bin"

echo ""
echo "OK. Instalacion local lista:"
echo "- ffmpeg:       $BIN_DIR/ffmpeg"
echo "- whisper.cpp:  $BIN_DIR/whisper-cli (o $BIN_DIR/main)"
echo "- modelo:       $MODELS_DIR/ggml-medium.bin"
echo ""
echo "Config sugerida en codexbot.env:"
echo "  BOT_TRANSCRIBE_AUDIO=1"
echo "  BOT_TRANSCRIBE_BACKEND=whispercpp"
echo "  BOT_TRANSCRIBE_FFMPEG_BIN=$BIN_DIR/ffmpeg"
echo "  BOT_TRANSCRIBE_WHISPERCPP_BIN=$BIN_DIR/whisper-cli"
echo "  BOT_TRANSCRIBE_WHISPERCPP_MODEL_PATH=$MODELS_DIR/ggml-medium.bin"
echo "  BOT_TRANSCRIBE_WHISPERCPP_THREADS=8"
