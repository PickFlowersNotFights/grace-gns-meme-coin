#!/usr/bin/env bash
# Fetch the Kokoro v1.0 neural TTS model used for the voiceover (Apache-2.0).
set -euo pipefail
DEST="${1:-$(dirname "$0")/../models}"
mkdir -p "$DEST"
BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  if [ ! -s "$DEST/$f" ]; then
    echo "fetching $f"
    curl -fSL --retry 4 -o "$DEST/$f.part" "$BASE/$f" && mv "$DEST/$f.part" "$DEST/$f"
  fi
done
echo "models ready in $DEST"
