#!/usr/bin/env bash
# luna_setup.sh — get everything luna_voice.py needs, and report what is
# missing rather than failing halfway.
#
#   cd ~/pi_assistant
#   source venv/bin/activate
#   bash luna_setup.sh
#
# Safe to re-run: every step checks before doing work.

cd "$(dirname "$0")"
PY=python3
OK="  [ok]"; BAD="  [MISSING]"
PROBLEMS=0

note() { echo; echo "── $1"; }

note "1. Python packages"
$PY - <<'EOF' 2>/dev/null || pip install numpy webrtcvad-wheels vosk
import numpy, webrtcvad, vosk
EOF
for m in numpy webrtcvad vosk; do
  if $PY -c "import $m" 2>/dev/null; then echo "$OK $m"
  else echo "$BAD $m"; PROBLEMS=$((PROBLEMS+1)); fi
done

note "2. Vosk wake-word model"
# Used for wake detection until a trained openWakeWord 'luna' model exists.
if [ -f models/vosk-small/am/final.mdl ]; then
  echo "$OK models/vosk-small"
else
  echo "  downloading (~40MB)..."
  mkdir -p models && cd models
  wget -q --show-progress https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  unzip -q vosk-model-small-en-us-0.15.zip
  rm -rf vosk-small && mv vosk-model-small-en-us-0.15 vosk-small
  rm -f vosk-model-small-en-us-0.15.zip
  cd ..
  [ -f models/vosk-small/am/final.mdl ] && echo "$OK models/vosk-small" \
    || { echo "$BAD vosk model"; PROBLEMS=$((PROBLEMS+1)); }
fi

note "3. Piper (text to speech)"
if command -v piper >/dev/null || [ -x /usr/local/bin/piper ]; then
  echo "$OK piper"
else
  echo "  installing piper for aarch64..."
  cd /tmp
  # Release asset naming changed between piper versions; try both.
  for URL in \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz" \
    "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz" ; do
    if wget -q -O piper.tgz "$URL"; then
      tar -xzf piper.tgz && break
    fi
  done
  if [ -f piper/piper ]; then
    sudo cp piper/piper /usr/local/bin/
    sudo cp piper/lib*.so* /usr/local/lib/ 2>/dev/null
    sudo cp -r piper/espeak-ng-data /usr/local/share/ 2>/dev/null
    sudo ldconfig
    echo "$OK piper installed"
  else
    echo "$BAD piper — download it manually from"
    echo "     https://github.com/rhasspy/piper/releases"
    PROBLEMS=$((PROBLEMS+1))
  fi
  cd - >/dev/null
fi

note "4. Voice model"
if ls voices/*.onnx >/dev/null 2>&1; then
  for v in voices/*.onnx; do echo "$OK $v"; done
else
  echo "  downloading en_US-amy-medium..."
  mkdir -p voices && cd voices
  B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium
  wget -q --show-progress $B/en_US-amy-medium.onnx
  wget -q $B/en_US-amy-medium.onnx.json
  cd ..
  ls voices/*.onnx >/dev/null 2>&1 && echo "$OK voice installed" \
    || { echo "$BAD voice"; PROBLEMS=$((PROBLEMS+1)); }
fi

note "5. CUDA Whisper"
$PY - <<'EOF'
try:
    import ctranslate2
    t = ctranslate2.get_supported_compute_types("cuda")
    print("  [ok] CTranslate2 CUDA:", sorted(t))
except Exception as e:
    print("  [MISSING] CTranslate2 has no CUDA support.")
    print("     Run build_ctranslate2_cuda.sh, or Whisper falls back to CPU")
    print("     silently and responses take seconds instead of ~400ms.")
EOF

note "6. ReSpeaker"
if arecord -l 2>/dev/null | grep -qi -E "arrayuac10|respeaker"; then
  CARD=$(arecord -l | grep -i -E "arrayuac10|respeaker" | head -1 \
         | sed 's/^card \([0-9]*\).*/\1/')
  echo "$OK capture on card $CARD"
  if aplay -l 2>/dev/null | grep -qi -E "arrayuac10|respeaker"; then
    echo "$OK playback available (needed later for barge-in)"
  else
    echo "  [note] no ReSpeaker playback device — barge-in unavailable"
  fi
else
  echo "$BAD ReSpeaker not detected"; PROBLEMS=$((PROBLEMS+1))
fi

note "7. luna_voice.py"
if [ -f luna_voice.py ]; then echo "$OK luna_voice.py"
else echo "$BAD luna_voice.py — copy it into $(pwd)"; PROBLEMS=$((PROBLEMS+1)); fi

echo
if [ "$PROBLEMS" = 0 ]; then
  echo "════════ ready ════════"
  echo "Stop anything holding the mic, then run it:"
  echo "    sudo systemctl stop piassistant   2>/dev/null"
  echo "    python3 luna_voice.py"
  echo
  echo 'Say "Luna", wait for "Yes?", then ask something.'
else
  echo "════════ $PROBLEMS thing(s) to fix above ════════"
fi
