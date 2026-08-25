"""
audio_utils.py — compatibility shim.

All real audio work now lives in luna_voice.py: continuous ReSpeaker capture,
VAD endpointing, CUDA Whisper, and streaming interruptible TTS. This module
exists only so existing callers (server.py, display_addon.py, main.py) keep
working without edits.

WHY THIS FILE IS A SHIM RATHER THAN AN IMPLEMENTATION

  ONE lock, ONE device. The previous speak() ran its own piper|sox|aplay
  pipeline with no lock at all. With luna_voice also driving the speaker, two
  independent paths would fight over one device and produce intermittent
  "Device or resource busy". Delegating means there is a single owner.

THREE BUGS THIS REPLACES

  1. Voice models were hardcoded to /home/ben/pi_assistant/voices/... on a
     machine whose home is /home/groot. Piper was handed a nonexistent model
     on every call, so TTS silently never worked.

  2. get_speaker_device() did not exist, yet main.py's do_shutdown() imports
     it. The ImportError was swallowed by a bare except, so shutdown.wav never
     played.

  3. speak() built its command as  echo "{text}" | piper ...  under
     shell=True. Any answer containing a double quote, backtick or $ would
     break the command or execute part of it — and GPT answers contain quotes
     routinely. luna_voice writes the text to piper's stdin instead, so no
     shell quoting is involved.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kept so old imports resolve; nothing writes to these any more.
RECORD_PATH = "/tmp/recorded.wav"
RAW_PATH = "/tmp/raw_recorded.wav"


# ── TTS ────────────────────────────────────────────────────────────────────

def speak(text: str):
    """Speak via the single TTS path in luna_voice."""
    try:
        from luna_voice import speak as _speak
        return _speak(text)
    except Exception as e:
        print(f"[TTS] shim failed: {e}")
        return False


def interrupt():
    """Stop speech immediately (e-stop, barge-in, shutdown)."""
    try:
        from luna_voice import interrupt as _interrupt
        _interrupt()
    except Exception:
        pass


# ── Device lookup ──────────────────────────────────────────────────────────

def get_speaker_device():
    """ALSA device string for playback, e.g. 'plughw:0,0'.

    Prefers the ReSpeaker's own 3.5mm jack, because the XVF-3000 can only
    cancel echo for audio it knows it played — routing TTS anywhere else means
    no AEC and therefore no barge-in.
    """
    try:
        from luna_voice import find_speaker
        return find_speaker()
    except Exception:
        return "default"


def get_card_number(keyword, mode='capture'):
    cmd = ['arecord', '-l'] if mode == 'capture' else ['aplay', '-l']
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        low = line.lower()
        if keyword.lower() in low and low.startswith('card '):
            try:
                return int(line.split(':')[0][5:].strip())
            except ValueError:
                continue
    return None


def get_mic_card():
    """Legacy helper. The ReSpeaker is found by name, not by number, because
    USB card indices shuffle between boots — it moved from card 2 to card 0
    across a single reboot on this machine."""
    for kw in ('arrayuac10', 'respeaker'):
        c = get_card_number(kw, 'capture')
        if c is not None:
            return c
    return get_card_number('usb', 'capture')


def get_speaker_card():
    for kw in ('arrayuac10', 'respeaker', 'usb pnp', 'usb audio'):
        c = get_card_number(kw, 'playback')
        if c is not None:
            return c
    return None


# ── Recording (retired) ────────────────────────────────────────────────────
# luna_voice keeps one continuous stream open and endpoints with VAD, so
# nothing records fixed-length clips any more. These stubs stay only so a
# stale import cannot crash the app; if one of them is ever called, the caller
# is on an old code path and should be updated.

def record_audio(duration=3):
    print("[MIC] record_audio() is retired — luna_voice streams continuously "
          "and endpoints with VAD. Caller needs updating.")
    return None


def record_question():
    print("[MIC] record_question() is retired — see record_audio().")
    return None
