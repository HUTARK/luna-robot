#!/usr/bin/env python3
"""
luna_voice_probe.py — establish the facts before writing any voice code.

    python3 luna_voice_probe.py

Answers four questions that determine the whole pipeline design:

  1. Which ReSpeaker is this, and what is its native capture format?
  2. WHICH CHANNEL is the DSP output (beamformed + AEC + noise suppressed)?
     Documentation says channel 0 for XVF-3000, but the XVF-3800 boards
     differ, and guessing wrong means feeding raw mic audio to the ASR and
     throwing away the entire reason for buying the board.
  3. Does the ReSpeaker expose a PLAYBACK device? Echo cancellation only
     works when TTS goes out its own 3.5mm jack, because the DSP needs the
     played audio as a reference. No playback device = no barge-in.
  4. Is CTranslate2 actually built with CUDA? On aarch64 pip installs a
     CPU-only build, and faster-whisper silently falls back — so an Orin
     performs no better than a Pi and nothing tells you why.

Run it in a quiet room, then again while speaking, and compare.
"""

import json
import os
import re
import subprocess
import sys
import wave

try:
    import numpy as np
except ImportError:
    sys.exit("Need numpy:  pip install numpy")

REC = "/tmp/luna_probe.wav"
SECS = 5


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"<failed: {e}>"


def hr(title):
    print(f"\n{'='*68}\n{title}\n{'='*68}")


# ===========================================================================
hr("1. Platform")
print("L4T / JetPack:", sh("cat /etc/nv_tegra_release 2>/dev/null | head -1")
      or "<not a Jetson?>")
print("Model:        ", sh("cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'"))
print("CUDA:         ", sh("nvcc --version 2>/dev/null | tail -2 | head -1")
      or "<nvcc not on PATH — check /usr/local/cuda/bin>")
print("Python:       ", sys.version.split()[0])


# ===========================================================================
hr("2. Capture devices")
arecord_l = sh("arecord -l")
print(arecord_l or "<none>")

card = None
label = None
for line in arecord_l.splitlines():
    low = line.lower()
    if not low.startswith("card "):
        continue
    if any(k in low for k in ("arrayuac10", "respeaker", "xvf", "seeed")):
        card = int(line.split(":")[0][5:].strip())
        label = line.split("[")[1].split("]")[0] if "[" in line else "?"
        break

if card is None:
    print("\n! No ReSpeaker found. Plug it in and rerun.")
    sys.exit(1)
print(f"\n-> ReSpeaker '{label}' on card {card}")

dev = f"plughw:{card},0"
print("\nHardware params (the AUTHORITATIVE format — do not trust docs):")
print(sh(f"arecord -D hw:{card},0 --dump-hw-params -d 1 /dev/null 2>&1 "
         f"| grep -E 'CHANNELS|RATE|FORMAT|ACCESS'"))

# Parse the real channel count rather than assuming 6.
raw = sh(f"arecord -D hw:{card},0 --dump-hw-params -d 1 /dev/null 2>&1")
m = re.search(r"^CHANNELS:\s*(.+)$", raw, re.M)
chans = 6
if m:
    nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if nums:
        chans = max(nums)
m = re.search(r"^RATE:\s*(.+)$", raw, re.M)
rate = 16000
if m:
    nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if nums:
        rate = nums[0]
print(f"\n-> Using channels={chans} rate={rate}")


# ===========================================================================
hr("3. Playback device (required for AEC / barge-in)")
aplay_l = sh("aplay -l")
print(aplay_l or "<none>")
has_pb = any(
    (("arrayuac10" in l.lower() or "respeaker" in l.lower() or "xvf" in l.lower())
     and l.lower().startswith("card "))
    for l in aplay_l.splitlines())
if has_pb:
    print("\n-> ReSpeaker HAS a playback device. Wire the speaker to its 3.5mm")
    print("   jack and route TTS there: the DSP then has a reference signal,")
    print("   AEC works, and the robot can be interrupted mid-sentence.")
else:
    print("\n-> No ReSpeaker playback device found. Without one the DSP has no")
    print("   echo reference, so AEC is inert and barge-in is not possible —")
    print("   you would have to gate the mic while speaking, as on Ben.")


# ===========================================================================
hr(f"4. Recording {SECS}s at {rate}Hz x {chans}ch — TALK NORMALLY NOW")
input("Press Enter, then speak continuously for 5 seconds... ")
sh(f"arecord -q -D {dev} -c {chans} -r {rate} -f S16_LE -d {SECS} {REC}", timeout=SECS+15)

if not os.path.exists(REC):
    sys.exit("Recording failed.")

with wave.open(REC) as w:
    n, ch, sw, fr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
    data = np.frombuffer(w.readframes(n), dtype=np.int16)
print(f"captured {n/fr:.1f}s, {ch} channels @ {fr}Hz")

sig = data.reshape(-1, ch).astype(np.float64)


def db(v):
    return -99.0 if v <= 0 else 20 * np.log10(v / 32768.0)


hr("5. Per-channel analysis")
print(f"{'CH':<4}{'RMS dBFS':>10}{'PEAK dBFS':>11}{'noise floor':>13}"
      f"{'crest':>8}   interpretation")
stats = []
for c in range(ch):
    x = sig[:, c]
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    # Quietest 10% of 20ms blocks approximates the noise floor. The DSP
    # output has a MUCH lower floor than the raw mics because noise
    # suppression is applied to it — that is the giveaway.
    blk = int(0.02 * fr)
    nb = len(x) // blk
    if nb:
        blocks = x[:nb * blk].reshape(nb, blk)
        floors = np.sqrt(np.mean(blocks ** 2, axis=1))
        floor = np.percentile(floors, 10)
    else:
        floor = rms
    crest = db(peak) - db(rms) if rms > 0 else 0
    stats.append((c, db(rms), db(peak), db(floor), crest))

SILENT_DB = -70.0
floors = [s[3] for s in stats]
rmss = [s[1] for s in stats]
# Only channels carrying signal can be the DSP output. A silent playback
# reference has the lowest noise floor of ALL channels, so it must be
# excluded first or it wins outright and gets misidentified.
active = [s for s in stats if s[1] > SILENT_DB]
silent = [s[0] for s in stats if s[1] <= SILENT_DB]
dsp_ch = min(active, key=lambda s: s[3])[0] if active else 0

for c, r, p, f, cr in stats:
    if c in silent:
        note = "silent — playback reference (nothing playing)"
    elif c == dsp_ch and len(active) > 1:
        note = "<== lowest noise floor of active channels: DSP OUTPUT"
    else:
        note = "raw mic (higher floor, unprocessed)"
    print(f"{c:<4}{r:>10.1f}{p:>11.1f}{f:>13.1f}{cr:>8.1f}   {note}")

hr("6. Conclusions")
print(f"* Feed channel {dsp_ch} to the wake word and ASR.")
print("  Confirm by ear — extract each channel and listen:")
for c in range(ch):
    print(f"      sox {REC} -c 1 /tmp/ch{c}.wav remix {c+1} && aplay /tmp/ch{c}.wav")
print("  The DSP channel sounds noticeably cleaner and slightly 'tighter'.")
if silent:
    print(f"* Channel(s) {silent} are near-silent: that is the playback")
    print("  reference. Rerun this while audio is playing and it should light")
    print("  up — if it stays silent, AEC has no reference and barge-in will")
    print("  not work.")
if max(rmss) > -3:
    print("* CLIPPING: reduce mic gain (see the usb_4_mic_array tuning tool).")
elif max(rmss) < -30:
    print("* Very quiet. Raise gain, or ASR accuracy will suffer regardless")
    print("  of which model you choose.")


# ===========================================================================
hr("7. Is CTranslate2 actually CUDA-enabled?")
try:
    import ctranslate2
    print("ctranslate2:", ctranslate2.__version__)
    types = ctranslate2.get_supported_compute_types("cuda")
    print("cuda compute types:", types)
    if types:
        print("\n-> CUDA build present. Use:")
        print('   WhisperModel("small.en", device="cuda", compute_type="float16")')
    else:
        print("\n-> CPU-ONLY BUILD. device='cuda' will silently fall back to CPU")
        print("   and your Orin will be no faster than the Pi. Fix with")
        print("   jetson-containers, or build CTranslate2 with -DWITH_CUDA=ON.")
except ImportError:
    print("ctranslate2 not installed (faster-whisper pulls it in)")
except Exception as e:
    print(f"could not query CUDA support: {e}")
    print("-> Usually means no CUDA in this build.")

print("\nDone. Paste this whole output back.")
