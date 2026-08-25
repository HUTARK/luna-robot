#!/usr/bin/env python3
"""
luna_stt_bench.py — pick the Whisper model for Luna, measured on this Orin
with this ReSpeaker.

    python3 luna_stt_bench.py record     record clips through ch0
    python3 luna_stt_bench.py bench      every clip x every config, with WER

Records from the ReSpeaker's DSP channel (channel 0, confirmed by
luna_voice_probe.py) at its native 16 kHz, so what is measured is what the
pipeline will actually receive.

Run after build_ctranslate2_cuda.sh. If CUDA is unavailable the script says so
and benchmarks CPU only, rather than quietly reporting CPU numbers as if they
were GPU ones.
"""

import json
import os
import re
import subprocess
import sys
import time
import wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE, "luna_stt_tests")
REFS = os.path.join(TEST_DIR, "refs.json")
RATE = 16000
CHANS = 6
DSP_CHANNEL = 0          # confirmed: 7dB lower noise floor than the raw mics
DURATION = 6

PROMPTS = [
    "Luna, what is your name?",
    "Raise your right arm slowly.",
    "Tell me about Tamil Nadu.",
    "Close your left hand and then open it.",
    "Who invented the telephone?",
    "Turn your head to the left and look up.",
    "My name is Arjun and I am from Chennai.",
    "Move the elbow to ninety degrees.",
    "What is the weather in Bengaluru today?",
    "Point at me with your index finger.",
]

# Domain vocabulary biases decoding toward the words this robot expects.
# Cheapest accuracy win available, and it matters most for joint names.
INITIAL_PROMPT = (
    "Luna is an assistant robot with a head, shoulders, elbows, wrists and "
    "fingers. Commands include raise, lower, rotate, open, close, point, "
    "turn head left, turn head right, look up, look down, shoulder, forearm, "
    "elbow, wrist, lateral, index finger, thumb. Names: Luna, Arjun, "
    "Chennai, Bengaluru, Tamil Nadu."
)

COMMON = dict(language="en", beam_size=5, condition_on_previous_text=False)


def cuda_ok():
    try:
        import ctranslate2
        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False


def configs():
    """CPU baselines plus, if available, the GPU options worth considering."""
    c = [
        ("tiny.en  cpu int8",  "tiny.en",  "cpu",  "int8",    {}),
        ("base.en  cpu int8",  "base.en",  "cpu",  "int8",    {}),
        ("small.en cpu int8",  "small.en", "cpu",  "int8",    {}),
    ]
    if cuda_ok():
        c += [
            ("base.en  cuda fp16",  "base.en",  "cuda", "float16", {}),
            ("small.en cuda fp16",  "small.en", "cuda", "float16", {}),
            ("small.en cuda +prompt", "small.en", "cuda", "float16",
             {"initial_prompt": INITIAL_PROMPT}),
            ("small.en cuda +prompt +vad", "small.en", "cuda", "float16",
             {"initial_prompt": INITIAL_PROMPT, "vad_filter": True}),
            ("distil-small.en cuda", "distil-small.en", "cuda", "float16",
             {"initial_prompt": INITIAL_PROMPT}),
            ("medium.en cuda fp16", "medium.en", "cuda", "float16",
             {"initial_prompt": INITIAL_PROMPT}),
        ]
    else:
        print("\n!! CTranslate2 has no CUDA support — CPU configs only.")
        print("   Run build_ctranslate2_cuda.sh first, or these numbers will")
        print("   understate what the Orin can do by roughly an order of")
        print("   magnitude.\n")
    return c


# ===========================================================================

def find_respeaker():
    out = subprocess.run(["arecord", "-l"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("card ") and ("arrayuac10" in low or "respeaker" in low):
            return f"plughw:{int(line.split(':')[0][5:].strip())},0"
    return None


def record_ch0(path, secs):
    """Capture all 6 channels, keep only the DSP channel. Recording 1 channel
    directly would make ALSA downmix all six — blending the processed output
    with the raw mics and the playback reference, which defeats the board."""
    dev = find_respeaker()
    if not dev:
        sys.exit("ReSpeaker not found.")
    raw = "/tmp/luna_6ch.wav"
    subprocess.run(["arecord", "-q", "-D", dev, "-c", str(CHANS),
                    "-r", str(RATE), "-f", "S16_LE", "-d", str(secs), raw],
                   check=False)
    with wave.open(raw) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    ch0 = data.reshape(-1, CHANS)[:, DSP_CHANNEL].copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(ch0.tobytes())
    peak = int(np.max(np.abs(ch0))) if ch0.size else 0
    return 20 * np.log10(max(peak, 1) / 32768.0)


def cmd_record():
    os.makedirs(TEST_DIR, exist_ok=True)
    refs = {}
    print(f"\n{len(PROMPTS)} clips, {DURATION}s each, from ReSpeaker channel "
          f"{DSP_CHANNEL}.\nSpeak naturally at your normal distance — "
          f"over-enunciating tunes for speech you won't use.\n")
    for i, prompt in enumerate(PROMPTS):
        name = f"clip{i:02d}.wav"
        input(f"[{i+1}/{len(PROMPTS)}] Enter, then say:  \"{prompt}\"  ")
        print("      recording... ", end="", flush=True)
        pk = record_ch0(os.path.join(TEST_DIR, name), DURATION)
        warn = "  ** very quiet" if pk < -25 else ("  ** clipping" if pk > -1 else "")
        print(f"peak {pk:.1f} dBFS{warn}")
        refs[name] = prompt
    with open(REFS, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"\nSaved to {TEST_DIR}/. Now: python3 luna_stt_bench.py bench")


# ===========================================================================

def normalise(s):
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def cmd_bench():
    from faster_whisper import WhisperModel
    if not os.path.isdir(TEST_DIR):
        sys.exit("No clips. Run: python3 luna_stt_bench.py record")
    files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".wav"))
    refs = json.load(open(REFS)) if os.path.isfile(REFS) else {}
    rows = []

    for label, size, device, ctype, extra in configs():
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        try:
            t0 = time.monotonic()
            model = WhisperModel(size, device=device, compute_type=ctype,
                                 cpu_threads=6)
            print(f"  loaded in {time.monotonic()-t0:.1f}s")
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        kwargs = dict(COMMON, **extra)
        # First call pays graph-build cost; burn it before timing.
        try:
            model.transcribe(np.zeros(RATE, dtype=np.float32) + 0.001,
                             language="en", beam_size=1)
        except Exception:
            pass

        times, wers = [], []
        for name in files:
            path = os.path.join(TEST_DIR, name)
            t0 = time.monotonic()
            try:
                segs, _ = model.transcribe(path, **kwargs)
                text = " ".join(s.text for s in segs).strip()
            except Exception as e:
                text = f"<error: {e}>"
            ms = (time.monotonic() - t0) * 1000
            times.append(ms)
            mark = ""
            if name in refs:
                w = wer(refs[name], text)
                wers.append(w)
                mark = f"  [WER {w*100:.0f}%]"
            print(f"  {name}  {ms:6.0f}ms  {text!r}{mark}")
        if times:
            rows.append((np.mean(wers) if wers else None,
                         label, np.mean(times), max(times)))
        del model

    if rows:
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(f"{'CONFIG':<30}{'MEAN WER':>10}{'MEAN ms':>10}{'MAX ms':>9}")
        for w, label, mt, mx in rows:
            ws = f"{w*100:9.1f}%" if w is not None else "        -"
            print(f"{label:<30}{ws}{mt:10.0f}{mx:9.0f}")
        scored = [r for r in rows if r[0] is not None]
        if scored:
            best = min(scored)
            print(f"\nMost accurate: {best[1]}  "
                  f"({best[0]*100:.1f}% WER, {best[2]:.0f}ms mean)")
        print("\nFor barge-in to feel instant you want MEAN ms under ~400 on a")
        print("3-4s utterance. Pick the most accurate config that clears that,")
        print("and remember vision will be competing for the same GPU later.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"record": cmd_record, "bench": cmd_bench}.get(
        cmd, lambda: print(__doc__))()
