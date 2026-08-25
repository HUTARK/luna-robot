"""
luna_voice.py — low-latency voice pipeline for Luna.
Jetson Orin Nano Super + ReSpeaker USB 4 Mic Array (XVF-3000).

Measured on this hardware (luna_stt_bench.py):
    small.en / cuda / float16 / +prompt   ->  ~0% real WER, 872ms on 6s clips
                                              (~0.14x realtime, so a 3s
                                               utterance lands near 400ms)
    medium.en                             ->  OOM on 8GB, do not use
    distil-small.en                       ->  71% WER, truncates, do not use
    cpu int8                              ->  unsupported by this CT2 build
                                              (needs -DWITH_RUY=ON to enable)

Hardware facts confirmed by luna_voice_probe.py:
    ReSpeaker on card 2, 6 channels @ 16 kHz, S16_LE
    CHANNEL 0 = DSP output (beamformed + AEC + noise suppressed).
      Its noise floor measured -47.6 dBFS vs ~-41 for the raw mics.
    Channel 5 = playback reference (silent when nothing is playing)
    The board exposes a PLAYBACK device on the same card.

BARGE-IN REQUIRES THE SPEAKER ON THE RESPEAKER'S 3.5mm JACK.
The XVF-3000 cancels echo by subtracting what it knows is being played. Route
TTS anywhere else and the DSP has no reference, channel 0 still contains the
robot's own voice, and wake detection during speech will trigger on itself.
Set ALLOW_BARGE_IN = False until the cable is moved.

Dependencies:
    pip install numpy webrtcvad-wheels vosk
    # faster-whisper needs the CUDA CTranslate2 built by
    # build_ctranslate2_cuda.sh — the PyPI aarch64 wheel is CPU-only and
    # falls back SILENTLY.
"""

import collections
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Capture (dictated by the XVF-3000; verified, do not guess) ─────────────
RATE          = 16000
CHANNELS      = 6
DSP_CHANNEL   = 0
FRAME_SAMPLES = 320                 # 20 ms — the only sizes webrtcvad accepts
FRAME_BYTES   = FRAME_SAMPLES * CHANNELS * 2
OWW_CHUNK     = 1280                # 80 ms — openWakeWord's window

MIC_KEYWORDS = ('arrayuac10', 'respeaker')

# ── Tuning ────────────────────────────────────────────────────────────────
ALLOW_BARGE_IN   = False    # True ONLY after the speaker is on the ReSpeaker jack
PREROLL_SEC      = 0.6      # audio retained from before the wake word fired
VAD_AGGRESSION   = 3        # 0-3, higher trims more aggressively
# A frame must be this many dB above the measured room floor to count as
# speech. Without it, AGC-boosted room noise reads as speech and utterances
# never end. Raise if capture runs long; lower if quiet speech gets ignored.
SPEECH_MARGIN_DB = 9.0
SPEECH_TIMEOUT   = 2.0      # give up if nobody speaks after the wake word
SILENCE_HANG_MS  = 650      # trailing silence that ends an utterance
MAX_UTTERANCE_S  = 15.0
MIN_SPEECH_SEC   = 0.4      # shorter than this, anything returned is invention
WAKE_COOLDOWN    = 1.5

# ── Follow-up conversation ────────────────────────────────────────────────
# After answering, listen once more WITHOUT requiring the wake word, so a
# conversation flows naturally. This is what Alexa's Follow-Up Mode and Siri's
# continuous conversation do, and it is deliberately NOT a long open window:
#
# A 30-second "listen to everything" window on a robot with arms is dangerous.
# It transcribes every side conversation in the room and tries to act on it —
# ordinary speech has already produced things like "take the apple and put it
# on the elbow" here. On a Q&A-only robot that is a wrong answer; on Luna it is
# a shoulder moving when nobody asked. It also leaves the user unable to tell
# whether they are being listened to.
#
# So: bounded turns, bounded total time, and it sleeps the moment nobody
# answers.
FOLLOWUP_ENABLED    = True
FOLLOWUP_WINDOW_SEC = 6.0   # how long to wait for the person to start speaking
FOLLOWUP_MAX_TURNS  = 6     # cap on turns per wake, so it cannot stay awake
FOLLOWUP_MAX_SEC    = 90.0  # cap on total conversation length

# ── Thinking fillers ──────────────────────────────────────────────────────
# Spoken only on the SLOW path, while the network call is in flight, so the
# pause is not dead air. A single fixed phrase gets grating fast — hearing
# "let me think" identically on every question is one of the things that makes
# an assistant feel mechanical. Keep them SHORT: the filler has to finish
# before the real answer arrives or it delays what the person actually asked
# for. Add your own; they are just strings.
THINKING_FILLERS = [
    "Let me think.",
    "One moment.",
    "Good question.",
    "Let me check.",
    "Hmm, let me see.",
    "Just a second.",
    "Looking that up.",
    "Let me find out.",
]
# Don't repeat the previous one — with 8 phrases, random choice alone still
# says the same thing twice in a row about one time in eight, which is exactly
# when it is most noticeable.
_last_filler = None


def thinking_filler():
    global _last_filler
    if len(THINKING_FILLERS) < 2:
        return THINKING_FILLERS[0] if THINKING_FILLERS else ""
    choices = [f for f in THINKING_FILLERS if f != _last_filler]
    _last_filler = random.choice(choices)
    return _last_filler

# ── STT ───────────────────────────────────────────────────────────────────
# Ordered fallback chain. The Orin Nano shares 8GB between CPU and GPU, so
# whether small.en fits depends on what else is loaded — the display, Vosk,
# and any vision model all compete for the same pool. Benchmarking standalone
# and then running alongside everything else is how you get an OOM at the
# worst possible moment, so try each in turn and settle for what fits.
#
# int8_float16 roughly halves weight memory at a small accuracy cost.
# base.en measured 5.1% WER vs small.en's 3.1%, and most of that gap was
# metric artifacts ("90" vs "ninety"), so base.en is a mild downgrade, not a
# broken one.
# Where Whisper weights live. Deliberately inside the project rather than
# ~/.cache/huggingface: a systemd service runs with a different HOME, so the
# user's cache is invisible to it and every load fails with "Cannot find an
# appropriate cached snapshot folder" — which looks like a model problem but is
# really a path problem. A project-local directory works for both, and means a
# manufactured unit ships with its models rather than downloading on first boot.
WHISPER_CACHE = os.path.join(BASE_DIR, "models", "whisper")

STT_CHAIN = [
    (os.environ.get("LUNA_STT_MODEL", "small.en"),
     os.environ.get("LUNA_STT_CTYPE", "float16"), "cuda"),
    ("small.en", "int8_float16", "cuda"),
    ("base.en",  "float16",      "cuda"),
    ("base.en",  "int8_float16", "cuda"),
    ("base.en",  "float32",      "cpu"),      # last resort: slow but works
]

# Biases decoding toward Luna's vocabulary. This is why joint names come back
# reliably; keep it short, a long prompt starts leaking into hallucinations.
STT_PROMPT = (
    "Luna is an assistant robot with a head, shoulders, elbows, wrists and "
    "fingers. Commands include raise, lower, rotate, open, close, point, "
    "turn head left, turn head right, look up, look down, shoulder, forearm, "
    "elbow, wrist, lateral, index finger, thumb. Names: Luna, Arjun, "
    "Chennai, Bengaluru, Tamil Nadu."
)

# ── TTS ───────────────────────────────────────────────────────────────────
PIPER_BIN   = shutil.which("piper") or "/usr/local/bin/piper"
VOICES_DIR  = os.path.join(BASE_DIR, "voices")
PIPER_VOICE = os.path.join(VOICES_DIR, "en_GB-jenny_dioco-medium.onnx")
PIPER_FLAGS = ['--length_scale', '0.95', '--sentence_silence', '0.15']

# ── Wake word ─────────────────────────────────────────────────────────────
# Two backends. openWakeWord is a discriminatively trained model: lower CPU
# and far fewer false triggers, but needs a model per phrase. Since Luna is a
# fixed product name, training one is the right long-term answer (Seeed/oww
# Colab notebook, ~1h from synthetic samples, no recordings needed).
# Until that exists, Vosk with open recognition accepts any name at runtime.
OWW_MODEL      = os.path.join(BASE_DIR, "wake_models", "luna.onnx")
VOSK_MODEL_DIR = os.path.join(BASE_DIR, "models", "vosk-small")
WAKE_NAME      = "luna"


# ==========================================================================
# Devices
# ==========================================================================

def _find_card(listing_cmd):
    try:
        out = subprocess.run(listing_cmd, capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        low = line.lower()
        if low.startswith('card ') and any(k in low for k in MIC_KEYWORDS):
            try:
                return int(line.split(':')[0][5:].strip())
            except ValueError:
                continue
    return None


def find_mic():
    c = _find_card(['arecord', '-l'])
    return f"plughw:{c},0" if c is not None else None


def find_speaker():
    """The ReSpeaker's own output. Using it is what makes AEC work.
    plughw handles the rate conversion from Piper's 22050."""
    c = _find_card(['aplay', '-l'])
    if c is not None:
        return f"plughw:{c},0"
    print("[SPK] ReSpeaker playback not found — AEC will not work")
    return "default"


# ==========================================================================
# Continuous capture
# ==========================================================================

class MicStream:
    """One long-lived arecord process yielding 20 ms frames of channel 0.

    Reads all 6 channels and deinterleaves. Asking arecord for -c 1 would
    make ALSA DOWNMIX all six, blending the DSP output with the four raw mics
    and the playback reference — which discards the entire benefit of the
    board and reintroduces the robot's own voice into the signal.
    """

    def __init__(self, device=None):
        self.device = device
        self._q = queue.Queue(maxsize=250)
        self.preroll = collections.deque(
            maxlen=int(PREROLL_SEC * RATE / FRAME_SAMPLES))
        self._proc = None
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._supervise, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self._kill()

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _spawn(self):
        dev = self.device or find_mic()
        if not dev:
            print("[MIC] ReSpeaker not found")
            return False
        # A previous run killed with Ctrl-C can leave its arecord holding the
        # device, which makes the new one exit immediately and shows up as
        # repeated "stream died, restarting".
        subprocess.run(['pkill', '-f', f'arecord -q -D {dev}'],
                       capture_output=True)
        time.sleep(0.2)
        try:
            self._proc = subprocess.Popen(
                ['arecord', '-q', '-D', dev, '-c', str(CHANNELS),
                 '-r', str(RATE), '-f', 'S16_LE', '-t', 'raw', '-'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            print(f"[MIC] streaming {dev} {CHANNELS}ch@{RATE} -> ch{DSP_CHANNEL}")
            return True
        except Exception as e:
            print(f"[MIC] spawn failed: {e}")
            return False

    def _supervise(self):
        while not self._stop.is_set():
            if not self._spawn():
                time.sleep(2)
                self.device = None
                continue
            self._pump()
            self._kill()
            if not self._stop.is_set():
                print("[MIC] stream died, restarting")
                self.device = None
                time.sleep(1)

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _pump(self):
        while not self._stop.is_set():
            raw = self._read_exact(FRAME_BYTES)
            if raw is None:
                return
            frame = np.frombuffer(raw, dtype=np.int16) \
                      .reshape(-1, CHANNELS)[:, DSP_CHANNEL].copy()
            self.preroll.append(frame)
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                # Consumer is behind (STT/LLM blocking). Drop the oldest so we
                # stay on live audio instead of building a backlog.
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(frame)
                except queue.Empty:
                    pass

    def read(self, timeout=1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def snapshot_preroll(self):
        f = list(self.preroll)
        return np.concatenate(f) if f else np.empty(0, dtype=np.int16)


# ==========================================================================
# Wake word
# ==========================================================================

class OpenWakeWord:
    def __init__(self, model_path=OWW_MODEL, threshold=0.5):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=[model_path],
                           inference_framework="onnx")
        self.threshold = threshold
        self._buf = np.empty(0, dtype=np.int16)
        self._last = 0.0

    def feed(self, frame):
        if time.monotonic() - self._last < WAKE_COOLDOWN:
            return None
        self._buf = np.concatenate([self._buf, frame])
        hit = None
        while len(self._buf) >= OWW_CHUNK:
            chunk, self._buf = self._buf[:OWW_CHUNK], self._buf[OWW_CHUNK:]
            for name, score in self.model.predict(chunk).items():
                if score > self.threshold:
                    hit = (name, float(score))
        if hit:
            self._last = time.monotonic()
        return hit

    def reset(self):
        self._buf = np.empty(0, dtype=np.int16)
        try:
            self.model.reset()
        except AttributeError:
            pass


class VoskWakeWord:
    """Open recognition, then a word-boundary search for the name.

    Deliberately NOT a restricted grammar: a grammar forces every sound onto
    an allowed path, so short names collapse onto acoustically similar common
    words. Open decoding lets the recogniser output "done" as done.
    """

    # Reset if nothing matches for this long. Left alone, the partial
    # hypothesis grows without bound — it costs memory, slows decoding, and
    # produced triggers like "that is a lot of apple's their luna" where the
    # name was buried in a minute of accumulated hypothesis.
    IDLE_RESET_SEC = 12.0
    # Only look at the tail of the partial, i.e. what was just said, rather
    # than anywhere in the whole accumulated phrase.
    TAIL_WORDS = 4

    def __init__(self, name=WAKE_NAME, model_dir=VOSK_MODEL_DIR,
                 require_prefix=False):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        self._model = Model(model_dir)
        self._rec = KaldiRecognizer(self._model, RATE)
        self._pending = 0
        self._last = 0.0
        self._last_partial_change = time.monotonic()
        self._prev_partial = ''
        n = re.escape(name.lower())
        # Word boundaries matter: a plain substring test on "luna" also fires
        # on "lunar", and on "ben" it fires on "bench" and "Bengaluru".
        pre = r'(?:hey\s+|ok\s+|okay\s+)' + ('' if require_prefix else '?')
        self._re = re.compile(rf'\b{pre}{n}\b')

    def feed(self, frame):
        if time.monotonic() - self._last < WAKE_COOLDOWN:
            return None
        self._rec.AcceptWaveform(frame.tobytes())
        self._pending += 1
        if self._pending < 4:          # check every 80 ms, not every 20
            return None
        self._pending = 0

        partial = json.loads(self._rec.PartialResult()).get('partial', '')
        now = time.monotonic()
        if partial != self._prev_partial:
            self._prev_partial = partial
            self._last_partial_change = now
        elif partial and now - self._last_partial_change > self.IDLE_RESET_SEC:
            # Stale hypothesis sitting around: clear it.
            self.reset()
            return None

        if not partial:
            return None
        tail = ' '.join(partial.lower().split()[-self.TAIL_WORDS:])
        if self._re.search(tail):
            self._last = now
            return (tail, 1.0)
        return None

    def reset(self):
        from vosk import KaldiRecognizer
        self._rec = KaldiRecognizer(self._model, RATE)
        self._pending = 0
        self._prev_partial = ''
        self._last_partial_change = time.monotonic()


def make_wake_detector():
    """openWakeWord if a Luna model exists, else Vosk."""
    if os.path.isfile(OWW_MODEL):
        try:
            d = OpenWakeWord()
            print(f"[WAKE] openWakeWord: {os.path.basename(OWW_MODEL)}")
            return d
        except Exception as e:
            print(f"[WAKE] openWakeWord failed ({e}), falling back to Vosk")
    d = VoskWakeWord()
    print(f"[WAKE] Vosk open recognition, listening for '{WAKE_NAME}'")
    return d


# ==========================================================================
# Endpointing
# ==========================================================================

class Endpointer:
    """webrtcvad plus an adaptive energy gate.

    VAD alone is not enough on this board. The XVF-3000's AGC amplifies a
    quiet room until the noise floor sits around -10 dBFS, and webrtcvad
    happily calls that speech — so an utterance never ends and capture runs
    to MAX_UTTERANCE_S, sweeping up everything said afterwards.

    So a frame counts as speech only if VAD says so AND it is meaningfully
    louder than the room. The floor is measured continuously rather than
    fixed, because it changes with AGC state, fans, and the room itself.
    """

    def __init__(self, aggression=VAD_AGGRESSION):
        import webrtcvad
        self.vad = webrtcvad.Vad(aggression)

    @staticmethod
    def _db(frame):
        x = frame.astype(np.float32)
        if x.size == 0:
            return -99.0
        rms = float(np.sqrt(np.mean(x * x)))
        return -99.0 if rms <= 0 else 20.0 * math.log10(rms / 32768.0)

    def capture(self, stream, preroll=False, start_timeout=None):
        """Collect one utterance. Returns float32 @16kHz, or None.

        start_timeout: how long to wait for speech to BEGIN. Short after the
        wake word (the person is already mid-sentence), longer for a follow-up
        turn where they need a moment to think of the next thing."""
        collected = []
        # Seed the floor estimate from the pre-roll: that is audio from just
        # before the wake word, i.e. the room at rest.
        recent = collections.deque(maxlen=100)          # ~2s of frames
        pr = stream.snapshot_preroll()
        if pr.size:
            for i in range(0, len(pr) - FRAME_SAMPLES, FRAME_SAMPLES):
                recent.append(self._db(pr[i:i + FRAME_SAMPLES]))
            if preroll:
                collected.append(pr)

        hang = int(SILENCE_HANG_MS / 20)
        cap = int(MAX_UTTERANCE_S * 1000 / 20)
        silence = 0
        seen = False
        deadline = time.monotonic() + (start_timeout or SPEECH_TIMEOUT)

        for _ in range(cap):
            frame = stream.read(timeout=1.0)
            if frame is None:
                break
            collected.append(frame)

            db = self._db(frame)
            floor = (float(np.percentile(recent, 20)) if len(recent) >= 10
                     else db - 30.0)
            loud = db > floor + SPEECH_MARGIN_DB
            voiced = loud and self.vad.is_speech(frame.tobytes(), RATE)

            # Only quiet frames update the floor, or a long utterance would
            # drag the floor up to speech level and nothing would ever end.
            if not loud:
                recent.append(db)

            if voiced:
                seen, silence = True, 0
            elif seen:
                silence += 1
                if silence >= hang:
                    break
            elif time.monotonic() > deadline:
                return None

        if not seen:
            return None
        audio = np.concatenate(collected).astype(np.float32) / 32768.0
        secs = len(audio) / RATE
        if secs >= MAX_UTTERANCE_S - 0.1:
            print(f"[VAD] hit the {MAX_UTTERANCE_S}s cap — the endpointer "
                  f"never saw silence. Lower AGCMAXGAIN on the ReSpeaker, or "
                  f"raise SPEECH_MARGIN_DB / VAD_AGGRESSION.")
        return audio


# ==========================================================================
# STT
# ==========================================================================

_NOISE_EXACT = {"", ".", "you", "thank you", "thanks", "okay", "ok", "oh",
                "hmm", "uh", "um", "bye", "hello", "hi", "yeah", "so"}
# Whisper was trained on subtitle files and emits credit lines on silence.
_NOISE_RE = [re.compile(p, re.I) for p in (
    r"\bsub(title)?s?\s+(by|from)\b", r"\bamara\.org\b", r"\bwww[\s.]?\w+",
    r"\b\w+\.(com|org|net|co\.uk|tv)\b", r"\bthanks?\s+for\s+watching\b",
    r"\b(please\s+)?(like\s+and\s+)?subscribe\b", r"^\W*$")]
_REPEAT = re.compile(r"(\b[\w'][\w' ]{1,40}?[.!?,]?)(?:\s+\1){2,}", re.I)


def is_noise(text):
    t = (text or "").strip().lower().rstrip(".!?,")
    return t in _NOISE_EXACT or any(r.search(t) for r in _NOISE_RE)


def collapse_repeats(text):
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT.sub(r'\1', text)
    return text.strip()


def _is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "cuda failed" in s


class Transcriber:
    def __init__(self, chain=None):
        import ctranslate2
        self.chain = chain or STT_CHAIN
        self.model = None
        cuda_ok = bool(ctranslate2.get_supported_compute_types("cuda"))
        if not cuda_ok:
            print("[STT] CTranslate2 has no CUDA support — CPU only. Run "
                  "build_ctranslate2_cuda.sh for ~10x faster transcription.")

        for size, ctype, device in self.chain:
            if device == "cuda" and not cuda_ok:
                continue
            try:
                self._load(size, ctype, device)
                return
            except Exception as e:
                msg = str(e)
                if "cached snapshot" in msg or "HF_HUB_OFFLINE" in msg:
                    # Not a model or memory problem: the weights are not in
                    # WHISPER_CACHE and downloads are blocked. Say so plainly,
                    # because the raw message sends people looking at CUDA.
                    print(f"[STT] {size}: weights not found in {WHISPER_CACHE} "
                          f"and downloads are disabled. Fetch them once with:")
                    print(f'      python3 -c "from faster_whisper import '
                          f'WhisperModel; WhisperModel(\'{size}\', '
                          f'device=\'cpu\', download_root=\'{WHISPER_CACHE}\')"')
                    continue
                print(f"[STT] {size}/{ctype}/{device} failed: "
                      f"{'OUT OF MEMORY' if _is_oom(e) else e}")
        raise RuntimeError(
            "No Whisper configuration could be loaded — see the messages above")

    def _load(self, size, ctype, device):
        from faster_whisper import WhisperModel
        print(f"[STT] loading {size} {ctype} on {device}...")
        os.makedirs(WHISPER_CACHE, exist_ok=True)
        m = WhisperModel(size, device=device, compute_type=ctype,
                         download_root=WHISPER_CACHE)
        # Warm up BEFORE declaring success. Loading the weights can succeed
        # while the first inference still OOMs on the activation buffers, and
        # discovering that on the user's first question is much worse than
        # discovering it now.
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(RATE * 3) * 0.01).astype(np.float32)
        t0 = time.monotonic()
        for _ in range(2):
            list(m.transcribe(noise, language="en", beam_size=1)[0])
        self.model = m
        self.size, self.ctype, self.device = size, ctype, device
        print(f"[STT] ready: {size} {ctype} {device}, "
              f"warm-up {(time.monotonic()-t0)*1000:.0f}ms")

    def _degrade(self):
        """Drop to the next configuration after a runtime OOM."""
        try:
            i = self.chain.index((self.size, self.ctype, self.device))
        except ValueError:
            i = 0
        for size, ctype, device in self.chain[i + 1:]:
            try:
                del self.model
                self.model = None
                self._load(size, ctype, device)
                return True
            except Exception as e:
                print(f"[STT] {size}/{ctype}/{device} failed: {e}")
        return False

    def transcribe(self, audio):
        if audio is None or len(audio) < MIN_SPEECH_SEC * RATE:
            return ""
        try:
            segs, _ = self.model.transcribe(
                audio, language="en", beam_size=5,
                vad_filter=True,                     # strips silence: faster
                vad_parameters=dict(                 # and stops invented filler
                    min_silence_duration_ms=300),
                condition_on_previous_text=False,    # stops "Hello. Hello."
                initial_prompt=STT_PROMPT)
            # The generator is lazy — decoding happens HERE, so an OOM
            # surfaces on iteration, not on the call above.
            text = " ".join(s.text for s in segs).strip()
        except Exception as e:
            if _is_oom(e):
                print("[STT] ran out of GPU memory mid-transcription — "
                      "dropping to a smaller configuration")
                if self._degrade():
                    return ""       # this utterance is lost; the next works
            print(f"[STT] error: {e}")
            return ""
        text = collapse_repeats(text)
        if is_noise(text):
            print(f"[STT] rejected as noise: {text!r}")
            return ""
        return text


# ==========================================================================
# Streaming TTS with interruption
# ==========================================================================

_speak_lock   = threading.Lock()
is_speaking   = threading.Event()
stop_speaking = threading.Event()
_SENT = re.compile(r'(?<=[.!?])\s+')
_speaker_cache = None


def _speaker():
    global _speaker_cache
    if _speaker_cache is None:
        _speaker_cache = find_speaker()
    return _speaker_cache


def speak(text, voice=PIPER_VOICE):
    """Synthesize and play sentence by sentence, piping Piper's raw output
    straight into aplay so audio starts while the rest is still being
    synthesized. Interruptible between and during sentences."""
    if not text or not text.strip():
        return False
    if not os.path.isfile(voice):
        print(f"[TTS] voice missing: {voice}")
        return False
    if not os.path.isfile(PIPER_BIN):
        print(f"[TTS] piper missing: {PIPER_BIN}")
        return False

    sentences = [s for s in _SENT.split(text.strip()) if s]
    stop_speaking.clear()
    is_speaking.set()
    # The display's caption and transcript are driven by these two events.
    # Without them the screen showed state changes but never any words, which
    # looked like a broken display when in fact nothing was ever sent.
    _notify("speech_start", text)
    ok = True
    with _speak_lock:
        try:
            for s in sentences:
                if stop_speaking.is_set():
                    break
                piper = subprocess.Popen(
                    [PIPER_BIN, '--model', voice, '--output_raw'] + PIPER_FLAGS,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL)
                player = subprocess.Popen(
                    ['aplay', '-q', '-t', 'raw', '-f', 'S16_LE',
                     '-r', '22050', '-c', '1', '-D', _speaker(), '-'],
                    stdin=piper.stdout, stderr=subprocess.DEVNULL)
                piper.stdout.close()
                piper.stdin.write(s.encode())
                piper.stdin.close()
                while player.poll() is None:
                    if stop_speaking.is_set():
                        player.terminate()
                        piper.terminate()
                        break
                    time.sleep(0.02)
                piper.wait(timeout=5)
        except Exception as e:
            print(f"[TTS] {e}")
            ok = False
        finally:
            is_speaking.clear()
            # In the finally so the caption always clears: the failure paths
            # above return early, and skipping this would leave the screen
            # showing speech that never happened.
            _notify("speech_end")
    return ok


def interrupt():
    stop_speaking.set()


# ==========================================================================
# Assistant
# ==========================================================================

def _notify(event, text=""):
    """Fire-and-forget event to the touch display.

    Wrapped in a bare try because the display is optional: a robot with no
    screen attached, or a display_addon that failed to import, must not be able
    to break the voice loop. This runs from the loop thread on every state
    change, so it also must not raise.
    """
    try:
        from display_addon import notify_display
        notify_display(event, text)
    except Exception:
        pass


class LunaVoice:
    """
    answer_local(text)  -> reply str, or None to fall through to answer_remote
    answer_remote(text) -> slow path (LLM). Optional.
    on_state(name)      -> 'wake'|'listening'|'thinking'|'speaking'|'idle'
    """

    def __init__(self, answer_local, answer_remote=None,
                 wake_ack="Yes?", on_state=None):
        self.answer_local = answer_local
        self.answer_remote = answer_remote
        self.wake_ack = wake_ack
        self._on_state = on_state
        self._paused = threading.Event()
        self._stop = threading.Event()

    def _state(self, s):
        # Two consumers, deliberately separate: on_state drives hardware (eyes,
        # LEDs) and must never be blocked by the display; _notify pushes the
        # same word to the touchscreen over SSE.
        #
        # The display previously only learned about 'question' and
        # 'speech_start', so it looked idle for the two seconds a network answer
        # takes — the reactor showed standby while Luna was actually thinking.
        # Now every transition reaches it: wake, listening, followup, thinking,
        # speaking, idle.
        if self._on_state:
            try:
                self._on_state(s)
            except Exception as e:
                print(f"[STATE] {e}")
        _notify("state", s)

    def start(self):
        self.stream = MicStream().start()
        self.wake = make_wake_detector()
        self.endpointer = Endpointer()
        self.stt = Transcriber()
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[LUNA] listening. barge-in="
              f"{'ON' if ALLOW_BARGE_IN else 'OFF'}")
        return self

    def stop(self):
        self._stop.set()
        interrupt()
        self.stream.stop()

    def pause(self):
        self._paused.set()
        interrupt()

    def resume(self):
        self.stream.drain()
        self.wake.reset()
        self._paused.clear()

    def _loop(self):
        was_speaking = False
        while not self._stop.is_set():
            frame = self.stream.read(timeout=1.0)
            if frame is None or self._paused.is_set():
                continue

            speaking = is_speaking.is_set()
            if speaking and not ALLOW_BARGE_IN:
                was_speaking = True
                continue
            if was_speaking:
                # Frames were dropped mid-utterance, so the decoder's partial
                # hypothesis is stale. Start clean.
                was_speaking = False
                self.wake.reset()
                self.stream.drain()
                continue

            hit = self.wake.feed(frame)
            if hit:
                self._handle(hit[0], interrupted=speaking)

    def _handle(self, matched, interrupted=False):
        print(f"[WAKE] {matched!r}{' (barge-in)' if interrupted else ''}")
        self.wake.reset()
        if interrupted:
            interrupt()                # cut the current sentence immediately
            time.sleep(0.15)           # let aplay die before we listen
            self.stream.drain()

        self._state('wake')
        if self.wake_ack:
            speak(self.wake_ack)
            self.stream.drain()        # do not transcribe our own ack

        started = time.monotonic()
        for turn in range(1, FOLLOWUP_MAX_TURNS + 1):
            first = (turn == 1)
            self._state('listening' if first else 'followup')

            # First turn: the person is already speaking, so a short window.
            # Later turns: give them time to think of the next thing.
            audio = self.endpointer.capture(
                self.stream,
                start_timeout=None if first else FOLLOWUP_WINDOW_SEC)

            if audio is None:
                # Nobody spoke. On turn 1 that means the wake word was a false
                # trigger; later it just means the conversation is over.
                print("[LUNA] no speech — sleeping" if not first
                      else "[LUNA] no speech after wake")
                break

            t0 = time.monotonic()
            text = self.stt.transcribe(audio)
            dt = (time.monotonic() - t0) * 1000
            if not text:
                # Empty or rejected as noise. Do not keep the session open on
                # noise, or a chatty room holds Luna awake indefinitely.
                break
            print(f"[STT] turn {turn}: {text!r}  "
                  f"({len(audio)/RATE:.1f}s, {dt:.0f}ms)")
            _notify("question", text)      # display logs it under YOU

            reply = None
            try:
                reply = self.answer_local(text)
            except Exception as e:
                print(f"[LOCAL] {e}")

            if reply is None and self.answer_remote:
                self._state('thinking')
                speak(thinking_filler())
                try:
                    reply = self.answer_remote(text)
                except Exception as e:
                    print(f"[REMOTE] {e}")
                    reply = "Sorry, I could not reach the network."

            if reply:
                self._state('speaking')
                speak(reply)
            self.stream.drain()        # drop anything captured while speaking

            if not FOLLOWUP_ENABLED:
                break
            if time.monotonic() - started > FOLLOWUP_MAX_SEC:
                print(f"[LUNA] {FOLLOWUP_MAX_SEC:.0f}s conversation cap — "
                      f"sleeping")
                break
        else:
            print(f"[LUNA] {FOLLOWUP_MAX_TURNS} turn cap — sleeping")

        self._state('idle')
        # Clear the wake detector: it has been fed nothing during the
        # conversation and its hypothesis is stale.
        self.wake.reset()
        self.stream.drain()


if __name__ == "__main__":
    LunaVoice(answer_local=lambda q: f"You said: {q}").start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
