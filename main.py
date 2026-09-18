import os
import subprocess
import threading
import time

from server import start_server, send_to_esp32
from qa_store import find_answer
from settings import load_settings

# The whole voice pipeline lives here now: continuous ReSpeaker capture, wake
# word, VAD endpointing, CUDA Whisper, streaming TTS, follow-up conversation.
from luna_voice import LunaVoice, speak, interrupt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Factory defaults ───────────────────────────────────────
FACTORY_ROBOT_NAME = "Luna"
FACTORY_WIFI_SSID  = "LunaRobot"
FACTORY_WIFI_PASS  = "luna1234"
FACTORY_IP         = "192.168.4.1"
HOTSPOT_IFACE      = "wlx105a95d1c4d6"   # external TP-Link adapter

assistant = None            # set in __main__; server.py routes reach it here

# ══════════════════════════════════════════════════════════════════════
# WHAT SHE SAYS WHEN SHE HEARS HER NAME
#
# A fixed "Yes?" every single time is the single most robot-like thing about
# the whole system — people notice it within three or four turns and it makes
# every subsequent answer feel canned, however good the answer is.
#
# Varied, with no immediate repeat, and grouped so the tone stays sensible:
# short and neutral most of the time, because a chatty acknowledgement wears
# out faster than a plain one.
# ══════════════════════════════════════════════════════════════════════
WAKE_ACKS = [
    "Yes?", "Yes?", "Yeah?",
    "Mm-hm?",
    "I'm listening.",
    "Go ahead.",
    "What's up?",
    "Yes, I'm here.",
    "Here.",
    "How can I help?",
]

_ack_bag = []
_last_ack = None


def next_wake_ack():
    """A different acknowledgement each time, cycling through all of them
    before any repeats.

    A shuffled BAG rather than random.choice: choosing randomly each time
    clumps badly. Measured over 14 wakes it used only 6 of the 10 phrases and
    said "Mm-hm?" four times, which sounds like a tic rather than variety. The
    bag guarantees every phrase is used once before any comes round again, and
    the reshuffle avoids the seam where the last of one bag could be the first
    of the next.

    Phrases listed twice in WAKE_ACKS appear twice per cycle, which is how the
    plain ones stay commoner than the chatty ones.
    """
    import random
    global _ack_bag, _last_ack
    if not _ack_bag:
        _ack_bag = list(WAKE_ACKS)
        random.shuffle(_ack_bag)
        if len(_ack_bag) > 1 and _ack_bag[-1] == _last_ack:
            _ack_bag[-1], _ack_bag[0] = _ack_bag[0], _ack_bag[-1]
    _last_ack = _ack_bag.pop()
    return _last_ack


# ── Vision trigger phrases ─────────────────────────────────
# Phrases that mean "answer this from the camera".
#
# Wider than before because a vision-language model can answer questions YOLO
# never could — colours, text, whether a door is open, how many people are
# wearing hats. The old list only caught "what do you see" style phrasings, so
# "is there a bottle in front of you" fell through to GPT, which answered
# confidently and blindly, having never seen the room.
VISION_TRIGGERS = [
    "what do you see", "what can you see", "what are you seeing",
    "what is in front of you", "what's in front of you",
    "describe what you see", "describe the scene", "describe the room",
    "describe where you are", "look around", "what are you looking at",
    "can you see", "do you see", "is there a", "are there any",
    "how many", "what colour", "what color", "what am i holding",
    "what is on the table", "what's on the table",
    "read this", "what does this say", "what is this",
]

# Words that make a phrase about the ROOM rather than the world in general.
# "how many" alone would hijack "how many days in a year", and "what colour"
# would hijack "what colour is the sky" — both belong to GPT, not the camera.
VISION_CONTEXT = [
    "you see", "in front of you", "front of you", "there", "this", "here",
    "table", "room", "holding", "camera", "looking", "scene", "around",
    "i am", "i'm", "my ",
]

AMBIGUOUS = ("can you see", "do you see", "is there a", "are there any",
             "how many", "what colour", "what color")

# Subjects that are never in the room, however the question is phrased.
# "can you see the future" and "is there a god" both matched an ambiguous
# phrase AND the bare word "there", so they were being sent to the camera —
# which would have answered by describing a wall. Checked before the context
# words, because an abstract subject settles it on its own.
NOT_IN_THE_ROOM = (
    "future", "past", "god", "heaven", "point in", "any point",
    "meaning of", "problem with", "difference between", "reason why",
)


def is_vision_request(text: str) -> bool:
    t = (text or "").lower()
    for phrase in VISION_TRIGGERS:
        if phrase not in t:
            continue
        # The unambiguous ones are enough on their own.
        if phrase not in AMBIGUOUS:
            return True
        # An abstract subject rules the camera out whatever else matched.
        if any(n in t for n in NOT_IN_THE_ROOM):
            return False
        # The rest need something anchoring them to the here and now.
        if any(c in t for c in VISION_CONTEXT):
            return True
    return False


# ── Answering ──────────────────────────────────────────────
# Everything the old qa_mode() did after transcription. LunaVoice does the
# listening; this decides what to say.
#
# Return a string to answer, or None to fall through to answer_remote (GPT).

def _mark_answer():
    """Tell the gesture engine the next thing spoken is an ANSWER.

    Called just before returning an answer, because this is the only place
    that knows the difference. speech_start looks identical for "Yes?" and
    for a paragraph about India, so without this the arms would start a
    gesture for a one-word acknowledgement and get half a pose in before the
    speech ended — a twitch rather than a gesture.
    """
    try:
        from gesture import gestures
        gestures.expect_answer()
    except Exception:
        pass


def answer_local(question: str):
    # ── turn towards whoever spoke, and look like you are listening ──────
    #
    # Read the DOA here, at the top of the turn, because this is the one moment
    # it is trustworthy: the person has just finished asking, and the DSP's
    # DOAANGLE register holds the bearing of the most recent speech it heard.
    # Sampled while Luna is talking it would point at her own speaker, which is
    # mounted on the robot, is the loudest thing in the room, and cannot be
    # cancelled — AEC measured 1-2.5 dB on this build where useful is 20-30.
    try:
        from sound_direction import last_direction
        import head_tracker
        bearing = last_direction()
        if bearing is not None:
            head_tracker.tracker.face_bearing(bearing)
        else:
            # No new bearing — the person has not moved, or the DSP heard
            # nothing new. Still push the go-home timer out, because a
            # question WAS just asked: the conversation is live even if the
            # direction is unchanged, and the head should not wander off
            # mid-exchange.
            head_tracker.tracker.keep_awake()
        # The nod is NOT started here. This runs after the question has been
        # transcribed, so it would nod while thinking and answering rather
        # than while listening. It is driven from the listening state instead
        # — see _speech_hook in display_addon.py.
    except Exception as e:
        # Never allowed to stop an answer. This is presentation, not function.
        print(f"[DOA] skipped: {e}")

    # Load the acknowledgement for the NEXT wake. This runs once per turn,
    # after the current one has already been spoken, so rotating here gives a
    # different phrase every time without LunaVoice needing to know anything
    # about it.
    if assistant is not None:
        try:
            assistant.wake_ack = next_wake_ack()
        except Exception:
            pass

    # ── video requests ──────────────────────────────────────────────────
    # try_video_command has been in display_addon.py all along — its own
    # docstring shows exactly this call — but main.py never made it. So
    # "play a video on elephants" fell through to GPT, which answered
    # honestly that it cannot play videos. The machinery was complete and
    # simply unreachable from the voice path.
    #
    # Checked BEFORE the vision triggers, because "show me a video of a dog"
    # contains "show me" and would otherwise be taken as a request to look at
    # something with the camera.
    try:
        from display_addon import try_video_command
        q = try_video_command(question)
        if q:
            _mark_answer()
            return f"Playing a video about {q} on my screen."
    except Exception as e:
        print(f"[MEDIA] video request failed: {e}")

    if is_vision_request(question):
        print("[MODE] Vision request")
        # Vision holds the camera and the GPU. Pause wake detection so a new
        # turn cannot start while YOLO runs.
        if assistant:
            assistant.pause()
        try:
            # Said before the request, not after: the round trip is 1-3
            # seconds and silence reads as the robot having missed the
            # question.
            # Said BEFORE the request. A description is 1-3 seconds; a search
            # sweeps the head and can take ten. Silence for that long reads as
            # the robot having missed the question entirely.
            from vision_vlm import is_find_request as _isfind
            speak("Let me have a look around." if _isfind(question)
                  else "Let me take a look.")
            # The VLM answers the QUESTION rather than listing objects, and
            # falls back to YOLO by itself when there is no network or no key
            # — so there is no need to decide here which to use.
            from vision_vlm import (ask_about_scene, is_find_request,
                                    extract_thing, find_object)
            # "can you see a bottle" makes her LOOK — the camera sees about 60
            # degrees, so a bottle two feet to one side is simply not in the
            # picture, and answering "no" from a single frame is wrong in a way
            # that feels careless. find_object sweeps the head and stops the
            # moment it finds it.
            # Wrapped. An exception here used to propagate into the voice
            # thread and kill it, so one bad vision request stopped the robot
            # answering anything at all until a restart. A vision failure
            # should cost one answer, not the conversation.
            try:
                if is_find_request(question):
                    _mark_answer()
                    return find_object(extract_thing(question))
                _mark_answer()
                return ask_about_scene(question)
            except Exception as e:
                import traceback
                print(f"[VISION] request failed: {e}")
                traceback.print_exc()
                return "Sorry, I had trouble looking at that."
        except Exception as e:
            print(f"[VISION] Error: {e}")
            try:
                from vision import describe_scene
                _mark_answer()
                return describe_scene()
            except Exception:
                return "Sorry, my camera is not available right now."
        finally:
            if assistant:
                assistant.resume()

    answer = find_answer(question)
    if answer:
        print("[QA] Found in local store")
        _mark_answer()
        return answer

    if not load_settings().get("chatgpt_enabled", True):
        # This is the "tell me about India" case. Nothing is broken: online
        # answers are switched OFF in settings.json, so anything not in the
        # local Q&A store gets refused. The old wording gave no clue which of
        # the two it was — an empty store or a disabled toggle.
        print("[QA] online answers are DISABLED in settings "
              "(chatgpt_enabled=false) — turn on Online Chat in the app, "
              "or set it in settings.json")
        return "I can't look that up right now. My online answers are switched off."

    print("[QA] Not found locally — asking GPT")
    return None                      # -> answer_remote


def answer_remote(question: str):
    # The GPT answer is an answer too.
    _mark_answer()
    from ai_fallback import ask_gpt
    return ask_gpt(question)


# ── Shutdown ───────────────────────────────────────────────
def do_shutdown():
    print("[SHUTDOWN] Starting safe shutdown")
    if assistant:
        assistant.pause()            # stop listening, cut any speech
    time.sleep(0.5)
    try:
        send_to_esp32("MOVE:stop")
        time.sleep(1)
    except Exception:
        pass
    try:
        send_to_esp32("HOME")
        time.sleep(2)
    except Exception:
        pass
    try:
        wav = os.path.join(BASE_DIR, "shutdown.wav")
        if os.path.exists(wav):
            from luna_voice import find_speaker
            subprocess.run(['aplay', '-q', '-D', find_speaker(), wav],
                           timeout=8, capture_output=True)
    except Exception:
        pass
    try:
        send_to_esp32("LATCH:OFF")
        print("[SHUTDOWN] LATCH:OFF sent")
        time.sleep(1)
    except Exception:
        pass
    print("[SHUTDOWN] Executing poweroff")
    # Call shutdown directly. The Popen("sleep 2 && shutdown") + os._exit(0)
    # pattern LOOKS safer but is not: os._exit kills the main process, systemd
    # then tears down the whole cgroup by default, and the scheduled child dies
    # before it runs — so the ESP32 cuts power 15s later on a machine that
    # never shut down. `shutdown -h now` hands off to systemd and returns at
    # once, so it does not block either.
    subprocess.run(["sudo", "/sbin/shutdown", "-h", "now"])


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    # No multiprocessing. Whisper is loaded ONCE, in-process, on the GPU.
    # The old design forked a child per utterance and reloaded the model every
    # time, costing seconds per question — and fork() after CUDA has been
    # initialised (the YOLO preload) is unsupported and fails unpredictably.

    threading.Thread(target=start_server, daemon=True).start()
    print("[HTTP] Server started on port 5000")
    time.sleep(1)

    s = load_settings()

    # wake_ack must be a STRING. LunaVoice stores it and calls
    # speak(self.wake_ack) directly, so passing a function does not fail at
    # construction — it fails later inside speak() with
    #     AttributeError: 'function' object has no attribute 'strip'
    # which kills the voice thread outright and stops Luna listening at all.
    # A try/except around the constructor does not catch it, because nothing
    # is called until the first wake.
    #
    # So the phrase is rotated by REPLACING the attribute after each turn —
    # see the end of answer_local. This turn's ack has already been spoken by
    # then, so what is being set is the one for next time.
    assistant = LunaVoice(
        answer_local=answer_local,
        answer_remote=answer_remote,
        wake_ack=next_wake_ack(),
    ).start()

    welcome = s.get("welcome_speech",
                    f"Hello, I am {FACTORY_ROBOT_NAME}, your robot assistant")
    speak(welcome)

    # Preload YOLO AFTER Whisper has claimed its GPU memory. Loading it first
    # leaves too little contiguous memory for small.en and forces the fallback
    # to base.en. If small.en still will not fit, run YOLO on the CPU: in
    # vision.py change device=0 to device='cpu'. YOLOv8n is small enough that
    # the six A78 cores handle it, and it frees the GPU entirely for Whisper.
    def preload_vision():
        try:
            from vision import _load_model
            _load_model()
        except Exception as e:
            print(f"[VISION] Preload skipped: {e}")
    threading.Thread(target=preload_vision, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        try:
            send_to_esp32("MOVE:stop")
        except Exception:
            pass
        print("\n[MAIN] Stopped.")
