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

# ── Vision trigger phrases ─────────────────────────────────
VISION_TRIGGERS = [
    "what do you see",
    "what can you see",
    "what is in front of you",
    "describe what you see",
    "look around",
    "what are you looking at",
]


def is_vision_request(text: str) -> bool:
    text = text.lower()
    return any(t in text for t in VISION_TRIGGERS)


# ── Answering ──────────────────────────────────────────────
# Everything the old qa_mode() did after transcription. LunaVoice does the
# listening; this decides what to say.
#
# Return a string to answer, or None to fall through to answer_remote (GPT).

def answer_local(question: str):
    print(f"[Q] {question}")

    # ── video on the display ────────────────────────────────────────────
    # try_video_command parses "play a video about lions" and starts playback
    # itself, returning the search terms. It was already in display_addon but
    # nothing called it, so asking Luna for a video did nothing.
    #
    # Checked BEFORE the Q&A store and before GPT: a video request should never
    # wait on a network answer, and "play a video about tigers" would otherwise
    # be sent to GPT as a question.
    try:
        from display_addon import try_video_command
        vq = try_video_command(question)
        # "play a video" with no subject parses to "a" and would search YouTube
        # for the letter A. Ask instead of guessing.
        if vq and len(vq) < 3:
            return "A video about what?"
        if vq:
            print(f"[VIDEO] playing: {vq}")
            return f"Playing a video about {vq} on my screen."
    except Exception as e:
        print(f"[VIDEO] check failed: {e}")

    if any(p in question for p in ("stop the video", "stop video",
                                   "close the video", "close video")):
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:5000/display/stop_play",
                                   timeout=5)
        except Exception as e:
            print(f"[VIDEO] stop failed: {e}")
        return "Stopped."

    if is_vision_request(question):
        print("[MODE] Vision request")
        # Vision holds the camera and the GPU. Pause wake detection so a new
        # turn cannot start while YOLO runs.
        if assistant:
            assistant.pause()
        try:
            speak("Let me take a look.")
            from vision import describe_scene
            return describe_scene()
        except Exception as e:
            print(f"[VISION] Error: {e}")
            return "Sorry, my camera is not available right now."
        finally:
            if assistant:
                assistant.resume()

    answer = find_answer(question)
    if answer:
        print(f"[A] (local) {answer}")
        return answer

    if not load_settings().get("chatgpt_enabled", True):
        return "I don't know that one yet."

    print("[QA] Not found locally — asking GPT")
    return None                      # -> answer_remote


def answer_remote(question: str):
    from ai_fallback import ask_gpt
    t0 = time.monotonic()
    reply = ask_gpt(question)
    # Log the answer AND how long the network call took. Without the timing
    # there is no way to tell a slow API from a slow robot.
    print(f"[A] (gpt {(time.monotonic()-t0)*1000:.0f}ms) {reply}")
    return reply


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

    assistant = LunaVoice(
        answer_local=answer_local,
        answer_remote=answer_remote,
        wake_ack="Yes?",
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
