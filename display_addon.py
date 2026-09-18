"""
display_addon.py — touch-display support for Luna.

Adds three routes to the existing Flask server:

  GET  /display          -> serves display.html (the kiosk touch UI)
  GET  /display/events   -> Server-Sent Events stream (live speech, notices)
  POST /display/say      -> push an event to the display
                            {"type":"speech_start","text":"Hello!"}
                            {"type":"speech_end"}
                            {"type":"notice","text":"WiFi updated"}

Wire-up (2 lines in your server file, after `app = Flask(__name__)`):

    from display_addon import init_display, notify_display
    init_display(app)

Then make your TTS report what it says. In audio_utils.speak():

    from display_addon import notify_display
    def speak(text):
        notify_display("speech_start", text)
        ...play the audio (blocking)...
        notify_display("speech_end")

notify_display() works from the same process OR any other process on the
Jetson (it falls back to an HTTP POST to localhost:5000), so it's safe to
call from your voice-assistant script even if it runs separately.
"""

import json
import os
import queue
import threading
import time

# Folder containing this file — display.html, three.min.js, logo.png and
# the videos/ library live next to it. Works on any machine (Jetson, Pi).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Wi-Fi client interface (internet). The AP interface is never touched.
#   Jetson: internal card  = "wlP1p1s0"
#   Raspberry Pi: usually  = "wlan0"     <-- change this per robot
# Internet radio (client). "wlan0" = Pi internal WiFi. Auto-detected at
# runtime if this name doesn't exist on the robot (e.g. Jetson: wlP1p1s0),
# so the same file works on every unit. The AP dongle is never touched.
WIFI_CLIENT_IFACE = "wlan0"

# The hotspot's address is FIXED — the phone app depends on it.
# Enforced on every AP update so NetworkManager can never drift to 10.42.0.1.
AP_ADDRESS = "192.168.4.1/24"
_subscribers = []
_lock = threading.Lock()


# ── broadcast core ─────────────────────────────────────────
def _broadcast(event: dict):
    dead = []
    with _lock:
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def _tracker_speech_hook(event_type):
    """Keep the head tracker informed about when Luna is talking.

    Hooked here rather than in luna_voice because every speech_start and
    speech_end already passes through this function on its way to the
    display — so the tracker gets told without luna_voice needing to know the
    tracker exists.

    It matters because the head's return-to-centre timer must be HELD while
    she speaks: a 20-second answer outran an 18-second timer and the head
    turned away mid-sentence.
    """
    if event_type not in ("speech_start", "speech_end"):
        return
    try:
        from head_tracker import tracker
        tracker.set_speaking(event_type == "speech_start")
    except Exception:
        pass


# Voice states, as luna_voice reports them, mapped to eye animations.
# followup is the pause after an answer when she is still listening for more,
# so it uses the listening eyes rather than idle — going dim between an answer
# and a follow-up reads as having stopped paying attention.
_EYE_FOR_STATE = {
    "wake":      "listening",
    "listening": "listening",
    "followup":  "listening",
    "thinking":  "thinking",
    "speaking":  "speaking",
    "idle":      "idle",
}


def _set_eyes(state_name):
    """Set the eye animation for a voice state. Never allowed to raise:
    the eyes are decoration and must not be able to stop an answer."""
    try:
        from eyes import set_state
        set_state(_EYE_FOR_STATE.get(state_name, "idle"))
    except Exception as e:
        print(f"[EYES] could not set {state_name!r}: {e}")


def _speech_hook(event_type, text=""):
    """Drive the head nod and the arm gestures from Luna's actual voice state.

    Hooked here rather than in luna_voice because every state and speech event
    already passes through notify_display on its way to the display — so the
    tracker and the gesture engine get told without luna_voice needing to know
    either exists.

    ── THE TIMING, WHICH WAS WRONG ─────────────────────────────────────
    The nod used to be started from answer_local in main.py. That runs AFTER
    the question has been transcribed, so Luna nodded while she was thinking
    and answering rather than while listening — the opposite of what a nod
    means. It has to come from the LISTENING state, which only these events
    report.

    So:
        state=listening / followup   nod, she is taking it in
        speech_start                 stop nodding, start gesturing
        speech_end                   stop gesturing
        state=idle                   settle the head home
    """
    kind = (text or "").strip().lower()

    if event_type == "state":
        # The eyes follow the conversation. Set FIRST, before the head and
        # arms: it is a single short serial line, and it is the thing a
        # person notices soonest — the eyes changing as you finish speaking
        # is what makes the pause before an answer feel intentional.
        _set_eyes(kind)

        if kind in ("listening", "followup"):
            # Nod for a good while and let the next event cut it short. Sized
            # to outlast a long question rather than a short one: a nod that
            # stops halfway through somebody speaking looks like the robot
            # losing interest.
            try:
                from head_tracker import tracker
                tracker.attend(12.0)
            except Exception:
                pass
            return

        if kind == "thinking":
            # Keep nodding briefly while she works on the answer, so the pause
            # does not read as a freeze — but shorter, because this is not
            # listening any more.
            try:
                from head_tracker import tracker
                tracker.attend(4.0)
            except Exception:
                pass
            return

        if kind == "idle":
            # The conversation has ended. Settle the head home NOW rather than
            # after SOUND_HOME_AFTER: there is nobody to look at, and a head
            # left pointing where the last voice came from reads as staring at
            # an empty chair.
            try:
                from head_tracker import tracker
                tracker.attend(0)          # stop nodding
                tracker.go_home_gently("back to listening")
            except Exception:
                pass
            return
        return

    if event_type not in ("speech_start", "speech_end"):
        return

    talking = (event_type == "speech_start")
    _set_eyes("speaking" if talking else "idle")
    try:
        from gesture import gestures
        gestures.speaking(talking)
    except Exception:
        pass
    try:
        from head_tracker import tracker
        tracker.set_speaking(talking)
        if talking:
            # Stop the nod while she talks. The arms carry the movement now,
            # and a head nodding along to her own speech reads as agreeing
            # with herself.
            tracker.attend(0)
    except Exception:
        pass


def notify_display(event_type: str, text: str = ""):
    _speech_hook(event_type, text)
    _tracker_speech_hook(event_type)
    """Send an event to the touch display.
    Tries in-process first; falls back to HTTP so separate
    processes (voice loop, etc.) can use the same call."""
    event = {"type": event_type, "text": text}
    if _subscribers or _registered:
        _broadcast(event)
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:5000/display/say",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # display not up yet — never block speech on this


def load_json_list_safe(path):
    """Read a JSON list file; [] on any problem."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ── voice hook: reminders ──────────────────────────────────
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "twenty five": 25, "thirty": 30, "forty": 40,
            "forty five": 45, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "a": 1, "an": 1}


def try_reminder_command(text: str):
    """If the text sets or cancels a reminder/alarm/timer, do it and
    return a confirmation sentence for Luna to speak. Returns None if the
    text is not a reminder command.

    Understands e.g.:
      "remind me in 10 minutes to check the oven"
      "remind me to take medicine in half an hour"
      "remind me at 5:30 to call amma"
      "set a timer for 2 minutes"
      "cancel my reminders"

    Usage in the voice loop (before video/Q&A checks):

        resp = try_reminder_command(user_text)
        if resp:
            speak(resp)
    """
    import re
    import time as _t
    import urllib.request
    import urllib.parse
    low = (text or "").lower().strip()
    if not any(k in low for k in ("remind", "reminder", "alarm", "timer")):
        return None

    def _call(path):
        try:
            urllib.request.urlopen("http://127.0.0.1:5000" + path, timeout=8)
            return True
        except Exception as e:
            print(f"[REMIND] request failed: {e}")
            return False

    # cancel
    if re.search(r"\b(cancel|stop|delete|clear)\b.*\b(reminder|alarm|timer)", low):
        _call("/display/reminder_cancel?id=all")
        return "Okay, I've cancelled your reminders."

    # duration: "10 minutes", "two hours", "half an hour", "90 seconds"
    def parse_duration(s):
        m = re.search(r"\b(?:(\d+(?:\.\d+)?)|(" + "|".join(_WORDNUM) +
                      r")|half an?)\s*(second|sec|minute|min|hour|hr)s?\b", s)
        if not m:
            return None, None
        if m.group(1):
            n = float(m.group(1))
        elif m.group(2):
            n = float(_WORDNUM[m.group(2)])
        else:
            n = 0.5
        unit = m.group(3)
        mins = n / 60 if unit.startswith("sec") else n * 60 if unit.startswith(("hour", "hr")) else n
        return mins, m.group(0)

    # clock time: "at 5:30", "at 5 30 pm", "at 7 am", "at 530" (dot stripped)
    def parse_at(s):
        m = re.search(r"\bat\s+(\d{1,2})[:. ](\d{2})\s*(am|pm)?\b", s)
        if m:
            h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        else:
            m = re.search(r"\bat\s+(\d{3,4})\s*(am|pm)?\b", s)
            if m:  # "530" -> 5:30, "1730" -> 17:30
                num = m.group(1)
                h, mi, ap = int(num[:-2]), int(num[-2:]), m.group(2)
            else:
                m = re.search(r"\bat\s+(\d{1,2})\s*(am|pm)?\b", s)
                if not m:
                    return None, None
                h, mi, ap = int(m.group(1)), 0, m.group(2)
        h %= 24
        if mi > 59:
            return None, None
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        now = _t.localtime()
        target = _t.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                            h, mi, 0, 0, 0, -1))
        if target <= _t.time() and not ap and h < 12:
            target += 12 * 3600            # "at 5" said in the evening = 5 pm
        if target <= _t.time():
            target += 24 * 3600            # otherwise tomorrow
        return target, m.group(0)

    mins, dphrase = parse_duration(low)
    at, aphrase = (None, None) if mins else parse_at(low)
    if mins is None and at is None:
        return None

    # the task = what's left after removing command words and the time phrase
    task = low
    for junk in filter(None, [dphrase, aphrase]):
        task = task.replace(junk, " ")
    task = re.sub(r"\b(please|hey|ben|remind me|set (a |an )?"
                  r"(reminder|alarm|timer)( for| in)?|reminder|alarm|timer"
                  r"|\bto\b|\bin\b|\bfor\b|\bat\b)\b", " ", task)
    task = re.sub(r"\s+", " ", task).strip(" .,!?")
    task = task or "time's up"

    if mins is not None:
        ok = _call("/display/reminder_add?minutes=" + str(mins)
                   + "&text=" + urllib.parse.quote(task))
        if mins < 5 and mins % 1:
            nice = f"{int(mins * 60)} seconds"
        elif mins < 1:
            nice = f"{int(mins * 60)} seconds"
        elif mins < 60:
            nice = f"{int(mins)} minute{'s' if mins >= 2 else ''}"
        else:
            nice = f"{mins / 60:g} hour{'s' if mins >= 120 else ''}"
        if not ok:
            return "Sorry, I couldn't set that reminder."
        if task == "time's up":
            return f"Okay — timer set for {nice}."
        return f"Okay — I'll remind you in {nice} to {task}."
    else:
        ok = _call("/display/reminder_add?at=" + str(at)
                   + "&text=" + urllib.parse.quote(task))
        tstr = _t.strftime("%I:%M %p", _t.localtime(at)).lstrip("0")
        return (f"Okay — I'll remind you at {tstr} to {task}." if ok
                else "Sorry, I couldn't set that reminder.")


def notify_face(name: str, greeting: str = ""):
    """Show a face-greeting card on the display (photo + name + greeting).
    Call from face recognition right before speaking the greeting:

        from display_addon import notify_face
        notify_face(name, f"Hello {name}, {greeting}")
        speak(f"Hello {name}, {greeting}")
    """
    event = {"type": "face_greet", "name": name or "", "text": greeting or ""}
    if _subscribers or _registered:
        _broadcast(event)
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:5000/display/say",
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


_registered = False


# ── voice hook: call this with the user's transcribed text ─
def try_video_command(text: str):
    """If the text is a video request ('play a video about lions',
    'show me a video of the taj mahal', 'lion video play pannu', ...),
    trigger online playback on the display and return the query string.
    Returns None if the text is not a video request, so your voice loop
    can fall through to Q&A / chat as usual.

    Works from any process — it calls the local server over HTTP.

    Usage in your voice assistant, right after speech-to-text:

        from display_addon import try_video_command
        q = try_video_command(user_text)
        if q:
            speak(f"Playing a video about {q} on my screen")
        else:
            ... normal answer flow ...
    """
    import re
    t = (text or "").lower().strip()
    if "video" not in t and "youtube" not in t:
        return None
    patterns = [
        r"(?:play|show|put on|open)\s+(?:me\s+)?(?:a\s+|the\s+|some\s+)?"
        r"(?:youtube\s+)?videos?\s+(?:of|about|on|for)\s+(.+)",
        r"(?:play|show|put on|open)\s+(?:me\s+)?(.+?)\s+videos?(?:\s+.*)?$",
        r"(.+?)\s+videos?\s+(?:play|show)",
    ]
    query = None
    for p in patterns:
        m = re.search(p, t)
        if m:
            query = m.group(1).strip(" .,!?")
            break
    if not query:
        return None
    import urllib.request
    import urllib.parse
    # 1) local videos with a matching trigger word win over online
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:5000/display/videos", timeout=6) as r:
            vids = json.loads(r.read())
        for v in vids:
            trig = (v.get("trigger") or "").lower()
            if trig and (trig in query or query in trig):
                urllib.request.urlopen(
                    "http://127.0.0.1:5000/display/play?name="
                    + urllib.parse.quote(v["name"]), timeout=8)
                print(f"[MEDIA] Voice matched local trigger "
                      f"'{trig}' -> {v['name']}")
                return query
    except Exception as e:
        print(f"[MEDIA] local trigger check failed: {e}")
    # 2) otherwise search online as before
    try:
        url = ("http://127.0.0.1:5000/display/play_online?query="
               + urllib.parse.quote(query))
        urllib.request.urlopen(url, timeout=25)
    except Exception as e:
        print(f"[ONLINE] Voice video request failed: {e}")
    return query


# ── flask wiring ───────────────────────────────────────────
def init_display(app):
    global _registered
    from flask import Response, request, send_from_directory

    @app.route("/display")
    def display_page():
        return send_from_directory(_BASE_DIR, "display.html")

    @app.route("/display/logo.png")
    def display_logo():
        # optional — drop your Technovation logo.png next to display.html
        return send_from_directory(_BASE_DIR, "logo.png")

    @app.route("/display/events")
    def display_events():
        def stream():
            q = queue.Queue(maxsize=32)
            with _lock:
                _subscribers.append(q)
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        event = q.get(timeout=25)
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                with _lock:
                    if q in _subscribers:
                        _subscribers.remove(q)

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── video library ──────────────────────────────────
    import os
    import re
    videos_dir = os.path.join(_BASE_DIR, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    allowed_ext = {".mp4", ".webm", ".ogv", ".ogg", ".m4v", ".mov", ".mkv"}

    def _safe_name(name):
        name = os.path.basename(name or "")
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
        root, ext = os.path.splitext(name)
        if not root or ext.lower() not in allowed_ext:
            return None
        return f"{root}{ext.lower()}"

    @app.route("/display/upload_video", methods=["POST"])
    def display_upload_video():
        f = request.files.get("video")
        if not f:
            return {"status": "error", "message": "No file received"}, 400
        name = _safe_name(f.filename)
        if not name:
            return {"status": "error",
                    "message": "Use MP4, WebM, OGV, M4V, MOV or MKV"}, 400
        path = os.path.join(videos_dir, name)
        f.save(path)
        trig = (request.form.get("trigger") or "").strip().lower()
        if trig:
            d = _trig_load(); d[name] = trig; _trig_save(d)
        print(f"[MEDIA] Uploaded {name} "
              f"({os.path.getsize(path) // 1024} KB)"
              + (f" trigger='{trig}'" if trig else ""))
        _broadcast({"type": "notice", "text": f"Video added: {name}"})
        return {"status": "ok", "name": name}, 200

    @app.route("/display/videos")
    def display_videos():
        out = []
        trigs = _trig_load()
        for n in sorted(os.listdir(videos_dir)):
            p = os.path.join(videos_dir, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in allowed_ext:
                out.append({"name": n, "size": os.path.getsize(p),
                            "trigger": trigs.get(n, "")})
        return json.dumps(out), 200, {"Content-Type": "application/json"}

    @app.route("/display/video/<path:name>")
    def display_video(name):
        name = _safe_name(name)
        if not name:
            return {"status": "error", "message": "Bad name"}, 400
        # conditional=True enables HTTP Range → seeking works in the player
        return send_from_directory(videos_dir, name, conditional=True)

    @app.route("/display/video_delete")
    def display_video_delete():
        name = _safe_name(request.args.get("name"))
        path = os.path.join(videos_dir, name) if name else None
        if path and os.path.isfile(path):
            os.remove(path)
            d = _trig_load()
            if name in d:
                d.pop(name); _trig_save(d)
            print(f"[MEDIA] Deleted {name}")
        return {"status": "ok"}, 200

    @app.route("/display/play")
    def display_play():
        name = _safe_name(request.args.get("name"))
        if not name or not os.path.isfile(os.path.join(videos_dir, name)):
            return {"status": "error", "message": "Video not found"}, 404
        _broadcast({"type": "play_video", "text": name})
        print(f"[MEDIA] Playing {name}")
        return {"status": "ok"}, 200

    @app.route("/display/stop_play")
    def display_stop_play():
        _broadcast({"type": "stop_video"})
        return {"status": "ok"}, 200

    # ── online videos (YouTube via yt-dlp, no API key) ─────
    def _yt_search(query, n=6):
        """Search YouTube. Returns list of {id,title,duration,channel}."""
        import subprocess as sp
        try:
            r = sp.run(
                ["yt-dlp", f"ytsearch{n}:{query}",
                 "--flat-playlist", "--dump-json", "--no-warnings"],
                capture_output=True, text=True, timeout=20)
        except FileNotFoundError:
            return None  # yt-dlp not installed
        except Exception as e:
            print(f"[ONLINE] Search error: {e}")
            return []
        out = []
        for line in r.stdout.splitlines():
            try:
                j = json.loads(line)
                out.append({
                    "id": j.get("id", ""),
                    "title": j.get("title", ""),
                    "duration": j.get("duration") or 0,
                    "channel": j.get("channel") or j.get("uploader") or "",
                })
            except Exception:
                pass
        return out

    @app.route("/display/search_online")
    def display_search_online():
        q = (request.args.get("query") or "").strip()
        if not q:
            return {"status": "error", "message": "Empty query"}, 400
        res = _yt_search(q)
        if res is None:
            return {"status": "error",
                    "message": "yt-dlp not installed — run: pip3 install yt-dlp"}, 503
        return json.dumps(res), 200, {"Content-Type": "application/json"}

    @app.route("/display/play_online")
    def display_play_online():
        """Play by ?id=VIDEOID or search-and-play-top-result by ?query=..."""
        vid   = (request.args.get("id") or "").strip()
        title = (request.args.get("title") or "").strip()
        if not vid:
            q = (request.args.get("query") or "").strip()
            if not q:
                return {"status": "error", "message": "Need id or query"}, 400
            res = _yt_search(q, n=1)
            if res is None:
                return {"status": "error",
                        "message": "yt-dlp not installed — run: pip3 install yt-dlp"}, 503
            if not res:
                _broadcast({"type": "notice", "text": f"No video found for: {q}"})
                return {"status": "error", "message": "No results"}, 404
            vid, title = res[0]["id"], res[0]["title"]
        _broadcast({"type": "play_youtube", "id": vid, "text": title})
        print(f"[ONLINE] Playing {vid} — {title}")
        return {"status": "ok", "id": vid, "title": title}, 200

    # ── Wi-Fi client mode: scan / connect / status ──────────
    # (different from your /settings AP endpoint — this joins an
    #  existing network so Luna gets internet)
    import subprocess as _sp

    def _nmcli(args, timeout=25):
        """Run nmcli. Tries passwordless sudo (-n = never prompt, fail
        fast) and falls back to plain nmcli, which usually works
        unprivileged for scanning. Returns the best CompletedProcess."""
        first = None
        try:
            first = _sp.run(["sudo", "-n", "nmcli"] + args,
                            capture_output=True, text=True, timeout=timeout)
            if first.returncode == 0:
                return first
        except Exception:
            first = None
        second = _sp.run(["nmcli"] + args,
                         capture_output=True, text=True, timeout=timeout)
        if second.returncode == 0 or first is None:
            return second
        return first

    def _client_iface():
        """The WiFi device used for internet. Prefers WIFI_CLIENT_IFACE if
        it exists; otherwise picks a wifi device that is NOT the AP."""
        try:
            r = _nmcli(["-t", "-f", "DEVICE,TYPE", "device"], timeout=8)
            devs = [l.split(":")[0] for l in (r.stdout or "").splitlines()
                    if l.endswith(":wifi")]
            if WIFI_CLIENT_IFACE in devs:
                return WIFI_CLIENT_IFACE
            # exclude whichever device the AP profile is running on
            act = _nmcli(["-t", "-f", "NAME,DEVICE",
                          "connection", "show", "--active"], timeout=8)
            ap_devs = set()
            for line in (act.stdout or "").splitlines():
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    m = _nmcli(["-t", "-f", "802-11-wireless.mode",
                                "connection", "show", parts[0]], timeout=6)
                    if (m.stdout or "").strip().endswith("ap"):
                        ap_devs.add(parts[1])
            for d in devs:
                if d not in ap_devs:
                    return d
            return devs[0] if devs else WIFI_CLIENT_IFACE
        except Exception:
            return WIFI_CLIENT_IFACE

    @app.route("/display/wifi_scan")
    def display_wifi_scan():
        try:
            r = _nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY",
                        "dev", "wifi", "list", "ifname", _client_iface(),
                        "--rescan", "yes"], timeout=25)
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "nmcli failed").strip()[:160]
            return {"status": "error", "message": err}, 500
        nets = {}
        for line in r.stdout.splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            sec  = parts[-1]
            sig  = parts[-2]
            ssid = ":".join(parts[:-2]).replace("\\:", ":").strip()
            if not ssid:
                continue
            try:
                s = int(sig)
            except ValueError:
                s = 0
            if ssid not in nets or s > nets[ssid]["signal"]:
                nets[ssid] = {"ssid": ssid, "signal": s,
                              "secured": bool(sec and sec != "--")}
        out = sorted(nets.values(), key=lambda n: -n["signal"])
        return json.dumps(out), 200, {"Content-Type": "application/json"}

    def _wifi_forget(ssid):
        """Delete a saved connection profile by name — but never an
        AP-mode profile (that would kill the robot's own hotspot)."""
        try:
            mode = _nmcli(["-t", "-f", "802-11-wireless.mode",
                           "connection", "show", ssid], timeout=10)
            if "ap" in (mode.stdout or "").lower():
                print(f"[WIFI] Not deleting '{ssid}' — it's the AP profile")
                return False
            r = _nmcli(["connection", "delete", ssid], timeout=10)
            ok = r.returncode == 0
            print(f"[WIFI] Forgot stale profile '{ssid}': {ok}")
            return ok
        except Exception as e:
            print(f"[WIFI] Forget failed: {e}")
            return False

    @app.route("/display/wifi_connect")
    def display_wifi_connect():
        ssid = (request.args.get("ssid") or "").strip()
        pwd  = request.args.get("password") or ""
        if not ssid:
            return {"status": "error", "message": "No network name"}, 400
        args = ["dev", "wifi", "connect", ssid,
                "ifname", _client_iface()]
        if pwd:
            args += ["password", pwd]
        try:
            r = _nmcli(args, timeout=50)
            blob = (r.stdout + r.stderr).lower()
            # stale/corrupt saved profile → forget it and retry once
            if r.returncode != 0 and ("key-mgmt" in blob
                                      or "property is missing" in blob):
                print(f"[WIFI] Stale profile for '{ssid}' — healing")
                if _wifi_forget(ssid):
                    r = _nmcli(args, timeout=50)
                    blob = (r.stdout + r.stderr).lower()
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
        if r.returncode == 0 and "successfully" in blob:
            print(f"[WIFI] Connected to '{ssid}'")
            _broadcast({"type": "notice", "text": f"Connected to {ssid}"})
            return {"status": "ok", "ssid": ssid}, 200
        # nmcli sometimes reports failure/timeout while NetworkManager
        # finishes connecting in the background — verify before failing
        import time as _wt
        for _ in range(4):
            _wt.sleep(3)
            try:
                st = _nmcli(["-t", "-f", "ACTIVE,SSID", "dev", "wifi",
                             "ifname", _client_iface()], timeout=8)
                for line in st.stdout.splitlines():
                    if (line.startswith("yes:")
                            and line[4:].replace("\\:", ":") == ssid):
                        print(f"[WIFI] Connected to '{ssid}' (late)")
                        _broadcast({"type": "notice",
                                    "text": f"Connected to {ssid}"})
                        return {"status": "ok", "ssid": ssid}, 200
            except Exception:
                pass
        msg = "Wrong password" if ("secrets" in blob or "password" in blob) \
              else (r.stderr.strip().splitlines() or ["Connection failed"])[-1][:120]
        print(f"[WIFI] Connect failed: {msg}")
        return {"status": "error", "message": msg}, 400

    @app.route("/display/net_status")
    def display_net_status():
        ssid = ""
        try:
            r = _nmcli(["-t", "-f", "ACTIVE,SSID",
                        "dev", "wifi", "ifname", _client_iface()], timeout=8)
            for line in r.stdout.splitlines():
                if line.startswith("yes:"):
                    ssid = line[4:].replace("\\:", ":")
                    break
        except Exception:
            pass
        online = False
        try:
            import urllib.request
            req = urllib.request.urlopen(
                "http://connectivitycheck.gstatic.com/generate_204", timeout=4)
            online = (req.status == 204)
        except Exception:
            online = False
        return {"online": online, "ssid": ssid}, 200

    # ── weather for the idle screen (auto location, no API key) ──
    # Optional override: save {"weather_city": "Mumbai"} in settings to pin
    # the city; empty/absent = automatic from the internet connection.
    _wx_cache = {"t": 0.0, "data": None, "key": ""}

    @app.route("/display/weather")
    def display_weather():
        import time as _time
        import urllib.request as _ur
        import urllib.parse as _up
        city_override = ""
        try:
            from settings import load_settings
            city_override = str(load_settings().get("weather_city") or "").strip()
        except Exception:
            pass
        if (_wx_cache["data"] and _wx_cache["key"] == city_override
                and _time.time() - _wx_cache["t"] < 900):
            return _wx_cache["data"], 200
        try:
            if city_override:
                g = json.loads(_ur.urlopen(
                    "https://geocoding-api.open-meteo.com/v1/search?name="
                    + _up.quote(city_override) + "&count=1",
                    timeout=8).read())
                hits = g.get("results") or []
                if not hits:
                    return {"error": f"City not found: {city_override}"}, 404
                lat, lon = hits[0]["latitude"], hits[0]["longitude"]
                city = hits[0].get("name", city_override)
            else:
                loc = json.loads(_ur.urlopen(
                    "http://ip-api.com/json/?fields=status,city,lat,lon",
                    timeout=6).read())
                if loc.get("status") != "success":
                    raise Exception("geolocation failed")
                lat, lon = loc["lat"], loc["lon"]
                city = loc.get("city", "")
            w = json.loads(_ur.urlopen(
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,weather_code",
                timeout=8).read())
            cur = w.get("current", {})
            data = {
                "city": city,
                "temp": round(cur.get("temperature_2m", 0)),
                "humidity": cur.get("relative_humidity_2m", 0),
                "code": cur.get("weather_code", 0),
            }
            _wx_cache["t"] = _time.time()
            _wx_cache["data"] = data
            _wx_cache["key"] = city_override
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 503

    # ── reminders / alarms ──────────────────────────────────
    import time as _rtime
    REMINDERS_FILE = os.path.join(_BASE_DIR, "reminders.json")
    _rem = {}          # id -> {"id","text","at"}
    _rem_lock = threading.Lock()

    def _rem_save():
        try:
            with open(REMINDERS_FILE, "w") as f:
                json.dump(list(_rem.values()), f, indent=2)
        except Exception as e:
            print(f"[REMIND] Save error: {e}")

    def _rem_fire(rid):
        with _rem_lock:
            rem = _rem.pop(rid, None)
            if rem:
                _rem_save()
        if not rem:
            return  # was cancelled
        print(f"[REMIND] FIRING: {rem['text']}")
        _broadcast({"type": "reminder", "text": rem["text"]})

        def talk():
            try:
                from audio_utils import speak
                speak(f"Reminder! {rem['text']}")
            except Exception as e:
                print(f"[REMIND] Speak failed: {e}")
        threading.Thread(target=talk, daemon=True).start()

    def _rem_schedule(rem):
        delay = max(1.0, rem["at"] - _rtime.time())
        t = threading.Timer(delay, _rem_fire, args=[rem["id"]])
        t.daemon = True
        t.start()

    # reload reminders that survived a restart
    for _r in load_json_list_safe(REMINDERS_FILE):
        if _r.get("at", 0) > _rtime.time():
            _rem[_r["id"]] = _r
            _rem_schedule(_r)
    if _rem:
        print(f"[REMIND] Restored {len(_rem)} pending reminder(s)")

    @app.route("/display/reminder_add")
    def display_reminder_add():
        text = (request.args.get("text") or "").strip() or "Time's up"
        at   = request.args.get("at")       # epoch seconds, OR:
        mins = request.args.get("minutes")  # minutes from now
        try:
            when = float(at) if at else _rtime.time() + float(mins) * 60
        except (TypeError, ValueError):
            return {"status": "error", "message": "Need at= or minutes="}, 400
        if when <= _rtime.time():
            return {"status": "error", "message": "Time is in the past"}, 400
        rem = {"id": str(int(_rtime.time() * 1000)), "text": text,
               "at": when}
        with _rem_lock:
            _rem[rem["id"]] = rem
            _rem_save()
        _rem_schedule(rem)
        print(f"[REMIND] Set: '{text}' in {int(when - _rtime.time())}s")
        return {"status": "ok", "id": rem["id"], "at": when}, 200

    @app.route("/display/reminders")
    def display_reminders():
        now = _rtime.time()
        with _rem_lock:
            out = sorted(
                ({"id": r["id"], "text": r["text"], "at": r["at"],
                  "in_s": int(r["at"] - now)} for r in _rem.values()),
                key=lambda r: r["at"])
        return json.dumps(out), 200, {"Content-Type": "application/json"}

    @app.route("/display/reminder_cancel")
    def display_reminder_cancel():
        rid = request.args.get("id")
        with _rem_lock:
            if rid == "all":
                n = len(_rem)
                _rem.clear()
            else:
                n = 1 if _rem.pop(rid, None) else 0
            _rem_save()
        print(f"[REMIND] Cancelled {n}")
        return {"status": "ok", "cancelled": n}, 200

    # ══════════════════════════════════════════════════════════════════
    # LIVE CAMERA FEED (head-mounted camera)
    #
    # These four routes imported a module named `face_tracker`, which does not
    # exist on this robot — the tracker is `head_tracker`. The import raised,
    # the generator yielded nothing, and the live view was BLACK while
    # everything else about tracking worked. A stream that fails this way
    # produces no error in the browser either: an <img> pointed at an empty
    # multipart response just stays blank.
    # ══════════════════════════════════════════════════════════════════
    @app.route("/display/camera")
    def display_camera():
        from flask import Response

        def gen():
            import time as _ct
            try:
                from head_tracker import tracker
            except Exception as e:
                print(f"[CAM] live feed unavailable: {e}")
                return
            # Opening the stream is itself the request for a picture. Without
            # this the panel was black until tracking was already running —
            # which is backwards, since you have to SEE a face before you can
            # tap one. Preview runs the camera and the detector but does not
            # move the head.
            try:
                tracker.preview(True)
            except Exception as e:
                print(f"[CAM] preview: {e}")
            idle = 0
            # Serve frames the TRACKER already captured rather than opening the
            # camera again. Two owners of one device is how black frames
            # appear, and the tracker holds it continuously while enabled.
            while True:
                f = tracker.last_jpeg()
                if f:
                    idle = 0
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(f)).encode()
                           + b"\r\n\r\n" + f + b"\r\n")
                else:
                    # No frame yet. Give the tracker a moment to produce one
                    # rather than closing immediately — the stream is often
                    # opened in the same instant the toggle is flipped, before
                    # the first frame exists.
                    idle += 1
                    if idle > 60:            # ~5s with nothing at all
                        print("[CAM] live feed: tracker produced no frames")
                        return
                if tracker.mode == 'off':
                    return
                # Matched to the tracker's capture rate. Sleeping longer
                # discards frames it has already encoded; sleeping shorter
                # just re-sends the same one.
                _ct.sleep(0.05)              # ~20fps, the capture rate

        def gen_guarded():
            # Closing the panel must not leave the camera held forever, and
            # must not stop a head that is mid-follow either.
            try:
                yield from gen()
            finally:
                try:
                    from head_tracker import tracker
                    tracker.preview(False)   # no-op unless mode is 'preview'
                except Exception:
                    pass

        return Response(gen_guarded(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store",
                                 "X-Accel-Buffering": "no"})

    @app.route("/display/faces_live")
    def display_faces_live():
        try:
            from head_tracker import tracker
            return tracker.live_state(), 200
        except Exception as e:
            return {"enabled": False, "tracking": False,
                    "boxes": [], "error": str(e)}, 200

    @app.route("/display/track_select")
    def display_track_select():
        """Tap on the live view: LOCK onto whatever is there and follow it.

        The tap is resolved against real detections on the next frame — a
        remembered (x, y) would be meaningless the instant the head moves,
        since the camera moves with it. What is stored is the object, and it is
        found again in each new frame."""
        try:
            from head_tracker import tracker
            x = max(0.0, min(1.0, float(request.args.get("x", 0.5))))
            y = max(0.0, min(1.0, float(request.args.get("y", 0.5))))
            tracker.select_at(x, y)
            return {"status": "ok"}, 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    @app.route("/display/track_clear")
    def display_track_clear():
        try:
            from head_tracker import tracker
            tracker.clear_lock()
            tracker.centre()
        except Exception as e:
            print(f"[CAM] track_clear: {e}")
        return {"status": "ok"}, 200

    @app.route("/display/track_flip")
    def display_track_flip():
        """Flip a tracking axis without editing the file and restarting.

        If the head turns away from the person instead of towards them, this
        is the fix, and it is the first thing to try. The tracker also works
        this out on its own after a second or two of following, but doing it
        by hand is instant."""
        try:
            from head_tracker import tracker
            r = tracker.flip_axis(request.args.get("axis", "ud"))
            if r is None:
                return {"status": "error", "message": "axis must be lr or ud"}, 400
            return dict(r, status="ok"), 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    @app.route("/display/track_calibrate")
    def display_track_calibrate():
        """Measure which way each head axis actually moves the picture.

        Stand in front of the camera and call this. It nudges each axis about
        eight degrees and watches which way the image slid, which settles the
        direction question by experiment rather than by flag-flipping."""
        try:
            from head_tracker import tracker
            tracker.calibrate()
            return {"status": "ok"}, 200
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500

    @app.route("/display/track_status")
    def display_track_status():
        """Live tracking numbers, for calibrating. error_x/error_y should
        shrink towards zero and stay. Alternating sign means the head is
        hunting; shrinking but never arriving means CAM_FOV_H is too small."""
        try:
            from head_tracker import tracker
            return tracker.status(), 200
        except Exception as e:
            return {"error": str(e)}, 200

    # ── local video trigger words ───────────────────────────
    TRIGGERS_FILE = os.path.join(_BASE_DIR, "video_triggers.json")

    def _trig_load():
        try:
            with open(TRIGGERS_FILE) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _trig_save(d):
        try:
            with open(TRIGGERS_FILE, "w") as f:
                json.dump(d, f, indent=2)
        except Exception as e:
            print(f"[VIDEO] trigger save error: {e}")

    @app.route("/display/video_trigger")
    def display_video_trigger():
        name = (request.args.get("name") or "").strip()
        trig = (request.args.get("trigger") or "").strip().lower()
        if not name:
            return {"status": "error", "message": "name required"}, 400
        d = _trig_load()
        if trig:
            d[name] = trig
        else:
            d.pop(name, None)
        _trig_save(d)
        print(f"[VIDEO] trigger for '{name}' -> '{trig or '(cleared)'}'")
        return {"status": "ok"}, 200

    # ── camera snapshot (Vision page: face capture and the live view) ────
    #
    # Serialises SNAPSHOTS against each other and nothing else.
    #
    # It deliberately does NOT take vision._camera_lock. vision.get_camera()
    # takes that lock itself, and threading.Lock is not reentrant — a route
    # that holds it and then calls get_camera() deadlocks on the very first
    # request while still holding the lock, after which every later request
    # times out and reports "Camera busy - a face scan is running". That
    # message is plausible and completely wrong: no scan is running and the
    # camera was never opened. `fuser /dev/video0` shows nothing, which sends
    # you looking for contention that does not exist.
    _snap_lock = threading.Lock()

    @app.route("/display/cam_snap")
    def display_cam_snap():
        from flask import Response
        import cv2

        # If the head tracker is running it already owns the camera — reuse
        # its most recent frame rather than opening the device a second time.
        try:
            import head_tracker
            f = head_tracker.tracker.last_jpeg()
            if f:
                resp = Response(f, mimetype="image/jpeg")
                resp.headers["Cache-Control"] = "no-store"
                return resp
        except Exception:
            pass

        # Non-blocking: if a snapshot is already running, say so rather than
        # queueing behind it and returning a stale frame later.
        if not _snap_lock.acquire(blocking=False):
            return {"status": "error",
                    "message": "Another snapshot is already running"}, 503
        cap = None
        try:
            # ── ASK THE TRACKER. Do not open the camera. ─────────────────
            # This used to fall back to cv2.VideoCapture(0) when vision.py
            # had no camera open. That made a SECOND reader whenever the
            # tracker was also running, and two threads calling read() on one
            # cv2.VideoCapture is not thread-safe — on V4L2 it blocks inside
            # the driver holding the device lock. The visible result was the
            # camera appearing unavailable to whichever caller came second,
            # and at worst the whole Jetson hanging.
            #
            # The tracker owns the device. It wakes itself if idle, waits for
            # a frame that is both fresh AND lit, and hands it over.
            frame = None
            try:
                import head_tracker
                t = head_tracker.tracker
                if getattr(t, "_run", False):
                    frame = t.grab_for_caller(timeout=5.0, max_age=0.5,
                                              min_brightness=8.0)
            except Exception as e:
                print(f"[CAM] tracker unavailable: {e}")

            if frame is None:
                # Only reached when the tracker is not running at all, so
                # there is no second reader to collide with.
                try:
                    import vision
                    cap = vision.get_camera()
                except Exception as e:
                    return {"status": "error",
                            "message": f"camera unavailable ({e})"}, 503
                if cap is None or not cap.isOpened():
                    return {"status": "error",
                            "message": "camera unavailable — the tracker is "
                                       "not running and no device opened"}, 503
                # Warm up by wall clock. UVC drivers hand back zeroed buffers
                # for the first few dozen frames, so a handful of reads gives
                # pure black. Reading is what advances the exposure
                # algorithm; sleeping does not.
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                        try:
                            if float(f.mean()) >= 8.0:
                                break
                        except Exception:
                            break

            if frame is None:
                return {"status": "error", "message": "no frame"}, 503

            brightness = float(frame.mean())
            if brightness < 6.0:
                # Refuse rather than return it. Enrolling a face from a black
                # frame writes a useless embedding, and the failure surfaces
                # much later as "face recognition is broken".
                return {"status": "error",
                        "message": "Camera returned a black frame - check "
                                   "lighting and the lens cover"}, 503

            ok2, buf = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok2:
                return {"status": "error", "message": "Encode failed"}, 500
            print(f"[CAM] snapshot {frame.shape[1]}x{frame.shape[0]}, "
                  f"{len(buf)} bytes, brightness {brightness:.0f}")
            resp = Response(buf.tobytes(), mimetype="image/jpeg")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except Exception as e:
            print(f"[CAM] snapshot error: {e}")
            return {"status": "error", "message": str(e)}, 500
        finally:
            # Nothing to release: this function no longer opens the camera.
            # In a finally so an exception cannot leak the lock — a leaked
            # lock here reproduces the misleading "camera busy" failure.
            _snap_lock.release()

    def _find_ap_profile():
        """Return the name of the NetworkManager profile in AP mode."""
        r = _nmcli(["-t", "-f", "NAME", "connection", "show"], timeout=10)
        for name in (r.stdout or "").splitlines():
            name = name.strip()
            if not name:
                continue
            m = _nmcli(["-t", "-f", "802-11-wireless.mode",
                        "connection", "show", name], timeout=8)
            if (m.stdout or "").strip().endswith(":ap") or \
               (m.stdout or "").strip() == "ap":
                return name
        return None

    @app.route("/display/ap_update")
    def display_ap_update():
        ssid = (request.args.get("ssid") or "").strip()
        pwd  = request.args.get("password") or ""
        if not ssid:
            return {"status": "error", "message": "Network name required"}, 400
        if len(pwd) < 8:
            return {"status": "error",
                    "message": "Password needs 8+ characters"}, 400
        ap = _find_ap_profile()
        if not ap:
            return {"status": "error",
                    "message": "No hotspot profile found on this robot"}, 404
        r1 = _nmcli(["connection", "modify", ap,
                     "802-11-wireless.ssid", ssid], timeout=15)
        r2 = _nmcli(["connection", "modify", ap,
                     "wifi-sec.psk", pwd], timeout=15)
        # pin the fixed hotspot address — phone app depends on 192.168.4.1
        r3 = _nmcli(["connection", "modify", ap,
                     "ipv4.method", "shared",
                     "ipv4.addresses", AP_ADDRESS], timeout=15)
        if r1.returncode != 0 or r2.returncode != 0 or r3.returncode != 0:
            err = (r1.stderr + r2.stderr + r3.stderr).strip()[:140]
            return {"status": "error", "message": err or "nmcli failed"}, 500
        # persist so factory reset / boot flows use the new values
        try:
            from settings import load_settings, save_settings
            s = load_settings()
            s["wifi_ssid"] = ssid
            s["wifi_password"] = pwd
            save_settings(s)
        except Exception as e:
            print(f"[AP] settings persist skipped: {e}")
        # restart the AP only if it is running right now (on a single-radio
        # Pi the hotspot may be off while wlan0 is on the internet — then
        # the new name simply applies the next time the hotspot starts)
        act = _nmcli(["-t", "-f", "NAME", "connection", "show", "--active"],
                     timeout=10)
        applied = ap in (act.stdout or "")
        if applied:
            def bounce():
                _nmcli(["connection", "down", ap], timeout=20)
                _nmcli(["connection", "up", ap], timeout=30)
                print(f"[AP] Hotspot '{ap}' restarted as SSID '{ssid}'")
            threading.Thread(target=bounce, daemon=True).start()
        print(f"[AP] '{ap}' -> SSID '{ssid}' (applied now: {applied})")
        return {"status": "ok", "profile": ap, "applied": applied}, 200

    @app.route("/display/say", methods=["POST"])
    def display_say():
        data = request.get_json(silent=True) or {}
        event = {
            "type": data.get("type", "notice"),
            "text": data.get("text", ""),
        }
        if data.get("name"):
            event["name"] = str(data["name"])
        _broadcast(event)
        return {"status": "ok"}, 200

    # ── face photo lookup for the greeting card ─────────────
    @app.route("/display/face_photo")
    def display_face_photo():
        import base64 as _b64
        from flask import send_file, Response
        name = (request.args.get("name") or "").strip().lower()
        if not name:
            return {"status": "error"}, 404

        candidates = []
        # 1) JSON stores: vision_data.json (Pi) / faces_db.json (Jetson)
        for store in ("vision_data.json", "faces_db.json"):
            path = os.path.join(_BASE_DIR, store)
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                continue
            entries = (data.items() if isinstance(data, dict)
                       else enumerate(data) if isinstance(data, list) else [])
            for key, entry in entries:
                if not isinstance(entry, dict):
                    continue
                # the person's name may live in any of these fields
                labels = [str(entry.get(f) or "").strip().lower()
                          for f in ("name", "label", "person", "speech")]
                labels.append(str(key).strip().lower())
                if not any(lb == name or (lb and name in lb.split())
                           for lb in labels):
                    continue
                for field in ("image", "photo", "img", "face", "path", "file"):
                    v = entry.get(field)
                    if isinstance(v, str) and v:
                        candidates.append(v)
        # 2) image files named after the person
        for d in ("faces", "known_faces", "vision", "images", "."):
            folder = os.path.join(_BASE_DIR, d)
            try:
                for fn in os.listdir(folder):
                    root, ext = os.path.splitext(fn)
                    if (root.strip().lower() == name
                            and ext.lower() in (".jpg", ".jpeg", ".png", ".webp")):
                        candidates.append(os.path.join(folder, fn))
            except Exception:
                pass

        for c in candidates:
            # base64 / data-URI stored image
            if c.startswith("data:image") or (len(c) > 500 and "/" not in c[:80]):
                try:
                    mt = "image/jpeg"
                    if c.startswith("data:"):
                        mt = c[5:c.index(";")] or mt
                    b64 = c.split(",", 1)[1] if "," in c else c
                    return Response(_b64.b64decode(b64), mimetype=mt)
                except Exception:
                    continue
            p = c if os.path.isabs(c) else os.path.join(_BASE_DIR, c)
            if os.path.isfile(p):
                return send_file(p)
        return {"status": "error", "message": "no photo"}, 404

    _registered = True
    print("[DISPLAY] Routes ready at /display")
