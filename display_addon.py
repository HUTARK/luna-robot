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

import time
import json
import queue
import threading

_BASE_DIR = "/home/groot/pi_assistant"

# Wi-Fi client interface (internet). The USB AP interface is never touched.
#   Internal WiFi (wlP1p1s0)       -> joins router/hotspot -> internet
#   External USB  (wlx105a95d1c4d6) -> LunaRobot AP @ 192.168.4.1
WIFI_CLIENT_IFACE = "wlP1p1s0"
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


def notify_display(event_type: str, text: str = ""):
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
    try:
        import urllib.request
        import urllib.parse
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

    # Kept for compatibility only — the Luna display no longer loads three.js.
    # The 3D model on the home page was replaced by the conversation transcript,
    # so this route will 404 harmlessly if three.min.js is not present.
    @app.route("/display/three.min.js")
    def display_three():
        return send_from_directory(_BASE_DIR, "three.min.js")

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
        print(f"[MEDIA] Uploaded {name} "
              f"({os.path.getsize(path) // 1024} KB)")
        _broadcast({"type": "notice", "text": f"Video added: {name}"})
        return {"status": "ok", "name": name}, 200

    @app.route("/display/videos")
    def display_videos():
        out = []
        for n in sorted(os.listdir(videos_dir)):
            p = os.path.join(videos_dir, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in allowed_ext:
                out.append({"name": n, "size": os.path.getsize(p)})
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

    @app.route("/display/wifi_scan")
    def display_wifi_scan():
        try:
            r = _nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY",
                        "dev", "wifi", "list", "ifname", WIFI_CLIENT_IFACE,
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

    @app.route("/display/wifi_connect")
    def display_wifi_connect():
        ssid = (request.args.get("ssid") or "").strip()
        pwd  = request.args.get("password") or ""
        if not ssid:
            return {"status": "error", "message": "No network name"}, 400
        args = ["dev", "wifi", "connect", ssid,
                "ifname", WIFI_CLIENT_IFACE]
        if pwd:
            args += ["password", pwd]
        try:
            r = _nmcli(args, timeout=50)
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
        blob = (r.stdout + r.stderr).lower()
        if r.returncode == 0 and "successfully" in blob:
            print(f"[WIFI] Connected to '{ssid}'")
            _broadcast({"type": "notice", "text": f"Connected to {ssid}"})
            return {"status": "ok", "ssid": ssid}, 200
        msg = "Wrong password" if ("secrets" in blob or "password" in blob) \
              else (r.stderr.strip().splitlines() or ["Connection failed"])[-1][:120]
        print(f"[WIFI] Connect failed: {msg}")
        return {"status": "error", "message": msg}, 400

    @app.route("/display/net_status")
    def display_net_status():
        ssid = ""
        try:
            r = _nmcli(["-t", "-f", "ACTIVE,SSID",
                        "dev", "wifi", "ifname", WIFI_CLIENT_IFACE], timeout=8)
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

    # ── camera snapshot, for enrolling a face from the robot itself ──────
    # Luna had no camera route at all, so the Faces page could only accept a
    # file the operator had taken on a phone. Enrolling from the robot's own
    # camera is better: it captures the face from the exact angle and lens the
    # recogniser will actually see it from, which is most of what makes
    # recognition work.
    @app.route("/display/cam_snap")
    def display_cam_snap():
        import cv2
        # Share vision.py's camera and its lock rather than opening the device
        # independently. Two owners of one camera is how "Camera unavailable"
        # happens while nothing is obviously wrong.
        try:
            import vision
        except Exception as e:
            return {"status": "error", "message": f"vision module: {e}"}, 503

        lock = getattr(vision, "_camera_lock", None)
        if lock is not None and not lock.acquire(timeout=4):
            return {"status": "error",
                    "message": "Camera busy — a face scan is running"}, 503
        try:
            cap = vision.get_camera()
            if cap is None or not cap.isOpened():
                return {"status": "error", "message": "Camera unavailable"}, 503

            # Warm up by wall clock, not frame count. UVC drivers hand back
            # ZEROED buffers for the first few dozen frames, so a handful of
            # reads returns a pure black image. Reading in a tight loop is what
            # advances the exposure algorithm; sleeping does not.
            frame = None
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                ok, f = cap.read()
                if ok:
                    frame = f
            ok, f = cap.read()
            if ok:
                frame = f
            if frame is None:
                return {"status": "error", "message": "No frame from camera"}, 503

            brightness = float(frame.mean())
            if brightness < 6.0:
                # Refuse rather than return it: enrolling from a black frame
                # writes a useless embedding and the failure surfaces later as
                # "face recognition is broken".
                return {"status": "error",
                        "message": "Camera returned a black frame — check "
                                   "lighting and the lens cover"}, 503

            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok2:
                return {"status": "error", "message": "Encode failed"}, 500
            print(f"[CAM] snapshot {frame.shape[1]}x{frame.shape[0]}, "
                  f"{len(buf)} bytes, brightness {brightness:.0f}")
            return Response(buf.tobytes(), mimetype="image/jpeg")
        except Exception as e:
            print(f"[CAM] snapshot error: {e}")
            return {"status": "error", "message": str(e)}, 500
        finally:
            if lock is not None:
                lock.release()

    @app.route("/display/say", methods=["POST"])
    def display_say():
        data = request.get_json(silent=True) or {}
        _broadcast({
            "type": data.get("type", "notice"),
            "text": data.get("text", ""),
        })
        return {"status": "ok"}, 200

    _registered = True
    print("[DISPLAY] Routes ready at /display")
