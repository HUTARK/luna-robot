from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from identity import load_identity
import control_session
import os, threading, serial, json, time, subprocess, base64, re

# Self-locating. The hardcoded path breaks the moment this is cloned under a
# different username, and silently — load_poses() just returns {}.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
POSES_FILE = f"{BASE_DIR}/poses.json"
QA_FILE    = f"{BASE_DIR}/qa_store.json"
FACES_DIR  = f"{BASE_DIR}/faces"
FACES_DB   = f"{BASE_DIR}/faces_db.json"

os.makedirs(FACES_DIR, exist_ok=True)

# ── Flask App ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Flask buffers a POST body in RAM. On an 8GB board shared with Whisper and
# YOLO, an oversized face upload can take the whole process out.
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

from display_addon import init_display
init_display(app)

# ── State flags ────────────────────────────────────────────
obstacle_avoidance_enabled = True
online_chat_enabled        = True
face_tracking_enabled      = False

# ── ESP32 Serial (ttyTHS1 — configure later) ───────────────
ESP32_PORT = '/dev/ttyTHS1'
try:
    esp32 = serial.Serial(ESP32_PORT, 115200, timeout=1)
    print(f"[ESP32] Connected on {ESP32_PORT}")
except Exception as e:
    esp32 = None
    print(f"[ESP32] Not connected: {e}")

def reconnect_esp32():
    global esp32
    try:
        if esp32: esp32.close()
        esp32 = serial.Serial(ESP32_PORT, 115200, timeout=1)
        print(f"[ESP32] Reconnected")
        return True
    except:
        esp32 = None
        return False

_esp_lock = threading.Lock()

def send_to_esp32(command: str, quiet: bool = False):
    """Write one line to the ESP32.

    Serialised with a lock: the move keepalive, the loop sequencer, the head
    tracker and Flask request threads all write to this port, and two
    interleaved writes produce a corrupted line the ESP32 discards.

    quiet suppresses the log — the keepalive sends three commands a second and
    would otherwise bury everything else in the journal.
    """
    global esp32
    with _esp_lock:
        try:
            if esp32 and esp32.is_open:
                esp32.write((command + '\n').encode())
                if not quiet:
                    print(f"[ESP32] Sent: {command}")
            else:
                if reconnect_esp32():
                    esp32.write((command + '\n').encode())
                else:
                    print(f"[ESP32] Offline — skipped: {command}")
        except Exception as e:
            print(f"[ESP32] Error: {e}")
            reconnect_esp32()

# ── JSON helpers ───────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except: pass
    return default

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_poses():  return load_json(POSES_FILE, {})
def save_poses(p): save_json(POSES_FILE, p)
def load_qa():     return load_json(QA_FILE, {})
def save_qa(q):    save_json(QA_FILE, q)
def load_faces_db():  return load_json(FACES_DB, {})
def save_faces_db(d): save_json(FACES_DB, d)

# ── Volume control ─────────────────────────────────────────
def _speaker_card():
    """The ALSA card number that audio actually plays through.

    Asks audio_utils first, because that is what TTS uses — setting the
    volume on a different card than the one being played through is a silent
    no-op, and the previous version did exactly that. It took the FIRST card
    whose `aplay -l` line contained "usb", which on this robot could be the
    ReSpeaker or the USB dongle depending on enumeration order. Moving the
    speaker from one to the other therefore broke volume control with no
    error anywhere.
    """
    # 1. Whatever TTS is configured to use, e.g. "plughw:2,0" -> card 2.
    try:
        import audio_utils
        if hasattr(audio_utils, "get_speaker_device"):
            dev = str(audio_utils.get_speaker_device() or "")
            m = re.search(r"(?:plug)?hw:(\d+)", dev)
            if m:
                return m.group(1), f"from audio_utils ({dev})"
    except Exception:
        pass

    # 2. Otherwise prefer the ReSpeaker, since the speaker is wired to its
    #    output jack on this build.
    try:
        out = subprocess.run(['aplay', '-l'], capture_output=True,
                             text=True).stdout
        for line in out.splitlines():
            if not line.startswith('card'):
                continue
            low = line.lower()
            if 'arrayuac' in low or 'respeaker' in low:
                return line.split(':')[0].replace('card ', '').strip(), \
                       "ReSpeaker"
        # 3. Any USB audio card, as before.
        for line in out.splitlines():
            if line.startswith('card') and 'usb' in line.lower():
                return line.split(':')[0].replace('card ', '').strip(), \
                       "first USB card"
    except Exception as e:
        print(f"[VOLUME] could not list cards: {e}")
    return None, "none found"


def set_volume(value):
    """Set the playback volume, 0-100."""
    card, why = _speaker_card()
    if card is None:
        print(f"[VOLUME] no playback card found ({why})")
        return False

    # Control names differ between devices — the ReSpeaker exposes PCM, most
    # USB dongles expose Speaker. Trying each in turn is why this works
    # across both without configuration.
    for ctl in ('PCM', 'Speaker', 'Master', 'Headphone', 'Playback'):
        r = subprocess.run(['amixer', '-c', card, 'sset', ctl, f'{value}%'],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"[VOLUME] {value}% on card {card} via '{ctl}' ({why})")
            return True

    # Say which controls DO exist. Without this the failure is just "no",
    # and finding the right name means knowing to run amixer by hand.
    try:
        avail = subprocess.run(['amixer', '-c', card, 'scontrols'],
                               capture_output=True, text=True).stdout.strip()
        print(f"[VOLUME] none of the usual controls exist on card {card}. "
              f"Available:\n{avail}")
    except Exception:
        print(f"[VOLUME] no usable control on card {card}")
    return False


def get_volume():
    """The current volume as a percentage, or None."""
    card, _why = _speaker_card()
    if card is None:
        return None
    for ctl in ('PCM', 'Speaker', 'Master', 'Headphone', 'Playback'):
        r = subprocess.run(['amixer', '-c', card, 'sget', ctl],
                           capture_output=True, text=True)
        if r.returncode == 0:
            m = re.search(r"\[(\d+)%\]", r.stdout)
            if m:
                return int(m.group(1))
    return None


# ── Shared shutdown (SIGSEGV-safe pattern) ─────────────────
def do_shutdown():
    print("[SHUTDOWN] Starting safe shutdown")
    try:
        send_to_esp32("MOVE:stop")
        time.sleep(1)
    except: pass
    try:
        send_to_esp32("HOME")
        time.sleep(2)
    except: pass
    try:
        wav = f"{BASE_DIR}/shutdown.wav"
        if os.path.exists(wav):
            from audio_utils import get_speaker_device
            subprocess.run(['aplay', '-D', get_speaker_device(), wav],
                           timeout=8, capture_output=True)
    except: pass
    send_to_esp32("LATCH:OFF")
    print("[SHUTDOWN] LATCH:OFF sent — the ESP32 cuts power in 15s")
    time.sleep(1)

    print("[SHUTDOWN] poweroff")
    # Call shutdown DIRECTLY.
    #
    # The Popen("sleep 2 && shutdown") + os._exit(0) pattern that was here does
    # not work under systemd: os._exit kills this process, systemd tears down
    # the whole cgroup, and the scheduled child dies before it runs. On Giya the
    # journal showed exactly that — "Deactivated successfully" then "Scheduled
    # restart job" — so the service came back while the ESP32 cut power 15
    # seconds later on a machine that had never shut down. Every button press
    # was an unclean power-off.
    subprocess.run(["sudo", "/sbin/shutdown", "-h", "now"])


# ══════════════════════════════════════════════════════════════════════
# MOVEMENT KEEPALIVE
#
# The ESP32 stops the motors if no MOVE: arrives for 500ms:
#     if (millis() - lastCmdTime > 500 && robotMoving) stopMotors();
# That watchdog is the only thing that halts the robot if this Jetson dies, so
# it stays. It does mean a single /move per press runs for half a second and
# then stops — which is the "it stops after a few seconds while I am still
# holding the button" problem.
#
# So the direction is resent every 300ms while held, bounded by
# MOVE_HOLD_TIMEOUT and refreshed by each /move request, so a client that dies
# mid-press cannot drive the robot indefinitely.
#
# MEASURED TRADE-OFF, because it is not obvious:
#   HOLD=8s  app sends once, held 5s ....  5.0s  correct
#            app dies mid-press .........  8.0s  runaway
#            client repeats 1s, dies .... 11.0s  runaway, and WORSE
#   HOLD=2s  app sends once, held 5s ....  2.0s  CUT OFF
#            client repeats 1s, held 5s ..  5.0s  correct
# A repeating client makes a crash last LONGER at a long timeout, because each
# repeat pushes the deadline out again. Keep 8.0 while the phone app sends one
# request per press; drop to 2.0 only once every client repeats.
# ══════════════════════════════════════════════════════════════════════
MOVE_KEEPALIVE_SEC = 0.3
MOVE_HOLD_TIMEOUT  = 8.0

_move_dir      = "stop"
_move_deadline = 0.0
# How often the client refreshes a held button, learned from it. None until
# a repeat is seen, which is what distinguishes a client that repeats from
# one that sends a single request per press.
_move_last_req = 0.0
_move_gap      = None
_move_lock     = threading.Lock()


def stop_movement():
    """Clear the keepalive and stop. Used by e-stop, shutdown and restart."""
    global _move_dir
    with _move_lock:
        _move_dir = "stop"
    send_to_esp32("MOVE:stop")


def _move_keepalive():
    global _move_dir
    while True:
        time.sleep(MOVE_KEEPALIVE_SEC)
        with _move_lock:
            d, deadline = _move_dir, _move_deadline
        if d == "stop":
            continue                       # idle: no serial traffic at all
        if time.time() > deadline:
            with _move_lock:
                _move_dir = "stop"
                learned = _move_gap
            # Says WHY, because a robot stopping on its own mid-press and a
            # robot stopping because the release was lost look identical from
            # across a room.
            if learned:
                print(f"[MOVE] no refresh for {learned*3:.1f}s "
                      f"(client repeats every {learned*1000:.0f}ms) — "
                      f"stopping")
            else:
                print(f"[MOVE] no refresh for {MOVE_HOLD_TIMEOUT:.0f}s — "
                      f"stopping. This client does not repeat while held, so "
                      f"a lost release costs the full timeout.")
            send_to_esp32("MOVE:stop")
            continue
        send_to_esp32(f"MOVE:{d}", quiet=True)


# ══════════════════════════════════════════════════════════════════════
# HOME RACE GUARD
#
# Pressing Home in the app fires HOME *and* a POS: for every joint at slider
# midpoint, all at once. Flask serves them on concurrent threads, so the order
# reaching the serial port is whatever the scheduler decides. Captured on the
# robot:
#     Sent: POS:shoulder:1000:left
#     Sent: POS:wristUD:1000:left
#     Sent: HOME                      <- landed in the MIDDLE of the burst
#     Sent: POS:wristLR:1000:left     <- so this one overrode HOME
# Nine joints homed and wristLR went to 1000. Which joint loses changes every
# press, which is why it looked random rather than plainly wrong.
#
# So HOME wins its own burst: POS: is ignored briefly afterwards. The real fix
# belongs in the app, but the robot should not depend on arrival order.
# ══════════════════════════════════════════════════════════════════════
HOME_GUARD_SEC = 1.0
_home_until    = 0.0


# ══════════════════════════════════════════════════════════════════════
# ESP32 READER
#
# There was no reader thread, so NOTHING the ESP32 sent was handled: the
# button shutdown, obstacle events, person detection and the battery all went
# into the void. Only the Jetson -> ESP32 direction worked, which is why the
# power button did nothing and /status had to hardcode the battery.
# ══════════════════════════════════════════════════════════════════════
battery_percent = None          # None until the ESP32 reports one
last_esp_rx     = 0.0
_esp_faults     = {}


def esp32_reader():
    global last_esp_rx
    print("[ESP32] reader thread started")
    buf = b""
    while True:
        try:
            if not (esp32 and esp32.is_open):
                time.sleep(2)
                reconnect_esp32()
                continue
            chunk = esp32.read(esp32.in_waiting or 1)
            if not chunk:
                if last_esp_rx and time.time() - last_esp_rx > 30:
                    print("[ESP32] silent for 30s — reconnecting")
                    reconnect_esp32()
                    last_esp_rx = time.time()
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    last_esp_rx = time.time()
                    _handle_esp_line(line)
            if len(buf) > 512:
                buf = b""                  # runaway with no newline: discard
        except Exception as e:
            print(f"[ESP32] reader error: {e}")
            time.sleep(1)
            reconnect_esp32()


def _handle_esp_line(line):
    global battery_percent, _move_dir

    # Heartbeat carries the battery: "Dist:143 BATT:78", or just "Dist:143"
    # when no divider is fitted.
    if line.startswith("Dist:"):
        for tok in line.split():
            if tok.startswith("BATT:"):
                try:
                    battery_percent = max(0, min(100, int(tok[5:])))
                except ValueError:
                    pass
        return                             # every 2s: do not log

    if line.startswith("BATT:"):
        try:
            battery_percent = max(0, min(100, int(line[5:])))
            print(f"[ESP32] battery {battery_percent}%")
        except ValueError:
            pass
        return

    print(f"[ESP32] << {line}")

    if line.startswith("SHUTDOWN"):
        # The ESP32 button was held. It cuts power in 15 seconds whether or not
        # this succeeds, so the Jetson MUST power off inside that window or the
        # card takes an unclean cut.
        print("[ESP32] button shutdown — powering off")
        threading.Thread(target=do_shutdown, daemon=True).start()

    elif line.startswith("PERSON_DETECTED:"):
        if not obstacle_avoidance_enabled:
            return
        def greet():
            try:
                from audio_utils import speak
                from settings import load_settings
                name = load_settings().get("robot_name", "Luna")
                speak(f"Hello, I am {name}. How can I help you?")
            except Exception as e:
                print(f"[GREET] {e}")
        threading.Thread(target=greet, daemon=True).start()

    elif line.startswith("BLOCKED:"):
        # The ESP32 already stopped. Clear the keepalive too, or it keeps
        # resending the direction the ESP32 just refused.
        with _move_lock:
            globals()['_move_dir'] = "stop"
        print("[ESP32] obstacle — movement cleared")

    elif line.startswith("FAULT:"):
        what = line[6:]
        _esp_faults[what] = _esp_faults.get(what, 0) + 1
        print(f"[ESP32] FAULT {what} (x{_esp_faults[what]})")

    elif line.startswith("OK:"):
        print(f"[ESP32] recovered: {line[3:]}")


# ══════════════════════════════════════════════════════════════════════
# HEAD AIMING
#
# All head movement lives in head_tracker.py — face tracking AND tap-to-look.
# The thread that used to be here did tap-to-look only, and face tracking was
# never connected to anything: /face_tracking set a flag, printed a line, and
# returned. Nothing read the flag, which is why the toggle appeared to do
# nothing.
#
# Keeping it in one module also means one owner of headLR/headUD. Two threads
# both writing head position fight each other: each works from a stale idea of
# where the head is, both send corrections, and the head jitters or drifts.
# ══════════════════════════════════════════════════════════════════════
from head_tracker import tracker as head


# ── Loop / Sequencer ───────────────────────────────────────
loop_running = False

def run_loop():
    global loop_running
    poses = load_poses()
    if not poses:
        print("[LOOP] No poses saved")
        return
    print(f"[LOOP] Starting with {len(poses)} poses")
    while loop_running:
        for pos_key in sorted(poses.keys(), key=lambda x: int(x)):
            if not loop_running:
                break
            pose   = poses[pos_key]
            hand   = pose.get('hand', 'left')
            servos = pose.get('servos', {})
            for part, value in servos.items():
                if not loop_running:      # inner check too, so Stop takes
                    break                 # effect within ~50ms
                send_to_esp32(f"POS:{part}:{value}:{hand}")
                if part in ("headLR", "headUD"):
                    try:                      # same reason as in /position
                        head.note_external_head_command(part, int(value))
                    except Exception:
                        pass
                time.sleep(0.05)
            time.sleep(1)
    print("[LOOP] Stopped")


# ══════════════════════════════════════════════════════════════════════
# CONTROL SESSION GUARD  (T-24)
#
# One controller at a time, enforced locally with no cloud dependency. The
# session lives in memory in control_session.py; a restart clears it, which is
# deliberate — a crash can never leave the robot falsely locked.
#
# ─── THE ON-ROBOT TOUCHSCREEN IS EXEMPT, AND MUST BE ────────────────────
# The display calls /move, /position, /home, /save_pose and the loop routes
# directly and sends no session_id — it has no acquire/heartbeat logic at all.
# Gating those routes without an exemption returns 423 to every one of them,
# so the D-pad, the joint sliders, Home and the pose tools all stop working.
# The app team could not have caught this: they verified against a Flask test
# client, where the display never enters the picture.
#
# The exemption is by SOURCE ADDRESS, not by a shared secret: the display runs
# in a kiosk browser on the Pi itself and therefore connects from 127.0.0.1.
# Nothing on the hotspot can forge that — a remote client cannot make its
# packets appear to originate from the loopback interface.
#
# It is also the right policy on its own terms. Someone using the touchscreen
# is standing in front of the robot with their hand on it, which is a stronger
# claim to control than holding a phone somewhere on the same Wi-Fi. Physical
# presence outranks the lock, the same way the E-stop does.
#
# NOTE: if the display is ever served to a remote browser, this exemption stops
# applying to it and the display would need real session handling.
# ══════════════════════════════════════════════════════════════════════

LOCAL_ADDRS = {"127.0.0.1", "::1", "localhost"}


def _is_local_request():
    return (request.remote_addr or "") in LOCAL_ADDRS


def require_control_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # The on-robot touchscreen. See the note above.
        if _is_local_request():
            return f(*args, **kwargs)

        session = control_session.get_active_session()
        if session is None:
            return jsonify({"status": "error",
                            "message": "no_active_control_session"}), 423
        body = request.get_json(silent=True) or {}
        provided = request.args.get('session_id') or body.get('session_id')
        if provided != session['session_id']:
            return jsonify({"status": "error",
                            "message": "invalid_or_foreign_session"}), 423
        return f(*args, **kwargs)
    return wrapper


# ── Control session endpoints ─────────────────────────────────────────
@app.route('/control/acquire', methods=['POST', 'GET'])
def control_acquire():
    body = request.get_json(silent=True) or {}
    app_id = (request.args.get('app_id') or body.get('app_id') or '').strip().lower()
    client_id = (request.args.get('client_id') or body.get('client_id') or '').strip()

    if not app_id or not client_id:
        return jsonify({"status": "error",
                        "message": "app_id and client_id are required"}), 400

    # The app-to-robot identity check: a Giya or Cabi app can never take
    # control of a Luna unit.
    identity = load_identity()
    robot_type = (identity or {}).get('robot_type', '').strip().lower()
    if robot_type and app_id != robot_type:
        print(f"[CONTROL] rejected acquire — app_id '{app_id}' "
              f"!= robot_type '{robot_type}'")
        return jsonify({"status": "error",
                        "message": "app_identity_mismatch"}), 403

    ok, session = control_session.acquire(app_id, client_id)
    if not ok:
        return jsonify({"status": "error", "message": "robot_busy"}), 409

    print(f"[CONTROL] acquired — client={client_id} app={app_id} "
          f"session={session['session_id']}")
    return jsonify({
        "status": "ok",
        "session_id": session['session_id'],
        "timeout_seconds": control_session.CONTROL_SESSION_TIMEOUT_SECONDS
    }), 200


@app.route('/control/heartbeat', methods=['POST', 'GET'])
def control_heartbeat():
    body = request.get_json(silent=True) or {}
    session_id = request.args.get('session_id') or body.get('session_id')
    if not session_id or not control_session.heartbeat(session_id):
        return jsonify({"status": "error", "message": "invalid_session"}), 423
    return jsonify({"status": "ok",
                    "timeout_seconds":
                        control_session.CONTROL_SESSION_TIMEOUT_SECONDS}), 200


@app.route('/control/release', methods=['POST', 'GET'])
def control_release():
    body = request.get_json(silent=True) or {}
    session_id = request.args.get('session_id') or body.get('session_id')
    if not session_id or not control_session.release(session_id):
        return jsonify({"status": "error", "message": "invalid_session"}), 423
    print(f"[CONTROL] released — session={session_id}")
    return jsonify({"status": "ok"}), 200


# ══════════════════════════════════════════════════════════
#  ENDPOINTS — Ben Max API
# ══════════════════════════════════════════════════════════

# ── Status ─────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    # "connected" means the ESP32 is actually TALKING, not merely that the
    # port opened. An unplugged cable leaves the port open and cheerful.
    link = (esp32 is not None and esp32.is_open
            and last_esp_rx > 0 and (time.time() - last_esp_rx) < 10)
    response = {
        "status": "ok",
        # 100 until the first heartbeat, so a robot with no battery sensor
        # shows full rather than an alarming 0. battery_known tells the truth.
        "battery": battery_percent if battery_percent is not None else 100,
        "battery_known": battery_percent is not None,
        "wifi": True,
        "connected": link,
        "esp_link": link,
        "faults": _esp_faults,
        "robot": "luna",
        "tier": "max",
    }

    # ── Factory identity, protocol v1 ────────────────────────────────────
    # Merged in from robot_identity.json via identity.py. Omitted ENTIRELY
    # when the file is absent — never a placeholder — so the app cannot
    # mistake an unconfigured unit for an identified one.
    #
    # Read from its own file, never from settings.json, so no customer-facing
    # request (/settings, /save-audio) can alter what the robot reports as its
    # identity. If the file supplies a "tier" it replaces the default above.
    # Control-session state. Booleans and counters only: the session_id is a
    # bearer token, and anyone able to read /status on the shared Wi-Fi could
    # otherwise capture it and impersonate the controller. It is returned
    # exactly once, to the client that just succeeded at /control/acquire.
    # The ACTUAL level, read back from ALSA rather than the saved setting.
    # They differ whenever something else changed it — amixer by hand, or a
    # reboot before the restore above existed — and a slider showing a value
    # the robot is not at is worse than no slider.
    try:
        vol = get_volume()
        if vol is not None:
            response["volume"] = vol
    except Exception:
        pass

    active = control_session.get_active_session()
    response["controller_active"] = active is not None
    response["session_expires_in"] = control_session.time_remaining()

    identity = load_identity()
    if identity:
        response.update(identity)
    return jsonify(response), 200

# ── Movement ───────────────────────────────────────────────
@app.route('/move', methods=['GET'])
@require_control_session
def handle_move():
    """Drive the robot, and hold it moving while the button is held.

    ── THE RUNAWAY, AND WHY A FIXED TIMEOUT CANNOT FIX IT ──────────────
    The app sends dir=stop when a finger lifts. If that one request is lost —
    a WiFi hiccup, the app backgrounded, the screen locked — nothing stops the
    robot, and the keepalive happily drives it for the full hold timeout.
    Eight seconds of a robot driving itself across a room is not acceptable,
    and pressing another direction to stop it is not a workaround anybody
    should have to know.

    Shortening the timeout is not a fix either: an app that sends ONE request
    per press needs a long hold, or the robot stops two seconds into a
    deliberate five-second press.

    So the timeout ADAPTS to what the app actually does. If the app repeats
    while the button is held — and most do, to keep the connection warm — the
    gap between repeats is measured and the deadline set to a small multiple
    of it. A lost stop then costs a fraction of a second rather than eight.
    If the app does NOT repeat, the long fallback applies, because there is
    nothing else to go on.

    Nothing has to be configured: the robot learns it from the first press.
    """
    global _move_dir, _move_deadline, _move_last_req, _move_gap
    direction = request.args.get('dir', 'stop')
    # App sends 'back' — ESP32 expects 'backward'
    if direction == 'back':
        direction = 'backward'

    now = time.time()
    with _move_lock:
        changed = (direction != _move_dir)

        if direction == "stop":
            _move_dir = "stop"
            _move_deadline = 0.0
            _move_last_req = 0.0
        else:
            # A repeat of the SAME direction tells us how often this client
            # refreshes. Only same-direction repeats count: the gap between
            # forward and left is the user's reaction time, not the app's
            # polling rate.
            if not changed and _move_last_req:
                gap = now - _move_last_req
                # Ignore absurd values: a gap over 2s is a new press, not a
                # repeat, and one under 20ms is a duplicate request.
                if 0.02 <= gap <= 2.0:
                    # Smoothed, so one slow packet does not widen the window
                    # permanently.
                    _move_gap = gap if _move_gap is None else \
                                (_move_gap * 0.7 + gap * 0.3)
            elif changed:
                _move_last_req = 0.0

            _move_last_req = now
            if _move_gap:
                # Three missed refreshes before stopping. Two is twitchy on a
                # busy network; four starts to feel like a runaway again.
                hold = min(MOVE_HOLD_TIMEOUT, max(0.6, _move_gap * 3.0))
            else:
                hold = MOVE_HOLD_TIMEOUT       # client does not repeat
            _move_dir = direction
            _move_deadline = now + hold

    # Sent at once so the robot reacts on the press, not on the next tick.
    send_to_esp32(f"MOVE:{direction}", quiet=not changed)
    if changed:
        if direction == "stop":
            print("[MOVE] stop")
        else:
            print(f"[MOVE] {direction} "
                  f"(hold {(_move_deadline - now):.1f}s"
                  f"{', learned from the app' if _move_gap else ''})")
    return jsonify({"status": "ok", "dir": direction}), 200

@app.route('/speed', methods=['GET'])
@require_control_session
def handle_speed():
    value = request.args.get('value', '50')
    print(f"[SPEED] {value}")
    send_to_esp32(f"SPEED:{value}")
    return jsonify({"status": "ok"}), 200

@app.route('/estop', methods=['GET'])
def handle_estop():
    global loop_running
    print("[ESTOP] EMERGENCY STOP")
    loop_running = False
    # Clears the keepalive FIRST. Without that the thread resends the old
    # direction 300ms later and the robot carries on after an e-stop.
    stop_movement()
    send_to_esp32("MOVE:stop")
    return jsonify({"status": "ok", "message": "Emergency stop"}), 200

@app.route('/obstacle_avoidance', methods=['GET'])
@require_control_session
def handle_obstacle_avoidance():
    global obstacle_avoidance_enabled
    value = request.args.get('value', 'up')
    obstacle_avoidance_enabled = (value == 'up')
    send_to_esp32(f"HARDWARE:{'ON' if obstacle_avoidance_enabled else 'OFF'}")
    print(f"[OBSTACLE] {'Enabled' if obstacle_avoidance_enabled else 'Disabled'}")
    return jsonify({"status": "ok", "obstacle_avoidance": obstacle_avoidance_enabled}), 200

# ── Servo Joints ───────────────────────────────────────────
@app.route('/hand', methods=['GET'])
@require_control_session
def handle_hand():
    value = request.args.get('value', 'left')
    print(f"[HAND] {value}")
    send_to_esp32(f"HAND:{value}")
    return jsonify({"status": "ok"}), 200

@app.route('/position', methods=['GET'])
@require_control_session
def handle_position():
    part  = request.args.get('part', '')
    value = request.args.get('value', '1000')
    hand  = request.args.get('hand', 'left')

    if time.time() < _home_until:
        # See the HOME RACE GUARD note. 200 not 4xx: the app is not doing
        # anything wrong from its own point of view, and an error would show a
        # pointless failure toast.
        print(f"[POSITION] ignoring {part} — homing")
        return jsonify({"status": "ok", "ignored": "homing"}), 200

    # Fingers can be JSON: {"thumb":1000,"index":800}
    if value.strip().startswith('{'):
        try:
            fingers = json.loads(value)
            for fname, fval in fingers.items():
                send_to_esp32(f"POS:{fname}:{int(fval)}:{hand}")
                time.sleep(0.02)
            print(f"[POSITION] fingers {fingers} ({hand})")
            return jsonify({"status": "ok"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # A person is posing the arm. Speech gestures stand down for a while:
    # their instruction outranks decoration, and two writers on one joint is
    # how a servo gets damaged.
    try:
        from gesture import gestures
        gestures.note_user_activity()
    except Exception:
        pass

    print(f"[POSITION] {part} -> {value} ({hand})")
    send_to_esp32(f"POS:{part}:{value}:{hand}")
    # The tracker keeps its own model of where the head is pointing, and it
    # cannot read the servos. If a slider moves the head and nobody tells it,
    # the model is wrong, and the FIRST command it sends after that snaps the
    # head from its real position to the imagined one — which looks like the
    # tracker lurching for no reason the moment you select a face.
    if part in ("headLR", "headUD"):
        try:
            head.note_external_head_command(part, int(value))
        except Exception as e:
            print(f"[TRACK] sync: {e}")
    return jsonify({"status": "ok"}), 200

@app.route('/home', methods=['GET'])
@require_control_session
def handle_home():
    global _home_until
    print("[HOME]")
    # Set the guard BEFORE sending, so a POS: arriving while this handler is
    # still running is already suppressed.
    _home_until = time.time() + HOME_GUARD_SEC
    send_to_esp32("HOME")
    try:
        head.note_external_head_command("home", 0)
    except Exception as e:
        print(f"[TRACK] sync: {e}")
    return jsonify({"status": "ok"}), 200

# ── Loop / Sequencer ───────────────────────────────────────
@app.route('/save_pose', methods=['GET'])
@require_control_session
def handle_save_pose():
    pos  = request.args.get('pos', str(int(time.time())))
    hand = request.args.get('hand', 'left')
    # wristLR and wristUD were absent from this list, so those two axes were
    # dropped from every saved pose — silently — and the arm came back in a
    # different shape on replay.
    #
    # The list stays a whitelist (an unknown part name must not reach the
    # ESP32) but it now REPORTS what it rejected. A silent drop is what made
    # the wrist bug invisible: adding a joint to the firmware and the display
    # while forgetting the server produced no error anywhere.
    parts = ['head','headLR','headUD','lateral','shoulder','forearm',
             'elbow','wrist','wristLR','wristUD',
             'fingers','thumb','index','middle','ring','pinky']

    # The display saves poses under short names while /position and the loop
    # replay use the full ones. Same joints, two spellings — and the whitelist
    # only knew one of them, so wrLR and wrUD fell through it on every save.
    #
    # Normalise here rather than renaming anything: the stored keys have to
    # stay the names the ESP32 accepts, because run_loop sends them straight
    # back out as POS:{part}: — a pose saved as 'wrLR' would replay as a
    # command the firmware does not recognise.
    aliases = {'hdLR': 'headLR', 'hdUD': 'headUD',
               'wrLR': 'wristLR', 'wrUD': 'wristUD'}

    skip = {'pos', 'hand', 'robot', 'tier', 'robot_id', 'session_id'}
    incoming, unknown = {}, []
    for key, val in request.args.items():
        name = aliases.get(key, key)
        if name in parts:
            incoming[name] = val
        elif key not in skip:
            unknown.append(key)
    if unknown:
        print(f"[POSE] ignoring unknown part(s): {', '.join(unknown)} "
              f"— add them to `parts` or `aliases` in handle_save_pose "
              f"if they are joints")

    pose_data = {'hand': hand, 'servos': {}}
    for part in parts:
        val = incoming.get(part)
        if val is not None:
            try:
                pose_data['servos'][part] = int(val)
            except: pass
    poses = load_poses()
    poses[str(pos)] = pose_data
    save_poses(poses)
    print(f"[POSE] Saved {pos}: {pose_data}")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_start', methods=['GET'])
@require_control_session
def handle_loop_start():
    global loop_running
    loop_running = True
    threading.Thread(target=run_loop, daemon=True).start()
    print("[LOOP] START")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_stop', methods=['GET'])
@require_control_session
def handle_loop_stop():
    global loop_running
    loop_running = False
    print("[LOOP] STOP")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_undo', methods=['GET'])
@require_control_session
def handle_loop_undo():
    poses = load_poses()
    if poses:
        last_key = max(poses.keys(), key=lambda x: int(x))
        del poses[last_key]
        save_poses(poses)
        print(f"[LOOP] Removed {last_key}")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_delete', methods=['GET'])
@require_control_session
def handle_loop_delete():
    save_poses({})
    print("[LOOP] Cleared all")
    return jsonify({"status": "ok"}), 200

# ── Voice AI & Q&A ─────────────────────────────────────────
@app.route('/save-audio', methods=['GET', 'POST'])
def save_audio():
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = request.args
        robot_name     = data.get('robotName', '').strip()
        welcome_speech = data.get('welcomeSpeech', '').strip()

        from settings import save_settings
        updates = {}
        if robot_name:
            updates['robot_name'] = robot_name
        if welcome_speech:
            updates['welcome_speech'] = welcome_speech
        if updates:
            save_settings(updates)

        # Play back the welcome speech immediately
        if welcome_speech:
            def play():
                from audio_utils import speak
                speak(welcome_speech)
            threading.Thread(target=play, daemon=True).start()

        print(f"[SAVE-AUDIO] Robot: {robot_name}, Welcome: {welcome_speech}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"[SAVE-AUDIO] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/qa/add', methods=['POST'])
def qa_add():
    try:
        data = request.get_json()
        qa = load_qa()
        added, deleted = 0, 0
        if isinstance(data, list):
            for item in data:
                qid = str(item.get('id', ''))
                q   = item.get('q', '').strip()
                a   = item.get('a', '').strip()
                if not qid:
                    continue
                if q == "" and a == "":
                    # Empty payload = delete this ID
                    if qid in qa:
                        del qa[qid]
                        deleted += 1
                else:
                    qa[qid] = {"q": q.lower(), "a": a}
                    added += 1
        save_qa(qa)
        print(f"[QA] {added} added/updated, {deleted} deleted")
        return jsonify({"status": "ok", "added": added, "deleted": deleted}), 200
    except Exception as e:
        print(f"[QA] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/online_chat', methods=['GET'])
def handle_online_chat():
    global online_chat_enabled
    enabled = request.args.get('enabled', 'true').lower()
    online_chat_enabled = (enabled == 'true')
    from settings import save_settings
    save_settings({"chatgpt_enabled": online_chat_enabled})
    print(f"[CHAT] Online chat {'enabled' if online_chat_enabled else 'disabled'}")
    return jsonify({"status": "ok", "online_chat": online_chat_enabled}), 200

# ── Settings (volume / wifi) ───────────────────────────────
@app.route('/settings', methods=['GET', 'POST'])
def handle_settings():
    from settings import load_settings, save_settings

    if request.method == 'POST':
        try:
            s = save_settings(request.get_json() or {})
            return jsonify({"status": "ok", "settings": s}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # GET — could be volume, wifi config, or plain read
    volume    = request.args.get('volume')
    wifi_ssid = request.args.get('wifi_ssid')
    wifi_pass = request.args.get('wifi_password')

    if volume is not None:
        try:
            v = max(0, min(100, int(volume)))
            set_volume(v)
            save_settings({"volume": v})
            return jsonify({"status": "ok", "volume": v}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    if wifi_ssid and wifi_pass:
        def update_wifi():
            time.sleep(2)  # let the HTTP response go out first
            try:
                result = subprocess.run(
                    ['sudo', 'nmcli', '-t', '-f', 'NAME,TYPE', 'connection', 'show'],
                    capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if ':wifi' in line:
                        subprocess.run(['sudo', 'nmcli', 'connection', 'delete',
                                        line.split(':')[0]], capture_output=True)
                subprocess.run([
                    'sudo', 'nmcli', 'connection', 'add',
                    'type', 'wifi', 'ifname', 'wlan0',
                    'con-name', wifi_ssid, 'autoconnect', 'yes',
                    'ssid', wifi_ssid,
                    '802-11-wireless.mode', 'ap',
                    '802-11-wireless.band', 'bg',
                    'ipv4.method', 'shared',
                    'ipv4.addresses', '192.168.4.1/24',
                    'wifi-sec.key-mgmt', 'wpa-psk',
                    'wifi-sec.psk', wifi_pass
                ], capture_output=True)
                subprocess.run(['sudo', 'nmcli', 'connection', 'up', wifi_ssid],
                               capture_output=True)
                print(f"[WIFI] Updated to '{wifi_ssid}' @ 192.168.4.1")
            except Exception as e:
                print(f"[WIFI] Error: {e}")
        threading.Thread(target=update_wifi, daemon=True).start()
        return jsonify({"status": "ok",
                        "message": f"WiFi updating to {wifi_ssid} — reconnect"}), 200

    return jsonify(load_settings()), 200

# ── Vision AI ──────────────────────────────────────────────
@app.route('/face_tracking', methods=['GET'])
def handle_face_tracking():
    global face_tracking_enabled
    value = request.args.get('value', 'false').lower()
    face_tracking_enabled = value in ('1', 'true', 'yes', 'on')
    # This line was missing. The flag was set and printed and nothing read it,
    # so the toggle moved but the head never did.
    head.enable_face(face_tracking_enabled)
    return jsonify({"status": "ok", "face_tracking": face_tracking_enabled}), 200

def _save_base64_face(face_id, b64data):
    """Decode data:image/jpeg;base64,... and save as faces/<id>.jpg"""
    if ',' in b64data:
        b64data = b64data.split(',', 1)[1]
    img_bytes = base64.b64decode(b64data)
    path = os.path.join(FACES_DIR, f"{face_id}.jpg")
    with open(path, 'wb') as f:
        f.write(img_bytes)
    return path

@app.route('/upload_face', methods=['POST'])
def handle_upload_face():
    try:
        data = request.get_json()
        face_id = str(data.get('id', '0'))
        b64     = data.get('data', '')
        if not b64:
            return jsonify({"status": "error", "message": "No image data"}), 400
        path = _save_base64_face(face_id, b64)
        db = load_faces_db()
        db.setdefault(face_id, {})['image'] = path
        save_faces_db(db)
        print(f"[FACE] Saved face {face_id}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"[FACE] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Older app builds POST to /upload-face with a hyphen and got a 404.
@app.route('/upload-face', methods=['POST'])
def handle_upload_face_alias():
    return handle_upload_face()


@app.route('/upload_faces', methods=['POST'])
def handle_upload_faces():
    try:
        data = request.get_json()
        db = load_faces_db()
        added, deleted = 0, 0
        if isinstance(data, list):
            for item in data:
                fid    = str(item.get('id', ''))
                face   = item.get('face', '')
                speech = item.get('speech', '')
                if not fid:
                    continue
                if face == "" and speech == "":
                    if fid in db:
                        img = db[fid].get('image')
                        if img and os.path.exists(img):
                            os.remove(img)
                        del db[fid]
                        deleted += 1
                else:
                    entry = db.setdefault(fid, {})
                    if face:
                        entry['image'] = _save_base64_face(fid, face)
                    if speech:
                        entry['speech'] = speech
                    added += 1
        save_faces_db(db)
        print(f"[FACES] {added} saved, {deleted} deleted")
        return jsonify({"status": "ok", "added": added, "deleted": deleted}), 200
    except Exception as e:
        print(f"[FACES] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Joint speed ────────────────────────────────────────────
@app.route('/topspeed', methods=['GET'])
@require_control_session
def handle_topspeed():
    """Servo speed, 0-2000 to match the app and the firmware's TOPSPEED:.

    THIS ROUTE DID NOT EXIST. The display's servo-speed slider calls it, so
    every change 404'd and TOPSPEED: never reached the ESP32 — which is why
    joint speed appeared to do nothing while homing (driven by the firmware's
    own HOME_DPS) worked perfectly.

    /speed is a different thing: that is the DRIVE motor.
    """
    try:
        v = max(0, min(2000, int(request.args.get('value', '1000'))))
    except ValueError:
        return jsonify({"status": "error", "message": "bad value"}), 400
    send_to_esp32(f"TOPSPEED:{v}")
    print(f"[SERVO-SPEED] {v}/2000")
    return jsonify({"status": "ok", "value": v}), 200


# ── Fingers ────────────────────────────────────────────────
@app.route('/fingers', methods=['GET', 'POST'])
@require_control_session
def handle_fingers():
    """The app POSTs here. The route did not exist, so every finger command
    from the app returned 404 and never reached the ESP32 — the
    'POST /fingers 404' lines in the journal.

    Accepts ?value=1000&hand=left for all fingers together, or a JSON body of
    per-finger values: {"thumb":1000,"index":800}
    """
    try:
        if request.method == 'POST':
            data = dict(request.get_json(silent=True) or {})
            if not data:
                data = request.args.to_dict()
        else:
            data = request.args.to_dict()

        hand = str(data.pop('hand', 'left'))

        if time.time() < _home_until:
            print("[FINGERS] ignoring — homing")
            return jsonify({"status": "ok", "ignored": "homing"}), 200

        if 'value' in data:
            send_to_esp32(f"POS:fingers:{int(data['value'])}:{hand}")
            print(f"[FINGERS] all -> {data['value']} ({hand})")
            return jsonify({"status": "ok"}), 200

        sent = 0
        for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            if name in data:
                send_to_esp32(f"POS:{name}:{int(data[name])}:{hand}")
                # Space the burst: the ESP32 reads a line at a time into a
                # 96-byte buffer, and five back-to-back commands can overrun it.
                time.sleep(0.02)
                sent += 1
        if not sent:
            return jsonify({"status": "error",
                            "message": "no finger values given"}), 400
        print(f"[FINGERS] {sent} finger(s) ({hand})")
        return jsonify({"status": "ok", "sent": sent}), 200
    except Exception as e:
        print(f"[FINGERS] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


# ── Hardware toggle (older app builds) ─────────────────────
@app.route('/hardware', methods=['GET'])
@require_control_session
def handle_hardware():
    """Older app builds call /hardware?state=on|off instead of
    /obstacle_avoidance. Unhandled, it returned 404."""
    global obstacle_avoidance_enabled
    state = request.args.get('state', request.args.get('value', 'on')).lower()
    obstacle_avoidance_enabled = state in ('on', 'true', 'up', '1')
    send_to_esp32(f"HARDWARE:{'ON' if obstacle_avoidance_enabled else 'OFF'}")
    return jsonify({"status": "ok",
                    "hardware": obstacle_avoidance_enabled}), 200


# ── Object tracking ────────────────────────────────────────
@app.route('/track/object', methods=['GET'])
@require_control_session
def handle_track_object():
    """Tapping the camera view sends NORMALISED coordinates (0-1), not pixels:
    the preview is scaled to fit the panel and its size has nothing to do with
    the camera's resolution, so pixels would be wrong on every screen but the
    one they were measured on."""
    global track_object_enabled, track_target
    en = request.args.get('enabled')
    if en is not None:
        track_object_enabled = en.lower() in ('1', 'true', 'yes', 'on')
        if not track_object_enabled:
            head.clear_lock()
        print(f"[TRACK] object following {'on' if track_object_enabled else 'off'}")
    x, y = request.args.get('x'), request.args.get('y')
    if x is not None and y is not None:
        try:
            fx = max(0.0, min(1.0, float(x)))
            fy = max(0.0, min(1.0, float(y)))
        except ValueError:
            return jsonify({"status": "error", "message": "bad coordinates"}), 400
        track_target = {"x": fx, "y": fy}
        # Locks onto whatever is at that point and FOLLOWS it. The point is
        # resolved against real detections on the next frame: a remembered
        # (x, y) is meaningless once the head moves, because the camera moves
        # with it. What gets stored is the object.
        head.select_at(fx, fy)
    return jsonify({"status": "ok", "enabled": track_object_enabled,
                    "target": track_target}), 200


@app.route('/eyes1', methods=['GET'])
@app.route('/eyes2', methods=['GET'])
@app.route('/eyes3', methods=['GET'])
def handle_eyes_numbered():
    """The app's eye buttons.

    It calls /eyes1, /eyes2 and /eyes3 — not /eyes?state=, which is what the
    display and the voice states use. All three were 404 on Luna: Giya and
    Lisa have these routes and Luna never did, so the buttons moved in the UI
    and nothing reached the robot.

    The state name comes from the path, so there is nothing to map.
    """
    state = request.path.lstrip('/')          # "eyes1", "eyes2", "eyes3"
    try:
        from eyes import set_state, get_state
        set_state(state)
        return jsonify({"status": "ok", "state": get_state()}), 200
    except Exception as e:
        print(f"[EYES] {state} failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/eyes', methods=['GET'])
def handle_eyes():
    """Set the eye animation.

    The RIGHT way in while the service is running: it owns /dev/ttyTHS1 and
    serialises writes to it. A separate process opening the same port gives
    two writers, and two interleaved writes produce a corrupted line the
    firmware discards — which looks exactly like the command never arriving.

    Not gated by a control session: the eyes move nothing and are useful for
    testing from any device.
    """
    state = (request.args.get('state') or request.args.get('value')
             or 'idle').strip().lower()
    try:
        from eyes import VALID, set_state, get_state
        if state not in VALID:
            return jsonify({"status": "error",
                            "message": f"unknown state {state!r}",
                            "known": list(VALID)}), 400
        set_state(state)
        return jsonify({"status": "ok", "state": get_state()}), 200
    except Exception as e:
        print(f"[EYES] {state!r} failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/loop', methods=['GET'])
@require_control_session
def handle_loop():
    """The app's Loop buttons — a dance with music, for a set duration.

    Was 404 on every press: the app sends /loop?value=Loop 1 and no such
    route existed. The dances live in gestures.json under "loops", with the
    music filename and how long to run.
    """
    value = (request.args.get('value') or '').strip()
    if not value:
        return jsonify({"status": "error", "message": "no loop given"}), 400
    try:
        from gesture import gestures
        ok, detail = gestures.play_loop(value)
        if not ok:
            print(f"[LOOP] {value!r} refused: {detail}")
            return jsonify({"status": "error", "message": detail}), 409
        return jsonify({"status": "ok", "loop": value, "action": detail}), 200
    except Exception as e:
        print(f"[LOOP] {value!r} failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/run', methods=['GET'])
@require_control_session
def handle_run():
    """The app's Run buttons — one-shot sequences, same as modes.

    Was returning 404 on every press: the app sends /run?value=Run 1 and no
    such route existed. The sequences live in gestures.json under "runs".
    """
    value = (request.args.get('value') or '').strip()
    if not value:
        return jsonify({"status": "error", "message": "no run given"}), 400
    try:
        from gesture import gestures
        ok, detail = gestures.play_run(value)
        if not ok:
            print(f"[RUN] {value!r} refused: {detail}")
            return jsonify({"status": "error", "message": detail}), 409
        return jsonify({"status": "ok", "run": value, "action": detail}), 200
    except Exception as e:
        print(f"[RUN] {value!r} failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/mode', methods=['GET'])
@require_control_session
def handle_mode():
    """The app's Mode buttons — one-shot arm actions.

    Was returning 404 on every press: the app sends
        /mode?value=Mode 1
    and no such route existed, so the buttons did nothing and nothing said
    why. The poses live in gestures.json under "modes", so they can be
    changed without touching code.
    """
    value = (request.args.get('value') or '').strip()
    if not value:
        return jsonify({"status": "error", "message": "no mode given"}), 400
    try:
        from gesture import gestures
        ok, detail = gestures.play_mode(value)
        if not ok:
            print(f"[MODE] {value!r} refused: {detail}")
            return jsonify({"status": "error", "message": detail}), 409
        return jsonify({"status": "ok", "mode": value,
                        "action": detail}), 200
    except Exception as e:
        print(f"[MODE] {value!r} failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/motion/status', methods=['GET'])
def handle_motion_status():
    """Combined state for the display's one motion control."""
    out = {"head": True, "hands": True, "head_moving": False,
           "gesturing": False, "user_active": False, "doa_available": False}
    try:
        from head_tracker import tracker
        out.update(tracker.motion_status())
    except Exception as e:
        out["head_error"] = str(e)
    try:
        from gesture import gestures
        g = gestures.status()
        out["hands"] = g.get("enabled", False)
        out["gesturing"] = g.get("gesturing", False)
        out["user_active"] = g.get("user_active", False)
    except Exception as e:
        out["hands_error"] = str(e)
    try:
        from sound_direction import doa_available
        out["doa_available"] = bool(doa_available())
    except Exception:
        pass
    return jsonify(out), 200


@app.route('/motion/enable', methods=['GET'])
def handle_motion_enable():
    """Head and hand motion, together or independently.

    Deliberately NOT gated by require_control_session, and neither is
    /gesture/enable. This is the switch somebody reaches for when the robot is
    suddenly surrounded by children — it has to work from any device without
    acquiring control first. Same reasoning as /estop.
    """
    def flag(name):
        v = request.args.get(name)
        if v is None:
            return None
        return v.lower() in ('1', 'true', 'yes', 'on')

    head, hands = flag('head'), flag('hands')
    changed = {}

    if head is not None:
        try:
            from head_tracker import tracker
            tracker.set_head_motion(head)
            changed['head'] = tracker.head_motion
        except Exception as e:
            print(f"[MOTION] head: {e}")
    if hands is not None:
        try:
            from gesture import gestures
            gestures.enabled = hands
            changed['hands'] = gestures.enabled
        except Exception as e:
            print(f"[MOTION] hands: {e}")

    # Both persisted, so the control survives a restart. In memory only they
    # would silently revert on the next boot, and someone who switched them
    # off for a school visit would find them back on in the morning.
    try:
        from settings import save_settings
        save = {}
        if 'head' in changed:
            save['head_motion_enabled'] = changed['head']
        if 'hands' in changed:
            save['gestures_enabled'] = changed['hands']
        if save:
            save_settings(save)
    except Exception as e:
        print(f"[MOTION] could not save: {e}")

    return jsonify(dict({"status": "ok"}, **changed)), 200


@app.route('/gesture/status', methods=['GET'])
def handle_gesture_status():
    try:
        from gesture import gestures
        return jsonify(gestures.status()), 200
    except Exception as e:
        return jsonify({"enabled": False, "error": str(e)}), 200


@app.route('/gesture/enable', methods=['GET'])
def handle_gesture_enable():
    """Turn speech gestures on or off.

    Deliberately NOT gated by require_control_session: this is the switch
    somebody reaches for when the robot is suddenly surrounded by children,
    and it must work from any device without acquiring control first. Same
    reasoning as /estop.
    """
    try:
        from gesture import gestures
        v = request.args.get('value', 'true').lower()
        gestures.enabled = v in ('1', 'true', 'yes', 'on')
        # Persisted, so it survives a restart.
        try:
            from settings import save_settings
            save_settings({"gestures_enabled": gestures.enabled})
        except Exception as e:
            print(f"[GESTURE] could not save the setting: {e}")
        print(f"[GESTURE] {'enabled' if gestures.enabled else 'DISABLED'}")
        return jsonify({"status": "ok", "enabled": gestures.enabled}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/track/status', methods=['GET'])
def handle_track_status():
    """Live tracking numbers. Worth watching while calibrating: error_x and
    error_y should shrink towards zero and STAY there. Alternating between
    positive and negative means the head is hunting — lower DAMPING. Shrinking
    but never arriving means CAM_FOV_H is too small."""
    return jsonify(head.status()), 200


# ── Arm imitation (STUB) ───────────────────────────────────
track_arm_enabled = False


@app.route('/track/arm', methods=['GET'])
@require_control_session
def handle_track_arm():
    """THIS IS A STUB. The route exists so the toggle has something to talk to
    and reports honestly, but nothing drives the arms from it yet.

    Making it real needs a pose pipeline that does not exist:
      1. a pose model (MediaPipe Pose, or a YOLO-pose variant) at 15-30fps
      2. shoulder/elbow/wrist angles from the landmarks, in 3D — a 2D
         projection cannot tell an arm reaching forward from one hanging down
      3. a mapping onto Luna's servo ranges, which is not one-to-one
      4. smoothing, or she jitters on every frame of landmark noise
      5. a rate limit, so a fast human movement does not become a fast servo
         movement — that one is safety, not polish

    There is also a memory question: an 8GB Orin already holding Whisper and
    YOLO may not have room for a pose model too.

    Returning available:false lets the display say so instead of pretending.
    """
    global track_arm_enabled
    en = request.args.get('enabled')
    if en is not None:
        want = en.lower() in ('1', 'true', 'yes', 'on')
        try:
            import mediapipe          # noqa: F401
            available = True
        except ImportError:
            available = False
        if want and not available:
            print("[MIMIC] requested but no pose model is installed")
            return jsonify({
                "status": "error", "available": False,
                "message": "Arm copying needs a pose model that is not "
                           "installed yet."}), 501
        track_arm_enabled = want
        print(f"[MIMIC] arm imitation {'on' if want else 'off'}")
    return jsonify({"status": "ok", "enabled": track_arm_enabled,
                    "available": False}), 200


# ── System ─────────────────────────────────────────────────
@app.route('/shutdown', methods=['GET'])
@require_control_session
def handle_shutdown():
    threading.Thread(target=do_shutdown, daemon=True).start()
    return jsonify({"status": "ok", "message": "Shutting down..."}), 200

@app.route('/restart', methods=['GET'])
@require_control_session
def handle_restart():
    def do_restart():
        stop_movement()
        send_to_esp32("HOME")
        time.sleep(3)
        # Same reasoning as do_shutdown.
        subprocess.run(["sudo", "/sbin/reboot"])
    threading.Thread(target=do_restart, daemon=True).start()
    return jsonify({"status": "ok", "message": "Restarting..."}), 200

# ── Start ──────────────────────────────────────────────────
def start_server():
    # Restore the saved volume. ALSA levels do not persist across a reboot on
    # this board, so without this the robot comes back at whatever the driver
    # defaults to — usually louder than anybody wanted — while the app's
    # slider still shows the old value.
    try:
        from settings import load_settings
        v = load_settings().get("volume")
        if v is not None:
            set_volume(int(v))
    except Exception as e:
        print(f"[VOLUME] could not restore the saved level: {e}")

    threading.Thread(target=esp32_reader,  daemon=True).start()
    threading.Thread(target=_move_keepalive, daemon=True).start()
    head.start(send_to_esp32)
    try:
        from settings import load_settings
        head.head_motion = bool(load_settings().get("head_motion_enabled", True))
        print(f"[TRACK] head motion {'on' if head.head_motion else 'OFF'} "
              f"(from settings)")
    except Exception as e:
        print(f"[TRACK] could not read head_motion_enabled: {e}")
    try:
        from gesture import gestures
        gestures.start(send_to_esp32)
    except Exception as e:
        print(f"[GESTURE] not started: {e}")
    print(f"[MOVE] keepalive {MOVE_KEEPALIVE_SEC}s, "
          f"hold timeout {MOVE_HOLD_TIMEOUT}s")
    try:
        # threaded=True matters: without it Flask serves one request at a
        # time, so a slow /position call can delay the button-release event
        # and the robot keeps driving.
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except OSError as e:
        # Do not carry on half-dead. Letting this bubble out of the thread
        # meant the robot printed "Server started on port 5000" while serving
        # nothing — the app and display unreachable, one line in the log.
        print(f"[HTTP] *** CANNOT BIND PORT 5000: {e} ***")
        print("[HTTP] The app and display will NOT work. Find the process:")
        print("[HTTP]     sudo lsof -i :5000")
        print("[HTTP] Note: systemctl stop does not kill an orphaned child.")
        os._exit(1)          # let systemd restart us rather than run broken

if __name__ == '__main__':
    start_server()
