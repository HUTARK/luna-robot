from flask import Flask, request, jsonify
from flask_cors import CORS
import os, threading, serial, json, time, subprocess, base64

BASE_DIR   = "/home/groot/pi_assistant"
POSES_FILE = f"{BASE_DIR}/poses.json"
QA_FILE    = f"{BASE_DIR}/qa_store.json"
FACES_DIR  = f"{BASE_DIR}/faces"
FACES_DB   = f"{BASE_DIR}/faces_db.json"

os.makedirs(FACES_DIR, exist_ok=True)

# ── Flask App ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

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

def send_to_esp32(command: str):
    global esp32
    try:
        if esp32 and esp32.is_open:
            esp32.write((command + '\n').encode())
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
def set_volume(value):
    """Set USB speaker volume 0-100"""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        card = None
        for line in result.stdout.splitlines():
            if 'usb' in line.lower() and line.startswith('card'):
                card = line.split(':')[0].replace('card ', '').strip()
                break
        if card is None:
            print("[VOLUME] No USB speaker card found")
            return False
        for ctl in ['Speaker', 'PCM', 'Master', 'Headphone']:
            r = subprocess.run(
                ['amixer', '-c', card, 'sset', ctl, f'{value}%'],
                capture_output=True
            )
            if r.returncode == 0:
                print(f"[VOLUME] {ctl} set to {value}% on card {card}")
                return True
        return False
    except Exception as e:
        print(f"[VOLUME] Error: {e}")
        return False

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
    print("[SHUTDOWN] LATCH:OFF sent")
    time.sleep(1)
    subprocess.Popen(["bash", "-c", "sleep 2 && sudo /sbin/shutdown -h now"])
    os._exit(0)

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
                send_to_esp32(f"POS:{part}:{value}:{hand}")
                time.sleep(0.05)
            time.sleep(1)
    print("[LOOP] Stopped")

# ══════════════════════════════════════════════════════════
#  ENDPOINTS — Ben Max API
# ══════════════════════════════════════════════════════════

# ── Status ─────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ok",
        "battery": 100,
        "wifi": True,
        "connected": True,
        "robot": "ben",
        "tier": "max"
    }), 200

# ── Movement ───────────────────────────────────────────────
@app.route('/move', methods=['GET'])
def handle_move():
    direction = request.args.get('dir', 'stop')
    # App sends 'back' — ESP32 expects 'backward'
    if direction == 'back':
        direction = 'backward'
    print(f"[MOVE] {direction}")
    send_to_esp32(f"MOVE:{direction}")
    return jsonify({"status": "ok"}), 200

@app.route('/speed', methods=['GET'])
def handle_speed():
    value = request.args.get('value', '50')
    print(f"[SPEED] {value}")
    send_to_esp32(f"SPEED:{value}")
    return jsonify({"status": "ok"}), 200

@app.route('/estop', methods=['GET'])
def handle_estop():
    print("[ESTOP] EMERGENCY STOP")
    send_to_esp32("MOVE:stop")
    send_to_esp32("MOVE:stop")
    send_to_esp32("MOVE:stop")
    global loop_running
    loop_running = False
    return jsonify({"status": "ok", "message": "Emergency stop"}), 200

@app.route('/obstacle_avoidance', methods=['GET'])
def handle_obstacle_avoidance():
    global obstacle_avoidance_enabled
    value = request.args.get('value', 'up')
    obstacle_avoidance_enabled = (value == 'up')
    send_to_esp32(f"HARDWARE:{'ON' if obstacle_avoidance_enabled else 'OFF'}")
    print(f"[OBSTACLE] {'Enabled' if obstacle_avoidance_enabled else 'Disabled'}")
    return jsonify({"status": "ok", "obstacle_avoidance": obstacle_avoidance_enabled}), 200

# ── Servo Joints ───────────────────────────────────────────
@app.route('/hand', methods=['GET'])
def handle_hand():
    value = request.args.get('value', 'left')
    print(f"[HAND] {value}")
    send_to_esp32(f"HAND:{value}")
    return jsonify({"status": "ok"}), 200

@app.route('/position', methods=['GET'])
def handle_position():
    part  = request.args.get('part', '')
    value = request.args.get('value', '1000')
    hand  = request.args.get('hand', 'left')

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

    print(f"[POSITION] {part} -> {value} ({hand})")
    send_to_esp32(f"POS:{part}:{value}:{hand}")
    return jsonify({"status": "ok"}), 200

@app.route('/home', methods=['GET'])
def handle_home():
    print("[HOME]")
    send_to_esp32("HOME")
    return jsonify({"status": "ok"}), 200

# ── Loop / Sequencer ───────────────────────────────────────
@app.route('/save_pose', methods=['GET'])
def handle_save_pose():
    pos  = request.args.get('pos', str(int(time.time())))
    hand = request.args.get('hand', 'left')
    parts = ['head','headLR','headUD','lateral','shoulder','forearm',
             'elbow','wrist','fingers','thumb','index','middle','ring','pinky']
    pose_data = {'hand': hand, 'servos': {}}
    for part in parts:
        val = request.args.get(part)
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
def handle_loop_start():
    global loop_running
    loop_running = True
    threading.Thread(target=run_loop, daemon=True).start()
    print("[LOOP] START")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_stop', methods=['GET'])
def handle_loop_stop():
    global loop_running
    loop_running = False
    print("[LOOP] STOP")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_undo', methods=['GET'])
def handle_loop_undo():
    poses = load_poses()
    if poses:
        last_key = max(poses.keys(), key=lambda x: int(x))
        del poses[last_key]
        save_poses(poses)
        print(f"[LOOP] Removed {last_key}")
    return jsonify({"status": "ok"}), 200

@app.route('/loop_delete', methods=['GET'])
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
    face_tracking_enabled = (value == 'true')
    print(f"[FACE-TRACK] {'ON' if face_tracking_enabled else 'OFF'}")
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

# ── System ─────────────────────────────────────────────────
@app.route('/shutdown', methods=['GET'])
def handle_shutdown():
    threading.Thread(target=do_shutdown, daemon=True).start()
    return jsonify({"status": "ok", "message": "Shutting down..."}), 200

@app.route('/restart', methods=['GET'])
def handle_restart():
    def do_restart():
        send_to_esp32("MOVE:stop")
        send_to_esp32("HOME")
        time.sleep(3)
        subprocess.Popen(["bash", "-c", "sleep 2 && sudo /sbin/reboot"])
        os._exit(0)
    threading.Thread(target=do_restart, daemon=True).start()
    return jsonify({"status": "ok", "message": "Restarting..."}), 200

# ── Start ──────────────────────────────────────────────────
def start_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    start_server()
