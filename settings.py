import json, os

# Self-locating rather than hardcoded to /home/groot: the same code runs on
# units built under other accounts, and a wrong path here fails silently —
# load_settings() just returns the defaults and every saved setting is lost.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "settings.json")

DEFAULTS = {
    "robot_name": "Luna",
    "welcome_speech": "Hello, I am Luna, your robot assistant",
    "language": "en",
    "voice": "female",
    "chatgpt_enabled": True,
    # Arm movement while she speaks. Stored here so the toggle survives a
    # restart — in memory only, it would silently revert on every reboot and
    # someone who turned it off for a school visit would find it back on.
    "gestures_enabled": True,
    # Turning the head towards whoever spoke. Same reasoning as gestures:
    # stored so the toggle survives a restart.
    "head_motion_enabled": True,
}

def load_settings():
    """Settings, with any missing key filled from DEFAULTS.

    The merge matters: a settings.json written before a key existed would
    otherwise make load_settings().get("gestures_enabled") return None, and
    every new setting would appear absent on already-built robots.
    """
    s = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                s.update(json.load(f))
        except Exception as e:
            print(f"[SETTINGS] {SETTINGS_FILE} unreadable, using defaults: {e}")
    return s

def save_settings(new_settings):
    s = load_settings()
    s.update(new_settings)
    # Atomic: write beside the file then rename. A crash or a power cut
    # part-way through a plain write leaves truncated JSON, and then every
    # setting on the robot reverts to defaults at once.
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)
    return s
