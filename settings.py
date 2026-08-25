import json, os

SETTINGS_FILE = "/home/groot/pi_assistant/settings.json"

DEFAULTS = {
    "robot_name": "Ben",
    "welcome_speech": "Hello, I am Ben, your robot assistant",
    "language": "en",
    "voice": "male",
    "chatgpt_enabled": True
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return dict(DEFAULTS)

def save_settings(new_settings):
    s = load_settings()
    s.update(new_settings)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(s, f, indent=2)
    return s
