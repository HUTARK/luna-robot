"""
identity.py — HUTARK factory/QC robot identity (read-only at runtime).

robot_identity.json is written ONCE, at manufacturing/QC time, by whatever
tooling performs that step (not this file). This module only ever READS it.
No route in server.py writes to this file — see robot_identity.example.json
and README.md for how to populate it.

Deliberately a separate file/module from settings.py: settings.json holds
customer-editable data (robot_name, welcome_speech, voice, chatgpt_enabled);
robot_identity.json holds the permanent factory identity. They must never
share a source, so a customer's /save-audio or /settings request can never
change what /status reports as robot_id/robot_type/tier.

If robot_identity.json is missing or invalid, load_identity() returns None
and /status omits the identity fields entirely — never a placeholder value,
which could otherwise be mistaken by the app for a genuine identity.
"""
import json, os

# Self-locating rather than hardcoded to /home/groot: the same code runs on
# units built under different usernames, and a wrong path here fails silently
# — load_identity() returns None and /status simply omits the identity, which
# looks exactly like a missing identity file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTITY_FILE = os.path.join(BASE_DIR, "robot_identity.json")

# Cached after the first successful read. The identity is written once at
# manufacturing and never changes at runtime, while /status is polled every
# few seconds by both the display and the app — re-reading and re-parsing the
# file on each of those is pointless work.
#
# A FAILED read is deliberately NOT cached: if the file is created after the
# service starts, the next call picks it up without a restart, which matters
# when someone is installing this by hand on a live unit.
_cache = None


def load_identity():
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(IDENTITY_FILE):
        return None
    try:
        with open(IDENTITY_FILE, "r") as f:
            data = json.load(f)
    except Exception as e:
        # Say so once rather than failing silently. An invalid identity file
        # and a missing one produce the same /status response, and without a
        # log line there is no way to tell which you have.
        print(f"[IDENTITY] {IDENTITY_FILE} is unreadable: {e}")
        return None
    if not isinstance(data, dict):
        return None
    robot_id = data.get("robot_id")
    if not isinstance(robot_id, str) or not robot_id.strip():
        return None
    _cache = {
        "robot_id": robot_id.strip(),
        "robot_type": str(data.get("robot_type", "")).strip() or "unknown",
        "tier": str(data.get("tier", "")).strip() or "unknown",
        "protocol_version": str(data.get("protocol_version", "1")).strip() or "1",
    }

    # Optional manufacturing fields. Included only when present and non-empty,
    # so a unit written before these existed keeps reporting exactly the four
    # original keys and nothing downstream has to change.
    for key in ("robot_model", "serial_number"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            _cache[key] = val.strip()
    print(f"[IDENTITY] {_cache['robot_id']} "
          f"({_cache['robot_type']}/{_cache['tier']}, "
          f"protocol {_cache['protocol_version']})")
    return _cache
