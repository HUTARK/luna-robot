"""
control_session.py — HUTARK T-24 local control-session lock.

Enforces "one controller at a time" entirely on the Pi, in memory, with
zero Supabase/Internet dependency — required because the robot may be
completely offline. See README.md for the full design rationale.

Design summary:
- A single global slot (`_session`). Only one active controller is possible
  at a time, by construction — there is nothing to compare against; if the
  slot is occupied by someone else, acquire() simply refuses.
- No disk persistence. A Pi/server restart clears this module's state back
  to "no active session" as a side effect of the process restarting — this
  is intentional (see README: "Pi restart" section), not an oversight.
- CONTROL_SESSION_TIMEOUT_SECONDS = 15. If the controlling app stops
  sending heartbeats (crash, force-close, Wi-Fi drop, phone powered off)
  for longer than this, the session is treated as abandoned and the next
  acquire() or heartbeat() call clears it lazily — no background thread
  needed. See README for why 15s was chosen.
- client_id (the Android app's per-install id, already generated client-side
  by getOrCreateAppId() in storage.js) lets the SAME app instance silently
  re-acquire/refresh its own session — e.g. after a brief network blip —
  without a stale session it owns permanently locking itself out.
"""
import threading
import time
import uuid

CONTROL_SESSION_TIMEOUT_SECONDS = 15

_lock = threading.Lock()
_session = None  # {"session_id","app_id","client_id","last_heartbeat"} or None


def _expire_if_stale_locked():
    """Must be called while holding _lock."""
    global _session
    if _session and (time.time() - _session["last_heartbeat"]) > CONTROL_SESSION_TIMEOUT_SECONDS:
        _session = None


def get_active_session():
    """Returns a copy of the active session dict, or None. Auto-expires stale sessions."""
    global _session
    with _lock:
        _expire_if_stale_locked()
        return dict(_session) if _session else None


def acquire(app_id, client_id):
    """
    Returns (True, session_dict) on success, (False, current_session_dict_or_None) on refusal.
    Idempotent for the same client_id: a client that already holds the session
    (or whose prior session just expired) can re-acquire/refresh freely.
    """
    global _session
    with _lock:
        _expire_if_stale_locked()
        if _session is None:
            _session = {
                "session_id": str(uuid.uuid4()),
                "app_id": app_id,
                "client_id": client_id,
                "last_heartbeat": time.time(),
            }
            return True, dict(_session)
        if _session["client_id"] == client_id:
            _session["last_heartbeat"] = time.time()
            return True, dict(_session)
        return False, dict(_session)


def heartbeat(session_id):
    """Returns True if session_id matches the active session (and refreshes it)."""
    global _session
    with _lock:
        _expire_if_stale_locked()
        if _session and _session["session_id"] == session_id:
            _session["last_heartbeat"] = time.time()
            return True
        return False


def release(session_id):
    """Returns True if session_id matched the active session (and cleared it)."""
    global _session
    with _lock:
        _expire_if_stale_locked()
        if _session and _session["session_id"] == session_id:
            _session = None
            return True
        return False


def time_remaining():
    """Seconds left before the active session auto-expires, or None if no active session."""
    with _lock:
        _expire_if_stale_locked()
        if not _session:
            return None
        remaining = CONTROL_SESSION_TIMEOUT_SECONDS - (time.time() - _session["last_heartbeat"])
        return max(0, round(remaining, 1))
