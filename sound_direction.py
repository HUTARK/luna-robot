"""
sound_direction.py — which way the voice came from, from the ReSpeaker array.

    from sound_direction import last_direction, doa_available
    ang = last_direction()          # degrees 0-359, or None

═══════════════════════════════════════════════════════════════════════
HOW THIS WORKS, AND WHEN TO READ IT
═══════════════════════════════════════════════════════════════════════
The XVF DSP on the ReSpeaker computes a direction of arrival continuously and
exposes it over USB HID as DOAANGLE. It is not a stream — it is a register
holding the bearing of the MOST RECENT speech the array heard.

That is what makes this usable without any threading or gating: read it just
after someone has finished asking a question, and it points at them. No need
to sample continuously, no need to decide which of several sounds mattered,
and no risk of the head chasing every door slam.

Reading it at the wrong moment is the trap. Sample it while Luna is talking
and it points at her own speaker — the speaker is mounted on the robot, is the
loudest thing in the room, and AEC does not work on this build (measured: 1-2.5
dB of cancellation, where useful is 20-30). So: read after the person speaks,
before Luna answers. In practice that means the top of answer_local.

═══════════════════════════════════════════════════════════════════════
WHAT THE NUMBER MEANS, AND WHY IT NEEDS CALIBRATING
═══════════════════════════════════════════════════════════════════════
DOAANGLE is measured from the array's own zero, which depends entirely on how
the board is screwed down. MOUNT_OFFSET_DEG turns it into a bearing relative
to where the head points at rest, and it has to be measured once per robot —
see the procedure at the bottom of this file.

Accuracy is roughly +-15 degrees in a quiet room and worse in a reverberant
one, because the array is 4 microphones a few centimetres apart. That is fine
for "turn towards the person who spoke" and useless for anything finer, which
is why this only ever nudges the head to face a general direction and the
camera takes over from there.
"""

import time

# ── Mounting ──────────────────────────────────────────────────────────
# Degrees to subtract from DOAANGLE so that 0 means "straight ahead of the
# head at rest". MEASURE THIS — see the procedure at the bottom. Wrong, and
# the head turns confidently to the wrong place, which looks far worse than
# not turning at all.
MOUNT_OFFSET_DEG = 0.0

# True if a larger DOAANGLE means further to the robot's LEFT. Depends on
# which way up the board is fitted. Flip it if the head turns the wrong way.
DOA_CCW = True

# Bearings to ignore, as (from, to) in corrected degrees. The robot's own
# speaker sits at a fixed bearing and is the loudest thing in the room; if the
# array latches onto it the head turns to stare at its own chest. Fill this in
# once you know where the speaker points.
IGNORE_SECTORS = []           # e.g. [(160, 200)] for a speaker directly behind

# Beyond this the head cannot help anyway, and turning that far on a bearing
# accurate to +-15 degrees is a guess. Better to stay put.
MAX_TURN_DEG = 75.0

# Ignore a reading that has not changed since last time: the register holds its
# previous value, so the same number twice usually means nothing new was heard,
# not that the person spoke twice from an identical spot.
STALE_AFTER = 6.0             # seconds

_tuning = None
_last_raw = None
_last_at = 0.0
_unavailable_reason = ""


def _open():
    """The ReSpeaker's tuning interface, or None.

    Imported lazily and never allowed to raise: this is a nice-to-have, and a
    missing mic or a missing udev rule must not stop the robot from answering
    questions.
    """
    global _tuning, _unavailable_reason
    if _tuning is not None:
        return _tuning
    if _unavailable_reason:
        return None                # already tried and failed
    try:
        import sys, os, glob
        # Same checkout used to set AGCMAXGAIN. Added to sys.path rather than
        # copied, so there is one version of tuning.py on the robot.
        #
        # NOT expanduser("~"). The service runs as root (User=root in
        # piassistant.service), so "~" becomes /root and the checkout in
        # /home/groot is invisible — which is why this worked from a shell and
        # failed inside the service with ModuleNotFoundError. Searching the
        # real home directories directly avoids depending on who is running.
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.path.dirname(here), "usb_4_mic_array"),
            "/opt/usb_4_mic_array",
        ]
        # Every user's home, so this works whichever account the robot is
        # built under and whoever the service runs as.
        for home in glob.glob("/home/*") + ["/root"]:
            candidates.append(os.path.join(home, "usb_4_mic_array"))
            candidates.append(os.path.join(home, "src", "usb_4_mic_array"))
        found = None
        for p in candidates:
            if os.path.isfile(os.path.join(p, "tuning.py")):
                found = p
                if p not in sys.path:
                    sys.path.insert(0, p)
                break
        if found is None:
            _unavailable_reason = (
                "tuning.py not found. Looked in: " + ", ".join(candidates[:4])
                + " ... Clone it with:  git clone "
                "https://github.com/respeaker/usb_4_mic_array")
            print(f"[DOA] {_unavailable_reason}")
            return None
        print(f"[DOA] using {found}/tuning.py")
        from tuning import find as find_tuning
        dev = find_tuning()
        if dev is None:
            _unavailable_reason = ("ReSpeaker tuning interface not found. "
                                   "Check the udev rule and replug the board.")
            print(f"[DOA] {_unavailable_reason}")
            return None
        _tuning = dev
        print("[DOA] ReSpeaker direction-of-arrival available")
        return _tuning
    except Exception as e:
        _unavailable_reason = f"{type(e).__name__}: {e}"
        print(f"[DOA] unavailable — {_unavailable_reason}")
        return None


def doa_available():
    return _open() is not None


def _corrected(raw):
    """Raw DOAANGLE to a signed bearing: negative left, positive right,
    0 straight ahead."""
    a = (float(raw) - MOUNT_OFFSET_DEG) % 360.0
    if a > 180.0:
        a -= 360.0             # -180..180
    if DOA_CCW:
        a = -a                 # make positive mean the robot's right
    return a


def _ignored(bearing):
    for lo, hi in IGNORE_SECTORS:
        lo_c, hi_c = _corrected(lo), _corrected(hi)
        if lo_c <= bearing <= hi_c:
            return True
    return False


def last_direction():
    """Bearing of the most recent speech, in degrees: negative left, positive
    right, 0 ahead. None when unavailable, stale, ignored or too far.

    Deliberately returns None rather than a best guess. A confident turn to
    the wrong place looks much worse than not turning.
    """
    global _last_raw, _last_at
    dev = _open()
    if dev is None:
        return None
    try:
        raw = dev.read("DOAANGLE")
    except Exception as e:
        print(f"[DOA] read failed: {e}")
        return None

    now = time.time()
    # An unchanged register usually means nothing new was heard. Two people
    # speaking from exactly the same bearing is possible but rare, and acting
    # on a stale value means turning towards where someone stood a minute ago.
    if raw == _last_raw and (now - _last_at) < STALE_AFTER:
        return None
    _last_raw, _last_at = raw, now

    bearing = _corrected(raw)
    if _ignored(bearing):
        print(f"[DOA] {raw} deg is in an ignored sector — probably our own "
              f"speaker")
        return None
    if abs(bearing) > MAX_TURN_DEG:
        print(f"[DOA] {bearing:+.0f} deg is beyond the {MAX_TURN_DEG:.0f} "
              f"degree limit — staying put")
        return None
    print(f"[DOA] voice at {bearing:+.0f} deg (raw {raw})")
    return bearing


def voice_activity():
    """1 while the DSP thinks someone is speaking, 0 otherwise, None if
    unavailable. Not used for turning — the DOA register is enough — but
    useful when checking the mount offset."""
    dev = _open()
    if dev is None:
        return None
    try:
        return int(dev.read("VOICEACTIVITY"))
    except Exception:
        return None


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════
    # CALIBRATION
    #
    # Guided rather than a raw dump, because reading the stream by eye is
    # genuinely hard: DOAANGLE only changes when the array hears something and
    # HOLDS its last value otherwise, so most of what scrolls past is stale.
    # This samples fast, keeps only the values that CHANGED — a change means
    # new sound — and reports the median of them, which throws out the odd
    # reflection without being pulled about by it.
    #
    # VOICEACTIVITY is ignored. On this board it reads 0 even while the angle
    # is tracking a voice, so gating on it would discard every good sample.
    #
    #   python3 sound_direction.py            guided, three positions
    #   python3 sound_direction.py raw        the old live stream
    # ═══════════════════════════════════════════════════════════════════
    import sys
    import statistics

    if not doa_available():
        raise SystemExit("DOA not available — see the message above")
    dev = _open()

    if len(sys.argv) > 1 and sys.argv[1] == "raw":
        print("raw stream. Ctrl-C to stop.\n")
        try:
            while True:
                print(f"  raw {dev.read('DOAANGLE'):4}   "
                      f"corrected {_corrected(dev.read('DOAANGLE')):+7.1f} deg")
                time.sleep(0.3)
        except KeyboardInterrupt:
            print()
        raise SystemExit

    def sample(seconds=6.0):
        """Median of the angles seen while the value was CHANGING."""
        seen, last = [], None
        end = time.time() + seconds
        while time.time() < end:
            try:
                raw = int(dev.read("DOAANGLE"))
            except Exception as e:
                print(f"    read failed: {e}")
                return None, 0
            if raw != last:
                seen.append(raw)
                last = raw
                print(f"    {raw:4}", end="", flush=True)
            time.sleep(0.12)
        print()
        if len(seen) < 3:
            return None, len(seen)
        # Circular median: angles wrap at 360, so a plain median of 350 and 10
        # gives 180 — the exact opposite direction.
        import math
        sx = sum(math.cos(math.radians(a)) for a in seen)
        sy = sum(math.sin(math.radians(a)) for a in seen)
        return math.degrees(math.atan2(sy, sx)) % 360.0, len(seen)

    print("=" * 62)
    print("  ReSpeaker direction calibration")
    print("=" * 62)
    print("\n  Luna must stay QUIET throughout — her own speaker is the")
    print("  loudest thing in the room and the array will lock onto it.")
    print("  Stand about a metre away, and keep clear of walls.\n")

    results = {}
    for label, hint in (
        ("FRONT", "directly in front of the robot, where the head points"),
        ("RIGHT", "to the robot's RIGHT"),
        ("LEFT",  "to the robot's LEFT"),
    ):
        input(f"  Stand {hint}, then press Enter...")
        print(f"  Talk continuously for 6 seconds — count out loud:")
        ang, n = sample()
        if ang is None:
            print(f"    only {n} changes — talk LOUDER and keep going, "
                  f"then run this again\n")
        else:
            results[label] = ang
            print(f"  -> {label}: {ang:.0f} deg  ({n} samples)\n")

    print("=" * 62)
    if "FRONT" not in results:
        raise SystemExit("  No usable FRONT reading. Nothing to calibrate.")

    front = results["FRONT"]
    print(f"\n  MOUNT_OFFSET_DEG = {front:.0f}")

    if "RIGHT" in results:
        # Which way the numbers run: is RIGHT clockwise or anticlockwise from
        # FRONT? Compared as a signed difference so the wrap at 360 does not
        # invert the answer.
        d = (results["RIGHT"] - front + 540) % 360 - 180
        ccw = d < 0
        print(f"  DOA_CCW = {ccw}")
        print(f"\n  (RIGHT read {results['RIGHT']:.0f}, which is {d:+.0f} deg "
              f"from FRONT)")
        if abs(d) < 20:
            print("  WARNING: RIGHT is barely different from FRONT. Either you")
            print("  did not move far enough, or the array cannot separate")
            print("  those directions. Stand further to the side and redo it.")
    else:
        print("  DOA_CCW = could not tell — no RIGHT reading")

    if "LEFT" in results:
        dl = (results["LEFT"] - front + 540) % 360 - 180
        print(f"  (LEFT read {results['LEFT']:.0f}, {dl:+.0f} deg from FRONT)")
        if "RIGHT" in results:
            dr = (results["RIGHT"] - front + 540) % 360 - 180
            if (dl < 0) == (dr < 0):
                print("\n  WARNING: LEFT and RIGHT came out on the SAME side of")
                print("  FRONT. That means the readings are not trustworthy —")
                print("  usually a reflection off a nearby wall. Move the robot")
                print("  away from surfaces and try again.")

    print("\n  Put those two values at the top of sound_direction.py,")
    print("  then re-run and check FRONT reads close to 0.")
    print()
