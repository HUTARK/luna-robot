"""
head_tracker.py — head aiming for Luna. Camera mounted ON the head.

    from head_tracker import tracker
    tracker.start(send_to_esp32)     # once, in start_server()
    tracker.preview(True)            # camera on, faces boxed, head STILL
    tracker.select_at(x, y)          # lock onto the face at that point
    tracker.clear_lock()             # let go and glide home

═══════════════════════════════════════════════════════════════════════
NOTHING MOVES UNTIL A FACE IS PICKED
═══════════════════════════════════════════════════════════════════════
There are four modes and only two of them move the head:

    off      camera released
    preview  camera live, faces boxed, HEAD DOES NOT MOVE   <- the default
    lock     following the one face that was tapped
    face     following the nearest face (opt-in, not the default)

The display's tracking toggle used to call /face_tracking?value=true, which
went straight to "face" and started chasing whoever the detector happened to
see. With a camera on the head that reads as the robot latching onto
strangers, furniture, whatever crosses the frame. Opening the camera view now
means preview, and only a tap starts the following.

═══════════════════════════════════════════════════════════════════════
SMOOTHNESS: TWO LOOPS, NOT ONE
═══════════════════════════════════════════════════════════════════════
The old loop measured and moved in the same breath: eight times a second it
sent one POS: per axis and the servo slewed there flat out. Eight discrete
lunges a second is exactly what jerky looks like, and no amount of tuning the
gain fixes it — the steps are the problem, not their size.

So there are now two threads:

  VISION   ~8 Hz. Looks, decides where the head SHOULD point, sets a goal.
           Never talks to the servos.
  MOTION   ~30 Hz. Eases the head towards that goal under a speed and an
           acceleration limit, and is the only thing that emits POS:.

The motion thread ramps velocity up and back down (a trapezoidal profile), so
every move starts and ends gently instead of snapping. MAX_SPEED and ACCEL
are the two knobs; the loop's gain no longer has anything to do with how the
movement feels.

═══════════════════════════════════════════════════════════════════════
WHY A HEAD-MOUNTED CAMERA MAKES THE LOOP DELICATE
═══════════════════════════════════════════════════════════════════════
Aiming is simple: the target's offset from the centre of the frame IS the
pointing error — centre it and the head is looking at it.

The trap is that moving the head moves the camera, so a measurement is only
meaningful together with where the head was pointing WHEN THE FRAME WAS
TAKEN. Correcting from wherever the head has drifted to since is how these
loops end up hunting. Two consequences run through this file:

  * a goal is `pose at capture + correction`, never `pose now + correction`.
  * the lock's remembered position is shifted by however far the head
    actually turned between frames — measured, not guessed.

With continuous motion the head is essentially always moving, so the old
trick of discarding frames captured during a move would discard everything.
The pose history replaces it.
"""

import math
import threading
import time


def _cv2():
    """cv2 or None. Wrapped because this module must still load and report a
    useful status when OpenCV is broken — and on this robot it HAS been: a
    NumPy 2 / NumPy 1 mismatch left cv2 importable but with most attributes
    missing, which surfaced as 'no attribute CascadeClassifier' and looked
    like a headless build rather than a failed import."""
    try:
        import cv2
        return cv2
    except Exception:
        return None


# ── Camera geometry ───────────────────────────────────────────────────
# The lens's field of view, degrees. The one number that must be roughly
# right; everything else derives from it.
#     typical USB webcam   ~60      wide-angle module  ~90-120
# Head falls short and creeps -> raise it.  Overshoots -> lower it.
CAM_FOV_H = 60.0
CAM_FOV_V = 45.0

UNITS_PER_DEG = 2000.0 / 180.0      # head sliders span 0-2000 over 180deg
SPAN_H = CAM_FOV_H * UNITS_PER_DEG  # app units across the frame, horizontally
SPAN_V = CAM_FOV_V * UNITS_PER_DEG

DAMPING  = 0.45      # fraction of the error corrected per decision
MAX_STEP = 125       # biggest single correction, app units (~11 degrees)

# The head holds still while the face is anywhere inside this fraction of the
# frame from the centre, and only then moves to re-centre it. A person is never
# perfectly still — they breathe, shift weight, gesture — and a plain deadband
# on a face detector still twitches, because the detector's own box jitters by
# a few pixels every frame. A hold zone means the head moves when the PERSON
# moves, not when the measurement wobbles. Bigger = calmer and lazier.
HOLD_ZONE = 0.11

# Aim where they are GOING, not only where they were. LEAD is roughly the
# delay being compensated: one decision (0.12s) plus the camera's own lag.
# Too high and the head swings past people when they stop; 0 disables it.
LEAD       = 0.28    # seconds of lead
VEL_SMOOTH = 0.45    # 0-1, how quickly the speed estimate follows the data
MIN_MOVE   = 10      # app units (~0.9 deg) — smaller goals are not worth it

# ── How the movement FEELS ────────────────────────────────────────────
# These, not DAMPING, are what to touch if the head looks jerky or sluggish.
MOTION_HZ  = 30.0    # servo update rate
MAX_SPEED  = 600.0   # app units per second (~54 deg/s) — cruising speed
ACCEL      = 2000.0  # units/s^2 — how briskly it gets up to speed and stops
EMIT_MIN   = 2       # don't send a POS: for less than this many units

# How stale a frame is by the time we see it. The pose used for a measurement
# is looked up this far back. Too low and the head hunts slightly; too high
# and it lags. 0.08-0.15 suits most USB cameras.
CAM_LATENCY = 0.10

# ── Mounting ──────────────────────────────────────────────────────────
# If the head turns AWAY from the face, flip the relevant one. This is the
# FIRST thing to check and it is one character.
INVERT_LR = False
INVERT_UD = True                    # image Y grows downward

# Watch whether the head's own movement shifts the picture the way the flags
# above predict, and correct them if it does not. A wrong flag is the classic
# symptom of "it lurches off in one direction the moment I select someone":
# the head moves the wrong way, the face slides FURTHER from centre, and the
# next correction is bigger. Nothing about that self-corrects, so it runs to
# the end stop.
AUTO_INVERT   = True
INV_SAMPLES   = 5      # evidence needed before flipping anything
INV_MIN_SHIFT = 0.035  # ignore head movements too small to judge by

# Until an axis has proved it moves the right way, the head may not stray more
# than this far from where it was when you picked the face. If the wiring IS
# backwards, every correction makes the error worse and the next one is bigger
# — without a leash the head walks itself into the end stop within a couple of
# seconds, which is the "it suddenly took off downwards" everyone reports. The
# leash comes off as soon as the direction is confirmed, which takes about a
# second of normal following.
PROBATION = 300        # app units (~27 degrees)

# Direction calibration: how far to nudge each axis, how long to let it settle,
# and how much the picture must visibly move for the result to mean anything.
CAL_STEP      = 90     # app units (~8 degrees)
CAL_SETTLE    = 0.45   # seconds after the move before trusting a frame
CAL_MIN_SHIFT = 0.03   # frame fractions

LR_MIN, LR_MAX, LR_HOME = 0, 2000, 1000
UD_MIN, UD_MAX, UD_HOME = 0, 2000, 1000

# ── Loop ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════
# CAPTURE RUNS IN ITS OWN THREAD — do not merge it back into the vision loop
#
# Grab, encode and detect in one pass costs ~200ms with InsightFace, which
# gives a 3-4fps preview and, worse, a LATE correction: the pose is recorded
# when the frame is grabbed but applied after detection, so the head has
# already moved and every correction is computed against a stale position.
# The error accumulates in one direction — the head drifts with nobody moving,
# loses the face, and homes.
#
# Damping cannot fix that. Damping controls how hard the loop pushes; it
# cannot help when the feedback itself is late.
#
# So: capture at CAPTURE_HZ in its own thread, recording the pose at the
# moment of capture. The vision thread takes the newest frame and its pose
# together, so the correction is always computed against where the head
# actually was when the shutter opened.
# ══════════════════════════════════════════════════════════════════════
CAPTURE_HZ      = 20.0   # capture thread: frames for the preview
PREVIEW_WIDTH   = 640    # downscale before ENCODING only. Detection still
                         # uses the full frame; this is the dominant cost in
                         # the capture thread at 20fps.
PERIOD          = 0.12   # vision thread: ~8 looks a second
LOST_AFTER      = 1.5    # seconds without the face before it reads as lost
HOME_AFTER      = 2.0    # seconds without the face before returning home

# There is deliberately no "give up after N seconds". A selection is a
# standing instruction: the tracker holds it, goes home while they are away,
# and picks them up again whenever they reappear — until Clear target. It used
# to drop the lock after 25 seconds, which meant stepping out of the room for
# half a minute silently ended the follow.
PREVIEW_IDLE    = 90.0   # preview with nobody watching releases the camera

# How far a face may move between frames and still be the same face, as a
# fraction of the frame.
MATCH_RADIUS = 0.22
# ...growing by this much per second while the face is missing, up to the cap.
# A bound matters more than the numbers: an unbounded search is what makes a
# tracker "follow things I did not select".
RADIUS_GROWTH    = 0.18
MATCH_RADIUS_MAX = 0.40
# A face does not halve or double in size between frames.
SIZE_RATIO_MIN = 0.45
SIZE_RATIO_MAX = 2.2

# ── Recognising the selected person again ─────────────────────────────
# Colour fingerprint, used when the detector gives no identity of its own.
SIG_BINS_H, SIG_BINS_S = 30, 32
SIG_MATCH   = 0.55     # minimum score to call it the same person
# A face in the right PLACE is still rejected if it looks like somebody else.
# Position alone will happily adopt whoever walks through the spot the selected
# person just left — which is most of what "it follows people I did not pick"
# actually is. Continuity earns some leniency, so this bar is lower than the
# one for recognising them from scratch, but it is not absent.
SIG_VETO    = 0.40
SIG_MARGIN  = 0.10     # ...and it must beat the runner-up by this much
SIG_REFRESH = 0.06     # how fast the stored fingerprint drifts with the light
# They may have walked nearer or further while out of view, but not by a lot.
REACQ_SIZE_MIN, REACQ_SIZE_MAX = 0.35, 2.8

# InsightFace gives an embedding per face — an identity, not a position. When
# available the lock survives the person leaving and coming back, and will not
# hop to whoever else is closest. Cosine similarity of normed embeddings.
EMB_SIM = 0.32

# After going home the remembered (cx, cy) means nothing, because the camera
# moved. With an identity embedding the person can still be recognised and
# picked up again. WITHOUT one there is no way to tell them from anyone else in
# the room, so the honest answer is to stay put and wait to be told again —
# turning this on re-acquires whoever is simply biggest, which is how a robot
# ends up following a stranger who walked past.
RELOCK_LARGEST = False

# Keep a lock alive using the top slice of a YOLO person box when no face is
# visible (someone turned round). Never used to CREATE a lock — a tap always
# resolves against a real face.
PERSON_FALLBACK = False
HEAD_SLICE      = 0.28
OBJ_CONF        = 0.35

# ── Turning towards a voice ─────────────────────────────────────────────
# How long after a voice turn, with nothing further heard, before the head
# settles back to centre. Held while she speaks, so a long answer cannot
# outrun it — a 20-second reply outran an 18-second timer in testing.
SOUND_HOME_AFTER = 18.0

# Beyond this a turn is not worth making: four microphones a few centimetres
# apart are accurate to about +-15 degrees, and swinging 90 degrees on a
# number that vague looks worse than staying put.
SOUND_MAX_TURN_DEG = 75.0

# Axis speed while returning home on request, app units per second. About a
# quarter of normal — slow enough to read as settling rather than a jerk.
GENTLE_SPEED = 160.0


def _cos(a, b):
    if not a or not b or len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(a, b))     # both are unit vectors


class _Axis:
    """One servo axis under a speed and acceleration limit.

    Position is a float carried between ticks. Rounding to int only at the
    moment of sending is what lets it creep smoothly at low speed — round the
    stored position instead and anything slower than one unit per tick stalls
    completely.
    """

    def __init__(self, home, lo, hi):
        self.pos = float(home)
        self.goal = float(home)
        self.vel = 0.0
        self.lo, self.hi = lo, hi
        self.sent = int(home)

    def step(self, dt):
        dist = self.goal - self.pos
        if abs(dist) < 0.25 and abs(self.vel) < 2.0:
            self.pos, self.vel = self.goal, 0.0
            return False
        # Fastest we may go and still stop in time: v = sqrt(2*a*d). Without
        # this the axis runs at full speed into its goal and stops dead, which
        # is the jerk you feel at the END of a move rather than the start.
        cruise = min(MAX_SPEED, math.sqrt(2.0 * ACCEL * abs(dist)))
        want = math.copysign(cruise, dist)
        dv = max(-ACCEL * dt, min(ACCEL * dt, want - self.vel))
        self.vel += dv
        move = self.vel * dt
        if abs(move) >= abs(dist):
            # One tick would carry us past the goal. Land on it exactly.
            # Stepping past and correcting back is a limit cycle: the head
            # sits at its destination flicking between 999 and 1001 forever,
            # which is both visible and a stream of pointless serial traffic.
            self.pos, self.vel = self.goal, 0.0
            return True
        self.pos = max(self.lo, min(self.hi, self.pos + move))
        return True

    def set_goal(self, v):
        self.goal = max(self.lo, min(self.hi, float(v)))

    @property
    def moving(self):
        return abs(self.goal - self.pos) > 0.5 or abs(self.vel) > 1.0


class HeadTracker:
    def __init__(self):
        self._run = False
        self._send = None
        self._vision_thread = None
        self._motion_thread = None

        self.mode = "off"
        # Instance copies so they can be flipped at runtime, from the display
        # or by the automatic check, without editing the file and restarting.
        self.inv_lr = INVERT_LR
        self.inv_ud = INVERT_UD
        self._inv_ev = {"x": 0.0, "y": 0.0}
        self._inv_n = {"x": 0, "y": 0}
        self._prev_target = None
        self._last_drift = (0.0, 0.0)
        self._dir_ok = {"x": False, "y": False}
        self._calib = None           # direction-calibration state machine
        self._anchor = None          # pose when the current lock was taken

        # Whether anything has ever told us where the head actually is.
        self._synced = False

        # Voice turning, the listening nod, and the gentle return home.
        # ALL initialised here. _home_gently_until was previously set only
        # inside the methods that use it, but the motion loop reads it on its
        # very first tick — so it raised AttributeError and killed the motion
        # thread, which meant no servo output at all with nothing in the log
        # to say so afterwards.
        self.head_motion = True
        self._paused_until = 0.0
        self._sound_turn_at = 0.0
        self._speaking = False
        self._attend_until = 0.0
        self._attend_bias = 0.0
        self._home_gently_until = 0.0

        self._lr = _Axis(LR_HOME, LR_MIN, LR_MAX)
        self._ud = _Axis(UD_HOME, UD_MIN, UD_MAX)

        # Where the head was pointing, recently. A measurement belongs to the
        # pose the camera had when the frame was captured, not to now.
        self._poses = [(time.time(), float(LR_HOME), float(UD_HOME))]
        self._poses_lock = threading.Lock()
        self._capture_pose = (float(LR_HOME), float(UD_HOME))
        self._prev_capture_pose = None

        self._lock = None            # {"cx","cy","w","h","emb","anchored"}
        self._last_seen = 0.0
        self._last_served = 0.0
        self._pending_pick = None    # (x, y, deadline)
        self._going_home = False

        # Speed estimate for the lead term, in app units per second.
        self._vel_lr = 0.0
        self._vel_ud = 0.0
        self._last_want = None
        self._last_meas_t = 0.0

        self.last_error = (0.0, 0.0)
        self.frames = 0
        self.detections = 0
        self.moves = 0
        self.last_err_msg = ""
        self.note = ""
        # Which face detector is actually running. Without insightface there
        # is no identity, so a lock can only be held by continuity — worth
        # knowing when the tracker refuses to pick someone up again.
        self.detector = "none"

        self._jpeg = None
        self._jpeg_lock = threading.Lock()
        self._frame = None
        self._frame_at = 0.0
        self._frame_pose = (LR_HOME, UD_HOME)
        self._frame_lock = threading.Lock()
        self._capture_pose = (LR_HOME, UD_HOME)
        self._cap_thread = None
        self._last_frame_at = 0.0
        self._boxes = []
        self._cascade = None
        self._yolo = None

    # convenience: plenty of existing code reads tracker.lr / tracker.ud
    @property
    def lr(self):
        return self._lr.pos

    @property
    def ud(self):
        return self._ud.pos

    # ══ public ════════════════════════════════════════════════════════
    def start(self, send_fn):
        if self._vision_thread:
            return
        self._send = send_fn
        self._run = True
        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()
        self._cap_thread = threading.Thread(target=self._capture_loop,
                                            daemon=True)
        self._cap_thread.start()
        self._vision_thread = threading.Thread(target=self._loop, daemon=True)
        self._vision_thread.start()
        print("[TRACK] head tracker started — faces only, moves on selection")

    def preview(self, on=True):
        """Camera on, faces boxed, head still. This is what makes the live
        view work BEFORE anything is selected — with the old code the panel
        you are meant to tap on stayed black until tracking was already
        running, which is backwards."""
        if on:
            if self.mode == "off":
                self.mode = "preview"
                self._last_served = time.time()
                self.note = "tap a face to follow"
                print("[TRACK] preview on — head will not move until you pick a face")
        elif self.mode == "preview":
            self.mode = "off"
            self.note = ""
            print("[TRACK] preview off")

    def enable_face(self, on):
        """Follow the nearest face with no selection. Opt-in: the display's
        camera view does NOT call this any more."""
        if on:
            self.mode = "face"
            self.note = "following the nearest face"
            self._lock = None
            self._last_seen = 0.0
            print("[TRACK] mode -> face (nearest)")
        else:
            self._lock = None
            self._last_seen = 0.0
            if self.mode in ("face", "lock"):
                self.mode = "preview"
                self._last_served = time.time()
                self.note = "tap a face to follow"
                self.centre()
            print(f"[TRACK] mode -> {self.mode}")

    def select_at(self, x, y):
        """Lock onto the FACE at this point of the live view (normalised 0-1).

        Resolved against real detections over the next second rather than
        immediately: a remembered (x, y) is meaningless the instant the head
        moves, and one frame's detector miss should not cost the user their
        tap."""
        self._pending_pick = (float(x), float(y), time.time() + 1.0)
        if self.mode == "off":
            self.mode = "preview"
        self._last_served = time.time()
        self.note = "picking a face..."
        print(f"[TRACK] picking the face at {x:.3f},{y:.3f}")

    def clear_lock(self):
        """Let go and glide home. Stays in preview so the live view survives —
        dropping to 'off' here is what used to kill the video stream."""
        self._lock = None
        self._pending_pick = None
        self._last_seen = 0.0
        self._forget_velocity()
        if self.mode != "off":
            self.mode = "preview"
        self._last_served = time.time()
        self.note = "tap a face to follow"
        self.centre()
        print("[TRACK] lock cleared — no longer watching for anyone")

    # ══ voice turning, the nod, and settling home ═════════════════════
    def go_home_gently(self, why="asked"):
        """Return to centre slowly, whatever is happening.

        Separate from centre() because the speed matters here: when head
        motion is switched off, or a conversation ends, the head should settle
        back over a second or two. Snapping to centre at full speed looks like
        a fault, and on a head carrying a camera it is also the movement most
        likely to loosen a mount.

        The speed cap is time-limited, so this cannot leave tracking slow if
        something goes wrong part-way through.
        """
        if self.at_home and not (self._lr.moving or self._ud.moving):
            return
        print(f"[TRACK] returning home gently ({why})")
        self._home_gently_until = time.time() + 4.0
        self._sound_turn_at = 0.0
        self._lr.set_goal(LR_HOME)
        self._ud.set_goal(UD_HOME)

    # ══ scripted looking, used by the vision search ═══════════════════
    def look_to(self, lr=None, ud=None, hold=True):
        """Point the head at specific coordinates and stay there.

        hold=True switches out of face/lock mode first, so the vision loop
        stops aiming — otherwise the tracker would immediately correct back
        towards whatever face it can see and fight the search. Preview keeps
        the camera running, so the live view stays up while the head sweeps.
        """
        if hold and self.mode in ("face", "lock"):
            self.mode = "preview"
            self._last_served = time.time()
        if self.mode == "off":
            self.mode = "preview"
            self._last_served = time.time()
        if lr is not None:
            self._lr.set_goal(max(LR_MIN, min(LR_MAX, float(lr))))
        if ud is not None:
            self._ud.set_goal(max(UD_MIN, min(UD_MAX, float(ud))))
        self._going_home = False
        self._home_gently_until = 0.0

    @property
    def settled(self):
        """True once both axes have stopped.

        Waiting for this before taking a picture matters: a frame captured
        mid-slew is motion-blurred, and a blurred frame is exactly when a
        vision model starts inventing things.
        """
        return not (self._lr.moving or self._ud.moving)

    @property
    def pose(self):
        """Where the head is now, in app units."""
        return (self._lr.pos, self._ud.pos)

    def wait_settled(self, timeout=2.5):
        """Block until the head stops, or the timeout expires.

        The timeout is not optional: with the ESP32 unplugged the axes never
        report arrival, and a search would hang forever — inside the voice
        thread, which would take the whole conversation down with it.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.settled:
                return True
            time.sleep(0.05)
        return False

    def pause(self, seconds):
        """Stop driving the head for a while.

        Called when a gesture sequence moves headLR or headUD. The tracker
        normally owns those joints — it nods while listening and turns
        towards a voice — so without this there would be two writers on one
        joint and the head would twitch between them.

        Implemented as a deadline rather than a flag, so a crash in the
        gesture code cannot leave the head permanently frozen.
        """
        self._paused_until = time.time() + max(0.0, min(120.0, seconds))
        print(f"[TRACK] head handed to a gesture for "
              f"{min(120.0, seconds):.0f}s")

    @property
    def paused(self):
        return time.time() < getattr(self, "_paused_until", 0.0)

    def set_head_motion(self, on):
        """Enable or disable head movement in response to a voice.

        Switching it off settles the head home, rather than leaving it
        wherever the last voice put it — otherwise turning the feature off
        appears to do nothing until the next conversation.
        """
        self.head_motion = bool(on)
        print(f"[TRACK] head motion {'on' if on else 'OFF'}")
        if not on:
            self.go_home_gently("head motion switched off")

    def face_bearing(self, deg):
        """Nudge the head towards a bearing: negative left, positive right.

        A nudge towards a general direction and nothing finer — the array is
        accurate to roughly +-15 degrees, so this puts the person in frame and
        the camera does the rest.

        Ignored while a face lock is active: the camera knows better than the
        microphones where somebody is, and letting the two fight gives a head
        that jerks between them.
        """
        if not self.head_motion:
            return False
        if self.paused:
            return False
        if self.mode == "lock" and self._lock:
            return False
        if abs(deg) > SOUND_MAX_TURN_DEG:
            print(f"[TRACK] {deg:+.0f} deg is too far to be worth turning")
            return False
        # One UNITS_PER_DEG for both axes in this file, which is right here:
        # a voice turn only moves left/right.
        units = deg * UNITS_PER_DEG
        target = max(LR_MIN, min(LR_MAX, self._lr.pos + units))
        if abs(target - self._lr.pos) < 40:
            return False                  # already close enough to bother
        self._lr.set_goal(target)
        self._sound_turn_at = time.time()
        self._home_gently_until = 0.0
        print(f"[TRACK] turning {deg:+.0f} deg towards the voice")
        return True

    def keep_awake(self):
        """Push the go-home timer out without moving. Called when a question
        arrives but the bearing has not changed: the conversation is plainly
        live, and a head that wanders off mid-exchange reads as having stopped
        listening."""
        if self._sound_turn_at:
            self._sound_turn_at = time.time()

    def set_speaking(self, on):
        """Tell the tracker Luna is talking, so the go-home timer is held.

        Without this an answer longer than SOUND_HOME_AFTER makes the head
        wander back mid-sentence. Resetting per question is not enough,
        because one question can produce a very long answer.
        """
        self._speaking = bool(on)
        if not on and self._sound_turn_at:
            self._sound_turn_at = time.time()   # count from when she FINISHED

    def attend(self, seconds=5.0):
        """A small slow nod, as though listening.

        A head perfectly still while somebody talks to it reads as switched
        off; the movement is what makes it look like it is paying attention.

        seconds=0 CANCELS an in-progress nod — used when she starts speaking,
        since the arms carry the movement then and a head nodding along to her
        own speech reads as agreeing with herself.
        """
        if seconds <= 0:
            self._attend_until = 0.0
            self._attend_bias = 0.0
            return
        if self.head_motion and not self.paused:
            self._attend_until = time.time() + seconds

    def motion_status(self):
        """What the display's motion control needs."""
        return {
            "head": self.head_motion,
            "head_moving": bool(self._lr.moving or self._ud.moving),
            "at_home": self.at_home,
        }

    def _attend_step(self, now):
        if now >= self._attend_until:
            self._attend_bias = 0.0
            return
        import math
        # Amplitude and period come from gestures.json, so everything
        # expressive is tunable in one file. Re-read at most once a second:
        # this runs 30 times a second and a file stat per tick is waste.
        if now - getattr(self, "_nod_cfg_at", 0) > 1.0:
            self._nod_cfg_at = now
            try:
                from gesture import gestures
                self._nod_amp, self._nod_period = gestures.nod_config()
            except Exception:
                self._nod_amp, self._nod_period = 35.0, 2.2
        amp = getattr(self, "_nod_amp", 35.0)
        per = getattr(self, "_nod_period", 2.2)
        self._attend_bias = amp * math.sin((now % per) / per * 2 * math.pi)

    def _sound_home_check(self, now):
        """Settle back to centre a while after a voice turn.

        In the MOTION loop, not the vision loop: the vision loop returns early
        when the mode is 'off', and 'off' is exactly the state the head is in
        after a voice turn with no camera tracking. There, the head would
        never come back.
        """
        if not self._sound_turn_at or self._speaking:
            return
        if self.mode == "lock" and self._lock:
            self._sound_turn_at = 0.0               # tracking took over
            return
        if now - self._sound_turn_at < SOUND_HOME_AFTER:
            return
        self._sound_turn_at = 0.0
        if not self.at_home:
            self.go_home_gently("quiet")

    def centre(self):
        """Home, at the same eased pace as everything else. It used to snap."""
        self._lr.set_goal(LR_HOME)
        self._ud.set_goal(UD_HOME)
        self._going_home = True

    def grab_for_caller(self, timeout=4.0, max_age=0.5, min_brightness=8.0):
        """A fresh, LIT frame for another module, captured by this thread only.

        Waits for the capture loop to produce one rather than reading the
        camera directly — two readers on one cv2.VideoCapture is not
        thread-safe and hangs inside the V4L2 driver.

        ── WHY IT CHECKS BRIGHTNESS ────────────────────────────────────
        If the tracker was idle, this wakes it, and the FIRST frames a UVC
        camera produces after being idle are zeroed buffers. A caller that
        accepts the first fresh frame gets a black picture and reports "check
        the lighting and the lens cover" — which sends somebody to look at
        the hardware when the camera is simply not awake yet.

        So it keeps waiting for a frame with some actual light in it, and
        only gives up at the timeout. Reading is what advances the exposure
        algorithm; the capture loop is already doing that.

        min_brightness=0 disables the check, for a caller that genuinely
        wants whatever is there.
        """
        woke = False
        if self.mode == "off":
            self.preview(True)
            woke = True
            # A cold camera needs longer than a warm one. Measured on this
            # robot: four frames was black, a 1.5-second read gave brightness
            # 146.
            timeout = max(timeout, 5.0)
        try:
            deadline = time.time() + timeout
            best = None
            while time.time() < deadline:
                frame, _pose, at = self._latest()
                if frame is not None and (time.time() - at) <= max_age:
                    if min_brightness <= 0:
                        return frame
                    try:
                        lit = float(frame.mean())
                    except Exception:
                        return frame
                    if lit >= min_brightness:
                        return frame
                    best = frame          # remember it, keep waiting
                time.sleep(0.05)

            if best is not None:
                print(f"[TRACK] no lit frame in {timeout:.0f}s — the camera "
                      f"may really be dark")
                return best
            # Nothing fresh at all. Hand back whatever there is; the caller
            # checks and reports.
            frame, _pose, at = self._latest()
            return frame
        finally:
            if woke:
                self._last_served = time.time()

    def last_jpeg(self):
        """The most recent frame. Serving this to the display means the camera
        has ONE owner — two openers on one device is how black frames appear."""
        if self.mode == "off":
            return None
        self._last_served = time.time()
        with self._jpeg_lock:
            return self._jpeg

    def live_state(self):
        return {
            "enabled": self.mode != "off",
            "mode": self.mode,
            "selecting": self._pending_pick is not None,
            "tracking": self.mode in ("lock", "face")
                        and (time.time() - self._last_seen) < LOST_AFTER,
            "boxes": list(self._boxes),
            "locked_label": "face" if self._lock else "",
            "at_home": self.at_home,
            "moving": self._lr.moving or self._ud.moving,
            "calibrating": self._calib is not None,
            "identity": self.detector == "insightface",
            "note": self.note,
        }

    @property
    def at_home(self):
        return (abs(self._lr.pos - LR_HOME) < 3 and abs(self._ud.pos - UD_HOME) < 3)

    def status(self):
        return {
            "mode": self.mode,
            "locked": "face" if self._lock else None,
            "lr": round(self._lr.pos), "ud": round(self._ud.pos),
            "goal_lr": round(self._lr.goal), "goal_ud": round(self._ud.goal),
            "moving": self._lr.moving or self._ud.moving,
            "error_x": round(self.last_error[0], 3),
            "error_y": round(self.last_error[1], 3),
            "seeing_target": (time.time() - self._last_seen) < LOST_AFTER,
            "at_home": self.at_home,
            "frames": self.frames,
            "detections": self.detections,
            "moves": self.moves,
            "note": self.note,
            "last_error_msg": self.last_err_msg,
            "fov_h": CAM_FOV_H, "damping": DAMPING,
            "max_speed": MAX_SPEED, "accel": ACCEL,
            "invert_lr": self.inv_lr, "invert_ud": self.inv_ud,
            "synced": self._synced,
            "detector": self.detector,
            "identity": self.detector == "insightface",
            "calibrating": self._calib is not None,
        }

    # ══ motion thread ═════════════════════════════════════════════════
    def _motion_loop(self):
        """The only thing that talks to the servos.

        The body is wrapped, because this thread dying is the worst failure in
        the file: the head stops moving permanently and NOTHING says so. The
        vision thread carries on detecting faces and aiming at them, the log
        looks healthy, and only the servos know. It has already happened once,
        from an uninitialised attribute read on the first tick. A thread that
        is the sole owner of a piece of hardware should not be one typo away
        from silence.
        """
        dt = 1.0 / MOTION_HZ
        last = time.time()
        while self._run:
            try:
                time.sleep(dt)
                now = time.time()
                real_dt = min(0.2, now - last)      # a scheduling hiccup must not
                last = now                          # translate into a huge lurch

                self._attend_step(now)
                self._sound_home_check(now)

                # Cap the speed while settling home. Applied to the velocity here
                # rather than by changing MAX_SPEED, so a fault part-way through
                # cannot leave tracking permanently slow.
                if now < self._home_gently_until:
                    for ax in (self._lr, self._ud):
                        ax.vel = max(-GENTLE_SPEED, min(GENTLE_SPEED, ax.vel))

                a = self._lr.step(real_dt)
                b = self._ud.step(real_dt)

                with self._poses_lock:
                    self._poses.append((now, self._lr.pos, self._ud.pos))
                    if len(self._poses) > 240:      # ~8 seconds of history
                        del self._poses[:120]

                if not (a or b) and abs(self._attend_bias) < 1.0:
                    if self._going_home and self.at_home:
                        self._going_home = False
                    continue

                for ax in (self._lr, self._ud):
                    # The nod is added HERE, not to ax.pos. Writing it into the
                    # axis would corrupt the tracker's model of where the head is
                    # pointing, and every drift calculation depends on that being
                    # right — each nod would be mistaken for the head having moved
                    # to follow something.
                    bias = self._attend_bias if ax is self._ud else 0.0
                    want = int(round(max(ax.lo, min(ax.hi, ax.pos + bias))))
                    if abs(want - ax.sent) >= EMIT_MIN or (want != ax.sent and not ax.moving):
                        ax.sent = want
                        self._emit(ax)
            except Exception as e:
                self._motion_errs = getattr(self, '_motion_errs', 0) + 1
                # Loud for the first few, then rare: a fault every 33ms would
                # fill the journal and bury everything else.
                if self._motion_errs <= 3 or self._motion_errs % 300 == 0:
                    print(f"[TRACK] motion loop error "
                          f"#{self._motion_errs}: {e}")
                    if self._motion_errs <= 3:
                        import traceback
                        traceback.print_exc()
                time.sleep(0.1)

    def _emit(self, ax):
        if not self._send:
            return
        if self.paused:
            # A gesture owns the head right now. Skipping the write rather
            # than the whole motion loop, so the axis model keeps tracking
            # where it believes the head is — otherwise it would resume with
            # a stale position and jump.
            return
        name = "headLR" if ax is self._lr else "headUD"
        # quiet: up to 30 a second per axis, and it would bury the log.
        self._send(f"POS:{name}:{ax.sent}:both", quiet=True)
        self.moves += 1

    def _pose_at(self, t):
        """Where the head was pointing at time t."""
        with self._poses_lock:
            best = self._poses[0]
            for p in self._poses:
                if p[0] <= t:
                    best = p
                else:
                    break
            return best[1], best[2]

    # ══ camera ════════════════════════════════════════════════════════
    def _read_frame(self):
        import vision
        cap = vision.get_camera()
        if cap is None or not cap.isOpened():
            self.last_err_msg = "camera unavailable"
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            self.last_err_msg = "no frame"
            return None
        return frame

    def _capture_loop(self):
        """Read, encode, publish. No detection — that is the whole point.

        Records the head's pose at the MOMENT of capture. The vision thread
        reads the frame and that pose together, so a correction is always
        computed against where the head actually was when the shutter opened,
        not where it has drifted to since.
        """
        period = 1.0 / CAPTURE_HZ
        while self._run:
            time.sleep(period)
            if self.mode == "off":
                continue
            try:
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.15)
                    continue
                pose = self._pose_at(time.time() - CAM_LATENCY)
                self.frames += 1
                with self._frame_lock:
                    self._frame = frame
                    self._frame_pose = pose
                    self._frame_at = time.time()
                self._store(frame)
            except Exception as e:
                self.last_err_msg = f"capture: {e}"
                time.sleep(0.2)

    def _grab(self):
        """One frame, plus the pose it belongs to.

        Does NOT take vision._camera_lock. get_camera() takes it itself, and
        threading.Lock is not reentrant — holding it here and then calling
        get_camera() deadlocks on the first call while still holding the lock,
        after which every camera user is told the camera is busy. That exact
        bug shipped in cam_snap.
        """
        try:
            frame = self._read_frame()
            if frame is None:
                return None
            self._capture_pose = self._pose_at(time.time() - CAM_LATENCY)
            self.frames += 1
            self._store(frame)
            # Keep the raw frame for anything else that needs to see through
            # Luna's eyes — the VLM search, describe_scene, cam_snap. They
            # must NOT open the camera themselves: two threads calling read()
            # on one cv2.VideoCapture is not thread-safe, and on V4L2 it
            # blocks in the driver holding the device lock. That is a hang,
            # not a crash, and it takes the display down with it because
            # Flask ends up waiting on the same camera.
            with self._frame_lock:
                self._frame = frame
                self._frame_at = time.time()
            return frame
        except Exception as e:
            self.last_err_msg = f"grab: {e}"
            return None

    def _latest(self):
        """The most recent frame, the pose it belongs to, and when it was
        taken. Returns (None, pose, 0.0) if nothing has been captured.

        This is the ONLY way another module should get a camera frame while
        the tracker is running. vision_vlm called it before it existed, the
        AttributeError was swallowed, and the fallback opened the camera
        directly — giving two readers and the hang described above.
        """
        with self._frame_lock:
            if self._frame is None:
                return None, self._frame_pose, 0.0
            return self._frame, self._frame_pose, self._frame_at

    def _store(self, frame):
        """Encode for the preview. Downscaled first: at 20fps this is the
        dominant cost in the capture thread — about 11ms a frame at 1280x720
        against 3ms at 640 — and the display cannot show the extra detail.
        DETECTION still uses the full frame."""
        cv2 = _cv2()
        if cv2 is None:
            return
        try:
            h, w = frame.shape[:2]
            if w > PREVIEW_WIDTH:
                scale = PREVIEW_WIDTH / float(w)
                frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = buf.tobytes()
        except Exception:
            pass

    # ══ detection — faces only ════════════════════════════════════════
    def _detect_faces(self, frame):
        """Faces as normalised boxes. InsightFace if vision.py has it (it also
        gives an identity embedding), otherwise a Haar cascade."""
        cv2 = _cv2()
        h, w = frame.shape[:2]
        out = []

        try:
            import vision
            app = getattr(vision, "_face_app", None)
            if app is not None:
                for f in app.get(frame) or []:
                    x1, y1, x2, y2 = [float(v) for v in f.bbox]
                    emb = getattr(f, "normed_embedding", None)
                    out.append({"label": "face", "conf": 1.0,
                                "x": x1/w, "y": y1/h,
                                "w": (x2-x1)/w, "h": (y2-y1)/h,
                                "cx": (x1+x2)/2/w, "cy": (y1+y2)/2/h,
                                "emb": [float(v) for v in emb] if emb is not None else None})
                if out:
                    if self.detector != "insightface":
                        self.detector = "insightface"
                        print("[TRACK] using insightface — faces have identities, "
                              "so the lock survives them leaving the room")
                    return out
        except Exception as e:
            self.last_err_msg = f"insightface: {e}"

        if cv2 is None:
            self.last_err_msg = ("cv2 unavailable — check the NumPy version: "
                                 "pip install 'numpy<2'")
            return out

        if hasattr(cv2, "CascadeClassifier"):
            try:
                if self._cascade is None:
                    path = getattr(getattr(cv2, "data", None),
                                   "haarcascades", "") \
                           + "haarcascade_frontalface_default.xml"
                    self._cascade = cv2.CascadeClassifier(path)
                if self._cascade is not None and not self._cascade.empty():
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grey = cv2.equalizeHist(grey)     # helps badly lit rooms
                    # minSize was 60px, which threw away anyone more than a
                    # couple of metres off — they never existed as far as the
                    # tracker was concerned.
                    for (x, y, bw, bh) in self._cascade.detectMultiScale(
                            grey, 1.15, 5, minSize=(40, 40)):
                        out.append({"label": "face", "conf": 1.0,
                                    "x": x/w, "y": y/h,
                                    "w": bw/w, "h": bh/h,
                                    "cx": (x+bw/2)/w, "cy": (y+bh/2)/h,
                                    "emb": None})
                    if out and self.detector != "cascade":
                        self.detector = "cascade"
                        print("[TRACK] using the Haar cascade — no identities, so "
                              "a lost lock has to be re-tapped rather than guessed")
            except Exception as e:
                self.last_err_msg = f"cascade: {e}"

        return out

    def _detect_heads_from_people(self, frame):
        """Top slice of each YOLO person box, as a stand-in head. Only reached
        when PERSON_FALLBACK is on and no face was found."""
        out = []
        h, w = frame.shape[:2]
        model = self._yolo
        if model is None:
            try:
                import vision
                for name in ("_model", "model", "_yolo", "yolo"):
                    m = getattr(vision, name, None)
                    if m is not None and hasattr(m, "predict"):
                        model = m
                        break
                if model is None and hasattr(vision, "_load_model"):
                    vision._load_model()
                    for name in ("_model", "model", "_yolo", "yolo"):
                        m = getattr(vision, name, None)
                        if m is not None and hasattr(m, "predict"):
                            model = m
                            break
            except Exception as e:
                self.last_err_msg = f"yolo lookup: {e}"
            self._yolo = model if model is not None else False
        if not model:
            return out
        try:
            res = model.predict(frame, verbose=False, conf=OBJ_CONF)
            if not res:
                return out
            r = res[0]
            names = getattr(r, "names", {}) or getattr(model, "names", {})
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if str(names.get(cls, "")).lower() != "person":
                    continue          # people only — this is not object tracking
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                bh = (y2 - y1) * HEAD_SLICE
                out.append({"label": "head",
                            "conf": float(b.conf[0]) if b.conf is not None else 0.0,
                            "x": x1/w, "y": y1/h,
                            "w": (x2-x1)/w, "h": bh/h,
                            "cx": (x1+x2)/2/w, "cy": (y1 + bh/2)/h,
                            "emb": None})
        except Exception as e:
            self.last_err_msg = f"yolo: {e}"
        return out

    # ══ control ═══════════════════════════════════════════════════════
    def _aim(self, cx, cy, now):
        """Set the goal so that (cx, cy) ends up in the centre of the frame.

        Measured FROM THE POSE AT CAPTURE. Adding the correction to wherever
        the head has slid to since would count the movement already under way
        twice over, which is overshoot, which is hunting.

        Plus a lead term. Position-only correction always trails a moving
        person by however far they travel during one decision plus the camera
        delay — at a walking pace that measured 16 degrees off centre, which
        looks like the robot watching someone's shoulder. Estimating how fast
        they are crossing and aiming where they are GOING is what turns that
        into following.
        """
        cap_lr, cap_ud = self._capture_pose
        ex, ey = cx - 0.5, cy - 0.5
        self.last_error = (ex, ey)

        # Where the head would have to point, right now, to centre them.
        want_lr = cap_lr + (-ex * SPAN_H if self.inv_lr else ex * SPAN_H)
        want_ud = cap_ud + (-ey * SPAN_V if self.inv_ud else ey * SPAN_V)

        dt = now - self._last_meas_t if self._last_meas_t else 0.0
        if 0.0 < dt < 0.5 and self._last_want is not None:
            # Smoothed, or detector noise becomes head jitter.
            vx = (want_lr - self._last_want[0]) / dt
            vy = (want_ud - self._last_want[1]) / dt
            self._vel_lr += VEL_SMOOTH * (vx - self._vel_lr)
            self._vel_ud += VEL_SMOOTH * (vy - self._vel_ud)
        else:
            self._vel_lr = self._vel_ud = 0.0
        self._vel_lr = max(-MAX_SPEED, min(MAX_SPEED, self._vel_lr))
        self._vel_ud = max(-MAX_SPEED, min(MAX_SPEED, self._vel_ud))
        self._last_want = (want_lr, want_ud)
        self._last_meas_t = now

        step_lr = (want_lr - cap_lr) * DAMPING + self._vel_lr * LEAD
        step_ud = (want_ud - cap_ud) * DAMPING + self._vel_ud * LEAD
        step_lr = max(-MAX_STEP, min(MAX_STEP, step_lr))
        step_ud = max(-MAX_STEP, min(MAX_STEP, step_ud))

        # Hold zone. While they are near enough to the middle, do nothing at
        # all — not a small move, none. This is what stops the head drifting
        # about while somebody stands there talking.
        if abs(ex) >= HOLD_ZONE and abs(step_lr) >= MIN_MOVE:
            self._lr.set_goal(self._leash(cap_lr + step_lr, 0, "x"))
            self._going_home = False
        if abs(ey) >= HOLD_ZONE and abs(step_ud) >= MIN_MOVE:
            self._ud.set_goal(self._leash(cap_ud + step_ud, 1, "y"))
            self._going_home = False

    def _leash(self, goal, idx, key):
        """Keep an unproven axis near where it started."""
        if self._dir_ok[key] or self._anchor is None:
            return goal
        a = self._anchor[idx]
        return max(a - PROBATION, min(a + PROBATION, goal))

    def sync_position(self, lr=None, ud=None):
        """Something else moved the head — a slider, Home, a pose replay.
        Believe it.

        THIS IS WHY THE HEAD USED TO LURCH ON THE FIRST COMMAND. The tracker
        kept its own idea of where the head was pointing, starting at
        1000/1000, and nothing ever corrected it. If the head was actually
        sitting anywhere else — because a slider moved it, a pose left it
        there, or it simply powered up somewhere else — then the first POS:
        the tracker sent snapped the servo from its real position to the
        tracker's imagined one. From in front of the camera that looks exactly
        like "I selected a face and it suddenly dived downwards": the size and
        direction of that jump has nothing to do with the face, only with how
        wrong the assumption was.
        """
        now = time.time()
        if lr is not None:
            self._lr.pos = self._lr.goal = float(lr)
            self._lr.vel = 0.0
            self._lr.sent = int(round(float(lr)))
        if ud is not None:
            self._ud.pos = self._ud.goal = float(ud)
            self._ud.vel = 0.0
            self._ud.sent = int(round(float(ud)))
        with self._poses_lock:
            self._poses.append((now, self._lr.pos, self._ud.pos))
        self._synced = True
        self._forget_velocity()
        self._prev_capture_pose = None
        self._prev_target = None
        self._last_drift = (0.0, 0.0)

    def note_external_head_command(self, part, value):
        """Call from wherever else the server sends POS:headLR / POS:headUD."""
        if part == "headLR":
            self.sync_position(lr=value)
        elif part == "headUD":
            self.sync_position(ud=value)
        elif part == "home":
            self.sync_position(lr=LR_HOME, ud=UD_HOME)

    def flip_axis(self, axis):
        """Runtime direction flip, so this can be sorted out in front of the
        robot instead of in an editor."""
        if axis in ("lr", "x", "pan"):
            self.inv_lr = not self.inv_lr
        elif axis in ("ud", "y", "tilt"):
            self.inv_ud = not self.inv_ud
        else:
            return None
        self._inv_ev = {"x": 0.0, "y": 0.0}
        self._inv_n = {"x": 0, "y": 0}
        self._forget_velocity()
        print(f"[TRACK] flipped {axis}: invert_lr={self.inv_lr} invert_ud={self.inv_ud}")
        return {"invert_lr": self.inv_lr, "invert_ud": self.inv_ud}

    def _check_direction(self, target):
        """Did the picture move the way our own movement said it should?

        We know how far the head turned between these two frames, so we know
        which way a stationary face ought to have slid in the image. If it
        consistently slides the OTHER way, the axis is wired backwards. The
        person's own movement is noise here — it is not correlated with what
        the servos did, so it averages out over a handful of samples while a
        wrong sign does not.
        """
        prev, self._prev_target = self._prev_target, (target["cx"], target["cy"])
        if not AUTO_INVERT or prev is None:
            return
        pred_x, pred_y = self._last_drift
        for key, pred, seen in (("x", pred_x, target["cx"] - prev[0]),
                                ("y", pred_y, target["cy"] - prev[1])):
            if abs(pred) < INV_MIN_SHIFT:
                continue          # too small a move to judge anything by
            self._inv_ev[key] += 1.0 if (pred * seen) > 0 else -1.0
            self._inv_n[key] += 1
            if self._inv_ev[key] >= 2 and not self._dir_ok[key]:
                self._dir_ok[key] = True     # proved itself; leash off
            # Unanimity is worth more than volume: three consecutive head
            # movements that all pushed the picture the wrong way is already a
            # wiring fault, and waiting for five costs seconds of the head
            # walking away from the person.
            unanimous = (self._inv_n[key] >= 3
                         and self._inv_ev[key] <= -self._inv_n[key])
            if not unanimous and self._inv_n[key] < INV_SAMPLES:
                continue
            if unanimous or self._inv_ev[key] <= -(INV_SAMPLES * 0.6):
                which = "lr" if key == "x" else "ud"
                print(f"[TRACK] the {which} axis moves the head the WRONG WAY "
                      f"— flipping it automatically")
                self.flip_axis(which)
                return
            self._inv_ev[key] = 0.0
            self._inv_n[key] = 0

    def calibrate(self):
        """Work out both axis directions by experiment, in about two seconds.

        Nudge the head a known amount and watch which way the picture moves.
        That is a measurement, not a guess: it settles "does headUD going up
        mean the head looks up or down" for this particular robot, whatever
        the servo horns and linkages happen to do. Needs one face in view —
        stand in front of the camera and press it.
        """
        if self.mode == "off":
            self.mode = "preview"
            self._last_served = time.time()
        self._calib = {"phase": "start", "t": time.time()}
        self.note = "calibrating — stay in view"
        print("[TRACK] calibrating directions — hold still, one face in view")
        return True

    def _cal_face(self, dets):
        return max(dets, key=lambda d: d["w"] * d["h"]) if dets else None

    def _cal_fail(self, why):
        print(f"[TRACK] calibration abandoned: {why}")
        self.note = f"calibration failed — {why}"
        c = self._calib
        if c and "base_pose" in c:
            self._lr.set_goal(c["base_pose"][0])
            self._ud.set_goal(c["base_pose"][1])
        self._calib = None

    def _run_calibration(self, dets):
        c = self._calib
        now = time.time()
        moving = self._lr.moving or self._ud.moving

        if c["phase"] == "start":
            f = self._cal_face(dets)
            if f is None:
                if now - c["t"] > 6.0:
                    self._cal_fail("no face in view")
                return
            c["base_pose"] = (self._lr.pos, self._ud.pos)
            c["base"] = (f["cx"], f["cy"])
            self._lr.set_goal(self._lr.pos + CAL_STEP)
            c["phase"], c["t"] = "lr_move", now
            return

        if c["phase"] in ("lr_move", "ud_move"):
            # Wait for the move to finish AND for a frame taken after it.
            if moving or now - c["t"] < CAL_SETTLE:
                if now - c["t"] > 4.0:
                    self._cal_fail("the head did not move — is the ESP32 connected?")
                return
            c["phase"] = "lr_read" if c["phase"] == "lr_move" else "ud_read"
            return

        if c["phase"] == "lr_read":
            f = self._cal_face(dets)
            if f is None:
                return self._cal_fail("lost the face mid-test")
            shift = f["cx"] - c["base"][0]
            if abs(shift) < CAL_MIN_SHIFT:
                return self._cal_fail("could not see the picture move")
            # Headed one way in servo units; the picture went the other. If the
            # picture moved the SAME way as the servo numbers, the axis is
            # inverted relative to the convention this file uses.
            self.inv_lr = shift > 0
            print(f"[TRACK] turn axis: headLR up moved the image {shift:+.3f} "
                  f"-> invert_lr={self.inv_lr}")
            self._lr.set_goal(c["base_pose"][0])
            c["phase"], c["t"] = "ud_prep", now
            return

        if c["phase"] == "ud_prep":
            if moving or now - c["t"] < CAL_SETTLE:
                return
            f = self._cal_face(dets)
            if f is None:
                return self._cal_fail("lost the face between tests")
            c["base"] = (f["cx"], f["cy"])       # fresh baseline, head is back
            self._ud.set_goal(self._ud.pos + CAL_STEP)
            c["phase"], c["t"] = "ud_move", now
            return

        if c["phase"] == "ud_read":
            f = self._cal_face(dets)
            if f is None:
                return self._cal_fail("lost the face mid-test")
            shift = f["cy"] - c["base"][1]
            if abs(shift) < CAL_MIN_SHIFT:
                return self._cal_fail("could not see the picture move")
            self.inv_ud = shift > 0
            print(f"[TRACK] tilt axis: headUD up moved the image {shift:+.3f} "
                  f"-> invert_ud={self.inv_ud}")
            self._ud.set_goal(c["base_pose"][1])
            self._dir_ok = {"x": True, "y": True}     # measured; leash off
            self._inv_ev = {"x": 0.0, "y": 0.0}
            self._inv_n = {"x": 0, "y": 0}
            self._calib = None
            self.note = (f"calibrated — turn {'flipped' if self.inv_lr else 'normal'}, "
                         f"tilt {'flipped' if self.inv_ud else 'normal'}")
            print(f"[TRACK] calibration done: invert_lr={self.inv_lr} "
                  f"invert_ud={self.inv_ud}")
            return

    # ══ appearance ════════════════════════════════════════════════════
    def _signature(self, frame, d):
        """A colour fingerprint of this person: the face patch, and the patch
        of body just below it.

        This exists because the Haar cascade gives no identity. Position alone
        cannot survive the head going home — the camera moved, so the old
        coordinates describe a view that no longer exists — and without
        SOMETHING to recognise them by, "start following them again when they
        come back" is not a solvable problem; the tracker would just be picking
        whoever is there.

        The body patch does most of the work. Faces at webcam resolution are a
        few dozen pixels of much the same colours from person to person; a
        shirt is not. It is a within-session cue, not face recognition: it
        holds while the lighting and their clothes do, which is exactly the
        span of "until I clear the target".
        """
        cv2 = _cv2()
        if cv2 is None or frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            x1, y1 = int(d["x"] * w), int(d["y"] * h)
            bw, bh = max(1, int(d["w"] * w)), max(1, int(d["h"] * h))
            cx = x1 + bw // 2

            def hist(px):
                if px is None or px.size == 0 or px.shape[0] < 4 or px.shape[1] < 4:
                    return None
                hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
                hi = cv2.calcHist([hsv], [0, 1], None, [SIG_BINS_H, SIG_BINS_S],
                                  [0, 180, 0, 256])
                cv2.normalize(hi, hi, 0, 1, cv2.NORM_MINMAX)
                return hi

            face = frame[max(0, y1):min(h, y1 + bh), max(0, x1):min(w, x1 + bw)]
            ty1, ty2 = y1 + bh, int(y1 + bh * 2.4)
            tx1, tx2 = int(cx - bw * 0.95), int(cx + bw * 0.95)
            torso = frame[max(0, ty1):min(h, ty2), max(0, tx1):min(w, tx2)]
            return (hist(face), hist(torso) if ty1 < h else None)
        except Exception as e:
            self.last_err_msg = f"signature: {e}"
            return None

    def _sig_score(self, a, b):
        """0-1, higher is more likely the same person."""
        cv2 = _cv2()
        if cv2 is None or a is None or b is None:
            return -1.0
        try:
            parts, weights = [], []
            if a[0] is not None and b[0] is not None:
                parts.append(cv2.compareHist(a[0], b[0], cv2.HISTCMP_CORREL))
                weights.append(0.45)
            if a[1] is not None and b[1] is not None:
                parts.append(cv2.compareHist(a[1], b[1], cv2.HISTCMP_CORREL))
                weights.append(0.55)
            if not parts:
                return -1.0
            total = sum(w for w in weights)
            return sum(p * w for p, w in zip(parts, weights)) / total
        except Exception:
            return -1.0

    def _refresh_signature(self, sig):
        """Drift with the lighting, slowly, so a long session does not end with
        a fingerprint taken under different lamps. Slow enough that it cannot
        walk onto somebody else one frame at a time."""
        if sig is None or self._lock is None or self._lock.get("sig") is None:
            return
        try:
            old = self._lock["sig"]
            merged = []
            for i in (0, 1):
                if old[i] is None or sig[i] is None:
                    merged.append(old[i])
                else:
                    merged.append(old[i] * (1.0 - SIG_REFRESH) + sig[i] * SIG_REFRESH)
            self._lock["sig"] = tuple(merged)
        except Exception:
            pass

    def _reacquire(self, dets):
        """Is our person among these faces? Used when position can no longer
        answer it — after the head has gone home, or after they were out of
        view long enough that where they were says nothing about where they
        are.

        Requires the best candidate to be both good enough AND clearly better
        than the runner-up. Two people who score 0.61 and 0.58 is not an
        identification, it is a coin toss, and a coin toss is how a robot ends
        up following the wrong person.
        """
        if not self._lock:
            return None
        scored = []
        for d in dets:
            if self._lock.get("emb") and d.get("emb"):
                s = _cos(self._lock["emb"], d["emb"])
                scored.append((s, d, EMB_SIM))
            elif self._lock.get("sig") and d.get("sig"):
                lsize = max(1e-6, self._lock["w"] * self._lock["h"])
                ratio = (d["w"] * d["h"]) / lsize
                if not (REACQ_SIZE_MIN <= ratio <= REACQ_SIZE_MAX):
                    continue          # far too near or far to be them
                scored.append((self._sig_score(self._lock["sig"], d["sig"]),
                               d, SIG_MATCH))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        best_s, best_d, threshold = scored[0]
        if best_s < threshold:
            return None
        if len(scored) > 1 and (best_s - scored[1][0]) < SIG_MARGIN:
            return None
        return best_d

    def _forget_velocity(self):
        """After a gap, the last measurement says nothing about speed —
        differencing across it invents a huge one and the head lunges."""
        self._vel_lr = self._vel_ud = 0.0
        self._last_want = None
        self._last_meas_t = 0.0
        self._prev_target = None

    def _drift(self):
        """How far the image slid because the HEAD turned since the last
        frame, in frame fractions. Measured from the pose history, so it stays
        right whether the head managed a full step or was still accelerating.

        Turn right and a stationary face moves left in the image; tilt up and
        it moves down. Getting this wrong does not look like a sign error, it
        looks like the tracker randomly losing people.
        """
        if self._prev_capture_pose is None:
            return 0.0, 0.0
        d_lr = self._capture_pose[0] - self._prev_capture_pose[0]
        d_ud = self._capture_pose[1] - self._prev_capture_pose[1]
        dx = (d_lr / SPAN_H) * (1.0 if self.inv_lr else -1.0)
        dy = (d_ud / SPAN_V) * (1.0 if self.inv_ud else -1.0)
        return dx, dy

    def _pick_target(self, dets):
        """Which detection is the face we are following."""
        if not dets:
            return None

        if self.mode == "face":
            return max(dets, key=lambda d: d["w"] * d["h"])   # nearest = biggest

        if not self._lock:
            return None

        identified = [d for d in dets if d.get("emb")]

        # 1. Identity, when the detector provides it. Survives the head moving,
        #    the person leaving the room, and somebody else standing closer.
        if self._lock.get("emb") and identified:
            best, best_s = None, EMB_SIM
            for d in identified:
                s = _cos(self._lock["emb"], d.get("emb"))
                if s > best_s:
                    best, best_s = d, s
            if best is not None:
                return best
            if len(dets) == len(identified):
                # Every face here was checked and none is our person. Saying
                # None is the whole point of having an identity.
                return None

        # 2. Position — but only near where we last saw them, and only at
        #    roughly the size they were. The radius grows while they are
        #    missing, because uncertainty about where they went grows too; it
        #    stops growing at MATCH_RADIUS_MAX so it can never mean "anywhere".
        if self._lock.get("anchored"):
            gap = max(0.0, time.time() - self._last_seen) if self._last_seen else 0.0
            radius = min(MATCH_RADIUS_MAX, MATCH_RADIUS + gap * RADIUS_GROWTH)
            lx, ly = self._lock["cx"], self._lock["cy"]
            lsize = max(1e-6, self._lock["w"] * self._lock["h"])
            best, best_d = None, 1e9
            for d in dets:
                dist = ((d["cx"] - lx) ** 2 + (d["cy"] - ly) ** 2) ** 0.5
                if dist > radius:
                    continue
                # A face does not change size quickly. Something at half or
                # double the size is a different face, or not a face at all.
                ratio = (d["w"] * d["h"]) / lsize
                if not (SIZE_RATIO_MIN <= ratio <= SIZE_RATIO_MAX):
                    continue
                if self._lock.get("sig") and d.get("sig") is not None:
                    if self._sig_score(self._lock["sig"], d["sig"]) < SIG_VETO:
                        continue          # right place, wrong person
                if dist < best_d:
                    best_d, best = dist, d
            if best is not None:
                return best

        # 3. Recognise them again. Position has failed — either they are far
        #    from where they were, or the head has been home since and the old
        #    coordinates describe a view that no longer exists. This is what
        #    makes "walks out, walks back in, following resumes" work.
        found = self._reacquire(dets)
        if found is not None:
            if not self._lock.get("anchored"):
                print("[TRACK] that is them again — resuming")
            return found

        # No candidate passed. That is the correct answer, not a failure.
        #
        #    There used to be a rule here that took the only face in the frame
        #    on the grounds that it could not be anyone else. It could: a Haar
        #    cascade fires happily on a poster, a chair back, a patterned
        #    cushion, and the moment the selected person turned their head or
        #    blinked out of the detector for a frame, the head swung onto
        #    whatever else had been found. There was a second rule that grabbed
        #    the LARGEST face after coming home, which adopted whoever happened
        #    to be nearest the robot. Both of them followed people nobody had
        #    selected. Fast movement is handled by the search radius growing
        #    above, which is bounded — unlike "anything, anywhere".
        return None

    def _go_home(self, why):
        print(f"[TRACK] {why} — returning to home position")
        self.centre()
        self._forget_velocity()
        # Say which of the two situations this is. With identities the tracker
        # can pick them up again by itself; without, it genuinely cannot tell
        # them from anyone else, and the user needs to know it is waiting for
        # a tap rather than silently doing nothing.
        self.note = "out of frame — home, watching for them to come back"
        if self._lock:
            # The remembered position was relative to a view that no longer
            # exists. Keep the identity, drop the coordinates.
            self._lock["anchored"] = False

    # ══ vision thread ═════════════════════════════════════════════════
    def _loop(self):
        while self._run:
            time.sleep(PERIOD)

            if self.mode == "off":
                continue

            # Nobody has asked for a frame in a while and nothing is being
            # followed: let go of the camera so cam_snap and vision can have it.
            if (self.mode == "preview" and self._pending_pick is None
                    and time.time() - self._last_served > PREVIEW_IDLE):
                self.mode = "off"
                self.note = ""
                print("[TRACK] preview idle — camera released")
                continue

            # Take what the capture thread has, WITH the pose recorded when
            # it was taken. The vision thread no longer touches the camera, so
            # detection cannot stall the preview and the correction is never
            # computed against a stale pose.
            frame, cap_pose, cap_at = self._latest()
            if frame is None:
                continue
            if cap_at == self._last_frame_at:
                continue                  # nothing new since the last look
            self._last_frame_at = cap_at
            self._capture_pose = cap_pose

            now = time.time()

            # ── detect (faces, and only faces) ──
            dets = self._detect_faces(frame)
            if not dets and PERSON_FALLBACK and self._lock:
                dets = self._detect_heads_from_people(frame)
            if dets:
                self.detections += 1

            # Fingerprint every face while we are holding a selection, so the
            # person can be recognised again after they leave the view. Only
            # while locked: it is wasted work in preview.
            if self.mode == "lock" and self._lock is not None:
                for d in dets:
                    d["sig"] = self._signature(frame, d)

            # ── direction calibration takes over the head while it runs ──
            if self._calib is not None:
                self._boxes = [dict(d, sel=False) for d in dets]
                for b in self._boxes:
                    b.pop("emb", None)
                    b.pop("sig", None)
                self._run_calibration(dets)
                continue

            # ── the lock lives in the world; the image moved under it ──
            # Compute the drift ONCE and keep it. It used to be recomputed
            # inside the direction check, by which point _prev_capture_pose had
            # already been advanced to _capture_pose — so it read zero every
            # time and the check silently never fired.
            self._last_drift = self._drift()
            if self._lock and self._lock.get("anchored"):
                dx, dy = self._last_drift
                self._lock["cx"] = min(1.0, max(0.0, self._lock["cx"] + dx))
                self._lock["cy"] = min(1.0, max(0.0, self._lock["cy"] + dy))
            self._prev_capture_pose = self._capture_pose

            # ── resolve a tap into a lock ──
            if self._pending_pick is not None:
                px, py, deadline = self._pending_pick
                hit = None
                # Prefer a box that CONTAINS the tap; fall back to the nearest
                # centre, because a finger on a touchscreen is not precise.
                for d in dets:
                    if d["x"] <= px <= d["x"] + d["w"] and \
                       d["y"] <= py <= d["y"] + d["h"]:
                        if hit is None or d["w"] * d["h"] < hit["w"] * hit["h"]:
                            hit = d
                if hit is None and dets:
                    near = min(dets, key=lambda d: (d["cx"]-px)**2 + (d["cy"]-py)**2)
                    if ((near["cx"]-px)**2 + (near["cy"]-py)**2) ** 0.5 < MATCH_RADIUS:
                        hit = near
                if hit is not None:
                    self._pending_pick = None
                    self._lock = {"cx": hit["cx"], "cy": hit["cy"],
                                  "w": hit["w"], "h": hit["h"],
                                  "emb": hit.get("emb"),
                                  "sig": self._signature(frame, hit),
                                  "anchored": True}
                    self.mode = "lock"
                    self._anchor = (self._lr.pos, self._ud.pos)
                    if not self._synced:
                        print("[TRACK] WARNING: nothing has told this tracker where "
                              "the head actually is; assuming "
                              f"{int(self._lr.pos)}/{int(self._ud.pos)}. If the head "
                              "jumps on the first move, wire sync_position() to "
                              "wherever else headLR/headUD are set.")
                    self._forget_velocity()
                    self._last_seen = now
                    self.note = "following your selection"
                    print(f"[TRACK] locked onto the face at "
                          f"{hit['cx']:.2f},{hit['cy']:.2f}"
                          f"{' (with identity)' if hit.get('emb') else ''}")
                elif now > deadline:
                    # Give the detector a second before saying no — one missed
                    # frame should not cost the user their tap. Do NOT aim at
                    # the bare point: that is what made a tap on a wall swing
                    # the head at nothing.
                    self._pending_pick = None
                    self.note = "no face there — tap on a face"
                    print("[TRACK] no face at that point — ignoring the tap")

            # ── overlay ──
            target = self._pick_target(dets) if self.mode in ("lock", "face") else None
            self._boxes = [dict(d, sel=(target is not None and d is target))
                           for d in dets]
            for b in self._boxes:
                b.pop("emb", None)        # 512 floats do not belong in JSON
                b.pop("sig", None)        # nor do two histograms

            if self.mode == "preview":
                continue                  # camera and boxes only; head stays put

            # ── nothing to follow ──
            if target is None:
                lost_for = now - self._last_seen if self._last_seen else 999
                if lost_for > HOME_AFTER and not (self.at_home or self._going_home):
                    self._go_home("face out of frame")
                continue

            # ── follow ──
            self._last_seen = now
            self.note = ("following your selection" if self.mode == "lock"
                         else "following the nearest face")
            if self.mode == "lock" and self._lock is not None:
                self._lock.update({"cx": target["cx"], "cy": target["cy"],
                                   "w": target["w"], "h": target["h"],
                                   "anchored": True})
                if target.get("emb") and not self._lock.get("emb"):
                    self._lock["emb"] = target["emb"]
                if target.get("sig") is not None:
                    if self._lock.get("sig") is None:
                        self._lock["sig"] = target["sig"]
                    else:
                        self._refresh_signature(target["sig"])
            self._check_direction(target)
            self._aim(target["cx"], target["cy"], now)


tracker = HeadTracker()
