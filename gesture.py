"""
gesture.py — arm movement while Luna speaks.

    from gesture import gestures
    gestures.start(send_to_esp32)
    gestures.speaking(True)      # from speech_start
    gestures.speaking(False)     # from speech_end

═══════════════════════════════════════════════════════════════════════
POSES LIVE IN gestures.json, IN APP VALUES
═══════════════════════════════════════════════════════════════════════
Every number in that file is an APP VALUE, 0-2000 — the same figure the
display's joint sliders show. So finding a gesture is: move the sliders until
the arm looks right, read the values off the screen, type them in.

An earlier version expressed poses as fractions of a per-joint amplitude. That
is tidier in code and useless in practice: there is no way to look at an arm
and know what "0.6 of the forearm's amplitude" means, so every pose had to be
found by trial and error with a rebuild in between.

The file is re-read when it changes — checked before each gesture, so poses
can be tuned while the robot is talking. No restart.

═══════════════════════════════════════════════════════════════════════
SAFETY
═══════════════════════════════════════════════════════════════════════
This is the only thing on the robot that moves a full-size arm without
anybody asking it to. Everything else happens because a person pressed
something; a gesture happens because Luna started a sentence, possibly with
somebody standing in front of her. So:

  · every joint has a hard clamp in gestures.json, and nothing in the file
    can exceed it — a typo of 2000 where 200 was meant cannot throw an arm
  · the shoulder's clamp is the tightest: it carries the arm's whole weight,
    has the most reach, and is the joint that can actually strike someone
  · movement is slow, and TOPSPEED is restored afterwards
  · it stops the moment speech ends and returns to neutral
  · it stands down entirely while anybody is using the app or the display,
    so it cannot fight a person posing the arm

GESTURE_ENABLED, or /gesture/enable, turns the whole thing off.

CALIBRATE THE ARM TABLES FIRST. App value 1000 is only a sensible neutral if
the firmware's ARM_HOME says so; on defaults it is merely mid-travel, which
may be inside the body.
"""

import json
import os
import random
import subprocess
import threading
import time

GESTURE_ENABLED = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_FILE = os.path.join(BASE_DIR, "gestures.json")
# Music for the dances. Drop .wav or .mp3 files here and name them in
# gestures.json.
MUSIC_DIR = os.path.join(BASE_DIR, "music")

RESTORE_SPEED = 1000       # TOPSPEED to put back afterwards
RETURN_SETTLE = 0.8        # pause after speech before returning to neutral
NEUTRAL_DEFAULT = 1000     # app value treated as "at rest"

# If a person has touched a joint within this many seconds, stand down. A
# gesture fighting somebody posing the arm is confusing and a way to strip a
# servo.
USER_ACTIVE_WITHIN = 20.0

# Used only if gestures.json is missing or unreadable. Deliberately minimal:
# a broken file should not mean a robot flailing, and it should be obvious
# from the log that the file did not load.
FALLBACK = {
    "limits": {"forearm": [900, 1100], "elbow": [900, 1100]},
    "speed": 420,
    "speaking": {"poses": [{"forearm": 1060, "hold": 0.6},
                           {"forearm": 1000, "hold": 0.5}]},
    "modes": {},
}


class Gestures:
    def __init__(self):
        self._run = False
        self._send = None
        self._thread = None
        self._speaking = False
        self._active = False
        self._last_user_move = 0.0
        self._touched = {}          # joint -> last value sent
        self.enabled = GESTURE_ENABLED

        self._cfg = dict(FALLBACK)
        self._cfg_mtime = 0.0
        self._cfg_error = ""
        self._last_action = None
        self._mode_running = False
        self._answer_pending = False
        self._mode_cancel = False

    # ══ public ════════════════════════════════════════════════════════
    def start(self, send_fn):
        if self._thread:
            return
        # Honour the saved setting, or the toggle reverts on every restart and
        # somebody who switched it off for a school visit finds it back on.
        try:
            from settings import load_settings
            self.enabled = bool(load_settings().get("gestures_enabled",
                                                    GESTURE_ENABLED))
        except Exception:
            self.enabled = GESTURE_ENABLED
        self._load_cfg(force=True)
        self._send = send_fn
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        n_sp = len(self._cfg.get("speaking", {}).get("poses", []))
        n_md = len([k for k in self._cfg.get("modes", {})
                    if not k.startswith("_")])
        print(f"[GESTURE] ready (enabled={self.enabled}, "
              f"{n_sp} speaking poses, {n_md} modes)")

    def speaking(self, on):
        """Called on speech_start and speech_end.

        Whether it actually gestures depends on expect_answer(): with
        answers_only set, only speech that IS an answer moves the arms. The
        wake acknowledgement is a single word, "Let me take a look" is three,
        and gesturing through those looks like a twitch — the first pose has
        barely arrived before the speech is over.
        """
        self._speaking = bool(on)
        if not on:
            # An answer is one utterance. Clear the flag so the next thing she
            # says — an acknowledgement, a reminder — does not inherit it.
            self._answer_pending = False

    def expect_answer(self):
        """Mark the NEXT thing she says as an answer to a question.

        Called from main.py just before an answer is returned, which is the
        only place that knows the difference. Nothing else in the speech path
        does: speech_start looks identical for "Yes?" and for a paragraph
        about India.
        """
        self._answer_pending = True

    def _should_gesture(self):
        if not self._speaking or not self.enabled:
            return False
        if self._cfg.get("speaking", {}).get("answers_only", True):
            return self._answer_pending
        return True

    def note_user_activity(self):
        """Called whenever a person moves a joint from the app or display."""
        self._last_user_move = time.time()

    def nod_config(self):
        """The nod settings from gestures.json, for the head tracker.

        Kept in the same file as the gestures so everything expressive is
        tunable in one place — otherwise the arm poses are editable without a
        restart and the nod needs a code change, which is a confusing split.
        """
        self._load_cfg()
        n = self._cfg.get("nod", {})
        try:
            amp = float(n.get("amplitude", 35))
        except (TypeError, ValueError):
            amp = 35.0
        try:
            per = float(n.get("period", 2.2))
        except (TypeError, ValueError):
            per = 2.2
        # Bounded: a nod of 400 units would be a violent head-swing, and a
        # period of 0.1 a vibration.
        return (max(0.0, min(120.0, amp)), max(1.0, min(6.0, per)))

    def reload(self):
        """Force a re-read of gestures.json."""
        return self._load_cfg(force=True)

    def status(self):
        return {
            "enabled": self.enabled,
            "speaking": self._speaking,
            "answer_pending": self._answer_pending,
            "answers_only": self._cfg.get("speaking", {})
                                .get("answers_only", True),
            "gesturing": self._active,
            "user_active": (time.time() - self._last_user_move)
                            < USER_ACTIVE_WITHIN,
            "modes": {k: v.get("name", k)
                      for k, v in self._cfg.get("modes", {}).items()
                      if not k.startswith("_") and isinstance(v, dict)},
            "runs": {k: v.get("name", k)
                     for k, v in self._cfg.get("runs", {}).items()
                     if not k.startswith("_") and isinstance(v, dict)},
            "mode_running": self._mode_running,
            "speaking_poses": len(self._cfg.get("speaking", {})
                                  .get("poses", [])),
            "pose_file": POSE_FILE,
            "pose_file_error": self._cfg_error,
        }

    # ══ configuration ═════════════════════════════════════════════════
    def _load_cfg(self, force=False):
        """Re-read gestures.json if it has changed.

        Checked by modification time before each gesture, so poses can be
        edited while the robot is running — which is how anybody would
        actually tune them: change a number, say something, watch the arm.
        """
        try:
            mtime = os.path.getmtime(POSE_FILE)
        except OSError:
            if force:
                self._cfg_error = f"{POSE_FILE} not found — using the fallback"
                print(f"[GESTURE] {self._cfg_error}")
            return False
        if not force and mtime == self._cfg_mtime:
            return False
        try:
            with open(POSE_FILE) as f:
                cfg = json.load(f)
            if not isinstance(cfg.get("speaking"), dict):
                raise ValueError("no 'speaking' block in the file")
            self._cfg = cfg
            self._cfg_mtime = mtime
            self._cfg_error = ""
            modes = [k for k in cfg.get("modes", {}) if not k.startswith("_")]
            print(f"[GESTURE] loaded {os.path.basename(POSE_FILE)}: "
                  f"{len(cfg.get('speaking', {}).get('poses', []))} speaking "
                  f"poses, modes {modes}")
            return True
        except Exception as e:
            # Keep whatever was working. A JSON typo mid-edit should not stop
            # the robot gesturing with the last good poses.
            self._cfg_error = f"{type(e).__name__}: {e}"
            print(f"[GESTURE] {os.path.basename(POSE_FILE)} is invalid "
                  f"({self._cfg_error}) — keeping the previous poses")
            self._cfg_mtime = mtime      # do not retry until it changes again
            return False

    def _seed_baseline(self):
        """Record the arms as being at home, if nothing is known yet.

        Both the speaking loop AND play_mode need this. Only the speaking
        path had it, so a mode found _touched empty, _travel_time could not
        tell how far anything had to travel, and it assumed a full 2000-unit
        sweep — a six-second wait after a gesture that moved three joints a
        short way.
        """
        if self._touched:
            return
        home = {j: self._clamp(j, v)
                for j, v in self._cfg.get("home", {}).items()
                if not j.startswith("_")}
        for j, v in home.items():
            if v is None:
                continue
            for side in ("left", "right", "both"):
                self._touched[(side, j)] = v

    def _pose_targets(self, pose, default_hand):
        """Yield (hand, joint, value) for one pose.

        TWO SHAPES are accepted, because mirrored arms are not symmetric once
        built — the horn splines land where they land, so the same physical
        position needs different numbers on each side. Measured on this
        robot: lateral 515 left against 588 right, shoulder 549 against 723.

            SHARED — one value used on whichever arm is playing:
                { "forearm": 1120, "elbow": 940, "hold": 0.6 }

            PER-HAND — different values per side, both arms move:
                { "left":  { "lateral": 515, "elbow": 1820 },
                  "right": { "lateral": 588, "elbow": 1669 },
                  "hold": 0.8 }

        A pose may use both: the per-hand blocks win for the joints they name,
        and anything at the top level applies to both.
        """
        per_hand = {k: v for k, v in pose.items()
                    if k in ("left", "right") and isinstance(v, dict)}

        # Top-level joints first, so a per-hand value can override one.
        shared = {}
        for joint, value in pose.items():
            if joint.startswith("_") or joint in ("hold", "hand",
                                                  "left", "right"):
                continue
            shared[joint] = value

        if per_hand:
            # An explicit per-hand pose drives BOTH arms, whatever `hand`
            # says: stating left and right separately is a statement that
            # both should move.
            for side, joints in per_hand.items():
                merged = dict(shared)
                merged.update({k: v for k, v in joints.items()
                               if not k.startswith("_")})
                for joint, value in merged.items():
                    yield side, joint, value
            return

        for joint, value in shared.items():
            yield default_hand, joint, value

    def _clamp(self, joint, value):
        """A joint's app value, held inside its configured limits.

        Two levels of protection and this is the first: nothing in the pose
        file can move a joint outside these numbers. The firmware clamps too,
        but it only knows what the servo survives, not what is sensible for
        an arm near a person.
        """
        lo, hi = self._cfg.get("limits", {}).get(joint, [0, 2000])[:2]
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        clamped = max(int(lo), min(int(hi), v))
        if clamped != v:
            print(f"[GESTURE] {joint}={v} is outside its limit "
                  f"{lo}..{hi} — using {clamped}")
        return clamped

    # ══ internals ═════════════════════════════════════════════════════
    HEAD_JOINTS = ("headLR", "headUD", "head")

    def _pause_head_tracker(self, seconds):
        """Stop the head tracker fighting a gesture that moves the head.

        The tracker owns headLR and headUD — it nods while listening and
        turns towards a voice. A sequence that also drives them means two
        writers on one joint, and whichever command lands last wins, so the
        head twitches between the two.

        Nothing else needs this: the tracker never touches the arms.
        """
        try:
            from head_tracker import tracker
            tracker.attend(0)                     # stop any nod
            tracker.pause(seconds)
        except Exception:
            pass

    def _send_pos(self, joint, value, hand):
        if not self._send:
            return
        # quiet: a couple of commands a second for the length of a sentence,
        # which would otherwise bury everything else in the log.
        # NOT quiet. Gesture commands were suppressed, so there was no way
        # to tell from the log whether anything reached the ESP32 — the only
        # lines were "mode X starting" and "arms home", with silence between.
        # A gesture is a couple of commands a second, which is not enough to
        # bury anything.
        self._send(f"POS:{joint}:{value}:{hand}")
        # Keyed by (hand, joint): the same joint can hold different values on
        # each arm, so a single key per joint would make the change-detection
        # skip a genuinely needed command on the other side.
        self._touched[(hand, joint)] = value
        if hand == "both":
            self._touched[("left", joint)] = value
            self._touched[("right", joint)] = value

    # The firmware maps TOPSPEED onto degrees per second like this, and
    # 2000 app units is the joint's full 180 degrees. Needed to work out how
    # long a move will take.
    @staticmethod
    def _dps(topspeed):
        return 12.0 + (320.0 - 12.0) * max(0, min(2000, topspeed)) / 2000.0

    def _travel_time(self, targets, topspeed):
        """Seconds for the furthest-travelling joint to arrive.

        Needed because TOPSPEED is a RATE, not a per-move setting: changing it
        while a joint is still moving changes the speed of the rest of that
        move. Restoring the speed too early is what produced a jerk — slow and
        smooth for a moment, then snapping the rest of the way, and the
        gentler home_speed was set the worse it got.
        """
        dps = self._dps(topspeed)
        worst = 0
        for (hand, joint), want in targets.items():
            now = self._touched.get((hand, joint))
            if now is None:
                # Unknown position: assume the worst, a full sweep. Better to
                # wait too long than to cut a move short.
                worst = 2000
                break
            worst = max(worst, abs(want - now))
        degrees = worst / 2000.0 * 180.0
        # A margin for the serial round trip and the firmware's own tick.
        return degrees / dps + 0.35

    def _return_home(self, why="speech ended"):
        """Return the arms to the rest pose the moment speech ends.

        Sends EVERY joint in the home block, not only the ones the last
        gesture happened to move. Otherwise a gesture that used the elbow
        leaves the forearm wherever an earlier one put it, and the rest pose
        drifts over the course of a conversation until the arms are somewhere
        nobody chose.

        Uses explicit per-joint values rather than a bare HOME by default,
        because HOME also moves the head and the fingers — which fights the
        head tracker and undoes a turn towards a voice. Set
        use_firmware_home in gestures.json if the head is not in use and you
        would rather trust the firmware's own ARM_HOME table.
        """
        if not self._send:
            return

        # Home slowly: the arms have just been somewhere deliberate, and
        # snapping back looks like a fault. On a full-size arm it is also the
        # movement most likely to loosen a mount.
        self._send(f"TOPSPEED:{int(self._cfg.get('home_speed', 220))}")

        if self._cfg.get("use_firmware_home"):
            # A bare HOME uses the firmware's own HOME_DPS, which is already
            # gentle, so no wait is needed here.
            print(f"[GESTURE] {why} — sending HOME")
            self._send("HOME")
            self._touched.clear()
            return

        home = {k: v for k, v in self._cfg.get("home", {}).items()
                if not k.startswith("_")}
        if not home:
            # No home block: fall back to the joints actually touched, which
            # at least undoes this gesture.
            home = {j: NEUTRAL_DEFAULT for (_h, j) in self._touched}

        # Work out the travel BEFORE sending, while _touched still holds
        # where the arms actually are.
        speed = int(self._cfg.get("home_speed", 220))
        targets = {}
        for joint, value in home.items():
            v = self._clamp(joint, value)
            if v is None:
                continue
            for side in ("left", "right"):
                if self._touched.get((side, joint)) != v:
                    targets[(side, joint)] = v
        wait = self._travel_time(targets, speed) if targets else 0.0

        sent = 0
        for joint, value in home.items():
            v = self._clamp(joint, value)
            if v is None:
                continue
            # Either arm being away from home is reason enough to send it.
            if (self._touched.get(("left", joint)) == v
                    and self._touched.get(("right", joint)) == v):
                continue          # both already home
            self._send_pos(joint, v, "both")
            sent += 1
            # Spaced: the ESP32 reads a line at a time, and a burst of
            # eleven can outrun it.
            time.sleep(0.05)

        # Hold the slow speed until the arms have ARRIVED. Restoring it early
        # makes the rest of the journey happen at the app's speed, which is
        # the jerk this is here to avoid.
        if wait > 0:
            print(f"[GESTURE] homing at speed {speed}, "
                  f"allowing {wait:.1f}s")
            time.sleep(wait)
        print(f"[GESTURE] {why} — arms home "
              f"({sent} of {len(home)} joints moved)")
        self._touched.clear()

    def _loop(self):
        while self._run:
            time.sleep(0.15)

            if not self.enabled:
                continue

            want = self._should_gesture()

            if want and (time.time() - self._last_user_move) < USER_ACTIVE_WITHIN:
                if self._active:
                    print("[GESTURE] standing down — someone is using the arm")
                    self._active = False
                    # Restore the speed, or every slider they touch afterwards
                    # crawls at gesture speed with no explanation. Deliberately
                    # NOT returning the arm home: they are posing it, and
                    # yanking it back would be worse than the gesture was.
                    self._send(f"TOPSPEED:{RESTORE_SPEED}")
                    self._touched.clear()
                continue

            # ── start ──
            if want and not self._active:
                self._active = True
                self._load_cfg()          # pick up any edits since last time
                # Assume the arms are at home when a gesture starts, which
                # they are: the previous one returned them there. Without
                # this, the change-detection has no baseline and the first
                # pose sends all eleven joints at once.
                self._seed_baseline()
                speed = int(self._cfg.get("speed", 320))
                self._send(f"TOPSPEED:{speed}")
                print(f"[GESTURE] speaking — gesturing at speed {speed}")

            # ── stop ──
            if not want and self._active:
                self._active = False
                time.sleep(RETURN_SETTLE)
                self._return_home()
                self._send(f"TOPSPEED:{RESTORE_SPEED}")
                continue

            if not self._active:
                continue

            # ── the speaking loop ──
            # One sequence, played start to finish, then again, for as long
            # as she is speaking. Not a random pick from a set: a repeating
            # cycle is predictable, which is what makes it tunable by eye —
            # you can watch one pass, change a number, and watch the same
            # pass again.
            seq = self._cfg.get("speaking", {})
            poses = seq.get("poses", [])
            if not poses:
                time.sleep(0.5)
                continue

            self._play(poses, "both", check_speaking=True)

            # A breath between cycles, so it does not read as one relentless
            # movement.
            try:
                pause = float(seq.get("loop_pause", 0.4))
            except (TypeError, ValueError):
                pause = 0.4
            end = time.time() + max(0.0, min(3.0, pause))
            while time.time() < end and self._should_gesture() and self._active:
                time.sleep(0.05)

    # ══ playing a sequence ════════════════════════════════════════════
    def _play(self, poses, hand, check_speaking):
        """Run a list of poses in order.

        check_speaking=True for the speech loop, so it abandons the moment
        she stops talking — a gesture that finishes its cycle after the
        sentence has ended looks like the robot talking to itself.

        check_speaking=False for a mode action, which plays to completion:
        half a handshake is worse than none.
        """
        for pose in poses:
            if not isinstance(pose, dict):
                continue

            # A step that says {"home": true} returns the arms to the rest
            # pose mid-sequence and waits for them to arrive. Run 1 needs it:
            # the left arm does its part, everything goes home, then the right
            # arm mirrors it. Without this, the second half would start from
            # wherever the first half finished.
            if pose.get("home"):
                self._return_home("mid-sequence home")
                speed = int(self._cfg.get("speed", 320))
                self._send(f"TOPSPEED:{speed}")
                try:
                    extra = float(pose.get("hold", 0.3))
                except (TypeError, ValueError):
                    extra = 0.3
                time.sleep(max(0.0, min(5.0, extra)))
                continue
            if not self._run or self._mode_cancel:
                return False
            if check_speaking and not (self._should_gesture() and self._active):
                return False
            # Checked every pose, not just at the top: a person grabbing a
            # slider mid-sequence should not keep receiving commands until
            # it finishes.
            if (time.time() - self._last_user_move) < USER_ACTIVE_WITHIN:
                return False

            for target_hand, joint, value in self._pose_targets(pose, hand):
                v = self._clamp(joint, value)
                if v is None:
                    continue
                key = (target_hand, joint)
                if self._touched.get(key) == v:
                    continue          # already there
                self._send_pos(joint, v, target_hand)
                time.sleep(0.02)

            try:
                hold = float(pose.get("hold", 0.5))
            except (TypeError, ValueError):
                hold = 0.5
            # Up to 30 seconds. It was capped at 3, which would have
            # silently truncated a deliberate 10-second hold — a pose held
            # for a third of the intended time, with nothing to say why.
            hold = max(0.1, min(30.0, hold))
            # Slept in slices so a stop is noticed within 50ms rather than at
            # the end of a long hold.
            end = time.time() + hold
            while time.time() < end:
                if not self._run or self._mode_cancel:
                    return False
                if check_speaking and not (self._should_gesture() and self._active):
                    return False
                time.sleep(0.05)
        return True

    # ══ mode actions ══════════════════════════════════════════════════
    # ══ music ═════════════════════════════════════════════════════════
    def _start_music(self, filename):
        """Play a file from the music folder, and return the process.

        Tries several players because the right one depends on the format and
        on what is installed: aplay handles WAV only, mpg123 is the lightest
        for MP3, mpv and ffplay handle anything. Reporting which are missing
        matters — a dance that moves in silence looks like a broken feature,
        and the cause is one apt-get away.
        """
        if not filename:
            return None
        path = os.path.join(MUSIC_DIR, os.path.basename(str(filename)))
        if not os.path.isfile(path):
            print(f"[DANCE] no music file at {path} — dancing in silence")
            return None

        # The ALSA device the rest of the robot speaks through, so the music
        # comes out of the same speaker rather than a default that may not
        # exist.
        dev = None
        try:
            import audio_utils
            if hasattr(audio_utils, "get_speaker_device"):
                dev = audio_utils.get_speaker_device()
        except Exception:
            pass

        attempts = []
        if path.lower().endswith(".wav"):
            attempts.append(["aplay", "-q"] + (["-D", dev] if dev else []) + [path])
        attempts += [
            ["mpg123", "-q"] + (["-a", dev] if dev else []) + [path],
            ["mpv", "--no-video", "--really-quiet"]
            + ([f"--audio-device=alsa/{dev}"] if dev else []) + [path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ]

        tried = []
        for cmd in attempts:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                print(f"[DANCE] music: {os.path.basename(path)} via {cmd[0]}")
                return p
            except FileNotFoundError:
                tried.append(cmd[0])
            except Exception as e:
                print(f"[DANCE] {cmd[0]} failed: {e}")
                tried.append(cmd[0])
        print(f"[DANCE] no player available (tried {', '.join(tried)}). "
              f"Install one:  sudo apt install mpg123")
        return None

    def _stop_music(self, proc):
        """Stop the music. Called whether the dance finished or was cut
        short — otherwise a cancelled dance leaves the track playing with
        nothing moving, which is worse than silence."""
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _set_eyes(self, state):
        """Set the eye animation. Never allowed to raise — the eyes are
        decoration and must not be able to stop a dance."""
        try:
            from eyes import set_state
            set_state(state)
        except Exception as e:
            print(f"[DANCE] could not set the eyes to {state!r}: {e}")

    def _dance_head(self, stop_event, poses):
        """Move the head on its OWN rhythm while the arms dance.

        A separate thread, not extra joints in the arm poses, because the two
        want different timing. The forearm section runs at 0.45s a pose; a
        head following that beat twitches rather than dances, and the head
        carries the camera so it is the most visible thing moving.

        Runs at whatever TOPSPEED the dance set — the firmware has one servo
        speed for everything, so the head cannot have its own. At 700 a
        600-unit head move takes about half a second, which fits comfortably
        inside a one-second hold.
        """
        try:
            while not stop_event.is_set():
                for pose in poses:
                    if stop_event.is_set():
                        break
                    for joint, value in pose.items():
                        if joint in ("hold",) or joint.startswith("_"):
                            continue
                        v = self._clamp(joint, value)
                        if v is not None:
                            self._send_pos(joint, v, "both")
                    try:
                        hold = float(pose.get("hold", 1.0))
                    except (TypeError, ValueError):
                        hold = 1.0
                    end = time.time() + max(0.1, min(5.0, hold))
                    while time.time() < end and not stop_event.is_set():
                        time.sleep(0.05)
        except Exception as e:
            print(f"[DANCE] head track stopped: {e}")

    # ══ base movement during a dance ══════════════════════════════════
    def _wiggle_base(self, stop_event, speed=25, step=0.7):
        """Rock the base left and right while a dance runs.

        ── WHY IT ALTERNATES IN EQUAL PAIRS ────────────────────────────
        Left then right for the SAME duration, so the net rotation over a
        dance is about zero and the robot finishes roughly where it started.
        Without that pairing it would creep around the room over twenty
        seconds, which at a demonstration means it ends up somewhere nobody
        chose — possibly off the edge of a stage.

        The speed is deliberately low. This is a sway, not driving, and a
        full-size robot rocking briskly near an audience is alarming rather
        than fun.

        It ALWAYS sends a stop on the way out, in a finally, because the one
        failure that matters here is a dance ending with the base still
        turning.
        """
        try:
            self._send(f"SPEED:{speed}")
            while not stop_event.is_set():
                for direction in ("left", "right"):
                    if stop_event.is_set():
                        break
                    self._send(f"MOVE:{direction}", quiet=True)
                    # The firmware stops the motors if no MOVE: arrives for
                    # 500ms, so a held direction has to be refreshed. Resent
                    # every 300ms, the same as the drive keepalive.
                    end = time.time() + step
                    while time.time() < end and not stop_event.is_set():
                        time.sleep(0.3)
                        if not stop_event.is_set():
                            self._send(f"MOVE:{direction}", quiet=True)
                    self._send("MOVE:stop", quiet=True)
                    # A pause between directions, so it rocks rather than
                    # grinding straight from one into the other.
                    end = time.time() + 0.25
                    while time.time() < end and not stop_event.is_set():
                        time.sleep(0.05)
        except Exception as e:
            print(f"[DANCE] base wiggle stopped: {e}")
        finally:
            # The important line in this method.
            self._send("MOVE:stop")
            print("[DANCE] base stopped")

    # ══ dances ════════════════════════════════════════════════════════
    def play_loop(self, loop_key):
        """Play one of the app's Loop actions: a dance with music.

        Unlike a mode or a run, this REPEATS its poses for a set duration
        rather than playing through once — a dance is a cycle, and the music
        decides how long it lasts.
        """
        return self._play_named("loops", loop_key, "dance")

    def play_run(self, run_key):
        """Play one of the app's Run actions. Same machinery as a mode."""
        return self._play_named("runs", run_key, "run")

    def play_mode(self, mode_key):
        """Play one of the app's Mode actions, once, then return home.

        Runs on its own thread so the HTTP request returns immediately — a
        handshake takes three seconds and the app should not be left waiting
        on it.

        Refuses while she is speaking: the speech loop owns the arms then,
        and two sequences writing the same joints would fight.
        """
        # Load the file FIRST. The lookup used to run against whatever was
        # already in memory, so a mode pressed before the first gesture — or
        # before start() had run — was matched against the fallback config,
        # which has no modes at all. It reported "no action for 'Mode 1'
        # (have: [])", which reads like the file being wrong rather than
        # simply unread.
        self._load_cfg()
        return self._play_named("modes", mode_key, "mode")

    def _play_named(self, section, key_wanted, label):
        """Look up and play a named one-shot action from `section`.

        Shared by modes and runs — they differ only in which block of the
        file they come from and what the log calls them.
        """
        # Load the file FIRST. The lookup used to run against whatever was
        # already in memory, so an action pressed before the first gesture was
        # matched against the fallback config and reported as missing.
        self._load_cfg()
        actions = self._cfg.get(section, {})
        # Tolerant lookup: the app sends "Run 1", and somebody editing by hand
        # will eventually write "run1" or "RUN 1".
        want = str(key_wanted).strip().lower().replace(" ", "")
        found = None
        for k, v in actions.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            if k.strip().lower().replace(" ", "") == want:
                found = (k, v); break
            if str(v.get("name", "")).strip().lower().replace(" ", "") == want:
                found = (k, v); break
        if not found:
            available = [k for k in actions if not k.startswith("_")]
            return False, f"no {label} for {key_wanted!r} (have: {available})"

        key, action = found
        # Deliberately NOT checking self.enabled. That toggle governs arm
        # movement WHILE SPEAKING. A mode or run is a button somebody has
        # just pressed on purpose.
        if self._speaking:
            return False, "busy speaking"
        if self._mode_running:
            # Cancel the one in flight rather than refuse. Somebody pressing
            # a button has no idea the previous action is still homing, so a
            # refusal reads as the button being dead.
            print("[GESTURE] interrupting the running action")
            self._mode_cancel = True
            deadline = time.time() + 2.0
            while self._mode_running and time.time() < deadline:
                time.sleep(0.02)
            if self._mode_running:
                print("[GESTURE] previous action did not stop in time")
            self._mode_cancel = False

        def run():
            self._mode_running = True
            music = None
            wiggle_stop = None
            head_stop = None
            try:
                self._load_cfg()
                self._seed_baseline()
                speed = int(action.get("speed", self._cfg.get("speed", 320)))
                self._send(f"TOPSPEED:{speed}")

                # A dance repeats its poses for a set time, with music. A mode
                # or a run plays through once. duration is what separates
                # them, so it lives in the action rather than in the code.
                try:
                    duration = float(action.get("duration", 0))
                except (TypeError, ValueError):
                    duration = 0.0
                if duration > 0:
                    music = self._start_music(action.get("music"))

                # If the sequence drives the head, stand the tracker down for
                # roughly as long as it will take, plus the home afterwards.
                poses = action.get("poses", [])
                if self._uses_head(poses):
                    est = sum(float(p.get("hold", 0.5)) for p in poses
                              if isinstance(p, dict)) + 12.0
                    self._pause_head_tracker(est)

                print(f"[GESTURE] {label} {key} — {action.get('name', key)} "
                      f"at speed {speed}"
                      + (f", {duration:.0f}s" if duration > 0 else ""))

                if duration > 0:
                    # The eyes, for the length of the dance.
                    if action.get("eyes"):
                        self._set_eyes(action["eyes"])

                    # The head, on its own track and its own timing.
                    head_poses = (action.get("head") or {}).get("poses") or []
                    if head_poses:
                        # The tracker owns headLR and headUD for the listening
                        # nod and for turning towards a voice. Hand them over
                        # for the whole dance plus the home afterwards, or the
                        # two fight and the head twitches between them.
                        self._pause_head_tracker(duration + 20.0)
                        head_stop = threading.Event()
                        threading.Thread(
                            target=self._dance_head,
                            args=(head_stop, head_poses),
                            daemon=True).start()

                    # Rock the base, if this dance asks for it.
                    if action.get("base_wiggle"):
                        wiggle_stop = threading.Event()
                        threading.Thread(
                            target=self._wiggle_base,
                            args=(wiggle_stop,
                                  int(action.get("base_speed", 25)),
                                  float(action.get("base_step", 0.7))),
                            daemon=True).start()

                    # INTRO, played once, to get into position. Separate from
                    # the looping section because getting into a pose and
                    # keeping a rhythm are different things — repeating the
                    # setup moves would make the dance lurch back to the
                    # start every cycle instead of flowing.
                    intro = action.get("intro") or []
                    if intro:
                        self._play(intro, action.get("hand", "both"),
                                   check_speaking=False)

                    # Then the repeating section, for `duration` seconds. The
                    # last cycle finishes rather than being cut mid-pose:
                    # stopping an arm halfway to hit an exact count looks
                    # worse than running a second over.
                    end = time.time() + duration
                    cycles = 0
                    while time.time() < end and not self._mode_cancel:
                        if not self._play(poses, action.get("hand", "both"),
                                          check_speaking=False):
                            break
                        cycles += 1
                    print(f"[GESTURE] {label} {key} — {cycles} cycles "
                          f"over {duration:.0f}s")

                    # Stop the base and settle the eyes HERE, as the dance
                    # ends — not in the finally, which runs after the arms
                    # have homed. Homing takes several seconds at the slow
                    # home speed, and a base still rocking while the arms
                    # quietly return looks like the dance half-ended.
                    if wiggle_stop is not None:
                        wiggle_stop.set()
                        wiggle_stop = None
                        time.sleep(0.3)      # let the stop reach the motors
                    if head_stop is not None:
                        head_stop.set()
                        head_stop = None
                    if action.get("eyes"):
                        self._set_eyes("idle")
                else:
                    self._play(poses, action.get("hand", "both"),
                               check_speaking=False)
                if self._mode_cancel:
                    # Cancelled by another action. Do NOT home: the incoming
                    # one is about to move the same joints, and homing first
                    # would make the two overlap.
                    print(f"[GESTURE] {label} {key} cancelled — "
                          f"leaving home to the next action")
                    return
                time.sleep(0.3)
                self._return_home(f"{label} {key} finished")
                self._send(f"TOPSPEED:{RESTORE_SPEED}")
            except Exception as e:
                import traceback
                print(f"[GESTURE] {label} {key} failed: {e}")
                traceback.print_exc()
            finally:
                # All in a finally, so a cancelled or crashed dance cannot
                # leave the music playing, the base turning or the eyes
                # flashing with nothing else happening.
                if wiggle_stop is not None:
                    wiggle_stop.set()
                if head_stop is not None:
                    head_stop.set()
                self._stop_music(music)
                if action.get("eyes"):
                    self._set_eyes("idle")
                self._mode_running = False

        threading.Thread(target=run, daemon=True).start()
        return True, action.get("name", key)

    def _uses_head(self, poses):
        for p in poses:
            if not isinstance(p, dict):
                continue
            for k, v in p.items():
                if k in self.HEAD_JOINTS:
                    return True
                if isinstance(v, dict) and any(j in self.HEAD_JOINTS for j in v):
                    return True
        return False

gestures = Gestures()
