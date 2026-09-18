"""
vision_vlm.py — answering arbitrary questions about what Luna can see.

    from vision_vlm import ask_about_scene
    ask_about_scene("is there a bottle in front of you")
    ask_about_scene("describe the scene")
    ask_about_scene("what colour is my shirt")

═══════════════════════════════════════════════════════════════════════
WHY THIS EXISTS ALONGSIDE describe_scene()
═══════════════════════════════════════════════════════════════════════
YOLO can only ever name the 80 classes it was trained on, and only as a list.
So describe_scene() can say "I can see a person, a chair and a bottle" and
that is the ceiling. It cannot answer:

    "what colour is my shirt"          no colour information at all
    "is the door open"                 'door' is not a COCO class
    "how many people are wearing hats"  no attributes, no relations
    "what is written on that box"      no text

A vision-language model answers the actual question asked. That is the whole
difference, and it is why this is worth the network round trip.

describe_scene() is KEPT as the offline path. A robot that can only see when
the internet is up is worse than one that always manages a rough answer, and
the fallback is genuinely useful at an event with bad WiFi.

═══════════════════════════════════════════════════════════════════════
WHY CLOUD RATHER THAN A LOCAL MODEL, FOR NOW
═══════════════════════════════════════════════════════════════════════
An 8GB Orin Nano shares that memory between CPU and GPU, and Whisper small.en
in fp16 plus YOLO are already resident. A 2B-parameter VLM in fp16 is roughly
4GB on top of that — it fits only by unloading something else, and the answers
from a 2B model are markedly worse than from a hosted one.

Local models worth revisiting when there is time: Moondream2 (1.8B),
Florence-2 (0.23B/0.77B, strong at captioning and grounding for its size), and
SmolVLM. All are plausible on this board; none is as good, and each needs the
memory question solved first.

The pragmatic order is: get the capability working over the network, learn what
people actually ask it, THEN decide whether a local model can serve those
questions well enough to be worth the fight.

═══════════════════════════════════════════════════════════════════════
LATENCY, AND WHY THE IMAGE IS SMALL
═══════════════════════════════════════════════════════════════════════
The round trip is 1-3 seconds, most of it the model thinking rather than the
upload — but the upload is the part under our control. A 1280x720 JPEG is
~180KB; downscaled to 768px it is ~60KB, which over the robot's WiFi is
roughly a second saved for no loss of answer quality at this kind of question.

Luna already says "Let me take a look." before this runs, which covers the
wait. Without that the silence reads as the robot having missed the question.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vision-capable and cheap. Both understand images; mini is several times
# cheaper and quite good enough to say what is on a table.
VLM_MODEL = "gpt-4o-mini"

MAX_EDGE      = 768     # longest side sent, pixels
JPEG_QUALITY  = 82
HTTP_TIMEOUT  = 25      # seconds; a 3s answer is normal, 25 means trouble
MAX_TOKENS    = 220     # a spoken answer, not an essay

# Spoken aloud, so the prompt asks for prose rather than lists or markdown.
SYSTEM_PROMPT = (
    "You are the eyes of a robot called Luna. You are shown a photograph from "
    "the camera in Luna's head and asked a question about it. Answer in one to "
    "three short spoken sentences, as Luna, in the first person. Be specific "
    "and concrete about what is actually visible. If you cannot tell, say so "
    "plainly rather than guessing. Never use lists, bullet points, markdown or "
    "emoji — every word you write will be read aloud."
)


def _api_key():
    """The OpenAI key, from the environment or ai_config.json.

    Same source as ai_fallback.py, deliberately: one key in one place. Never
    from settings.json, which is customer-editable and gets sent to the app.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(os.path.join(BASE_DIR, "ai_config.json")) as f:
            return str(json.load(f).get("openai_key", "")).strip()
    except Exception:
        return ""


def _grab_jpeg():
    """A frame from the camera, JPEG-encoded and downscaled.

    ── NEVER OPENS THE CAMERA ITSELF ───────────────────────────────────
    It asks the head tracker, which owns the device. Two threads calling
    read() on one cv2.VideoCapture is not thread-safe, and on V4L2 it blocks
    inside the driver while holding the device lock — a hang rather than a
    crash, which then takes the touchscreen down too because Flask ends up
    waiting on the same camera.

    That is exactly what happened: this function called tracker._latest(),
    which did not exist, the AttributeError was swallowed by a bare except,
    and the fallback opened the camera directly while the search had already
    put the tracker into preview mode. The whole Jetson appeared to freeze.

    So there is no fallback to opening the device. If the tracker cannot
    supply a frame, that is an error worth reporting.
    """
    cv2 = None
    try:
        import cv2 as _cv2mod
        cv2 = _cv2mod
    except Exception as e:
        return None, f"OpenCV unavailable ({e})"

    try:
        import head_tracker
        t = head_tracker.tracker
    except Exception as e:
        return None, f"head tracker unavailable ({e})"

    if not getattr(t, "_run", False):
        return None, ("the head tracker is not running, so nothing owns the "
                      "camera")

    frame = t.grab_for_caller(timeout=2.5, max_age=0.5)
    if frame is None:
        return None, (f"no frame from the camera "
                      f"({getattr(t, 'last_err_msg', '') or 'unknown'})")

    try:
        if float(frame.mean()) < 6.0:
            # Refuse rather than describe a black rectangle. A vision model
            # will confidently invent a scene from noise.
            return None, ("the camera returned a black frame — check "
                          "lighting and the lens cover")
    except Exception:
        pass

    try:
        h, w = frame.shape[:2]
        if max(h, w) > MAX_EDGE:
            s = MAX_EDGE / float(max(h, w))
            frame = cv2.resize(frame, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None, "could not encode the image"
        return buf.tobytes(), None
    except Exception as e:
        return None, f"encode failed ({e})"


def _fallback(reason):
    """YOLO's list of objects, phrased as a sentence.

    Used when the VLM cannot be reached. Says nothing about WHY out loud —
    "my internet is down" is not an answer to "what can you see" — but logs
    the reason so it is diagnosable.
    """
    print(f"[VLM] falling back to YOLO: {reason}")
    try:
        from vision import describe_scene
        return describe_scene()
    except Exception as e:
        print(f"[VLM] fallback also failed: {e}")
        return "Sorry, I cannot see anything at the moment."


def ask_about_scene(question=None):
    """Answer a question about what the camera can see.

    question=None asks for a general description. Returns a string that is
    always safe to speak: on any failure it falls back to YOLO rather than
    raising or returning an error message.
    """
    q = (question or "").strip() or "Describe what you can see."

    key = _api_key()
    if not key:
        return _fallback("no API key in ai_config.json")

    # Show the POV panel for a plain description too — it is only a second or
    # two, but seeing what she looked at is how you tell a wrong answer from a
    # wrong camera.
    try:
        import head_tracker
        head_tracker.tracker.preview(True)
        # Warm the camera before the picture that matters. See the note in
        # find_object.
        head_tracker.tracker.grab_for_caller(timeout=5.0, min_brightness=8.0)
    except Exception as e:
        print(f"[VLM] could not wake the camera: {e}")
    _notify("vision_start", "looking")
    try:
        jpeg, err = _grab_jpeg()
    finally:
        pass
    if jpeg is None:
        _notify("vision_end")
        # A camera problem is worth saying out loud: it is actionable by the
        # person standing there, unlike a network failure.
        print(f"[VLM] {err}")
        return f"I could not get a picture — {err}."

    b64 = base64.b64encode(jpeg).decode()
    body = {
        "model": VLM_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    # "low" is one fixed tile: fast and cheap, and plenty for
                    # "what is on the table". Raise to "high" only for reading
                    # small text, which costs several times as much.
                    "detail": "low",
                }},
            ]},
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        # Distinguish these, because the fixes are completely different and
        # a single "request failed" wastes an afternoon.
        if e.code == 401:
            _notify("vision_end")
        return _fallback(f"the API key was rejected: {detail}")
        if e.code == 429:
            _notify("vision_end")
        return _fallback(f"rate limited or out of quota: {detail}")
        if e.code == 404:
            _notify("vision_end")
        return _fallback(f"model '{VLM_MODEL}' not available on this key")
        _notify("vision_end")
        return _fallback(f"HTTP {e.code}: {detail}")
    except Exception as e:
        _notify("vision_end")
        return _fallback(f"network error ({type(e).__name__}: {e})")

    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        _notify("vision_end")
        return _fallback(f"unexpected response shape: {str(data)[:180]}")

    if not text:
        # An empty content field with a length finish_reason means the answer
        # was truncated before any text arrived — raise MAX_TOKENS.
        why = data.get("choices", [{}])[0].get("finish_reason", "?")
        _notify("vision_end")
        return _fallback(f"empty answer (finish_reason={why})")

    _notify("vision_end")
    kb = len(jpeg) / 1024.0
    print(f"[VLM] {time.time()-t0:.1f}s, {kb:.0f}KB image -> {text[:90]}")

    # Strip anything that would be read aloud as punctuation noise.
    for ch in ("*", "#", "`", "_"):
        text = text.replace(ch, "")
    return " ".join(text.split())


# ══════════════════════════════════════════════════════════════════════
# LOOKING FOR SOMETHING
#
# "Can you see a water bottle" should make Luna LOOK, not just answer from
# whatever happens to be in frame. The camera sees perhaps 60 degrees; a
# bottle two feet to her left is not in the picture, and answering "no" from
# one frame is wrong in a way that feels careless.
#
# So: ask at the current position, and if the answer is no, sweep the head and
# ask again at each stop. Stop the moment it is found, and leave the head
# pointing at it — which is more useful than naming a direction, and avoids
# the mirror problem of deciding whether +LR means left or right on this
# particular robot.
#
# WHY THE ANSWER IS PARSED, NOT READ
# Free text cannot be tested reliably: "I don't see a bottle, though there is a
# glass" contains both "see a bottle" and a negation. So the search prompt
# demands a strict prefix — YES or NO on its own line — and anything that does
# not start with one is treated as not-found rather than guessed at. Costs
# nothing and removes the whole class of parsing errors.
# ══════════════════════════════════════════════════════════════════════

# Where to look, as offsets from wherever the head already is, in app units.
# ~11 degrees each. Centre first, so a thing already in view is found without
# any movement at all.
# Four looks, not six. The camera sees about 60 degrees, and +-300 units is
# +-27 degrees, so three horizontal stops already cover roughly 114 degrees of
# view — wider than a person turning their head without moving their shoulders.
# The extra two stops in an earlier version pushed the worst case to 17
# seconds, which is long enough that people assume the robot has frozen.
#
#   found straight ahead   ~1.6s
#   found to one side      ~4-7s
#   not found at all       ~9.5s
# Where to look, as offsets from wherever the head already is, in app units.
# Five positions: ahead, left, right, up, down.
#
# Offsets in APP UNITS because that is what the firmware takes. With
# HEAD_UD_A/B at 0/180 the conversion is 11.1 units per degree, so:
#     300 units = 27 degrees      400 units = 36 degrees
#
# The up/down offsets are within Luna's measured mechanical travel: the head
# tilts 25-145 degrees, which is app 278-1611, so home 1000 +-400 stays
# comfortably inside. Widen HEAD_UD_MIN/MAX before increasing these, not
# after — the firmware clamps silently and the head would simply stop short.
#
# Centre first, so something already in view is found with no movement at all.
SEARCH_SWEEP = [
    (0, 0),          # straight ahead
    (-300, 0),       # left, ~27 degrees
    (300, 0),        # right
    (0, 400),        # up
    (0, -400),       # down, where things on tables live
]

# Five looks at roughly 1.6s each plus head movement is about 12 seconds worst
# case. Long, but "Let me have a look around" is spoken first and the POV
# panel shows the sweep happening, so the wait reads as the robot working
# rather than as the robot having frozen.
SEARCH_TIMEOUT = 20.0      # whole search, seconds
SETTLE_PAUSE   = 0.25      # after the head stops, before the picture

FIND_PROMPT = (
    "You are the eyes of a robot. Look at this photograph and decide whether "
    "the thing described is visible in it.\n"
    "Reply with exactly one line, starting with YES or NO.\n"
    "If it is visible: YES followed by a short phrase saying where it is and "
    "what it looks like.\n"
    "If it is not visible: NO and nothing else.\n"
    "Do not explain. Do not hedge. Judge only what is actually in the picture."
)


def _ask_once(jpeg, question, system):
    """One VLM call. Returns (text, error)."""
    key = _api_key()
    if not key:
        return None, "no API key"
    b64 = base64.b64encode(jpeg).decode()
    body = {
        "model": VLM_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            ]},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read())
        return (data["choices"][0]["message"]["content"] or "").strip(), None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _notify(kind, text=""):
    """Tell the display what is happening, so it can show the live view while
    the head sweeps. Never allowed to break the search."""
    try:
        from display_addon import notify_display
        notify_display(kind, text)
    except Exception:
        pass


def find_object(thing, question=None):
    """Look around for something and say whether it is there.

    thing: what to look for, e.g. "a water bottle".
    Returns a sentence safe to speak.
    """
    thing = (thing or "").strip() or "it"

    try:
        import head_tracker
        head = head_tracker.tracker
    except Exception as e:
        print(f"[FIND] no head control ({e}) — looking straight ahead only")
        head = None

    # Wake the camera BEFORE the sweep, not at the first picture.
    #
    # A UVC camera returns zeroed buffers for the first second or so after
    # being idle. Waking it here means the warm-up overlaps the first head
    # movement instead of delaying — and stops the first look coming back
    # black, which reported "check the lighting and the lens cover" and sent
    # somebody to inspect a perfectly good camera.
    if head:
        try:
            head.preview(True)
            # Pull one frame and discard it, so the exposure has settled by
            # the time the first real picture is taken.
            head.grab_for_caller(timeout=5.0, min_brightness=8.0)
        except Exception as e:
            print(f"[FIND] could not wake the camera: {e}")
    _notify("vision_start", f"looking for {thing}")
    started = time.time()

    # Reading the head's position is INSIDE the guard, not outside it.
    #
    # It used to be a bare `head.pose`, and when the tracker did not have
    # that property the AttributeError propagated out of answer_local into
    # the voice thread and killed it — the robot stopped responding
    # altogether and the head never moved, which looked like two separate
    # faults rather than one missing method.
    #
    # Nothing in a search is worth losing the conversation for. If the head
    # cannot be read, the search still runs from wherever it is pointing.
    try:
        base = head.pose if head else (None, None)
    except Exception as e:
        print(f"[FIND] cannot read the head position ({e}) — "
              f"searching from here without moving")
        head = None
        base = (None, None)
    looked = 0

    try:
        for dx, dy in SEARCH_SWEEP:
            if time.time() - started > SEARCH_TIMEOUT:
                print("[FIND] out of time")
                break

            try:
                if head and (dx or dy):
                    head.look_to(base[0] + dx, base[1] + dy)
                    # A frame taken mid-slew is motion-blurred, and blur is
                    # precisely when a vision model starts inventing things.
                    head.wait_settled()
                    time.sleep(SETTLE_PAUSE)
                elif head:
                    head.look_to(base[0], base[1])
                    head.wait_settled(1.0)
            except Exception as e:
                # Carry on looking from wherever the head is. A search that
                # cannot move the head is worth less, but it still answers.
                print(f"[FIND] head move failed ({e}) — looking from here")
                head = None

            jpeg, err = _grab_jpeg()
            if jpeg is None:
                print(f"[FIND] {err}")
                return f"I could not get a picture — {err}."

            looked += 1
            text, err = _ask_once(jpeg, f"Is there {thing} in this picture?",
                                  FIND_PROMPT)
            if err:
                print(f"[FIND] {err}")
                return _fallback(err)

            try:
                head_pose = head.pose if head else (0, 0)
            except Exception:
                head_pose = (0, 0)
            print(f"[FIND] look {looked} at lr={head_pose[0]:.0f} "
                  f"ud={head_pose[1]:.0f} -> {text[:70]}")

            up = text.upper()
            if up.startswith("YES"):
                where = text[3:].lstrip(" :,-").strip()
                moved = bool(dx or dy)
                # The head is left pointing at it, which is more use than
                # naming a direction — and it sidesteps having to know whether
                # +LR means left or right on this particular build.
                if moved:
                    return (f"Yes — I had to look around, but I can see "
                            f"{thing}. {where}" if where else
                            f"Yes, I found {thing} after looking around.")
                return (f"Yes, I can see {thing}. {where}" if where
                        else f"Yes, I can see {thing} right in front of me.")
            # Anything not starting with YES counts as not found. A model that
            # ignored the format is not evidence of anything.

        # Nothing anywhere. Say that it LOOKED, because "no" alone sounds like
        # it did not bother.
        if head:
            try:
                head.look_to(base[0], base[1])
            except Exception:
                pass
        if looked <= 1:
            return f"No, I cannot see {thing} from here."
        return (f"No — I looked around and I cannot see {thing} anywhere "
                f"nearby.")
    finally:
        _notify("vision_end")


# Phrases that mean "go and look for a thing" rather than "describe what you
# see". Kept here rather than in main.py so the same list serves the display.
FIND_PATTERNS = (
    "can you see", "do you see", "can you find", "find me", "look for",
    "is there a", "is there any", "are there any", "search for",
    "where is", "where's", "do you have",
)


# Questions that ask for a DESCRIPTION even though they contain a find
# phrase. "what do you see" contains "do you see" and would otherwise send
# Luna sweeping the room looking for an object called "it".
DESCRIBE_FIRST = (
    "what do you see", "what can you see", "what are you seeing",
    "what do you see in front", "describe", "what is in front of you",
    "what's in front of you", "look around", "what are you looking at",
    "tell me what you", "what is on the table", "what's on the table",
)


def is_find_request(text):
    t = (text or "").lower()
    # A description request wins. Checked first because these phrases contain
    # the find ones, not the other way round.
    if any(d in t for d in DESCRIBE_FIRST):
        return False
    if not any(p in t for p in FIND_PATTERNS):
        return False
    # There must be something left to look FOR. "can you see" with nothing
    # after it is a description request phrased oddly, not a search.
    return extract_thing(text) not in ("it", "")


def extract_thing(text):
    """The object out of a question like "can you see a water bottle".

    Deliberately crude: it strips the leading phrase and trailing filler. A
    vision model copes easily with a slightly untidy noun phrase, so parsing
    this precisely would be effort spent for no gain.
    """
    t = (text or "").lower().strip(" ?.!")
    for p in FIND_PATTERNS:
        i = t.find(p)
        if i >= 0:
            t = t[i + len(p):]
            break
    for tail in (" in front of you", " in front of me", " anywhere",
                 " near you", " around you", " over there", " here",
                 " on the table", " nearby", " somewhere"):
        t = t.replace(tail, "")
    t = t.strip(" ?.!,")
    for lead in ("a ", "an ", "the ", "any ", "some "):
        if t.startswith(lead):
            t = t[len(lead):]
            break
    return t.strip() or "it"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or None
    if q and is_find_request(q):
        print(find_object(extract_thing(q)))
    else:
        print(ask_about_scene(q))