"""
eyes.py — the NeoPixel eyes, driven from the ESP32.

Same interface as the Pi version, so nothing that imports it needs changing:

    from eyes import set_state
    set_state("listening")

═══════════════════════════════════════════════════════════════════════
WHY THIS NO LONGER TALKS TO THE LEDS DIRECTLY
═══════════════════════════════════════════════════════════════════════
WS2812 encodes each bit as a pulse width — 0.4us high for a zero, 0.8us for a
one — with roughly 150ns of tolerance, at 800kHz.

The Pi met that using DMA driven by its PWM hardware, so the timing never
touched the CPU and Linux scheduling could not disturb it. That is what the
`neopixel` library uses, and it is Pi-specific.

The Jetson Orin has no equivalent. Adafruit's library does not support Jetson
for NeoPixels, and that is a deliberate limitation rather than an oversight:
bit-banging from a preemptible kernel cannot hold 150ns, and a single
scheduler tick mid-frame latches garbage into the strip — visible as
flickering or wrong colours. There is no correct way to do it from the Orin's
CPU.

So the rings moved to the ESP32, which drives WS2812 from its RMT peripheral,
hardware-timed in the same way the Pi's DMA was. It was already on the robot
and already wired to the Jetson, so the only thing crossing the link is a
state name.

The animation itself lives in the firmware. This module only says which one
to play.

═══════════════════════════════════════════════════════════════════════
WIRING
═══════════════════════════════════════════════════════════════════════
    left ring  DIN -> ESP32 GPIO 9
    right ring DIN -> ESP32 GPIO 10
    both rings 5V and GND -> the SERVO supply, not the ESP32's 3V3

Eighteen LEDs at full white is around a watt, which the ESP32's regulator
cannot supply. Share the servo rail, and make sure the grounds are common —
without a shared ground the data line has no reference and the colours come
out wrong.

A 300-470 ohm resistor in each data line, close to the ring, is worth
fitting. It damps reflections on a long lead, which otherwise show up as the
first LED behaving erratically while the rest are fine.
"""

import threading

# States the firmware knows. Anything else falls back to idle at the ESP32
# and logs there, but checking here means a typo is reported on the machine
# somebody is actually looking at.
VALID = ("idle", "listening", "thinking", "speaking",
         "eyes1", "eyes2", "eyes3", "off")

DEBUG = False          # True to see which route each send took

_state = "idle"
_lock = threading.Lock()


def _send(cmd):
    """Send one command to the ESP32, and SAY which route was used.

    The route matters and is easy to get wrong. Inside the service,
    server.send_to_esp32 holds the serial lock — several threads write to
    that port and two interleaved writes produce a corrupted line the
    firmware discards. Run standalone, `import server` may succeed and open a
    SECOND handle on a port the service already owns, at which point both
    processes write to it and neither knows.

    Silence was the problem: this used to swallow every failure, so a command
    that never arrived looked exactly like one that did.
    """
    # ── ORDER MATTERS ──────────────────────────────────────────────
    # In-process FIRST. The HTTP route used to be first, and inside the
    # service that calls /eyes, which calls set_state, which calls this —
    # straight back into itself. The second set_state sees the state is
    # already set and returns, the handler returns 200, and this reports
    # success having never touched the serial port. The eyes stayed on
    # whatever they were last showing while everything claimed to work.
    #
    # So: if this process IS the service, write to the port directly.
    try:
        import sys
        srv = sys.modules.get("server")
        if srv is not None and hasattr(srv, "send_to_esp32"):
            srv.send_to_esp32(cmd)
            _dbg(f"{cmd} via server.send_to_esp32")
            return True
    except Exception as e:
        _dbg(f"in-process route failed: {e}")

    # Only now try the service over HTTP — this is a DIFFERENT process, so
    # there is no loop. It is the right route for a standalone script,
    # because the service owns the port and serialises writes to it.
    try:
        import urllib.request
        with urllib.request.urlopen(
                f"http://127.0.0.1:5000/eyes?state={cmd[5:]}",
                timeout=2) as r:
            if r.status == 200:
                print(f"[EYES] {cmd} via the service")
                return True
    except Exception as e:
        _dbg(f"service route unavailable ({type(e).__name__}), trying serial")

    # Last resort: the port directly. Correct only when the service is NOT
    # running — otherwise two processes write to one port and the lines
    # interleave into something the firmware discards.
    try:
        import serial
        with serial.Serial("/dev/ttyTHS1", 115200, timeout=1) as s:
            s.write((cmd + "\n").encode())
        print(f"[EYES] {cmd} via the serial port directly")
        return True
    except Exception as e:
        print(f"[EYES] COULD NOT SEND {cmd!r}: {type(e).__name__}: {e}")
        print("[EYES] Is the service running? If so it owns /dev/ttyTHS1 "
              "and the /eyes route is the way in.")
        return False


def _dbg(msg):
    if DEBUG:
        print(f"[EYES] {msg}")


def set_state(state):
    """Set the eye animation. Safe to call repeatedly with the same value."""
    global _state
    state = str(state or "idle").strip().lower()
    if state not in VALID:
        print(f"[EYES] unknown state {state!r} — using idle. "
              f"Known: {', '.join(VALID)}")
        state = "idle"
    with _lock:
        if state == _state:
            # Resending restarts the animation at its first frame, so a status
            # poll calling this every second would freeze the breathing effect
            # at its starting point — lit, but never changing.
            return
        _state = state
    _send(f"EYES:{state}")


def get_state():
    return _state


# The Pi version exposed these; kept so nothing that imports them breaks.
def start():
    set_state("idle")


def stop():
    set_state("off")


if __name__ == "__main__":
    import sys
    import time
    if len(sys.argv) > 1:
        set_state(sys.argv[1])
    else:
        print("cycling every state, 4 seconds each. Ctrl-C to stop.\n")
        try:
            for s in VALID:
                print(f"  {s}")
                _state = None          # force the send
                set_state(s)
                time.sleep(4)
        except KeyboardInterrupt:
            print()
        _state = None
        set_state("idle")
