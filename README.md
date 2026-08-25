# Luna — Jetson Orin Assistant Robot

Luna is a humanoid assistant robot: continuous-listening voice interaction with a
wake word and follow-up conversation, GPU speech recognition, offline Q&A with a
GPT fallback, face recognition, scene description, a touch display, and two arms
with ten joints each driven from an ESP32-S3.

```
                    ┌──────────── Phone App ─────────────┐
                    │   joins LunaRobot WiFi              │
                    │   http://192.168.4.1:5000           │
                    └───────────────┬────────────────────┘
                                    │ REST
┌────────────────────┐  USB  ┌──────▼───────────────┐ UART ┌──────────────┐
│ ReSpeaker 4-Mic    │◄─────►│  Jetson Orin Nano    ├─────►│  ESP32-S3    │
│  · 6ch capture     │       │  Flask + faster-     │ 5/6  │ 2x PCA9685   │
│  · 3.5mm out       │       │  whisper (CUDA) +    │      │ 20 arm servos│
└────────────────────┘       │  Piper + YOLO +      │      │ 4 direct     │
                             │  InsightFace         │      │ TF-Luna      │
┌────────────────────┐       └──────┬───────────────┘      │ motors+latch │
│ USB Camera         │◄─────────────┤                      └──────────────┘
└────────────────────┘              │ HTTP (localhost)
                             ┌──────▼───────────────┐
                             │ Touch display        │
                             │ Brave kiosk /display │
                             └──────────────────────┘
```

---

## 1. Hardware

| Component | Details |
|---|---|
| Compute | **Jetson Orin Nano Super**, JetPack 6.2 (L4T R36.4.7), CUDA 12.6 |
| Microphone | **ReSpeaker USB 4-Mic Array** (XMOS XVF-3000). The only mic — no INMP441 |
| Speaker | Amplified speaker on the **ReSpeaker's own 3.5 mm jack** |
| Camera | USB webcam (UVC) |
| Display | HDMI/DSI touchscreen, Brave in kiosk mode |
| WiFi (internet) | Jetson internal, `wlP1p1s0` |
| WiFi (hotspot) | TP-Link TL-WN823N USB, `wlx…` — see §7 |
| MCU | ESP32-S3, UART2 on GPIO 5/6 ↔ Jetson `/dev/ttyTHS1` |
| Arm servos | 2× PCA9685 (0x40 left, 0x41 right), 10 channels per arm |
| Direct servos | lateral L/R, head left-right, head up-down |
| Distance | TF-Luna on I2C, address 0x10 |
| Power | ESP32 latch holds system power; button-initiated shutdown |

### The ReSpeaker matters more than it looks

It is **fixed at 16 kHz, 6 channels**, and cannot be asked for anything else.

| Channel | Contents |
|---|---|
| **0** | DSP output — beamformed, noise-suppressed. **Use this for everything** |
| 1–4 | raw microphone capsules |
| 5 | playback reference (silent unless audio is playing through the board) |

Channel 0 measured a **7 dB lower noise floor** than the raw capsules on this
build. Asking `arecord` for `-c 1` would make ALSA downmix all six, blending the
processed output with the raw mics *and* the playback reference — throwing away
the entire benefit of the board. The code captures all six and extracts channel 0.

---

## 2. Software stack

| Layer | Choice | Why |
|---|---|---|
| Wake word | Vosk small, open recognition | accepts any name at runtime |
| STT | **faster-whisper `small.en`, CUDA float16** | ~0% real WER on the command set |
| Endpointing | webrtcvad + an adaptive energy gate | see §6 |
| TTS | Piper `en_US-amy-medium`, streamed per sentence | first audio in ~250 ms |
| Q&A | `qa_store.json` → GPT fallback | local answers never wait on network |
| Vision | YOLOv8n (scene) + InsightFace (faces) | |
| Server | Flask on :5000, started from `main.py` | one process owns the port |
| Display | `display_addon.py` + `display.html` | glass UI, SSE-driven reactor |

Measured on this hardware, 6-second clips:

| Config | Real WER | Mean |
|---|---|---|
| `tiny` (multilingual) | 31% | 890 ms |
| `base.en` | ~0%* | 430 ms |
| **`small.en` cuda fp16 + prompt** | **~0%** | **870 ms** |
| `medium.en` | — | **OOM on 8 GB** |
| `distil-small.en` | 71% | truncates badly, do not use |

\* the residual WER is metric artifacts — "90" vs "ninety", "Bangalore" vs
"Bengaluru" — not errors. `small.en` transcribed the whole command set correctly.

---

## 3. Base system setup

### 3.1 Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git ffmpeg sox unzip \
    alsa-utils libportaudio2 python3-dev cmake build-essential \
    libopenblas-dev pkg-config brave-browser
sudo reboot
```

### 3.2 Performance mode

The Orin boots throttled. Everything below is measured at MAXN:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Add both to your boot sequence — a robot that responds slowly for its first
minute after power-on reads as broken.

### 3.3 Clone and venv

```bash
git clone https://github.com/HUTARK/luna-robot.git ~/pi_assistant
cd ~/pi_assistant
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. CUDA Whisper — the one hard dependency

**PyPI has no CUDA aarch64 wheel for CTranslate2.** `pip install faster-whisper`
gives you a CPU-only build, and passing `device="cuda"` then falls back to CPU
**silently** — the Orin performs like a Raspberry Pi with nothing in the logs to
explain it.

Check which you have:

```bash
python3 -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

An empty set or an error means CPU-only. Build it:

```bash
./build_ctranslate2_cuda.sh
```

That builds only `sm_87` (Orin's architecture) to keep compile time down, uses
OpenBLAS because MKL is x86-only, and verifies CUDA at the end. Expect 30–90
minutes. If nvcc runs out of memory on 8 GB, the script prints swapfile commands.

**Alternative:** NVIDIA's `jetson-containers` project ships prebuilt
faster-whisper images for JetPack 6. Less satisfying, but it works.

### 4.1 Download the weights INTO the project

```bash
python3 -c "
from faster_whisper import WhisperModel
for m in ('small.en','base.en'):
    WhisperModel(m, device='cpu', compute_type='float32',
                 download_root='models/whisper')
    print('cached', m)
"
```

**Not optional.** `~/.cache/huggingface` belongs to your user, but the systemd
service runs with a different `HOME` and cannot see it — the failure reads as
"Cannot find an appropriate cached snapshot folder", which looks like a model
problem but is a path problem. `models/whisper` works for both, and it means a
built unit ships with its weights rather than downloading on first boot at a
customer site.

### 4.2 Vosk wake-word model

```bash
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-small
rm vosk-model-small-en-us-0.15.zip
cd ..
```

A missing model directory is a **silent total failure**. Put this on the build
checklist.

---

## 5. Piper TTS — and the library conflict

Piper ships its own `libespeak-ng`, and the system one lacks a symbol it needs.
Scatter Piper's libraries into `/usr/local/lib` and the loader finds the system
copy first:

```
piper: symbol lookup error: libpiper_phonemize.so.1:
       undefined symbol: espeak_TextToPhonemesWithTerminator
```

Keep it self-contained instead:

```bash
cd /tmp
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
sudo mkdir -p /opt/piper && sudo cp -r piper/* /opt/piper/

sudo tee /usr/local/bin/piper > /dev/null <<'EOF'
#!/bin/bash
# Piper ships its own espeak-ng and onnxruntime. Without pinning
# LD_LIBRARY_PATH the loader finds the SYSTEM libespeak-ng first, which lacks
# espeak_TextToPhonemesWithTerminator, and piper dies at startup.
exec env LD_LIBRARY_PATH=/opt/piper:$LD_LIBRARY_PATH \
     ESPEAK_DATA_PATH=/opt/piper/espeak-ng-data \
     /opt/piper/piper "$@"
EOF
sudo chmod +x /usr/local/bin/piper
```

Voice:

```bash
cd ~/pi_assistant && mkdir -p voices && cd voices
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium
wget $B/en_US-amy-medium.onnx $B/en_US-amy-medium.onnx.json
cd ..
```

Test:

```bash
echo "test one two three" | piper --model voices/en_US-amy-medium.onnx --output_raw \
  | aplay -q -t raw -f S16_LE -r 22050 -c 1 -D plughw:0,0 -
```

---

## 6. Audio routing

### 6.1 Identify the ReSpeaker — by name, never by number

```bash
arecord -l
aplay -l
```

**Card numbers shuffle between boots.** On this build the ReSpeaker moved from
card 2 to card 0 across a single reboot. Every lookup in the code matches on
`ArrayUAC10`, then reads the number off whatever it found. Never hardcode it.

### 6.2 Speaker on the ReSpeaker's jack

Wire the amplified speaker into the ReSpeaker's 3.5 mm output, not the Jetson.
This is what gives the DSP a playback reference — see §6.4.

```bash
speaker-test -D plughw:0,0 -c 2 -t wav -l 1
```

### 6.3 Browser audio

Luna's voice goes through `aplay` directly. **Brave goes through PulseAudio**,
which ignores `/etc/asound.conf` entirely — so kiosk video plays silently until
the sink is set:

```bash
pactl list short sinks
pactl set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo
mkdir -p ~/.config/pulse
echo "set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo" \
  >> ~/.config/pulse/default.pa
```

`set-default-sink` only affects **new** streams, so restart Brave afterwards.

That sink reports 16 kHz, so PulseAudio resamples every video down. Speech is
clear, music is dull. Nothing to fix — it is the hardware.

### 6.4 Mic gain — the setting that breaks endpointing

The XVF-3000's AGC will amplify a quiet room until its noise floor sits around
−10 dBFS. `webrtcvad` then classifies that noise as speech, an utterance never
ends, and capture runs to the 15-second cap — sweeping up everything said
afterwards.

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/respeaker/usb_4_mic_array
cd usb_4_mic_array
~/pi_assistant/venv/bin/pip install pyusb
```

`tuning.py` predates Python 3.9 and uses a removed method:

```bash
sed -i 's/\.tostring()/\.tobytes()/g' tuning.py
```

Writing needs USB access:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/60-respeaker.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# UNPLUG AND REPLUG the board — udev only applies on device add
```

Then:

```bash
~/pi_assistant/venv/bin/python tuning.py AGCMAXGAIN 5
```

Verify — record 5 seconds of **silence**:

```bash
arecord -D plughw:0,0 -c 6 -r 16000 -f S16_LE -d 5 /tmp/q.wav
python3 -c "
import wave, numpy as np
w=wave.open('/tmp/q.wav'); d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
x=d.reshape(-1,6)[:,0].astype(float)
f=lambda v:-99 if v<=0 else 20*np.log10(v/32768)
print(f'ch0 silence peak {f(np.max(np.abs(x))):.1f} rms {f(np.sqrt(np.mean(x**2))):.1f}')
"
```

Want peak below −18, RMS around −30. At `AGCMAXGAIN 1000` this measured
peak 0.0 / RMS −9.7 — saturated on room noise.

> **These settings live in RAM and reset on every power cycle.** Without a
> service to reapply them, the 15-second captures come back after each reboot
> and the cause is invisible. Make it persistent.

### 6.5 Barge-in is OFF, and why

`ALLOW_BARGE_IN = False` in `luna_voice.py`. Interrupting Luna mid-sentence
requires the DSP to cancel its own voice from channel 0. Measured on this build:

| Test | Cancellation |
|---|---|
| pink noise, 3 s | 1.0 dB |
| real speech, 4 s convergence | 2.5 dB |
| after raising GAMMA_E/ETAIL/ENL and NLATTENONOFF | **1.4 dB — worse** |

Useful AEC is 20–30 dB. Channel 5 *does* carry a playback reference (−27.9 dBFS
while playing, −99 when silent), and `AECFREEZEONOFF` is 0 and `ECHOONOFF` is 1,
so it is enabled and adapting — it simply is not working on this board. Likely
the 3.5 mm jack is an analogue path the DSP has no matched reference for.

Turning barge-in on regardless makes Luna hear herself and interrupt herself
constantly. Leave it off; everything else works.

---

## 7. Networking — dual WiFi

```
Internal  wlP1p1s0      → router/phone → internet (GPT, OTA)
External  wlx…          → LunaRobot AP @ 192.168.4.1 → the app connects here
```

The internal interface is `wlP1p1s0`, **not** `wlan0`. The external name is
MAC-derived and differs per adapter — find it with `nmcli device status`.

### 7.1 RTL8192EU driver

Not in the Tegra kernel. Must be built:

```bash
sudo apt install -y dkms build-essential git
git clone https://github.com/clnhub/rtl8192eu-linux.git
cd rtl8192eu-linux
```

**Jetson-specific:** DKMS fails with `arch/aarch64/Makefile: No such file` —
Tegra kernel source uses `arm64`, not `aarch64`:

```bash
make ARCH=arm64 -j6
sudo make ARCH=arm64 install
sudo modprobe 8192eu
echo "8192eu" | sudo tee /etc/modules-load.d/8192eu.conf
```

> **Kernel updates break this.** The module is built for the exact running
> kernel. After any `apt upgrade` that installs a new one, the `wlx…` interface
> vanishes on reboot and must be rebuilt. Put it on the update checklist.

### 7.2 Internet on the internal adapter

```bash
sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASS" ifname wlP1p1s0
ping -c 3 8.8.8.8
```

### 7.3 Hotspot on the external adapter

Set the IP **at creation**, or NetworkManager assigns 10.42.0.1:

```bash
IFACE=$(nmcli -t -f DEVICE device status | grep '^wlx' | head -1)
sudo nmcli connection add type wifi ifname "$IFACE" con-name LunaRobot \
  autoconnect yes ssid LunaRobot \
  802-11-wireless.mode ap 802-11-wireless.band bg \
  ipv4.method shared ipv4.addresses 192.168.4.1/24 \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk 'luna1234'
sudo nmcli connection modify LunaRobot connection.autoconnect-priority 100
sudo nmcli connection up LunaRobot
ip addr show "$IFACE" | grep inet     # must be 192.168.4.1
```

Password minimum 8 characters. `autoconnect-priority 100` makes it win over
stale profiles on boot. These values must match `FACTORY_WIFI_*` in `main.py`,
or a factory reset moves the hotspot out from under the customer's phone.

> RTL8192EU AP mode is weaker than the internal radio and logs
> `rtl8xxxu_send_beacon_frame: Failed to read beacon valid bit` under load. If
> the app disconnects, swap the roles: internal = AP, external = internet.

---

## 8. OpenAI key

**Never in source.** Ben's key was committed to a public repo and scraped.

```bash
cd ~/pi_assistant
echo '{"openai_key": "sk-proj-YOUR-KEY"}' > ai_config.json
chmod 600 ai_config.json
grep -q '^ai_config.json$' .gitignore || echo "ai_config.json" >> .gitignore
```

`ai_fallback.py` reads `OPENAI_API_KEY` from the environment, then this file.
The app's settings page writes the same file.

---

## 9. Display kiosk

Launched from `~/.config/autostart/luna-display.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Luna Display
Exec=bash -c "sleep 10 && brave-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble --autoplay-policy=no-user-gesture-required --app=http://localhost:5000/display"
X-GNOME-Autostart-enabled=true
```

`--autoplay-policy=no-user-gesture-required` is **required**: Luna starts videos
from a voice command, so there is no user gesture on the page, and browsers block
unmuted autoplay without one — video plays silently.

Screen blanking:

```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

### Touch orientation

If the panel is mounted inverted, touch will be 180° out. Find the device name
with `libinput list-devices`, then:

```bash
echo 'SUBSYSTEM=="input", ATTRS{idVendor}=="0712", ATTRS{idProduct}=="000a", ENV{LIBINPUT_CALIBRATION_MATRIX}="-1 0 1 0 -1 1"' \
  | sudo tee /etc/udev/rules.d/99-touch-rotate.rules
sudo udevadm control --reload-rules && sudo reboot
```

Matching on USB IDs rather than the device name survives a firmware change that
alters the reported name.

---

## 10. ESP32-S3 firmware

Flash `esp32/luna_esp32_v3.ino` — board "ESP32S3 Dev Module", USB CDC On Boot
enabled. Libraries: **Adafruit PWM Servo Driver**, **ESP32Servo**.

### 10.1 Serial link

```
ESP32 GPIO 5 (TX) ──► Jetson RXD        /dev/ttyTHS1 @ 115200
ESP32 GPIO 6 (RX) ◄── Jetson TXD
ESP32 GND ────────── Jetson GND         required, not optional
```

TX goes to RX. Straight-through wiring is silent in both directions with no
error anywhere. `begin()` takes **RX first, then TX**.

Prove it before flashing the full firmware:

```bash
sudo systemctl stop piassistant
python3 serial_test.py PING
```

Loopback (jumper Jetson pin 8 to pin 10) isolates the Jetson's UART from
everything else — worth doing first if nothing arrives.

### 10.2 Joint map

Ten channels per arm, per PCA9685:

| Ch | Joint | Servo travel |
|---|---|---|
| 0 | shoulder | 180° |
| 1 | forearm | 180° |
| 2 | elbow | 180° |
| 3 | wrist left-right | 180° |
| 4 | wrist up-down | 180° |
| 5–9 | thumb, index, middle, ring, pinky | **360° positional** |

Direct GPIO: lateral L (1), lateral R (2), head LR (7), head UD (4).

**The fingers are 360° POSITIONAL servos**, not continuous-rotation. The
distinction matters: a continuous servo looks identical but its pulse sets
*speed*, and commanding one to an "angle" makes it spin until it stalls against
the finger's stop. If a finger ever runs and never stops, that is what you have.

`CH_SPAN[]` in the firmware carries each channel's travel, so 360° fingers use
the full pulse range. Mapping them over 0–180 like the other joints reaches only
half their travel.

### 10.3 Calibrating joint limits

Five tables, left and right kept separate because mirrored arms are never quite
symmetric once built:

```c
ARM_OUT_A   angle at app value 0        (put the LARGER angle here to reverse)
ARM_OUT_B   angle at app value 2000
ARM_HOME    park angle
ARM_MIN     hard clamp — nothing gets past these
ARM_MAX
```

Defaults reproduce the old behaviour exactly, so flashing changes nothing until
you measure. Per joint, arm supported by hand:

1. `POS:elbow:1000:left` and watch where it lands
2. creep the value until it reaches its mechanical stop
3. **the moment it buzzes or strains, back off** — that is the stop, and pushing
   through strips gears in seconds
4. back off ~5° from each end and write those into MIN/MAX

Send `LIMITS` over serial to print the whole table with current positions.

### 10.4 Known pin conflicts — read before wiring

**GPIO 22 and 23 do not exist on the ESP32-S3** (it has 0–21 and 26–48). With
`SDA_PIN 22` the I2C bus cannot start, so both PCA9685s and the TF-Luna are
unreachable and **no arm joint will move**. `LATCH_PIN 23` cannot hold or cut
power. Also `RPWM_L 35` and `SHUTDOWN_BTN 35` are the same pin.

Free on the S3, avoiding USB (19/20), strapping (0/3/45/46) and PSRAM (33–37):
**8, 9, 10, 11, 12, 13, 14, 17, 18, 39, 40, 41, 42, 47, 48**.

> A lesson from Ben worth carrying: initialising **UART2 on GPIO 5/6 killed the
> I2C bus** on that build — same pins, same board, only the order differed.
> Luna's firmware brings I2C up *before* the UARTs for that reason, and probes
> the PCA after each peripheral so a conflict names itself in one boot.

### 10.5 Eyes

WS2812 rings are driven from the **ESP32, not the Jetson**. NeoPixels need
sub-microsecond bit timing; on a Pi that comes from DMA into BCM PWM hardware,
and the Jetson has no equivalent — Blinka's Tegra backend cannot do it and
userspace bit-banging cannot hold the timing.

`eyes.py` is a shim that sends `EYES:<state>` down the existing serial link.
Paste `luna_eyes_esp32.ino` into the firmware for the animation.

Feed the rings from the **servo 5 V rail**, not the ESP32's 5 V pin — 18 LEDs at
full white is about 1 A and the onboard regulator will brown out mid-sentence.

---

## 11. Service

```bash
sudo tee /etc/systemd/system/piassistant.service > /dev/null <<'EOF'
[Unit]
Description=Luna Assistant
After=network.target

[Service]
User=root
WorkingDirectory=/home/groot/pi_assistant
ExecStartPre=/bin/sleep 10
ExecStart=/home/groot/pi_assistant/venv/bin/python3 /home/groot/pi_assistant/main.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=LD_LIBRARY_PATH=/home/groot/pi_assistant/venv/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now piassistant
```

**Only ever run one thing on port 5000.** `main.py` starts the Flask server
itself. A second unit running `server.py` separately will win the port at boot
and the real service silently serves nothing:

```bash
sudo lsof -i :5000        # expect exactly one python3 running main.py
```

`start_server()` currently swallows the bind error into a dead thread, so the
robot reports "Server started on port 5000" while starting nothing. Worth making
it exit loudly instead — see §14.

---

## 12. Verify the build

```bash
sudo journalctl -u piassistant -f
```

Expect, in order:

```
[MIC] streaming plughw:0,0 6ch@16000 -> ch0
[WAKE] Vosk open recognition, listening for 'luna'
[STT] loading small.en float16 on cuda...
[STT] ready: small.en float16 cuda, warm-up 1900ms
[LUNA] listening. barge-in=OFF
[VISION] YOLOv8 loaded
```

Then work through:

- [ ] say "Luna" → she answers "Yes?"
- [ ] ask a saved question → `[A] (local)` and the right answer
- [ ] ask something unsaved → `[A] (gpt …ms)` with a real answer
- [ ] ask a follow-up **without** the wake word → it is heard
- [ ] `[STT] turn 1:` duration is your speaking time plus ~0.65 s, **not 15 s**
- [ ] "Luna, what do you see" → scene description
- [ ] "Luna, play a video about lions" → picture **and sound**
- [ ] display: reactor goes amber during a GPT wait, transcript fills in
- [ ] display: face row **Capture** produces a non-black photo
- [ ] phone joins `LunaRobot`, app reaches `http://192.168.4.1:5000`
- [ ] app moves every joint; `LIMITS` matches the robot
- [ ] hold the ESP32 button 4 s → homes, then the Jetson **actually powers off**

That last one is worth being pedantic about. If the journal shows
`Scheduled restart job` instead of systemd stopping units, the Jetson never shut
down and the ESP32 cut its power anyway — an unclean power-off every time.

---

## 13. Troubleshooting

| Symptom | Cause |
|---|---|
| `[STT] Heard: ''` every time, ~15 s per attempt | Whisper being reloaded per utterance. Load once at module level |
| Utterances always exactly 15.0 s | AGC amplifying room noise so VAD never sees silence. `AGCMAXGAIN 5` (§6.4) |
| Endpointing broke after a reboot | ReSpeaker settings are volatile — reapply, and make it persistent |
| `Cannot find an appropriate cached snapshot folder` | Whisper weights in `~/.cache`, invisible to the service. Use `models/whisper` (§4.1) |
| Orin no faster than a Pi | CTranslate2 is a CPU build; `device="cuda"` fell back silently (§4) |
| `CUDA failed with error out of memory` | YOLO and Whisper competing for 8 GB, or fragmentation after long uptime. Reboot, or let the fallback chain drop to `base.en` |
| `piper: undefined symbol: espeak_TextToPhonemesWithTerminator` | System libespeak-ng shadowing Piper's. Use the `/opt/piper` wrapper (§5) |
| Kiosk video silent | PulseAudio sink not set (§6.3), or autoplay policy missing (§9) |
| `Address already in use` on :5000 | A second process. Note `systemctl stop` does not kill an already-orphaned child — find the PID with `lsof` |
| Touch inverted | Display rotated but the panel was not (§9) |
| `wlx…` gone after reboot | `8192eu` not loading, or a kernel update needs the driver rebuilt (§7.1) |
| Hotspot at 10.42.0.1 | Profile created before the address was set. Delete and recreate (§7.3) |
| No arm joint moves | GPIO 22 does not exist on the S3 — I2C never started (§10.4) |
| A finger runs and never stops | It is a continuous-rotation servo, not a 360° positional one (§10.2) |
| `POST /fingers 404` | The app calls an endpoint `server.py` does not have |
| Wake word fires on "lunar" | Vosk open recognition. Fix with a trained `wake_models/luna.onnx` |

---

## 14. Known issues

- **`POST /fingers` returns 404** — the app's finger control never reaches the
  ESP32. `server.py` needs the route.
- **`start_server()` swallows bind failures** into a dead thread. The robot
  claims the server started while the app and display are unreachable. It should
  `os._exit(1)` and let systemd retry.
- **`ai_fallback.py` `max_tokens=100`** — if the model is a reasoning model,
  reasoning tokens count against that and `content` comes back empty, which the
  bare `except` reports as a network failure.
- **Barge-in unavailable** — AEC measured 1–2.5 dB on this board (§6.5).
- **Wake word is Vosk open recognition**, which runs full ASR continuously and
  triggers on conversational mentions. A trained openWakeWord model at
  `wake_models/luna.onnx` is picked up automatically if present.
- **`STT_PROMPT` is hardcoded** in `main.py` while `robot_name` is configurable
  from the app, so renaming Luna does not update the recogniser's vocabulary.
- **`vision.get_camera()` holds the camera open** while the display's snapshot
  route opens it separately. They share a lock now, but any new camera user must
  use the same one.
- **ESP32 pin conflicts** unresolved — §10.4.

---

## 15. Safety

Luna has two arms with powered shoulders. Software limits (§10.3) stop a
*commanded* over-travel, and `estop` halts motion and holds position.

Neither protects against a servo driver latching on or the ESP32 hanging. For
that you need a **hardware contactor that cuts servo power**, with the arms
counterbalanced or mechanically supported so they do not fall when power is
removed. `relax`-style commands that simply stop PWM leave a raised arm limp,
which is worse than stopping.

On a bench this is theoretical. Around people — particularly children — it is a
matter of when. Design it in while the frame is still being built; retrofitting a
power-cut path into an assembled robot is much harder.
