#include <HardwareSerial.h>
#include <Wire.h>
#include <Adafruit_NeoPixel.h>
#include <ESP32Servo.h>
#include <Adafruit_PWMServoDriver.h>

/* ═══════════════════════════════════════════════════════════════════════
 * LUNA — ESP32-S3 firmware
 *
 * CHANGES IN THIS VERSION
 *   1. Pi link moved to GPIO 5 (TX) / 6 (RX) on UART2 — the pair you proved
 *      working in the loopback test.
 *   2. New joints the app is already sending: headUD and wristUD.
 *      From the Jetson log:
 *        POS:headUD:1000:left     POS:wristUD:1000:left
 *        POS:headLR:1000:left     POS:wristLR:1000:left
 *      The previous firmware had only one wrist and one head axis, so those
 *      two commands were silently dropped by the "[IGNORE]" branch.
 *   3. Ben's latch system: power held on at boot, multi-function button,
 *      LATCH:OFF from the Pi, slow homing before power is cut.
 *   4. Per-joint travel and safety limits, needed because the new joints
 *      would otherwise snap to 90 degrees at boot with nothing supporting
 *      them.
 *
 * PINS ARE UNCHANGED as requested, EXCEPT the two additions that had no pin
 * at all: HEAD_UD_PIN (GPIO 4) and the wristUD PCA channel.
 *
 * ─── PINS, ALIGNED WITH THE WORKING BEN BUILD ──────────────────────────
 *   I2C     SDA 14 / SCL 21     real pins, same as Ben
 *   LATCH   GPIO 8              was 23, which DOES NOT EXIST on the S3
 *   BUTTON  GPIO 11, pull-up    was 35, which is also RPWM_L
 *   Pi link GPIO 5 TX / 6 RX    UART2, proven by the loopback test
 *   Battery GPIO 12             carried over from Ben; the display polls it
 *
 * ─── TABLES ARE INDEXED BY PCA CHANNEL, NOT BY JOINT ───────────────────
 * ARM_CH is 13 because the electronics team wired up to channel 12:
 *     thumb 2   index 3   middle 4   ring 5   pinky 6
 *     forearm 8   elbow 9   wristLR 10   wristUD 11   shoulder 12
 *     (0, 1 and 7 unused)
 *
 * So every table below has TWELVE entries in CHANNEL order, not ten in joint
 * order. Writing ten leaves the last two zero-filled by C++, which set
 * wristLR and wristUD to MIN=0 MAX=0 — both wrists frozen solid — and shifted
 * every other joint onto another joint's limits. The column header above each
 * table names the joint on each channel; keep it aligned when editing.
 *
 * PCA_USED marks channels 1 and 7 as empty so nothing is written to them.
 */


// ── MOTOR PINS ─────────────────────────────────────────────
#define RPWM_L  35
#define LPWM_L  36
#define RPWM_R  37
#define LPWM_R  38
#define EN_L    16
#define EN_R    15
#define PWM_FREQ  1000
#define PWM_RES   8
/* The PWM at which the motors just begin to turn under load. Everything
 * below it only makes them buzz, so the slider's useful range starts here.
 *
 * SET THIS BY MEASUREMENT, because it decides how much of the slider does
 * anything. Too high and the whole range is compressed into a narrow band of
 * real speeds — at 120 the slider spans PWM 120-255, so "20%" and "50%" are
 * 145 and 188, which on a geared motor under load feel nearly the same.
 *
 * To find it: put the robot on blocks, send SPEED: values from 1 upward and
 * watch for the wheels starting to turn reliably WITH the robot's weight on
 * them — a motor that spins freely at PWM 60 may stall at 100 on the floor.
 * Set this a little above that point.
 */
#define MIN_MOVE_PWM 120
#define MAX_MOVE_PWM 255

// ── DIRECT GPIO SERVOS ─────────────────────────────────────
#define LATERAL_L_PIN 1
#define LATERAL_R_PIN 2
#define HEAD_LR_PIN   7
// NEW: head up/down had no pin. GPIO 4 is free on the S3 and clashes with
// nothing else in this file.
#define HEAD_UD_PIN   4

// ── PI LINK (UART2) ────────────────────────────────────────
// GPIO 5/6, confirmed working. NOTE the begin() argument order is RX first,
// then TX — reversing it gives a silent link with no error anywhere.
#define PI_TX_PIN     5
#define PI_RX_PIN     6

// ── I2C BUS (TF-Luna + 2x PCA9685) ─────────────────────────
#define SDA_PIN       14
#define SCL_PIN       21
#define LUNA_ADDR     0x10
#define OBSTACLE_DIST 50
#define DETECT_DIST   150
#define COOLDOWN_MS   10000

// ── LATCH + BUTTON ─────────────────────────────────────────
#define LATCH_PIN     8       // HIGH holds system power (Ben uses 8)
#define SHUTDOWN_BTN  11      // to GND, internal pull-up (Ben uses 11)

/* ── EYES ───────────────────────────────────────────────────────────────
 * Two WS2812 rings, nine LEDs each, moved here from the Pi.
 *
 * WHY THEY CANNOT RUN ON THE JETSON
 * WS2812 encodes each bit as a pulse width — 0.4us high for a zero, 0.8us
 * for a one — with about 150ns of tolerance at 800kHz. The Pi met that using
 * DMA driven by its PWM hardware, so the timing never touched the CPU. The
 * Orin has no equivalent, which is why Adafruit's library does not support
 * Jetson for NeoPixels: bit-banging from a preemptible kernel cannot hold
 * 150ns, and one scheduler tick mid-frame latches garbage into the strip.
 *
 * The ESP32 drives WS2812 from its RMT peripheral, which is hardware-timed
 * in the same way. It is also already wired to the Jetson, so the only thing
 * crossing the link is a state name.
 *
 * GPIO 9 and 10 are free on this board — every other pin is spoken for.
 * ───────────────────────────────────────────────────────────────────── */
#define EYE_L_PIN     9
#define EYE_R_PIN     10
#define EYE_LEDS      9
#define EYE_BRIGHT    77      // of 255. 0.3 on the Pi was 0.3*255 = 77.
#define LONG_PRESS_MS 3000
#define SHORT_MIN_MS  100
#define LATCH_COUNTDOWN 15      // seconds before power is cut

// ── PCA9685 — one per arm ──────────────────────────────────
Adafruit_PWMServoDriver pcaL = Adafruit_PWMServoDriver(0x40); // LEFT arm
Adafruit_PWMServoDriver pcaR = Adafruit_PWMServoDriver(0x41); // RIGHT arm (A0 jumper!)
#define SERVO_FREQ   50
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2400

/* ── SERVO SPEED ────────────────────────────────────────────────────────
 * The app sends TOPSPEED: on a 0-2000 scale, default 1000. The firmware used
 * to do constrain(value, 0, 100) — so EVERYTHING from 100 upward became 100,
 * i.e. maximum. The app's own default of 1000 was already flat out, and 95%
 * of the slider did nothing. That was the speed control problem.
 *
 * servoSpeed is now stored as the app's 0-2000 value and mapped LINEARLY to
 * degrees per second between the two limits below.
 *
 * WHY THIS CHANGED. The old code derived a tick interval AND an integer step
 * from servoSpeed, which produced:
 *      speed   0 ->   20 deg/s
 *      speed  50 ->  148 deg/s
 *      speed  75 ->  375 deg/s
 *      speed  90 ->  778 deg/s     faster than the servo can move
 *      speed 100 -> 1600 deg/s     meaningless
 * A hobby servo flat out is about 400 deg/s, so everything above ~75 was
 * identical in practice and the slider appeared to do nothing over most of
 * its range. The step was also an integer, so the slowest possible motion
 * was one whole degree per tick — visibly steppy.
 *
 * SERVO_DPS_MAX is deliberately inside what the servo can actually deliver,
 * so the whole slider does something.
 */
#define SERVO_TICK_MS   20        // 50Hz, the servo's own refresh rate.
                                  // Updating faster achieves nothing.
#define SPEED_SCALE   2000        // the app's slider maximum
#define SERVO_DPS_MIN   12.0f     // at app value 0
#define SERVO_DPS_MAX  320.0f     // at app value 2000 (a hobby servo flat out
                                  // is roughly 400 deg/s, so this stays inside
                                  // what the hardware can actually deliver)

// Homing has its own speed, used by both the app's HOME and the boot/shutdown
// home. A full-size arm swinging to home at full speed with nothing
// supporting it is violent, and at shutdown the power is about to be cut.
#define HOME_DPS        45.0f

// Per-channel ceiling, so a heavy joint cannot be driven fast even if the app
// asks for it. A shoulder carrying the weight of an arm at 320 deg/s can hurt
// someone; a finger at the same speed cannot. 0 means no extra limit.

// ── PCA CHANNEL MAP, as wired by electronics ───────────────────────────
// These ARE the hardware channel numbers, so they double as table indices.
#define CH_SHOULDER  12
#define CH_THUMB     2
#define CH_INDEX     3
#define CH_MIDDLE    4
#define CH_RING      5
#define CH_PINKY     6
#define CH_FOREARM   8
#define CH_ELBOW     9
#define CH_WRIST_LR  10
#define CH_WRIST_UD  11
#define ARM_CH       13       // must cover the HIGHEST channel used, + 1

// MUST come after #define ARM_CH. A macro does not exist before its
// #define, so declaring this above the channel map gave
//     error: 'ARM_CH' was not declared in this scope
// and then a cascade of 'CH_MAX_DPS was not declared' from every use.
//
// Sized by ARM_CH rather than a literal. It was [12] while ARM_CH became 13,
// so
// reading CH_MAX_DPS[12] for the shoulder was out of bounds — undefined
// behaviour in C++, not an error. It would have read whatever sits next in
// memory and applied it as a speed cap: possibly 0, possibly enormous.
//
// The shoulder's 90 deg/s cap has also moved from index 7 to index 12,
// following the wiring. Left at 7 it would have capped an unused channel
// while the shoulder ran uncapped — the one joint that most needs limiting.
//                     ch0    ch1   ch2   ch3   ch4   ch5   ch6   ch7   ch8   ch9  ch10  ch11  ch12
//                       --     --  thmb  indx   mid  ring  pnky    --  fore  elbw  wrLR  wrUD  shld
const float CH_MAX_DPS[ARM_CH] = {
                        0,     0,    0,    0,    0,    0,    0,    0,  140,  160,  220,  220,   90
};

// Channels 1 and 7 have nothing on them. Skipping them saves two I2C writes
// per servo tick and makes the gap deliberate rather than looking like a bug.
//                     ch0    ch1   ch2   ch3   ch4   ch5   ch6   ch7   ch8   ch9  ch10  ch11  ch12
//                       --     --  thmb  indx   mid  ring  pnky    --  fore  elbw  wrLR  wrUD  shld
const bool PCA_USED[ARM_CH] = {
  false, false, true , true , true , true , true , false, true , true , true , true , true
};

#define ARM_L 0
#define ARM_R 1

// Names only for the LIMITS printout, so the table is readable against the
// robot in front of you.
// Twelve entries, in CHANNEL order. The previous version had ten, so
// CH_NAME[10] and [11] were null pointers and printLimits() would pass NULL
// to printf("%s") — undefined behaviour, usually a crash.
const char* CH_NAME[ARM_CH] = {
  "--",       "--",      "thumb",   "index",
  "middle",   "ring",    "pinky",   "--",
  "forearm",  "elbow",   "wristLR", "wristUD", "shoulder"
};

/* ── HOW WIDE EACH JOINT'S TRAVEL IS ────────────────────────────────────
 * The fingers use 360-degree POSITIONAL servos. Same pulse range as a normal
 * servo, but that range spans 360 degrees instead of 180 — so mapping them
 * over 0-180 like everything else reached only HALF their travel, and a
 * finger asked for its maximum stopped halfway.
 *
 * Worth being clear these are POSITIONAL 360s, not continuous-rotation
 * servos. A continuous-rotation servo looks identical but its pulse sets
 * SPEED, not angle — 1500us is stop — and commanding one to an "angle" makes
 * it spin until it stalls against the finger's stop, drawing high current.
 * If a finger ever runs and never stops, that is what you have.
 */

const int CH_SPAN[ARM_CH] = {
  //  ch0  ch1  ch2  ch3  ch4  ch5  ch6  ch7  ch8  ch9 ch10 ch11
  //  shld  --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
      180, 180, 180, 180, 360, 360, 360, 180, 180, 180, 180, 180,180
};

/* ── PER-JOINT TRAVEL + LIMITS ──────────────────────────────────────────
 *   OUT_A   angle at app value 0
 *   OUT_B   angle at app value 2000   (put the LARGER angle in OUT_A to
 *                                      reverse a joint — no wiring change)
 *   HOME    park angle, used at boot and on HOME
 *   MIN/MAX hard clamp. Nothing gets past these: not a slider, not a pose,
 *           not HOME. Tighten them where an arm can hit the body; never
 *           widen them to make a pose reachable.
 *
 * Left and right are separate tables because mirrored arms are never quite
 * symmetric once built — horn splines land where they land.
 *
 * DEFAULTS BELOW reproduce the previous behaviour exactly (0-180, home 90),
 * so flashing this changes nothing until you measure and fill them in. Find
 * them one joint at a time with the arm supported: creep the value until the
 * joint reaches its stop, and THE MOMENT IT BUZZES back off — that is the
 * mechanical stop, and pushing through strips gears in seconds.
 */
//     ch:  0    1    2    3    4    5    6    7    8    9   10   11
//        shld   --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
int ARM_OUT_A[2][ARM_CH] = {
  /* LEFT  */ {   180,   0,   60,   0,   0,   0,   0,   90,   180,   90,   180,   0, 180 },
  /* RIGHT */ {   0,   0,   60,   0,   0,   0,   0,   180,   0,   90,   0,   0,  180 }
};
//     ch:  0    1    2    3    4    5    6    7    8    9   10   11
//        shld   --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
int ARM_OUT_B[2][ARM_CH] = {
  /* LEFT  */ { 90, 180, 0, 60, 360, 360, 360, 180, 0, 0, 0, 180, 70 },
  /* RIGHT */ { 90, 180, 0, 60, 360, 360, 360, 90, 180, 0, 180, 180, 70 }
};
//     ch:  0    1    2    3    4    5    6    7    8    9   10   11
//        shld   --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
int ARM_HOME[2][ARM_CH] = {
  /* LEFT  */ {  180,  0,   60,   0,   0,   0,   0,  90,  90,  90,  90,  90, 180 },
  /* RIGHT */ {  0,  0,   60,   0,   0,   0,   0,  180,  90,  90,  90,  90, 180 }
};
//     ch:  0    1    2    3    4    5    6    7    8    9   10   11
//        shld   --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
int ARM_MIN[2][ARM_CH] = {
  /* LEFT  */ {   0,   0,   0,   0,   0,   0,   0,   180,   0,   0,   0,   0, 0 },
  /* RIGHT */ {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0 }
};
//     ch:  0    1    2    3    4    5    6    7    8    9   10   11
//        shld   --  thmb indx  mid ring pnky   --  fore elbw wrLR wrUD
int ARM_MAX[2][ARM_CH] = {
  /* LEFT  */ { 180, 180, 180, 180, 360, 360, 360, 90, 180, 180, 180, 180, 180 },
  /* RIGHT */ { 180, 180, 180, 180, 360, 360, 360, 180, 180, 180, 180, 180, 180 }
};
// Direct GPIO servos. The right lateral is mirror-mounted, expressed as
// A > B rather than a 180-degree subtraction buried in moveServos().
#define LAT_L_A     180
#define LAT_L_B   0
#define LAT_L_HOME 180
#define LAT_L_MIN   0
#define LAT_L_MAX 180

#define LAT_R_A   0
#define LAT_R_B     180
#define LAT_R_HOME 0
#define LAT_R_MIN   0
#define LAT_R_MAX 180

#define HEAD_LR_A     0
#define HEAD_LR_B   180
#define HEAD_LR_HOME 90
#define HEAD_LR_MIN   0
#define HEAD_LR_MAX 180

#define HEAD_UD_A     50       // NEW
#define HEAD_UD_B   130
#define HEAD_UD_HOME 90
#define HEAD_UD_MIN   0
#define HEAD_UD_MAX 180

HardwareSerial PiSerial(2);
Servo lateralL, lateralR, headLRServo, headUDServo;

// ── STATE ──────────────────────────────────────────────────
int  motorSpeed = 255;
int  servoSpeed = 1000;      // the app's 0-2000 scale, default 1000
int  lunaDistance = 999;
bool obstacleAhead = false;
char currentDir[12] = "stop";
char savedDir[12]  = "stop";
unsigned long lastGreeted=0, lastLuna=0, lastServoUpdate=0, lastCmdTime=0;
bool robotMoving=false, hardwareEnabled=true, shutdownTriggered=false;
unsigned long btnPressStart=0;
bool btnWasPressed=false;
bool pcaOk = false;

/* Positions are FLOAT, not int.
 *
 * At the slowest speed a joint moves 0.24 degrees per 20ms tick. With integer
 * positions that rounds to zero and the joint never moves at all — so slow
 * speeds would simply not work. Float also removes the visible stepping that
 * whole-degree movement gives on a long slow sweep.
 */
float armCurrent[2][ARM_CH];
float armTarget[2][ARM_CH];
float latCurrentL=LAT_L_HOME,   latTargetL=LAT_L_HOME;
float latCurrentR=LAT_R_HOME,   latTargetR=LAT_R_HOME;
float headLRCurrent=HEAD_LR_HOME, headLRTarget=HEAD_LR_HOME;
float headUDCurrent=HEAD_UD_HOME, headUDTarget=HEAD_UD_HOME;

// Non-zero while homing: overrides servoSpeed so HOME is always gentle,
// whatever speed the app last set. Cleared when every joint has arrived.
float homingDps = 0.0f;

// ── HELPERS ────────────────────────────────────────────────
int appSpeedToPWM(int v) {
  // The app sends 1-100, default 50. This mapped over 0-200, so the app's
  // maximum reached PWM 187 of 255 — about 73% of full power — and the top
  // quarter of the range was unreachable. Ben has the same bug.
  v = constrain(v, 0, 100);
  if (v == 0) return 0;
  // MIN_MOVE_PWM is the stall threshold: below it the motors buzz and do not
  // turn, so the useful range starts there rather than at 0.
  return map(v, 1, 100, MIN_MOVE_PWM, MAX_MOVE_PWM);
}

int mapJoint(int value, int outA, int outB) {
  value = constrain(value, 0, 2000);
  long deg = outA + (long)(outB - outA) * value / 2000L;
  int lo = min(outA, outB), hi = max(outA, outB);
  return constrain((int)deg, lo, hi);
}

// The single place a PCA angle is written, so the clamp cannot be bypassed.
void pcaWriteDeg(int arm, int ch, float deg) {
  if (arm < 0 || arm > 1 || ch < 0 || ch >= ARM_CH) return;
  if (!PCA_USED[ch]) return;              // nothing wired to channels 1 and 7
  deg = constrain(deg, (float)ARM_MIN[arm][ch], (float)ARM_MAX[arm][ch]);
  const int span = CH_SPAN[ch];              // 180 for joints, 360 for fingers
  deg = constrain(deg, 0.0f, (float)span);
  // The pulse range is identical either way; only the number of degrees it
  // represents differs. Mapping a 360 servo over 0-180 uses half the pulse
  // range and therefore reaches half the travel.
  // Float maths, so a fraction of a degree still shifts the pulse width. The
  // integer map() would have thrown the fraction away.
  float us = SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * deg / (float)span;
  int ticks = (int)(us * 4096.0f / 20000.0f + 0.5f);
  // ch IS the hardware channel — the defines above are channel numbers.
  if (arm == ARM_L) pcaL.setPWM(ch, 0, ticks);
  else              pcaR.setPWM(ch, 0, ticks);
}

void setMotor(int pinFwd, int pinBwd, int speed, bool fwd) {
  if (fwd) { ledcWrite(pinFwd, speed); ledcWrite(pinBwd, 0); }
  else     { ledcWrite(pinFwd, 0);     ledcWrite(pinBwd, speed); }
}

void stopMotors() {
  digitalWrite(EN_L, LOW); digitalWrite(EN_R, LOW);
  ledcWrite(RPWM_L,0); ledcWrite(LPWM_L,0);
  ledcWrite(RPWM_R,0); ledcWrite(LPWM_R,0);
  strcpy(currentDir, "stop");
  robotMoving = false;
  Serial.println("[MOTOR] STOP");
}

void executeMove(const char* dir) {
  strncpy(currentDir, dir, 11); currentDir[11]='\0';
  if (strcmp(dir,"stop") != 0) {
    strncpy(savedDir, dir, 11); savedDir[11]='\0';
    robotMoving = true;
  } else robotMoving = false;

  if (strcmp(dir,"forward")==0 && obstacleAhead && hardwareEnabled) {
    stopMotors(); strcpy(savedDir,"stop");
    PiSerial.print("BLOCKED:"); PiSerial.println(lunaDistance);
    return;
  }
  if (strcmp(dir,"stop")==0) { stopMotors(); return; }

  digitalWrite(EN_L, HIGH); digitalWrite(EN_R, HIGH);
  if      (strcmp(dir,"forward")==0)  { setMotor(RPWM_L,LPWM_L,motorSpeed,true);  setMotor(RPWM_R,LPWM_R,motorSpeed,true);  }
  else if (strcmp(dir,"backward")==0) { setMotor(RPWM_L,LPWM_L,motorSpeed,false); setMotor(RPWM_R,LPWM_R,motorSpeed,false); }
  else if (strcmp(dir,"left")==0)     { setMotor(RPWM_L,LPWM_L,motorSpeed,false); setMotor(RPWM_R,LPWM_R,motorSpeed,true);  }
  else if (strcmp(dir,"right")==0)    { setMotor(RPWM_L,LPWM_L,motorSpeed,true);  setMotor(RPWM_R,LPWM_R,motorSpeed,false); }
  Serial.print("[MOTOR] "); Serial.println(dir);
}

// ── SMOOTH SERVO UPDATE ────────────────────────────────────
void updateServos() {
  unsigned long now = millis();
  if (now - lastServoUpdate < SERVO_TICK_MS) return;
  // Use the ACTUAL elapsed time, not the nominal tick. If the loop is held up
  // by a long I2C write or a serial burst, a fixed step would make the joint
  // move slower than asked; using real elapsed time keeps the commanded speed
  // honest.
  float dt = (now - lastServoUpdate) / 1000.0f;
  if (dt > 0.2f) dt = 0.2f;          // after a long stall, do not lurch
  lastServoUpdate = now;

  // servoSpeed 0-100 -> degrees per second, linear. Homing overrides it.
  float dps = (homingDps > 0.0f) ? homingDps
            : SERVO_DPS_MIN + (SERVO_DPS_MAX - SERVO_DPS_MIN)
                * constrain(servoSpeed, 0, SPEED_SCALE) / (float)SPEED_SCALE;

  for (int a = 0; a < 2; a++)
    for (int ch = 0; ch < ARM_CH; ch++) {
      if (!PCA_USED[ch]) continue;
      float d = armTarget[a][ch] - armCurrent[a][ch];
      if (fabsf(d) < 0.05f) {
        if (armCurrent[a][ch] != armTarget[a][ch]) {
          armCurrent[a][ch] = armTarget[a][ch];
          pcaWriteDeg(a, ch, armCurrent[a][ch]);
        }
        continue;
      }
      // Per-joint ceiling: a shoulder stays slow even at servoSpeed 100.
      float jdps = dps;
      if (CH_MAX_DPS[ch] > 0.0f && jdps > CH_MAX_DPS[ch]) jdps = CH_MAX_DPS[ch];
      float step = jdps * dt;
      armCurrent[a][ch] += (d > 0) ? min(step, d) : max(-step, d);
      pcaWriteDeg(a, ch, armCurrent[a][ch]);
    }

  // Direct GPIO servos. Same velocity model, written as whole degrees because
  // the Servo library takes an int.
  struct Direct { float *cur, *tgt; Servo *s; int lo, hi; };
  Direct dsv[4] = {
    { &latCurrentL,   &latTargetL,   &lateralL,    LAT_L_MIN,   LAT_L_MAX   },
    { &latCurrentR,   &latTargetR,   &lateralR,    LAT_R_MIN,   LAT_R_MAX   },
    { &headLRCurrent, &headLRTarget, &headLRServo, HEAD_LR_MIN, HEAD_LR_MAX },
    { &headUDCurrent, &headUDTarget, &headUDServo, HEAD_UD_MIN, HEAD_UD_MAX },
  };
  for (int i = 0; i < 4; i++) {
    float d = *dsv[i].tgt - *dsv[i].cur;
    if (fabsf(d) < 0.05f) { *dsv[i].cur = *dsv[i].tgt; continue; }
    float step = dps * dt;
    *dsv[i].cur += (d > 0) ? min(step, d) : max(-step, d);
    dsv[i].s->write((int)(constrain(*dsv[i].cur, (float)dsv[i].lo,
                                    (float)dsv[i].hi) + 0.5f));
  }

  // Homing finished? Drop the override so the app's speed applies again.
  if (homingDps > 0.0f && servosAtTarget()) {
    homingDps = 0.0f;
    Serial.println("[SERVO] home reached");
  }
}

// Arrival tolerance in degrees. NOT called EPS: the Xtensa headers already
// define EPS as 192 (a special-register number), so "const float EPS" expands
// to "const float 192" and the file will not compile. Short generic names are
// a hazard in a file that includes the ESP32 core.
#define ARRIVE_DEG 0.1f

bool servosAtTarget() {
  // Tolerance, not equality: these are floats now, and exact comparison would
  // occasionally never be true.
  for (int a = 0; a < 2; a++)
    for (int ch = 0; ch < ARM_CH; ch++)
      if (PCA_USED[ch] && fabsf(armCurrent[a][ch]-armTarget[a][ch]) > ARRIVE_DEG)
        return false;
  return fabsf(latCurrentL-latTargetL) <= ARRIVE_DEG
      && fabsf(latCurrentR-latTargetR) <= ARRIVE_DEG
      && fabsf(headLRCurrent-headLRTarget) <= ARRIVE_DEG
      && fabsf(headUDCurrent-headUDTarget) <= ARRIVE_DEG;
}

// ── PART → PCA CHANNEL ─────────────────────────────────────
int partToChannel(const char* part) {
  if (strcmp(part,"shoulder")==0) return CH_SHOULDER;
  if (strcmp(part,"forearm")==0)  return CH_FOREARM;
  if (strcmp(part,"elbow")==0)    return CH_ELBOW;
  // "wrist" with no axis is treated as left/right, so an older app that only
  // knows one wrist keeps working.
  if (strcmp(part,"wrist")==0 || strcmp(part,"wristLR")==0) return CH_WRIST_LR;
  if (strcmp(part,"wristUD")==0)  return CH_WRIST_UD;
  if (strcmp(part,"thumb")==0  || strcmp(part,"finger_thumb")==0)  return CH_THUMB;
  if (strcmp(part,"index")==0  || strcmp(part,"finger_index")==0)  return CH_INDEX;
  if (strcmp(part,"middle")==0 || strcmp(part,"finger_middle")==0) return CH_MIDDLE;
  if (strcmp(part,"ring")==0   || strcmp(part,"finger_ring")==0)   return CH_RING;
  if (strcmp(part,"pinky")==0  || strcmp(part,"finger_pinky")==0)  return CH_PINKY;
  return -1;
}

void moveServos(const char* part, int value, const char* hand) {
  value = constrain(value, 0, 2000);
  bool doL = (strcmp(hand,"left")==0  || strcmp(hand,"both")==0);
  bool doR = (strcmp(hand,"right")==0 || strcmp(hand,"both")==0);

  if (strcmp(part,"lateral")==0) {
    if (doL) latTargetL = constrain(mapJoint(value, LAT_L_A, LAT_L_B), LAT_L_MIN, LAT_L_MAX);
    if (doR) latTargetR = constrain(mapJoint(value, LAT_R_A, LAT_R_B), LAT_R_MIN, LAT_R_MAX);
    Serial.printf("[SERVO] lateral(%s) app=%d -> L%.0f R%.0f\n",
                  hand, value, latTargetL, latTargetR);
    return;
  }
  // Head is a single physical head: "both" and either hand all drive it.
  if (strcmp(part,"headLR")==0 || strcmp(part,"head")==0) {
    headLRTarget = constrain(mapJoint(value, HEAD_LR_A, HEAD_LR_B),
                             HEAD_LR_MIN, HEAD_LR_MAX);
    Serial.printf("[SERVO] headLR app=%d -> %.0f\n", value, headLRTarget);
    return;
  }
  if (strcmp(part,"headUD")==0) {          // NEW
    headUDTarget = constrain(mapJoint(value, HEAD_UD_A, HEAD_UD_B),
                             HEAD_UD_MIN, HEAD_UD_MAX);
    Serial.printf("[SERVO] headUD app=%d -> %.0f\n", value, headUDTarget);
    return;
  }

  if (strcmp(part,"fingers")==0) {
    // Channels 2-6 are the five fingers, contiguous, so a simple range works.
    for (int ch = CH_THUMB; ch <= CH_PINKY; ch++) {
      if (doL) armTarget[ARM_L][ch] =
        constrain(mapJoint(value, ARM_OUT_A[ARM_L][ch], ARM_OUT_B[ARM_L][ch]),
                  ARM_MIN[ARM_L][ch], ARM_MAX[ARM_L][ch]);
      if (doR) armTarget[ARM_R][ch] =
        constrain(mapJoint(value, ARM_OUT_A[ARM_R][ch], ARM_OUT_B[ARM_R][ch]),
                  ARM_MIN[ARM_R][ch], ARM_MAX[ARM_R][ch]);
    }
    Serial.printf("[SERVO] fingers(%s) app=%d\n", hand, value);
    return;
  }

  int ch = partToChannel(part);
  if (ch < 0) { Serial.printf("[IGNORE] %s\n", part); return; }
  if (doL) armTarget[ARM_L][ch] =
    constrain(mapJoint(value, ARM_OUT_A[ARM_L][ch], ARM_OUT_B[ARM_L][ch]),
              ARM_MIN[ARM_L][ch], ARM_MAX[ARM_L][ch]);
  if (doR) armTarget[ARM_R][ch] =
    constrain(mapJoint(value, ARM_OUT_A[ARM_R][ch], ARM_OUT_B[ARM_R][ch]),
              ARM_MIN[ARM_R][ch], ARM_MAX[ARM_R][ch]);
  Serial.printf("[SERVO] %s(%s) app=%d -> L%.0f R%.0f%s\n",
                part, hand, value, armTarget[ARM_L][ch], armTarget[ARM_R][ch],
                pcaOk ? "" : "  [PCA NOT OK]");
}

void setHomeTargets() {
  for (int a = 0; a < 2; a++)
    for (int ch = 0; ch < ARM_CH; ch++)
      armTarget[a][ch] = constrain(ARM_HOME[a][ch],
                                   ARM_MIN[a][ch], ARM_MAX[a][ch]);
  latTargetL   = constrain(LAT_L_HOME,   LAT_L_MIN,   LAT_L_MAX);
  latTargetR   = constrain(LAT_R_HOME,   LAT_R_MIN,   LAT_R_MAX);
  headLRTarget = constrain(HEAD_LR_HOME, HEAD_LR_MIN, HEAD_LR_MAX);
  headUDTarget = constrain(HEAD_UD_HOME, HEAD_UD_MIN, HEAD_UD_MAX);
}

void homeServos() {
  // The app's HOME. Sets the override so homing is always gentle regardless
  // of whatever speed the app last sent, and returns immediately — loop()
  // keeps running, so serial and the obstacle sensor stay live while the arm
  // moves. The override clears itself in updateServos() on arrival.
  setHomeTargets();
  homingDps = HOME_DPS;
  Serial.printf("[SERVO] homing at %.0f deg/s\n", HOME_DPS);
}

/* ── STARTUP SEQUENCE ───────────────────────────────────────────────────
 * A slow, symmetric movement when the robot powers on.
 *
 * It is also a check. Watching it tells you, before anybody asks the robot
 * anything, whether every joint is alive and moving the right way — which is
 * otherwise only discovered when a gesture looks wrong halfway through a
 * demonstration.
 *
 * ── WHY IT FLOWS RATHER THAN STEPS ──────────────────────────────────
 * The obvious way is: set a target, wait for arrival, set the next. That
 * reads as a machine working through a list — every joint stops dead before
 * the next begins.
 *
 * Instead each movement is given a DURATION and the next begins partway
 * through, so joints overlap and the whole thing reads as one continuous
 * gesture. calibHold() waits a fixed time; calibSettle() waits for arrival,
 * and is used only where a pose must be complete before the next begins.
 *
 * ── SYMMETRY ────────────────────────────────────────────────────────
 * calibArm moves both arms to the same FRACTION of their own travel, not to
 * the same angle. The arms are mirrored, so ARM_OUT_A/B differ between them —
 * the same angle would send one arm out and the other into the body. A
 * fraction maps through each arm's own table and comes out symmetric.
 *
 * If the two arms do NOT look symmetric during this sequence, the OUT_A/OUT_B
 * tables are wrong for one of them, and every gesture will be lopsided too.
 * That is the single most useful thing this sequence tells you.
 * ───────────────────────────────────────────────────────────────────── */
#define CALIB_ON_BOOT   true
#define CALIB_DPS       18.0f    /* degrees per second. HOME_DPS is 45, so
                                    this is deliberately much gentler. Lower
                                    is slower and smoother; the whole
                                    sequence scales with it. */
#define CALIB_REACH     0.30f    /* how far the arm joints travel, as a
                                    fraction of their range. This is a check
                                    that a joint MOVES, not a demonstration
                                    of full travel — and a full-size arm
                                    sweeping its whole range on power-up,
                                    with nobody expecting it, is alarming. */

// Keep the servos updating for a fixed time, without waiting for arrival.
// This is what lets one movement begin while the last is still travelling.
static void calibHold(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    updateServos();
    eyesTick();              // keep the eyes alive during the boot sequence
    delay(2);
  }
}

// Wait until every joint has arrived, or give up. Used where a pose must be
// complete — the pause at full extension, and the final settle.
static void calibSettle(unsigned long limitMs) {
  unsigned long t0 = millis();
  while (millis() - t0 < limitMs) {
    updateServos();
    if (servosAtTarget()) break;
    delay(2);
  }
}

// Set one arm channel on BOTH arms to a fraction of its own travel.
// Does not wait: the caller decides how much to overlap.
static void calibArm(int ch, float f) {
  if (ch < 0 || ch >= ARM_CH || !PCA_USED[ch]) return;
  for (int a = 0; a < 2; a++) {
    int lo = ARM_OUT_A[a][ch], hi = ARM_OUT_B[a][ch];
    float deg = lo + (hi - lo) * f;
    armTarget[a][ch] = constrain((int)(deg + 0.5f),
                                 ARM_MIN[a][ch], ARM_MAX[a][ch]);
  }
}

static void calibFingers(int appValue) {
  for (int ch = CH_THUMB; ch <= CH_PINKY; ch++)
    for (int a = 0; a < 2; a++)
      armTarget[a][ch] = constrain(mapJoint(appValue, ARM_OUT_A[a][ch],
                                            ARM_OUT_B[a][ch]),
                                   ARM_MIN[a][ch], ARM_MAX[a][ch]);
}

static void calibHead(float lrF, float udF) {
  headLRTarget = constrain(mapJoint((int)(lrF * 2000), HEAD_LR_A, HEAD_LR_B),
                           HEAD_LR_MIN, HEAD_LR_MAX);
  headUDTarget = constrain(mapJoint((int)(udF * 2000), HEAD_UD_A, HEAD_UD_B),
                           HEAD_UD_MIN, HEAD_UD_MAX);
}

void startupSequence() {
  if (!CALIB_ON_BOOT) return;

  Serial.println("[CALIB] startup - both arms should move symmetrically");
  homingDps = CALIB_DPS;          // overrides servoSpeed for the whole run

  // ── 1. Waking. The head lifts on its own, which reads as the robot
  //       coming to rather than machinery starting.
  Serial.println("[CALIB] waking");
  calibHead(0.50f, 0.64f);
  calibHold(2600);

  // ── 2. The arms open. Shoulder first, then the forearm joins while the
  //       shoulder is still travelling, then the elbow. Each overlaps the
  //       last, so the arm unfolds in one continuous movement instead of
  //       three separate ones.
  Serial.println("[CALIB] opening");
  calibArm(CH_SHOULDER, CALIB_REACH);
  calibHold(1800);
  calibArm(CH_FOREARM,  CALIB_REACH);
  calibHold(1800);
  calibArm(CH_ELBOW,    CALIB_REACH);
  calibHold(2200);

  // ── 3. Wrists turn outward, palms opening. Both wrists together, both
  //       arms together.
  Serial.println("[CALIB] wrists");
  calibArm(CH_WRIST_LR, 0.62f);
  calibArm(CH_WRIST_UD, 0.60f);
  calibHold(2600);

  // ── 4. Fingers open. On 360-degree servos this is the longest single
  //       movement in the sequence, so it gets the time to complete.
  Serial.println("[CALIB] fingers");
  calibFingers(620);
  calibSettle(9000);

  // ── 5. A moment held at full extension, then the head looks slowly left
  //       and right — surveying the room while the arms stay open. Two
  //       things moving at different rates is what stops it looking
  //       mechanical.
  Serial.println("[CALIB] looking around");
  calibHead(0.34f, 0.56f);
  calibHold(3400);
  calibHead(0.66f, 0.56f);
  calibHold(4200);
  calibHead(0.50f, 0.50f);
  calibHold(2600);

  // ── 6. Everything settles home together, in one slow movement. Sent as
  //       a single set of targets rather than joint by joint, so the whole
  //       body closes at once rather than in sequence.
  Serial.println("[CALIB] settling");
  setHomeTargets();
  calibSettle(20000);

  homingDps = 0.0f;               // hand the speed back to servoSpeed
  Serial.println("[CALIB] done");
}

void homeServosSlow(const char* why) {
  // Boot and shutdown: home slowly AND block until done, because at shutdown
  // the power is about to be cut and a joint must not be left mid-travel.
  setHomeTargets();
  homingDps = HOME_DPS;
  Serial.printf("[SERVO] slow home (%s) at %.0f deg/s...\n", why, HOME_DPS);
  unsigned long t0 = millis();
  // Timeout scaled to the speed: at 45 deg/s a 180-degree joint needs 4s, so
  // a fixed 6s was uncomfortably tight and would abandon a long sweep.
  const unsigned long limit = (unsigned long)(360.0f / HOME_DPS * 1000.0f) + 2000;
  while (millis() - t0 < limit) {
    updateServos();
    if (servosAtTarget()) break;
    delay(2);
  }
  homingDps = 0.0f;
  Serial.printf("[SERVO] HOMED in %lums\n", millis() - t0);
}

void printLimits() {
  Serial.println("[LIMITS] arm  joint      ch span    A    B  HOME  MIN  MAX  now");
  for (int a = 0; a < 2; a++) {
    for (int ch = 0; ch < ARM_CH; ch++) {
      if (!PCA_USED[ch]) continue;          // channels 1 and 7 are empty
      Serial.printf("[LIMITS]  %s   %-9s %3d %4d %4d %4d %5d %4d %4d %4d\n",
        a == ARM_L ? "L" : "R", CH_NAME[ch], ch, CH_SPAN[ch],
        ARM_OUT_A[a][ch], ARM_OUT_B[a][ch], ARM_HOME[a][ch],
        ARM_MIN[a][ch], ARM_MAX[a][ch], (int)(armCurrent[a][ch] + 0.5f));
    }
  }
  Serial.printf("[LIMITS]  -   lateralL  -  180 %4d %4d %5d %4d %4d %4d\n",
    LAT_L_A, LAT_L_B, LAT_L_HOME, LAT_L_MIN, LAT_L_MAX, (int)(latCurrentL+0.5f));
  Serial.printf("[LIMITS]  -   lateralR  -  180 %4d %4d %5d %4d %4d %4d\n",
    LAT_R_A, LAT_R_B, LAT_R_HOME, LAT_R_MIN, LAT_R_MAX, (int)(latCurrentR+0.5f));
  Serial.printf("[LIMITS]  -   headLR    -  180 %4d %4d %5d %4d %4d %4d\n",
    HEAD_LR_A, HEAD_LR_B, HEAD_LR_HOME, HEAD_LR_MIN, HEAD_LR_MAX, (int)(headLRCurrent+0.5f));
  Serial.printf("[LIMITS]  -   headUD    -  180 %4d %4d %5d %4d %4d %4d\n",
    HEAD_UD_A, HEAD_UD_B, HEAD_UD_HOME, HEAD_UD_MIN, HEAD_UD_MAX, (int)(headUDCurrent+0.5f));
}

// ── TF-LUNA (on the same I2C bus) ──────────────────────────
void readLuna() {
  if (!hardwareEnabled) return;
  if (millis() - lastLuna < 100) return;
  lastLuna = millis();

  Wire.beginTransmission(LUNA_ADDR);
  Wire.write(0x00);
  if (Wire.endTransmission(false) != 0) return;
  Wire.requestFrom((uint8_t)LUNA_ADDR, (uint8_t)2);
  if (Wire.available() < 2) return;
  // Named variables, because C++ does not define the evaluation order of the
  // operands of |, so "Wire.read() | (Wire.read() << 8)" can take the bytes
  // either way round and the distance comes out byte-swapped at random.
  uint8_t lo = Wire.read();
  uint8_t hi = Wire.read();
  int dist = lo | (hi << 8);
  if (dist <= 0 || dist > 800) return;

  lunaDistance  = dist;
  obstacleAhead = (dist < OBSTACLE_DIST);

  if (obstacleAhead && strcmp(currentDir,"forward")==0) {
    stopMotors(); strcpy(savedDir,"stop");
    PiSerial.print("BLOCKED:"); PiSerial.println(dist);
  }
  if (dist < DETECT_DIST && !robotMoving && strcmp(currentDir,"stop")==0
      && millis() - lastGreeted > COOLDOWN_MS) {
    lastGreeted = millis();
    PiSerial.print("PERSON_DETECTED:"); PiSerial.println(dist);
  }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LATCH SYSTEM  (as Ben)
 *
 * The ESP32 holds its own power on through LATCH_PIN. Shutdown is always the
 * same sequence, whoever starts it:
 *     stop motors -> home slowly -> tell the Pi -> count down -> cut power
 *
 * The countdown exists so the Pi can finish its own shutdown. Cutting power
 * on a running Linux system corrupts SD cards, and that is exactly what
 * happens if the Pi's shutdown silently fails — so the Pi side must actually
 * power off within LATCH_COUNTDOWN seconds.
 * ═══════════════════════════════════════════════════════════════════════ */

void doLatchShutdown(const char* why) {
  if (shutdownTriggered) return;
  shutdownTriggered = true;
  Serial.printf("[LATCH] Shutdown (%s)\n", why);
  stopMotors();
  homeServosSlow("shutdown");
  PiSerial.println("SHUTDOWN");          // the Pi begins its own poweroff
  for (int i = LATCH_COUNTDOWN; i > 0; i--) {
    Serial.printf("[LATCH] %ds\n", i);
    delay(1000);
  }
  Serial.println("[LATCH] power off");
  digitalWrite(LATCH_PIN, LOW);
}

void checkButton() {
  if (shutdownTriggered) return;
  // INPUT_PULLUP with the button to GND, so pressed reads LOW.
  bool pressed = (digitalRead(SHUTDOWN_BTN) == LOW);

  if (pressed && !btnWasPressed) {
    btnPressStart = millis();
    btnWasPressed = true;
  }
  else if (pressed && btnWasPressed &&
           millis() - btnPressStart >= LONG_PRESS_MS) {
    Serial.println("[BTN] long press");
    doLatchShutdown("button");
  }
  else if (!pressed && btnWasPressed) {
    unsigned long held = millis() - btnPressStart;
    btnWasPressed = false;
    if (held >= SHORT_MIN_MS && held < LONG_PRESS_MS) {
      // A short press does NOT factory reset. On Ben a 0.1s tap wiped the
      // Q&A store, settings and every saved WiFi connection with no
      // confirmation — a brush against the button destroyed customer data.
      Serial.printf("[BTN] short press (%lums) — ignored\n", held);
    }
  }
}

// ── SERIAL HANDLER ─────────────────────────────────────────
// Reads BOTH the Pi link and USB, so diagnostics can be typed straight into
// the Serial Monitor without unplugging anything.
void processCommand(char* buf);

void handleSerial() {
  static char piBuf[96];
  static int  piLen = 0;
  while (PiSerial.available()) {
    char c = PiSerial.read();
    if (c == '\n' || c == '\r') {
      if (piLen > 0) { piBuf[piLen] = '\0'; processCommand(piBuf); piLen = 0; }
      continue;
    }
    if (c < 32 || c > 126) continue;
    if (piLen < 95) piBuf[piLen++] = c; else piLen = 0;
  }
  // NOTE: no buffer flush here. The old code did
  //     while (PiSerial.available()) PiSerial.read();
  // after each line, which threw away any command that arrived while the
  // first was being handled — and the app sends finger commands in bursts,
  // so most of a gesture was lost.

  static char usbBuf[96];
  static int  usbLen = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (usbLen > 0) { usbBuf[usbLen] = '\0'; processCommand(usbBuf); usbLen = 0; }
      continue;
    }
    if (c < 32 || c > 126) continue;
    if (usbLen < 95) usbBuf[usbLen++] = c; else usbLen = 0;
  }
}

void processCommand(char* buf) {
  Serial.printf("[CMD] %s\n", buf);

  if (strncmp(buf,"MOVE:",5)==0) { lastCmdTime = millis(); executeMove(buf+5); }
  else if (strncmp(buf,"SPEED:",6)==0) {
    int app = atoi(buf+6);
    motorSpeed = appSpeedToPWM(app);
    // Report the PWM actually applied. Without this there is no way to tell
    // a speed command that was ignored from one that landed but made little
    // difference — and those need opposite fixes.
    Serial.printf("[SPEED] motor %d/100 -> PWM %d/255 (band %d-%d)\n",
                  app, motorSpeed, MIN_MOVE_PWM, MAX_MOVE_PWM);
    if (robotMoving) executeMove(currentDir);
  }
  else if (strncmp(buf,"TOPSPEED:",9)==0) {
    servoSpeed = constrain(atoi(buf+9), 0, SPEED_SCALE);
    float d = SERVO_DPS_MIN + (SERVO_DPS_MAX - SERVO_DPS_MIN)
                              * servoSpeed / (float)SPEED_SCALE;
    Serial.printf("[SPEED] servo %d/%d = %.0f deg/s\n",
                  servoSpeed, SPEED_SCALE, d);
  }
  else if (strcmp(buf,"SPEED?")==0) {
    float d = SERVO_DPS_MIN + (SERVO_DPS_MAX - SERVO_DPS_MIN)
                              * servoSpeed / (float)SPEED_SCALE;
    Serial.printf("[SPEED] servo %d/%d = %.0f deg/s (range %.0f-%.0f), "
                  "home %.0f deg/s%s\n",
                  servoSpeed, SPEED_SCALE, d, SERVO_DPS_MIN, SERVO_DPS_MAX,
                  HOME_DPS, homingDps > 0 ? ", HOMING now" : "");
    Serial.printf("[SPEED] motor %d/255 PWM\n", motorSpeed);
    for (int ch = 0; ch < ARM_CH; ch++)
      if (PCA_USED[ch] && CH_MAX_DPS[ch] > 0)
        Serial.printf("[SPEED]   %-9s capped at %.0f deg/s\n",
                      CH_NAME[ch], CH_MAX_DPS[ch]);
  }
  else if (strncmp(buf,"HOME",4)==0) homeServos();
  else if (strcmp(buf,"LIMITS")==0)  printLimits();
  else if (strcmp(buf,"RESUME")==0)  executeMove(savedDir);
  else if (strncmp(buf,"POS:",4)==0) {
    char* p1 = strchr(buf+4, ':'); if (!p1) return;
    char* p2 = strchr(p1+1, ':');  if (!p2) return;
    *p1='\0'; *p2='\0';
    moveServos(buf+4, atoi(p1+1), p2+1);
  }
  else if (strcmp(buf,"HARDWARE:ON")==0)  { hardwareEnabled=true;  obstacleAhead=false; PiSerial.println("HW:ON");  }
  else if (strcmp(buf,"HARDWARE:OFF")==0) { hardwareEnabled=false; obstacleAhead=false; PiSerial.println("HW:OFF"); }
  else if (strncmp(buf, "EYES:", 5) == 0) {
    // From the Pi: EYES:idle, EYES:listening, EYES:thinking, EYES:speaking,
    // EYES:eyes1, EYES:eyes2, EYES:eyes3, EYES:off
    //
    // There used to be a SECOND handler for this prefix, earlier in the
    // chain, left over from before the eyes existed:
    //     else if (strncmp(buf,"EYES:",5)==0) Serial.printf("[EYES] %s\n", buf+5);
    // It only printed. Being first in an else-if chain it swallowed every
    // command and this branch was unreachable — so the serial monitor showed
    // "[EYES] eyes2" while the rings never changed, which looked like the
    // animation being broken rather than the command never arriving here.
    setEyes(buf + 5);
  }
  else if (strcmp(buf,"LATCH:OFF")==0) doLatchShutdown("pi request");
}


/* ── EYE ANIMATION ──────────────────────────────────────────────────────
 * A port of the Pi's eyes.py, with the same states and the same look.
 *
 * NON-BLOCKING, and that is the important part. The Pi ran this in its own
 * thread with time.sleep() between frames. Here there is one loop shared with
 * the servos, the ultrasonic sensor and the motor watchdog — so a sleep would
 * stall all of them. Instead each state advances only when its own interval
 * has elapsed, and eyesTick() returns immediately the rest of the time.
 *
 * A frame is 18 LEDs of SPI-like RMT output, about 540us. Called at most
 * every 50ms, that is roughly 1% of the loop.
 * ───────────────────────────────────────────────────────────────────── */
Adafruit_NeoPixel eyeL(EYE_LEDS, EYE_L_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel eyeR(EYE_LEDS, EYE_R_PIN, NEO_GRB + NEO_KHZ800);

char          eyeState[16]   = "idle";
unsigned long eyeStep        = 0;
unsigned long eyeLastFrame   = 0;
unsigned int  eyeInterval    = 50;      // ms between frames, per state

static void eyeShow(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t c = eyeL.Color(r, g, b);
  for (int i = 0; i < EYE_LEDS; i++) { eyeL.setPixelColor(i, c);
                                       eyeR.setPixelColor(i, c); }
  eyeL.show();
  eyeR.show();
}

void setEyes(const char* state) {
  if (!state || !*state) return;
  strncpy(eyeState, state, sizeof(eyeState) - 1);
  eyeState[sizeof(eyeState) - 1] = 0;
  eyeStep = 0;                 // restart the animation, not mid-cycle
  eyeLastFrame = 0;            // draw the first frame immediately
  Serial.printf("[EYES] %s\n", eyeState);
}

void eyesTick() {
  unsigned long now = millis();
  if (now - eyeLastFrame < eyeInterval) return;
  eyeLastFrame = now;

  if (!strcmp(eyeState, "idle")) {
    // A slow blue breath. max(5,...) keeps a dim floor so the eyes never
    // go fully dark, which reads as switched off rather than resting.
    int v = (int)((1.0f + sinf(eyeStep * 0.10f)) * 60.0f);
    eyeShow(0, 0, max(5, v));
    eyeInterval = 50;
  }
  else if (!strcmp(eyeState, "listening")) {
    // Brighter and a touch quicker than idle, so it is visibly attending.
    int v = (int)((1.0f + sinf(eyeStep * 0.16f)) * 90.0f);
    eyeShow(0, (uint8_t)(v / 5), max(20, v));
    eyeInterval = 45;
  }
  else if (!strcmp(eyeState, "thinking")) {
    // Amber, faster: working rather than waiting.
    int v = (int)((1.0f + sinf(eyeStep * 0.28f)) * 90.0f);
    eyeShow(max(25, v), (uint8_t)(v * 0.55f), 0);
    eyeInterval = 40;
  }
  else if (!strcmp(eyeState, "speaking")) {
    // Warm and lively, roughly in time with speech.
    int v = (int)((1.0f + sinf(eyeStep * 0.42f)) * 100.0f);
    eyeShow(max(40, v), (uint8_t)(v * 0.62f), (uint8_t)(v * 0.12f));
    eyeInterval = 35;
  }
  else if (!strcmp(eyeState, "eyes1")) {
    int v = (int)((1.0f + sinf(eyeStep * 0.06f)) * 80.0f);
    eyeShow(0, 0, max(5, v));
    eyeInterval = 50;
  }
  else if (!strcmp(eyeState, "eyes2")) {
    // The heartbeat: two quick beats, then a long dark rest.
    int phase = eyeStep % 20;
    if      (phase <  2) eyeShow(255, 20, 80);
    else if (phase <  4) eyeShow( 80,  5, 25);
    else if (phase <  6) eyeShow(255, 20, 80);
    else if (phase < 12) eyeShow(  0,  0,  0);
    else                 eyeShow( 15,  0,  6);
    eyeInterval = 80;
  }
  else if (!strcmp(eyeState, "eyes3")) {
    static const uint8_t RB[8][3] = {
      {255,0,0},{255,127,0},{255,255,0},{0,255,0},
      {0,0,255},{75,0,130},{148,0,211},{255,0,255}};
    const uint8_t* c = RB[eyeStep % 8];
    eyeShow(c[0], c[1], c[2]);
    eyeInterval = 120;
  }
  else if (!strcmp(eyeState, "disco")) {
    // Saturated colours jumping at random, for the dances. Deliberately
    // random rather than a sequence: a fixed cycle reads as a pattern within
    // a few seconds, and stops looking like a disco.
    //
    // Fully saturated — one channel at full, one mid, one off — because
    // pastels wash out under room lighting and end up looking like one
    // colour flickering.
    static const uint8_t DISCO[8][3] = {
      {255,0,60},{255,80,0},{255,220,0},{0,255,60},
      {0,180,255},{80,0,255},{255,0,255},{0,255,200}};
    static uint8_t last = 0;
    uint8_t i = random(8);
    if (i == last) i = (i + 1) % 8;    // never the same twice running
    last = i;
    eyeShow(DISCO[i][0], DISCO[i][1], DISCO[i][2]);
    // Roughly two changes a second. Faster reads as a strobe, which is
    // unpleasant to stand near and a genuine problem for anybody
    // photosensitive.
    eyeInterval = 220;
  }
  else if (!strcmp(eyeState, "off")) {
    eyeShow(0, 0, 0);
    eyeInterval = 500;        // nothing changing: stop redrawing constantly
  }
  else {
    // An unknown state should not leave the eyes frozen on whatever was
    // last drawn — that looks like a crash. Fall back to idle and say so.
    Serial.printf("[EYES] unknown state '%s' - using idle\n", eyeState);
    strcpy(eyeState, "idle");
  }
  eyeStep++;
}

// ── SETUP ──────────────────────────────────────────────────
void setup() {
  // LATCH FIRST, before anything else: if the PCA9685s and servos are fed
  // from this rail, every millisecond spent here is time they have to come up
  // before we talk to them.
  pinMode(LATCH_PIN, OUTPUT);
  digitalWrite(LATCH_PIN, HIGH);

  Serial.begin(115200);
  delay(1200);                       // USB CDC + rail settling
  Serial.println("\n[LATCH] ON");

  pinMode(SHUTDOWN_BTN, INPUT_PULLUP);   // button to GND, as Ben

  // ── I2C BEFORE the UARTs ────────────────────────────────
  // On Ben, bringing up a UART before I2C took the I2C bus down with it —
  // measured, not guessed. Same pins, same board, only the order differed.
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Serial.printf("[I2C] SDA=%d SCL=%d @ 100kHz\n", SDA_PIN, SCL_PIN);

  // Retry rather than give up on the first attempt: a slow rail succeeds on
  // the second try, and a robot whose arms are dead one boot in five is worse
  // than one that waits half a second.
  pcaOk = false;
  for (int attempt = 1; attempt <= 5 && !pcaOk; attempt++) {
    Wire.beginTransmission(0x40);
    if (Wire.endTransmission() == 0) {
      pcaOk = pcaL.begin() && pcaR.begin();
      Serial.printf("[PCA] attempt %d: 0x40 responded, begin() %s\n",
                    attempt, pcaOk ? "ok" : "FAILED");
    } else {
      Serial.printf("[PCA] attempt %d/5: no response at 0x40\n", attempt);
      delay(400);
    }
  }
  if (pcaOk) {
    pcaL.setPWMFreq(SERVO_FREQ);
    pcaR.setPWMFreq(SERVO_FREQ);
    Serial.println("[PCA] both boards ready");
  } else {
    Serial.println("[PCA] *** NOT FOUND — arms and TF-Luna will not work ***");
    Serial.printf("[PCA] check SDA=%d SCL=%d, common GND, VCC and V+\n",
                  SDA_PIN, SCL_PIN);
    Serial.println("[PCA] both boards must ACK: 0x40 and 0x41 (A0 bridged)");
  }

  // ── Pi link on GPIO 5/6 ─────────────────────────────────
  // Argument order is RX, then TX.
  PiSerial.begin(115200, SERIAL_8N1, PI_RX_PIN, PI_TX_PIN);
  delay(100);
  while (PiSerial.available()) PiSerial.read();     // flush boot garbage
  Serial.printf("[PI] UART2 RX=GPIO%d TX=GPIO%d\n", PI_RX_PIN, PI_TX_PIN);

  // Seed positions from the home table BEFORE attaching, so the first pulse a
  // servo sees is its own home angle rather than a jump.
  for (int a = 0; a < 2; a++)
    for (int ch = 0; ch < ARM_CH; ch++)
      armCurrent[a][ch] = armTarget[a][ch] =
        (float)constrain(ARM_HOME[a][ch], ARM_MIN[a][ch], ARM_MAX[a][ch]);
  // setHomeTargets() and the seed above cover unused channels harmlessly;
  // pcaWriteDeg refuses to write them.

  // Two timers for ESP32Servo (four direct servos share one 50Hz timer),
  // leaving the rest for the motor LEDC channels. Allocating all four starves
  // ledcAttach and can silently kill motor PWM.
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  lateralL.setPeriodHertz(50);
  lateralR.setPeriodHertz(50);
  headLRServo.setPeriodHertz(50);
  headUDServo.setPeriodHertz(50);
  lateralL.attach(LATERAL_L_PIN, SERVO_MIN_US, SERVO_MAX_US);
  lateralR.attach(LATERAL_R_PIN, SERVO_MIN_US, SERVO_MAX_US);
  headLRServo.attach(HEAD_LR_PIN, SERVO_MIN_US, SERVO_MAX_US);
  headUDServo.attach(HEAD_UD_PIN, SERVO_MIN_US, SERVO_MAX_US);

  if (pcaOk)
    for (int a = 0; a < 2; a++)
      for (int ch = 0; ch < ARM_CH; ch++)
        pcaWriteDeg(a, ch, armCurrent[a][ch]);
  lateralL.write(latCurrentL);
  lateralR.write(latCurrentR);
  headLRServo.write(headLRCurrent);
  headUDServo.write(headUDCurrent);
  delay(300);

  ledcAttach(RPWM_L, PWM_FREQ, PWM_RES);
  ledcAttach(LPWM_L, PWM_FREQ, PWM_RES);
  ledcAttach(RPWM_R, PWM_FREQ, PWM_RES);
  ledcAttach(LPWM_R, PWM_FREQ, PWM_RES);
  ledcWrite(RPWM_L,0); ledcWrite(LPWM_L,0);
  ledcWrite(RPWM_R,0); ledcWrite(LPWM_R,0);
  pinMode(EN_L,OUTPUT); digitalWrite(EN_L,LOW);
  pinMode(EN_R,OUTPUT); digitalWrite(EN_R,LOW);

  lastCmdTime = millis();
  motorSpeed = appSpeedToPWM(50);      // the app's default, not full power

  eyeL.begin(); eyeR.begin();
  eyeL.setBrightness(EYE_BRIGHT);
  eyeR.setBrightness(EYE_BRIGHT);
  setEyes("idle");
  eyesTick();                 // draw something at once, so a long boot does
                              // not look like dead eyes

  homeServosSlow("boot");

  // The startup sequence runs AFTER homing, so it begins from a known pose
  // rather than from wherever the servos powered up. It blocks, so the robot
  // cannot be commanded mid-sweep.
  startupSequence();

  Serial.printf("[SYSTEM] Luna ready (PCA %s)\n", pcaOk ? "ok" : "FAULT");
  Serial.println("[SYSTEM] type LIMITS here to print the joint table");
  PiSerial.println("Luna ready.");
}

// ── LOOP ───────────────────────────────────────────────────
void loop() {
  checkButton();
  readLuna();
  handleSerial();
  updateServos();
  // The eyes MUST be ticked here, in the main loop. An earlier edit put this
  // call inside calibHold() instead — the first place updateServos() appears
  // in the file — so the eyes animated during the boot sequence and then
  // froze on whatever frame was last drawn. Lit, but never changing.
  eyesTick();

  if (millis() - lastCmdTime > 500 && robotMoving) stopMotors();

  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 2000) {
    lastPrint = millis();
    Serial.printf("Dist:%d Obs:%d Dir:%s\n",
                  lunaDistance, obstacleAhead, currentDir);
    PiSerial.printf("Dist:%d\n", lunaDistance);   // heartbeat to the Jetson
  }
}
