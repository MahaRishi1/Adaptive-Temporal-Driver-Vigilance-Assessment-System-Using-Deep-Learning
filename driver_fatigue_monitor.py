import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import threading
import time
from collections import deque

# ============================================================
# Continuous Fatigue Index Class
# ============================================================
class ContinuousFatigueIndex:
    """
    Fatigue score: 0 = Fully Alert, 100 = Severe Fatigue.

    Fuses four normalized signals via exponential smoothing:
      - PERCLOS   (35%) : fraction of frames eyes are closed
      - Blink drop (20%): reduction from calibrated baseline
      - Yawn freq  (20%): yawn frequency in a sliding window
      - Head pose  (25%): pitch/yaw deviation from neutral
    """

    def __init__(self):
        self.score = 0.0
        self.alpha = 0.975         # EMA persistence (higher = slower decay)

        # Rolling windows (60 seconds @ ~1 update/frame)
        self.yawn_window  = deque(maxlen=60)
        self.blink_window = deque(maxlen=300)   # 2-min history for calibration

        # Blink baseline
        self.baseline_blink_rate = 18.0         # typical blinks/min
        self.calibrated = False

        # Signal weights (must sum to 1.0)
        self.w_perclos = 0.35
        self.w_blink   = 0.30
        self.w_yawn    = 0.30
        self.w_head    = 0.30

        # --- Blink tracking (managed externally via update_blink) ---
        self.eye_closed_frames  = 0             # consecutive closed frames
        self.total_blinks       = 0             # blinks in current minute
        self.blink_rate_start   = time.time()   # timer for 1-min window

        # --- PERCLOS (rolling window, reset every interval) ---
        self.closed_eye_frames  = 0
        self.total_eye_frames   = 0
        self.perclos_interval   = 300           # frames per PERCLOS window

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #
    def add_blink_rate_sample(self, blink_rate: float):
        """Call once per minute with the observed blink rate."""
        self.blink_window.append(blink_rate)
        if len(self.blink_window) >= 10:        # calibrate after 10 samples
            self.baseline_blink_rate = float(np.median(self.blink_window))
            self.calibrated = True

    # ------------------------------------------------------------------ #
    # Clamp helper (generic range)                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    # ------------------------------------------------------------------ #
    # Blink update (call every frame)                                      #
    # Returns True when a new blink is registered                          #
    # ------------------------------------------------------------------ #
    def update_blink(self, eye_closed: bool) -> bool:
        """
        Track consecutive closed frames.
        A blink is 2–15 frames of closure (< 0.5 s at 30 fps).
        Long closure (>= 15 frames) is drowsiness, not a blink.
        """
        new_blink = False
        if eye_closed:
            self.eye_closed_frames += 1
            self.closed_eye_frames += 1     # for PERCLOS
        else:
            if 2 <= self.eye_closed_frames < 15:
                self.total_blinks += 1
                new_blink = True
            self.eye_closed_frames = 0      # reset streak

        self.total_eye_frames += 1

        # Reset PERCLOS window periodically
        if self.total_eye_frames >= self.perclos_interval:
            self.closed_eye_frames = 0
            self.total_eye_frames  = 0

        return new_blink

    # ------------------------------------------------------------------ #
    # Current blink rate & per-minute reset                                #
    # ------------------------------------------------------------------ #
    def get_current_blink_rate(self) -> float:
        """Blinks per minute based on elapsed time since last reset."""
        elapsed = time.time() - self.blink_rate_start
        if elapsed <= 0:
            return 0.0
        return (self.total_blinks / elapsed) * 60.0

    def maybe_reset_blink_counter(self):
        """Call every frame; resets counter and calibrates every 60 s."""
        elapsed = time.time() - self.blink_rate_start
        if elapsed >= 60.0:
            rate = self.get_current_blink_rate()
            self.add_blink_rate_sample(rate)
            self.total_blinks    = 0
            self.blink_rate_start = time.time()

    # ------------------------------------------------------------------ #
    # PERCLOS                                                              #
    # ------------------------------------------------------------------ #
    @property
    def perclos(self) -> float:
        if self.total_eye_frames == 0:
            return 0.0
        return self.closed_eye_frames / self.total_eye_frames

    # ------------------------------------------------------------------ #
    # Main update (call once per frame after blink update)                 #
    # ------------------------------------------------------------------ #
    def update(self,
               blink_rate: float,
               yawn_detected: bool,
               pitch_deg: float,
               yaw_deg: float) -> float:

        # --- Yawn frequency ---
        self.yawn_window.append(1 if yawn_detected else 0)
        yawns_in_window = sum(self.yawn_window)

        # --- Normalize signals to [0, 1] ---

        # PERCLOS: meaningful above 0.25, severe at 0.60
        perclos_norm  = self.clamp((self.perclos - 0.25) / 0.35)

        # Blink drop: compare to calibrated baseline
        if self.calibrated:
            blink_drop = self.clamp(
                (self.baseline_blink_rate - blink_rate) / max(self.baseline_blink_rate, 1)
            )
        else:
            blink_drop = 0.0

        # Yawn intensity: 0 yawns ��� 0, 5+ yawns → 1
        yawn_norm = self.clamp((yawns_in_window - 1) / 4.0)

        # Head pose: pitch drop > 20° and yaw > 15° are concerning
        pitch_norm = self.clamp((abs(pitch_deg) - 20.0) / 20.0)
        yaw_norm_h = self.clamp((abs(yaw_deg)   - 15.0) / 20.0)
        head_norm  = max(pitch_norm, yaw_norm_h)

        # --- Weighted fusion ---
        signal = (
            self.w_perclos * perclos_norm +
            self.w_blink   * blink_drop   +
            self.w_yawn    * yawn_norm     +
            self.w_head    * head_norm
        )

        # --- Exponential moving average ---
        target_score = signal * 100.0

        # Slow recovery but fast increase
        if target_score > self.score:
            rate = 0.08
        else:
            rate = 0.02

        self.score = self.score + rate * (target_score - self.score)

        self.score = self.clamp(self.score, 0.0, 100.0)
        self.score = self.clamp(self.score, 0.0, 100.0)

        return self.score

    # ------------------------------------------------------------------ #
    # Alert level                                                          #
    # ------------------------------------------------------------------ #
    def get_alert_level(self) -> str:
        if   self.score < 30: return "NORMAL"
        elif self.score < 55: return "LOW"
        elif self.score < 75: return "MEDIUM"
        else:                 return "HIGH"


# ============================================================
# Eye Aspect Ratio
# ============================================================
def calculate_EAR(eye_pts: list) -> float:
    """
    eye_pts: 6 (x,y) tuples in MediaPipe order
    [outer, top1, top2, inner, bot2, bot1]
    """
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ============================================================
# Mouth Aspect Ratio (robust, corrected landmarks)
# ============================================================
# Use canonical stable points:
#   61 (left mouth corner), 291 (right mouth corner)
#   13 (upper inner lip center), 14 (lower inner lip center)
MOUTH_LANDMARKS = [61, 291, 13, 14]  # corners and inner centers



def calculate_MAR(mouth_pts: list) -> float:
    """
    Improved MAR calculation.
    More stable for yawning detection.
    """

    if len(mouth_pts) < 4:
        return 0.0

    left = mouth_pts[0]
    right = mouth_pts[1]
    top = mouth_pts[2]
    bottom = mouth_pts[3]

    width = dist.euclidean(left, right)

    if width < 1:
        return 0.0

    vertical = dist.euclidean(top, bottom)

    # Stronger scaling for yawning
    mar = vertical / width

    return mar



# ============================================================
# Alarm System
# ============================================================
_alarm_active  = False
_alarm_lock    = threading.Lock()

def _alarm_loop():
    while True:
        with _alarm_lock:
            if not _alarm_active:
                break
        try:
            playsound("alarm.wav")
        except Exception:
            pass
        time.sleep(1.5)

def start_alarm():
    global _alarm_active
    with _alarm_lock:
        if not _alarm_active:
            _alarm_active = True
            threading.Thread(target=_alarm_loop, daemon=True).start()

def stop_alarm():
    global _alarm_active
    with _alarm_lock:
        _alarm_active = False


# ============================================================
# MediaPipe Setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces         = 1,
    refine_landmarks      = True,
    min_detection_confidence = 0.6,
    min_tracking_confidence  = 0.7,
)

# Landmark index sets
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Head pose reference landmarks
NOSE_TIP    = 1
FOREHEAD    = 10
CHIN        = 152
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER  = 61
RIGHT_MOUTH_CORNER = 291

# ============================================================
# Smoothing Buffers
# ============================================================
EAR_BUF   = deque(maxlen=5)
PITCH_BUF = deque(maxlen=5)
YAW_BUF   = deque(maxlen=5)
MAR_BUF   = deque(maxlen=5)

# ============================================================
# CALIBRATION PHASE
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("=" * 55)
print("  CALIBRATION  –  Keep eyes OPEN and face forward")
print("=" * 55)

calib_ears = []
calib_start = time.time()
CALIB_SECS  = 20

while time.time() - calib_start < CALIB_SECS:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        def pt(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        le = [pt(i) for i in LEFT_EYE]
        re = [pt(i) for i in RIGHT_EYE]
        ear = (calculate_EAR(le) + calculate_EAR(re)) / 2.0
        if ear > 0.18:
            calib_ears.append(ear)

    remaining = int(CALIB_SECS - (time.time() - calib_start))
    cv2.putText(frame,
                f"CALIBRATING... KEEP EYES OPEN  ({remaining}s)",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2)
    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Calibration")

# Compute EAR threshold: 80% of median open-eye EAR
if calib_ears:
    EAR_THRESHOLD = float(np.median(calib_ears)) * 0.80
else:
    EAR_THRESHOLD = 0.23    # safe fallback

print(f"EAR Threshold : {EAR_THRESHOLD:.4f}")
print("Starting monitoring...  Press 'q' to quit")
print("-" * 55)

# ============================================================
# Adaptive MAR Threshold
# ============================================================
MAR_THRESHOLD         = None     # None = not yet calibrated
mar_calib_buf         = deque(maxlen=150)
MAR_CALIB_MULTIPLIER  = 1.6      # yawn is ~60% above resting

# Temporal yawn confirmation
YAWN_FRAME_THR = 12
current_yawn_frames = 0

# ============================================================
# Initialize Fatigue Tracker
# ============================================================
fatigue_index = ContinuousFatigueIndex()

# Drowsiness: consecutive frames EAR < threshold
# At 30 fps, 30 frames ≈ 1 second — more robust than 20
CONSEC_FRAMES = 30
consec_counter = 0

# Head-down debouncing
HEAD_DOWN_FRAMES_THR = 12   # ~0.4s at 30fps
head_down_frames = 0

# ============================================================
# MONITORING LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    state              = "NORMAL"
    yawn_detected      = False
    smooth_mar         = 0.0
    smooth_ear         = EAR_THRESHOLD + 0.01   # default: eyes open
    pitch_deg          = 0.0
    yaw_deg            = 0.0

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        def pt(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # ---- Face bounding box ----
        xs = [int(l.x * w) for l in lm]
        ys = [int(l.y * h) for l in lm]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)

        # ---- Eyes ----
        le = [pt(i) for i in LEFT_EYE]
        re = [pt(i) for i in RIGHT_EYE]

        for p in le: cv2.circle(frame, p, 2, (0, 255, 0), -1)
        for p in re: cv2.circle(frame, p, 2, (0, 255, 0), -1)

        raw_ear = (calculate_EAR(le) + calculate_EAR(re)) / 2.0
        EAR_BUF.append(raw_ear)
        smooth_ear = float(np.mean(EAR_BUF))

        # ---- Mouth / Yawn ----
        mouth_pts = [pt(i) for i in MOUTH_LANDMARKS]
        for p in mouth_pts: cv2.circle(frame, p, 2, (255, 0, 0), -1)

        raw_mar = calculate_MAR(mouth_pts)
        MAR_BUF.append(raw_mar)
        smooth_mar = float(np.mean(MAR_BUF))

        # Adaptive MAR calibration (first 150 frames)
        if MAR_THRESHOLD is None:
            mar_calib_buf.append(smooth_mar)
            if len(mar_calib_buf) >= 150:
                baseline_mar  = float(np.median(mar_calib_buf))
                MAR_THRESHOLD = baseline_mar * MAR_CALIB_MULTIPLIER
                print(f"MAR Threshold : {MAR_THRESHOLD:.4f}")
        else:
            # Stronger Yawn Detection

            if MAR_THRESHOLD is not None:

                # Must be significantly above threshold
                # TRUE YAWN DETECTION (robust against talking)

                mouth_width = dist.euclidean(mouth_pts[0], mouth_pts[1])

                if smooth_mar > MAR_THRESHOLD * 1.35 and mouth_width > 40:

                    current_yawn_frames += 1

                else:

                    current_yawn_frames = max(0, current_yawn_frames - 2)


                # Confirm real yawn (must last long)
                if current_yawn_frames >= 20:

                    yawn_detected = True

                    cv2.putText(frame,
                                "YAWNING!",
                                (x_min, y_min - 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 0, 255),
                                2)

        # ---- Head Pose (solvePnP-based with fallback) ----
        nose        = pt(NOSE_TIP)
        chin_pt     = pt(CHIN)
        l_eye_o     = pt(LEFT_EYE_OUTER)
        r_eye_o     = pt(RIGHT_EYE_OUTER)
        l_mouth     = pt(LEFT_MOUTH_CORNER)
        r_mouth     = pt(RIGHT_MOUTH_CORNER)
        forehead    = pt(FOREHEAD)

        # Camera intrinsics approximation from frame size (fx=fy)
        fx = fy = 1.2 * w
        cx = w / 2.0
        cy = h / 2.0
        cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coef = np.zeros((4, 1))

        # 3D model points (LearnOpenCV generic face model, scaled)
        model_3d = np.array([
            [0.0,    0.0,    0.0],     # Nose tip
            [0.0,  -330.0,  -65.0],    # Chin
            [-225.0, 170.0, -135.0],   # Left eye left corner
            [225.0,  170.0, -135.0],   # Right eye right corner
            [-150.0,-150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0],   # Right mouth corner
        ], dtype=np.float64)

        image_pts = np.array([
            [nose[0],   nose[1]],
            [chin_pt[0],chin_pt[1]],
            [l_eye_o[0],l_eye_o[1]],
            [r_eye_o[0],r_eye_o[1]],
            [l_mouth[0],l_mouth[1]],
            [r_mouth[0],r_mouth[1]],
        ], dtype=np.float64)

        pitch_deg = 0.0
        yaw_deg   = 0.0
        pnp_success = False
        if np.all(np.isfinite(image_pts)):
            try:
                success, rvec, tvec = cv2.solvePnP(model_3d, image_pts, cam_mtx, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    rmat, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                    pitch = np.arctan2(-rmat[2,0], sy)
                    yaw   = np.arctan2(rmat[1,0], rmat[0,0])
                    pitch_deg = float(np.degrees(pitch))
                    yaw_deg   = float(np.degrees(yaw))
                    pnp_success = True
            except Exception:
                pnp_success = False

        # Fallback: ratio-based proxy if PnP failed
        if not pnp_success:
            face_h = abs(chin_pt[1] - forehead[1])
            if face_h > 1:
                nose_rel_y = (nose[1] - forehead[1]) / face_h  # 0=top, 1=bottom
                pitch_deg = (nose_rel_y - 0.55) * 60.0
            yaw_deg = 0.0

        YAW_BUF.append(yaw_deg)
        PITCH_BUF.append(pitch_deg)
        smooth_yaw   = float(np.mean(YAW_BUF))
        smooth_pitch = float(np.mean(PITCH_BUF))

        # ---- Blink detection (single, correct path) ----
        eye_closed = smooth_ear < EAR_THRESHOLD
        fatigue_index.update_blink(eye_closed)

        if eye_closed:
            consec_counter += 1
        else:
            consec_counter = 0

        # ---- Blink rate (reset every 60 s) ----
        fatigue_index.maybe_reset_blink_counter()
        current_blink_rate = fatigue_index.get_current_blink_rate()

        # ---- Update fatigue score ----
        fatigue_score = fatigue_index.update(
            blink_rate    = current_blink_rate,
            yawn_detected = yawn_detected,
            pitch_deg     = smooth_pitch,
            yaw_deg       = smooth_yaw,
        )
        alert_level = fatigue_index.get_alert_level()

        # ---- State machine ----
        # ---- Improved Head Down Detection ----

        # Nose relative position inside face
        face_height = abs(chin_pt[1] - forehead[1])

        if face_height > 10:

            nose_relative = (nose[1] - forehead[1]) / face_height

            # Normal ≈ 0.55
            # Head Down ≈ 0.70+

            # MUCH MORE RELIABLE COLLAPSE DETECTION

            if nose_relative > 0.72:

                head_down_frames += 2

            elif smooth_pitch > 18:

                head_down_frames += 1

            else:

                head_down_frames = max(0, head_down_frames - 2)

        # ---- State Machine ----
        if consec_counter >= CONSEC_FRAMES:

            state = "DROWSY"

        else:

            if head_down_frames >= HEAD_DOWN_FRAMES_THR:
                state = "HEAD DOWN"

            elif abs(smooth_yaw) > 18:
                state = "DISTRACTED"

                

        # ---- Alarm ----
        if state in ("DROWSY", "HEAD DOWN") or alert_level in ("MEDIUM", "HIGH"):
            start_alarm()
        else:
            stop_alarm()

        # ================================================================
        # HUD
        # ================================================================
        state_colors = {
            "NORMAL":     (0,   255,   0),
            "DROWSY":     (0,     0, 255),
            "HEAD DOWN":  (255,   0, 255),
            "DISTRACTED": (0,   165, 255),
        }
        state_color = state_colors.get(state, (255, 255, 255))

        # State label
        cv2.putText(frame, f"STATE: {state}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 3)

        # EAR
        cv2.putText(frame, f"EAR: {smooth_ear:.3f}  (thr {EAR_THRESHOLD:.3f})",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # MAR
        thr_str = f"{MAR_THRESHOLD:.3f}" if MAR_THRESHOLD else "calibrating"
        cv2.putText(frame, f"MAR: {smooth_mar:.3f}  (thr {thr_str})",
                    (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if yawn_detected:
            cv2.rectangle(frame, (x_min, y_min-65), (x_min+180, y_min-35), (0,0,0), -1)
            cv2.putText(frame, "YAWN DETECTED", (x_min+5, y_min-45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Fatigue score bar
        fatigue_color = (
            (0, 0, 255)   if fatigue_score > 75 else
            (0, 165, 255) if fatigue_score > 55 else
            (0, 255, 255) if fatigue_score > 30 else
            (0, 255, 0)
        )
        cv2.putText(frame, f"FATIGUE: {fatigue_score:.1f}  [{alert_level}]",
                    (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fatigue_color, 2)

        # Draw fatigue bar
        bar_x, bar_y, bar_w, bar_h = 30, 160, 200, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * fatigue_score / 100.0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), fatigue_color, -1)

        # Blink rate & PERCLOS
        cv2.putText(frame, f"Blink: {current_blink_rate:.1f}/min",
                    (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"PERCLOS: {fatigue_index.perclos:.3f}",
                    (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Head pose
        cv2.putText(frame,
                    f"Pitch: {smooth_pitch:+.1f}°  Yaw: {smooth_yaw:+.1f}°",
                    (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if state == "HEAD DOWN":
            cv2.putText(frame, "COLLAPSE RISK", (30, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calibration badge
        calib_text  = "Blink Cal: OK" if fatigue_index.calibrated else "Blink Cal: ..."
        calib_color = (0, 255, 0)      if fatigue_index.calibrated else (0, 200, 200)
        cv2.putText(frame, calib_text,
                    (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, calib_color, 1)

    else:
        # No face detected
        stop_alarm()
        cv2.putText(frame, "NO FACE DETECTED",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        # Still update blink tracker so PERCLOS doesn't freeze
        fatigue_index.update_blink(False)
        fatigue_index.maybe_reset_blink_counter()

    cv2.imshow("Advanced Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# Cleanup
# ============================================================
stop_alarm()
cap.release()
cv2.destroyAllWindows()
print(f"\nMonitoring stopped.")
print(f"Final fatigue score : {fatigue_index.score:.1f}")
print(f"Alert level         : {fatigue_index.get_alert_level()}")
