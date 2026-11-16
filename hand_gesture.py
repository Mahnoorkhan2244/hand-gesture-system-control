import cv2
import mediapipe as mp
import numpy as np
import time
import os
import platform
import webbrowser
import pyautogui
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except ImportError:
    BRIGHTNESS_AVAILABLE = False
    print("[WARN] 'screen_brightness_control' not installed. "
          "Run: pip install screen-brightness-control")

pyautogui.FAILSAFE = False

COOLDOWN = 1.0
HOLD_TIME = 1.0               
CONFIDENCE_THRESHOLD = 0.60   
SCREENSHOT_DIR = r"C:\Users\HP\OneDrive\Desktop\virtual assistant\ml\screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

running = True

# MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

TIP_IDS = [4, 8, 12, 16, 20]


def get_finger_states(landmarks, handedness_label):
    fingers = []

    # Thumb
    if handedness_label == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # Other 4 fingers
    for tip_id in TIP_IDS[1:]:
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return fingers


def thumb_direction(landmarks):
    wrist_y = landmarks[0].y
    thumb_tip_y = landmarks[4].y
    thumb_mcp_y = landmarks[2].y
    if thumb_tip_y < thumb_mcp_y < wrist_y:
        return "up"
    elif thumb_tip_y > thumb_mcp_y > wrist_y:
        return "down"
    return "neutral"


def is_ok_sign(landmarks):
    # Geometric check for the 👌 Ok gesture
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    pinch_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_scale = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5

    if hand_scale == 0:
        return False
    return (pinch_dist / hand_scale) < 0.4


# ML MODEL — Ensemble: KNN + Random Forest + SVM (Soft Voting)
GESTURE_PATTERNS = {
    "OPEN_PALM":  [1, 1, 1, 1, 1],   # 🖐️
    "POINT_UP":   [0, 1, 0, 0, 0],   # ☝️
    "PEACE":      [0, 1, 1, 0, 0],   # ✌️
    "OK":         [0, 0, 1, 1, 1],   # 👌
    "THUMBS":     [1, 0, 0, 0, 0],   # 👍 / 👎
    "PINKY_THUMB": [1, 0, 0, 0, 1],  # 🤙
}


def build_training_data():
    X, y = [], []
    samples_per_class = 60
    rng = np.random.default_rng(42)
    for label, pattern in GESTURE_PATTERNS.items():
        for _ in range(samples_per_class):
            noisy = [p if rng.random() > 0.05 else 1 - p for p in pattern]
            X.append(noisy)
            y.append(label)
    return np.array(X), np.array(y)


def build_ensemble():
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(kernel="rbf", probability=True, random_state=42)

    return VotingClassifier(
        estimators=[("knn", knn), ("rf", rf), ("svm", svm)],
        voting="soft",         
    )


print("[INFO] Training gesture recognition ensemble (KNN + Random Forest + SVM)...")
X_train, y_train = build_training_data()
model = build_ensemble()
model.fit(X_train, y_train)
print(f"[INFO] Ensemble trained on {len(X_train)} samples across {len(GESTURE_PATTERNS)} gestures.")


def detect_gesture(landmarks, handedness_label):
    feature_vector = get_finger_states(landmarks, handedness_label)

    probs = model.predict_proba([feature_vector])[0]
    best_idx = int(np.argmax(probs))
    predicted = model.classes_[best_idx]
    confidence = probs[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return None

    if predicted == "THUMBS":
        direction = thumb_direction(landmarks)
        if direction == "up":
            predicted = "THUMBS_UP"
        elif direction == "down":
            predicted = "THUMBS_DOWN"
        else:
            predicted = None
    elif is_ok_sign(landmarks):
        predicted = "OK"

    return predicted


def volume_up():
    pyautogui.press("volumeup")
    print("[ACTION] 👍 Volume Up")


def volume_down():
    pyautogui.press("volumedown")
    print("[ACTION] 👎 Volume Down")


def brightness_up():
    if BRIGHTNESS_AVAILABLE:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(min(current + 10, 100))
            print("[ACTION] 👍👍 Brightness Up")
        except Exception as e:
            print("[WARN] Brightness control failed:", e)
    else:
        print("[WARN] Brightness control library not available")


def brightness_down():
    if BRIGHTNESS_AVAILABLE:
        try:
            current = sbc.get_brightness(display=0)[0]
            sbc.set_brightness(max(current - 10, 0))
            print("[ACTION] 👎👎 Brightness Down")
        except Exception as e:
            print("[WARN] Brightness control failed:", e)
    else:
        print("[WARN] Brightness control library not available")


def take_screenshot():
    filename = os.path.join(SCREENSHOT_DIR, f"screenshot_{int(time.time())}.png")
    img = pyautogui.screenshot()
    img.save(filename)
    print("[ACTION] ✌️ Screenshot saved:", filename)


def open_browser():
    webbrowser.open("https://www.google.com")
    print("[ACTION] 👌 Opening Browser")


def open_youtube():
    webbrowser.open("https://www.youtube.com")
    print("[ACTION] 👌 Opening YouTube")


def open_playlist():
    playlist_url = (
        "https://www.youtube.com/watch?v=fmxmNwAB0_Y"
        "&list=PLxCzCOWd7aiEXg5BV10k9THtjnS48yI-T&index=1"
    )
    webbrowser.open(playlist_url)
    print("[ACTION] 🖐️🖐️ Opening ML Course Playlist")


def open_file_explorer():
    system_name = platform.system()
    home_dir = os.path.expanduser("~")
    try:
        if system_name == "Windows":
            os.startfile(home_dir)
        elif system_name == "Darwin":
            os.system(f'open "{home_dir}"')
        elif system_name == "Linux":
            os.system(f'xdg-open "{home_dir}"')
        print("[ACTION] 🖐️ Opening File Explorer")
    except Exception as e:
        print("[WARN] Could not open file explorer:", e)


def lock_pc():
    system_name = platform.system()
    if system_name == "Windows":
        os.system("rundll32.exe user32.dll,LockWorkStation")
    elif system_name == "Darwin":
        os.system("pmset displaysleepnow")
    elif system_name == "Linux":
        os.system("gnome-screensaver-command -l || xdg-screensaver lock")
    print("[ACTION] ☝️ Locking PC")


def open_settings():
    system_name = platform.system()
    try:
        if system_name == "Windows":
            os.system("start ms-settings:")
        elif system_name == "Darwin":
            os.system("open /System/Library/PreferencePanes/")
        elif system_name == "Linux":
            os.system("gnome-control-center || xdg-open settings:// 2>/dev/null")
        print("[ACTION] 🤙 Opening Settings")
    except Exception as e:
        print("[WARN] Could not open settings:", e)


def minimize_all():
    system_name = platform.system()
    try:
        if system_name == "Windows":
            pyautogui.hotkey("win", "d")
        elif system_name == "Darwin":
            pyautogui.hotkey("command", "f3")
        else:
            pyautogui.hotkey("super", "d")
        print("[ACTION] ✌️✌️ Minimizing All Windows")
    except Exception as e:
        print("[WARN] Minimize all failed:", e)


def exit_program():
    global running
    running = False
    print("[ACTION] 👌👌 Exiting Program")


GESTURE_ACTIONS = {
    "THUMBS_UP":    ("👍 Volume Up",        volume_up),
    "THUMBS_DOWN":  ("👎 Volume Down",      volume_down),
    "PEACE":        ("✌️ Screenshot",       take_screenshot),
    "OK":           ("👌 Open YouTube",     open_youtube),
    "OPEN_PALM":    ("🖐️ File Explorer",    open_file_explorer),
    "POINT_UP":     ("☝️ Lock PC",          lock_pc),
    "PINKY_THUMB":  ("🤙 Open Settings",    open_settings),
}

DOUBLE_GESTURE_ACTIONS = {
    "THUMBS_UP":   ("👍👍 Brightness Up",       brightness_up),
    "THUMBS_DOWN": ("👎👎 Brightness Down",     brightness_down),
    "PEACE":       ("✌️✌️ Minimize All",        minimize_all),
    "OPEN_PALM":   ("🖐️🖐️ Open Playlist",       open_playlist),
    "OK":          ("👌👌 Exit Program",        exit_program),
}


def print_gesture_guide():
    rows = [
        ("👍", "Thumbs Up", "Volume Increase"),
        ("👎", "Thumbs Down", "Volume Decrease"),
        ("✌️", "Victory", "Take Screenshot"),
        ("👌", "Ok", "Open YouTube"),
        ("🖐️", "Open Hand", "File Explorer"),
        ("☝️", "Point Up", "Lock PC"),
        ("🤙", "Pinky Thumb", "Open Settings"),
        ("👍👍", "Double Thumbs", "Brightness +"),
        ("👎👎", "Double Thumbs", "Brightness -"),
        ("✌️✌️", "Double Victory", "Minimize All"),
        ("🖐️🖐️", "Double Open Hand", "Open ML Playlist"),
        ("👌👌", "Double Ok", "Exit Program"),
    ]

    line = "=" * 69
    print("\n" + line)
    print(" GESTURE GUIDE")
    print(line + "\n")
    for emoji, name, action in rows:
        print(f" {emoji} {name:<16} - {action}")
    print("\n" + line + "\n")


# UI HELPERS
WINDOW_NAME = "Hand Gesture Control"

# BGR colors
STATE_COLORS = {
    "IDLE":      (170, 170, 170),
    "DETECTING": (0, 165, 255),
    "DONE":      (80, 200, 60),
    "WARN":      (60, 60, 230),
}
STATE_LABELS = {
    "IDLE":      "WAITING",
    "DETECTING": "DETECTED - HOLD STEADY",
    "DONE":      "ACTION EXECUTED",
    "WARN":      "NOT RECOGNIZED",
}

GUIDE_ROWS = [
    ("Thumbs Up : Volume +",          ("single", "THUMBS_UP")),
    ("Thumbs Down : Volume -",        ("single", "THUMBS_DOWN")),
    ("Victory : Screenshot",          ("single", "PEACE")),
    ("Ok : Open YouTube",             ("single", "OK")),
    ("Open Hand : File Explorer",     ("single", "OPEN_PALM")),
    ("Point Up : Lock PC",            ("single", "POINT_UP")),
    ("Pinky Thumb : Settings",        ("single", "PINKY_THUMB")),
    ("2x Thumbs Up : Brightness +",   ("double", "THUMBS_UP")),
    ("2x Thumbs Down : Brightness -", ("double", "THUMBS_DOWN")),
    ("2x Victory : Minimize All",     ("double", "PEACE")),
    ("2x Open Hand : ML Playlist",    ("double", "OPEN_PALM")),
    ("2x Ok : Exit Program",          ("double", "OK")),
]


def clean_text(text):
    return "".join(c for c in text if ord(c) < 128).strip()


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def blend_rect(img, p1, p2, color, alpha):
    # Semi-transparent rectangle
    x1, y1 = max(p1[0], 0), max(p1[1], 0)
    x2, y2 = min(p2[0], img.shape[1]), min(p2[1], img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    fill = np.full_like(roi, color)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)


def draw_ui(frame, state, title, progress, remaining, active_key, fps):
    h, w = frame.shape[:2]
    s = w / 640.0

    def sc(v):
        return int(v * s)

    color = STATE_COLORS[state]

    x1, y1, x2, y2 = sc(15), sc(15), sc(345), sc(115)
    blend_rect(frame, (x1, y1), (x2, y2), (25, 25, 25), 0.65)
    cv2.rectangle(frame, (x1, y1), (x1 + sc(6), y2), color, -1)
    put_text(frame, STATE_LABELS[state], (x1 + sc(18), y1 + sc(24)),
             0.48 * s, color, 1)
    if remaining is not None:
        put_text(frame, f"{max(remaining, 0):.1f}s", (x2 - sc(55), y1 + sc(24)),
                 0.55 * s, color, 1)
    put_text(frame, title, (x1 + sc(18), y1 + sc(58)), 0.7 * s, (255, 255, 255), 2)

    bx1, bx2 = x1 + sc(18), x2 - sc(18)
    by1, by2 = y2 - sc(26), y2 - sc(14)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
    if progress > 0:
        fill_x = bx1 + int((bx2 - bx1) * min(progress, 1.0))
        cv2.rectangle(frame, (bx1, by1), (fill_x, by2), color, -1)

    pw = sc(250)
    px1 = w - pw - sc(15)
    py1 = sc(15)
    row_h = sc(24)
    py2 = py1 + sc(34) + row_h * len(GUIDE_ROWS)
    blend_rect(frame, (px1, py1), (px1 + pw, py2), (25, 25, 25), 0.65)
    put_text(frame, "GESTURE GUIDE", (px1 + sc(12), py1 + sc(22)),
             0.5 * s, (255, 255, 255), 1)
    cv2.line(frame, (px1 + sc(12), py1 + sc(30)), (px1 + pw - sc(12), py1 + sc(30)),
             (110, 110, 110), 1)

    for i, (text, key) in enumerate(GUIDE_ROWS):
        ry = py1 + sc(34) + i * row_h
        is_active = (key == active_key)
        if is_active:
            blend_rect(frame, (px1 + sc(6), ry), (px1 + pw - sc(6), ry + row_h - sc(2)),
                       color, 0.55)
        put_text(frame, text, (px1 + sc(14), ry + sc(16)), 0.42 * s,
                 (255, 255, 255) if is_active else (200, 200, 200), 1)

    # --- Bottom bar ---
    blend_rect(frame, (0, h - sc(30)), (w, h), (25, 25, 25), 0.7)
    put_text(frame, f"Hold a gesture for {HOLD_TIME:.0f}s to confirm",
             (sc(15), h - sc(10)), 0.48 * s, (220, 220, 220), 1)
    right_text = f"FPS: {fps:.0f}   |   Q: Quit"
    tw = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48 * s, 1)[0][0]
    put_text(frame, right_text, (w - tw - sc(15), h - sc(10)),
             0.48 * s, (220, 220, 220), 1)


def main():
    global running
    running = True

    print_gesture_guide()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_action_time = 0
    pending_key = None      
    pending_start = 0.0     
    action_done = False    

    prev_time = time.time()
    fps = 0.0
    hand_state = "IDLE"

    print("[INFO] Camera started. Show a gesture. Press 'q' or 👌👌 to quit.")

    while running:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # UI state
        state = "IDLE"
        title = "No hand detected"
        progress = 0.0
        remaining = None
        active_key = None

        if result.multi_hand_landmarks and result.multi_handedness:
            hands_data = list(zip(result.multi_hand_landmarks, result.multi_handedness))
            detected_gestures = []

            point_color = STATE_COLORS[hand_state]
            for hand_landmarks, handedness in hands_data:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=point_color, thickness=2),
                )
                label = handedness.classification[0].label
                landmarks = hand_landmarks.landmark

                predicted = detect_gesture(landmarks, label)
                detected_gestures.append(predicted)

            current_time = time.time()

            candidate = None
            if len(detected_gestures) == 2 and detected_gestures[0] is not None \
                    and detected_gestures[0] == detected_gestures[1] \
                    and detected_gestures[0] in DOUBLE_GESTURE_ACTIONS:
                g = detected_gestures[0]
                candidate = (("double", g),) + DOUBLE_GESTURE_ACTIONS[g]
            elif len(detected_gestures) == 1 and detected_gestures[0] in GESTURE_ACTIONS:
                g = detected_gestures[0]
                candidate = (("single", g),) + GESTURE_ACTIONS[g]
            elif len(detected_gestures) == 2:
                state, title = "WARN", "Gestures don't match"
            else:
                state, title = "WARN", "Unknown gesture"

            if candidate:
                key, action_name, action_func = candidate
                active_key = key
                name = clean_text(action_name)

                if key != pending_key:
                    pending_key = key
                    pending_start = current_time
                    action_done = False

                remaining = HOLD_TIME - (current_time - pending_start)

                if action_done:
                    state, title, progress, remaining = "DONE", name, 1.0, None
                elif remaining > 0:
                    state, title = "DETECTING", name
                    progress = 1 - remaining / HOLD_TIME
                else:
                    if (current_time - last_action_time) > COOLDOWN:
                        action_func()
                        last_action_time = current_time
                    action_done = True
                    state, title, progress, remaining = "DONE", name, 1.0, None
            else:
                pending_key = None
                action_done = False
        else:
            pending_key = None
            action_done = False

        hand_state = state

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

        draw_ui(frame, state, title, progress, remaining, active_key, fps)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()