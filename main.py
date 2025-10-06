# main_opencv_lbph_with_presence.py
import os
import time
import cv2
import numpy as np
import pygame
from datetime import datetime

# -------- Configuration --------
KNOWN_FACES_DIR = "known_faces"
ALERT_SOUND = "alert.mp3"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 70.0      # lower = stricter
ALERT_COOLDOWN = 3.0             # seconds between sound alerts
SAVE_COOLDOWN = 2.0              # per-name fallback (not used — we use presence + global)
CAPTURES_DIR = "captures"
ALERT_ONLY_FOR = ["mom", "dad", "sister", "brother"]  # names for which a sound is played

# New parameters based on your request:
MIN_PRESENCE_SECONDS = 1.5       # minimum seconds a face must be visible before saving
GLOBAL_SAVE_DELAY = 4.0          # global cooldown between any saved captures (seconds)

# -------- Sound initialization --------
try:
    pygame.mixer.init()
except Exception as e:
    print("⚠️ Warning: pygame.mixer.init() failed:", e)

def play_alert():
    if not os.path.exists(ALERT_SOUND):
        print("⚠️ alert.mp3 not found, no sound will be played.")
        return
    try:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except Exception as e:
        print("⚠️ Error while playing sound:", e)

# -------- Haar cascade ----------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Cannot load Haar cascade: " + cascade_path)

os.makedirs(CAPTURES_DIR, exist_ok=True)

# -------- (same LBPH training as before) --------
print("🧩 Searching for known faces in folder:", os.path.abspath(KNOWN_FACES_DIR))
if not os.path.isdir(KNOWN_FACES_DIR):
    raise SystemExit("❌ Folder 'known_faces' does not exist. Create it and add photos (mom.jpeg, dad.jpeg, ...).")

faces = []
labels = []
label2name = {}
name2label = {}
next_label = 0

for entry in sorted(os.listdir(KNOWN_FACES_DIR)):
    full = os.path.join(KNOWN_FACES_DIR, entry)
    if os.path.isdir(full):
        name = entry
        for fn in sorted(os.listdir(full)):
            fp = os.path.join(full, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue
            img = cv2.imread(fp)
            if img is None:
                print("❌ Cannot load", fp)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(rects) == 0:
                print("⚠️ No face found in", fp)
                continue
            x, y, w, h = max(rects, key=lambda r: r[2]*r[3])
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            if name not in name2label:
                name2label[name] = next_label
                label2name[next_label] = name
                next_label += 1
            faces.append(roi)
            labels.append(name2label[name])
            print("✔ Added (subfolder):", name, fn)
    else:
        ext = os.path.splitext(entry)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue
        name = os.path.splitext(entry)[0]
        fp = full
        img = cv2.imread(fp)
        if img is None:
            print("❌ Cannot load", fp)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(rects) == 0:
            print("⚠️ No face found in", fp)
            continue
        x, y, w, h = max(rects, key=lambda r: r[2]*r[3])
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        if name not in name2label:
            name2label[name] = next_label
            label2name[next_label] = name
            next_label += 1
        faces.append(roi)
        labels.append(name2label[name])
        print("✔ Added (file):", name, entry)

if len(faces) == 0:
    print("❌ No training faces found. Make sure images contain a clear face.")
    raise SystemExit

faces_np = np.array(faces)
labels_np = np.array(labels)

print(f"Training LBPH model on {len(faces)} images, {len(label2name)} persons...")
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    print("❌ cv2.face.LBPHFaceRecognizer_create() failed. Do you have opencv-contrib-python installed?")
    raise
recognizer.train(faces_np, labels_np)
print("✅ Training complete. Label mapping:", label2name)

# -------- Variables for saving and presence tracking --------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open camera index " + str(CAMERA_INDEX))
print("🎥 Camera is running. Press 'q' to quit.")

last_alert_time = 0.0
last_global_save_time = 0.0

# trackers: key -> { first_seen, last_seen, saved_flag }
presence = {}  # e.g. { key: {"first":ts, "last":ts, "saved":False} }

# helper function to build a unique key
def make_key(name, x, y, w, h, bucket=50):
    cx = x + w//2
    cy = y + h//2
    bx = cx // bucket
    by = cy // bucket
    return f"{name}_{bx}_{by}"

def save_capture(frame_color, name, confidence):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = name.replace(" ", "_")
    fname = f"{ts}_{safe_name}_{confidence:.1f}.jpg"
    path = os.path.join(CAPTURES_DIR, fname)
    try:
        cv2.imwrite(path, frame_color)
        print(f"💾 Saved: {path}")
    except Exception as e:
        print("⚠️ Error while saving:", e)

# -------- Main loop --------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Cannot read from camera.")
        time.sleep(0.1)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    now = time.time()
    seen_keys_this_frame = set()

    for (x, y, w, h) in rects:
        roi = gray[y:y+h, x:x+w]
        try:
            roi_resized = cv2.resize(roi, (200,200))
        except Exception:
            continue
        label, confidence = recognizer.predict(roi_resized)  # lower = better
        name = label2name.get(label, "Unknown")
        text = f"{name} ({confidence:.1f})"
        color = (0,255,0) if confidence < CONFIDENCE_THRESHOLD else (0,165,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        key = make_key(name, x, y, w, h, bucket=40)
        seen_keys_this_frame.add(key)

        if key not in presence:
            presence[key] = {"first": now, "last": now, "saved": False}
        else:
            presence[key]["last"] = now

        duration = now - presence[key]["first"]
        time_since_global = now - last_global_save_time
        if (duration >= MIN_PRESENCE_SECONDS) and (not presence[key]["saved"]) and (time_since_global >= GLOBAL_SAVE_DELAY):
            x0, y0, x1, y1 = max(0,x), max(0,y), min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            crop_color = frame[y0:y1, x0:x1]
            save_capture(crop_color, name, confidence)
            presence[key]["saved"] = True
            last_global_save_time = now

        if confidence < CONFIDENCE_THRESHOLD:
            if name.lower() in [n.lower() for n in ALERT_ONLY_FOR]:
                if now - last_alert_time > ALERT_COOLDOWN:
                    print(f"🔔 Recognized: {name} — confidence={confidence:.1f}")
                    play_alert()
                    last_alert_time = now
            else:
                print(f"(🎧 Recognized {name}, but no sound) confidence={confidence:.1f}")

    # cleanup old presence entries
    stale_keys = []
    for key, info in presence.items():
        if now - info["last"] > 1.0:
            stale_keys.append(key)
    for k in stale_keys:
        del presence[k]

    cv2.imshow("Recognition (LBPH) with saving", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
