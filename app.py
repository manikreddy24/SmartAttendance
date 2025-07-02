import cv2
import os
import time
from datetime import datetime
import pyttsx3

# 🔊 Init TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load known faces
known_faces_dir = 'faces'
known_faces = {}
for file in os.listdir(known_faces_dir):
    path = os.path.join(known_faces_dir, file)
    name = os.path.splitext(file)[0]
    img = cv2.imread(path)
    if img is not None:
        known_faces[name] = cv2.resize(img, (100, 100))  # Normalize size

# Log attendance
def mark_attendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        if name not in f.read():
            time = datetime.now().strftime('%H:%M:%S,%d-%m-%Y')
            f.write(f'{name},{time}\n')

# Start webcam
cap = cv2.VideoCapture(0)
last_mark_time = {}
display_success = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_time = time.time()

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_color, (100, 100))
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        label = "Unknown"

        for name, ref_img in known_faces.items():
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(roi_gray, ref_gray)
            score = diff.mean()
            print(f"[DEBUG] {name} match score: {score:.2f}")

            if score < 100:
                label = name
                mark_attendance(name)

                if name not in last_mark_time or current_time - last_mark_time[name] > 10:
                    mark_attendance(name)
                    speak(f"Hello {name}, your attendance is marked.")
                    last_mark_time[name] = current_time
                    display_success[name] = current_time
                break
                '''# Show success message for 2 seconds
                cv2.putText(frame, f"✅ {name} Marked Present!", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow('Smart Attendance', frame)
                #cv2.waitKey(2000)
                break'''


        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Smart Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

