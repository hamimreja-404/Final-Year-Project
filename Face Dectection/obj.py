import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Offline TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)   # speaking speed
engine.setProperty('volume', 1.0) # volume
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Open webcam
cap = cv2.VideoCapture(0)

last_spoken_phrase = None
last_time = 0
cooldown = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            labels.append(label)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if labels:
        phrase = ", ".join(sorted(set(labels)))

        # Speak only if new phrase or cooldown expired
        now = time.time()
        if phrase != last_spoken_phrase or now - last_time > cooldown:
            engine.say(phrase)
            engine.runAndWait()

            last_spoken_phrase = phrase
            last_time = now

    cv2.imshow("YOLOv8 Detection with Voice", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()