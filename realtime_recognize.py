import cv2 as cv
import numpy as np
import os
import pyttsx3
import time

# Speech settings----
engine = pyttsx3.init()
engine.setProperty('rate', 150)  
engine.setProperty('volume', 1)  
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) 

# link to other deep neural networks 
modelFile = "DNN/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "DNN/deploy.prototxt"
detection_confidence = 0.5  
recognition_threshold = 70.0 
face_resize = (200, 200)  

# paths of all the trained faces 
trained_model_path = "Trained_Faces/face_trained.yml"
people_path = "Trained_Faces/people.npy"

# Validation of all files 
if not os.path.exists(modelFile) or not os.path.exists(configFile):
    raise FileNotFoundError("DNN model/config not found. Check DNN files paths.")

if not os.path.exists(trained_model_path) or not os.path.exists(people_path):
    raise FileNotFoundError("Trained model or people.npy not found. Run training first.")


net = cv.dnn.readNetFromCaffe(configFile, modelFile)

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(trained_model_path)

people = np.load(people_path, allow_pickle=True).tolist()
if isinstance(people, np.ndarray):
    people = people.tolist()

# Start Live camera for dectection
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Starting real-time recognition. Press 'q' to quit.")

last_spoken_name = None
cooldown_time = 3 
last_spoken_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
                                1.0, (300, 300),
                                (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    max_conf = 0.0
    best_box = None
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > max_conf:
            max_conf = conf
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")

    if best_box is not None and max_conf > detection_confidence:
        (startX, startY, endX, endY) = best_box
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        face_roi = frame[startY:endY, startX:endX]
        if face_roi.size != 0 and (endX - startX) > 10 and (endY - startY) > 10:
            gray = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
            try:
                gray_resized = cv.resize(gray, face_resize)
            except:
                gray_resized = gray

            label, conf = recognizer.predict(gray_resized)

            if conf < recognition_threshold and 0 <= label < len(people):
                name = people[label]
                label_text = f"{name} ({conf:.1f})"
            else:
                name = "Unknown"
                label_text = f"{name} ({conf:.1f})"

            # speech codes
            current_time = time.time()
            if name != last_spoken_name or (current_time - last_spoken_time) > cooldown_time:
                engine.say(f"This is {name}")
                engine.runAndWait()
                last_spoken_name = name
                last_spoken_time = current_time

            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            (text_w, text_h), _ = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv.rectangle(frame, (startX, startY - 25), (startX + text_w, startY), (0, 255, 0), -1)
            cv.putText(frame, label_text, (startX, startY - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 2)

    cv.imshow("Real-time Recognition", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
