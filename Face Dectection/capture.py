import cv2 as cv
import os
from datetime import datetime
import numpy as np
import time

# all settings
dataset_dir = "known_faces"
modelFile = "DNN/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "DNN/deploy.prototxt"
confidence_threshold = 0.5
capture_interval = 1  # seconds
images_to_capture = 50  # images per person

# Name
person_name = input("Enter the name of the person: ").strip()
save_path = os.path.join(dataset_dir, person_name)
os.makedirs(save_path, exist_ok=True)

# Camera start
net = cv.dnn.readNetFromCaffe(configFile, modelFile)
cap = cv.VideoCapture(0)

print(f"[INFO] Capturing {images_to_capture} images for '{person_name}'...")
print("[INFO] Please face the camera...")

captured_count = 0
last_capture_time = 0

while captured_count < images_to_capture:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
                                1.0, (300, 300),
                                (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    max_confidence = 0
    best_box = None

    # Find the face with highest confidence 
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            best_box = box.astype("int")

    # Draw only one box (highest confidence)
    if best_box is not None and max_confidence > confidence_threshold:
        (startX, startY, endX, endY) = best_box
        cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = f"{max_confidence*100:.1f}%"
        cv.putText(frame, text, (startX, startY-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Auto-capture
        if time.time() - last_capture_time >= capture_interval:
            time_str = datetime.now().strftime("%H_%M_%S")
            file_path = os.path.join(save_path, f"{time_str}.jpg")

            face_img = frame[startY:endY, startX:endX]
            gray_face = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
            cv.imwrite(file_path, gray_face)

            captured_count += 1
            last_capture_time = time.time()
            print(f"[SAVED] {file_path} ({captured_count}/{images_to_capture})")

    cv.imshow("Face Capture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print("[INFO] Image capture complete. Starting training...")

# TRAIN MODEL with those images
def train_model():
    print("[INFO] Training model...")
    people = []
    features = []
    labels = []

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue
        people.append(person)
        label = people.index(person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            features.append(img)
            labels.append(label)

    features = [np.array(f, dtype='uint8') for f in features]
    labels = np.array(labels, dtype='int32')

    os.makedirs("Trained_Faces", exist_ok=True)
    face_recognizer = cv.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8
    )
    face_recognizer.train(features, labels)
    face_recognizer.save("Trained_Faces/face_trained.yml")
    np.save("Trained_Faces/people.npy", np.array(people))

    print(f"[INFO] Training complete. Trained on {len(people)} people and {len(features)} images.")

train_model()
