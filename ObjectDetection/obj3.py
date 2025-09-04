import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time
from queue import Queue, Empty, Full
import numpy as np

# --- Configuration ---
FRAME_SKIP = 2
OBJECT_COOLDOWN = 5  # seconds
YOLO_MODEL_PATH = 'yolov10n.pt'
CONFIDENCE_THRESHOLD = 0.2
IMG_SIZE = 320

class RealTimeObjectDetector:
    def __init__(self):
        # Queues for video frames and detection results
        self.frame_queue = Queue(maxsize=1)
        self.results_queue = Queue(maxsize=1)
        
        # State management
        self.last_spoken_time = {}
        self.latest_detections_for_drawing = []
        self.stop_event = threading.Event()

        # Initialize components
        self.model = self._init_yolo()
        self.engine = self._init_tts() # Initialize TTS engine once
        self.class_names = self.model.names

    def _init_yolo(self):
        """Initializes the YOLO model."""
        try:
            print("Loading model... 🚀")
            model = YOLO(YOLO_MODEL_PATH)
            model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False) # Warm up
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            exit()
            
    def _init_tts(self):
        """Initializes the text-to-speech engine."""
        try:
            print("Initializing TTS engine... 🗣️")
            return pyttsx3.init()
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {e}")
            return None

    def _clear_queue(self, q):
        """Helper function to empty a queue."""
        with q.mutex:
            q.queue.clear()

    # --- VOICE LOGIC AS REQUESTED ---
    def speak(self, text):
        """ Speaks the given text by creating a new thread for each call. """
        if not self.engine:
            print("⚠️ TTS engine not available.")
            return

        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Create and start a new thread for every speech request
        threading.Thread(target=run, daemon=True).start()

    def detection_worker(self):
        """Worker thread that consumes frames, runs YOLO, and produces results."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            current_detections = []
            results = self.model(frame, stream=True, imgsz=IMG_SIZE, verbose=False)

            for r in results:
                for box in r.boxes:
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        cls_id = int(box.cls[0])
                        class_name = self.class_names[cls_id]

                        current_detections.append({
                            "box": box.xyxy[0].cpu().numpy(),
                            "class_name": class_name,
                            "confidence": box.conf[0].item()
                        })

                        current_time = time.time()
                        if current_time - self.last_spoken_time.get(class_name, 0) > OBJECT_COOLDOWN:
                            self.last_spoken_time[class_name] = current_time
                            # Directly call the speak method
                            self.speak(f"I see a {class_name}")
            
            self._clear_queue(self.results_queue)
            self.results_queue.put(current_detections)

    def draw_boxes(self, frame):
        """Draws bounding boxes on the frame."""
        for det in self.latest_detections_for_drawing:
            x1, y1, x2, y2 = map(int, det["box"])
            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            color = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run(self):
        """Main function to start threads and run the primary video loop."""
        # Only the detection worker thread is needed now
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return

        cap.set(3, 640)
        cap.set(4, 480)
        window_name = "Object Detection"
        cv2.namedWindow(window_name)
        print("🟢 Starting webcam feed. Press 'q' or close the window to quit.")
        frame_count = 0

        while not self.stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("⚠️ Error: Failed to capture image.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                self._clear_queue(self.frame_queue)
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except Full:
                    pass

            try:
                self.latest_detections_for_drawing = self.results_queue.get_nowait()
            except Empty:
                pass

            display_frame = self.draw_boxes(frame)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Exiting...")
                break
        
        self.stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam feed stopped and resources released.")

if __name__ == "__main__":
    detector = RealTimeObjectDetector()
    detector.run()