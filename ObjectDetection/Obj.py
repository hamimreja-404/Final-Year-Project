import cv2
import torch
import pyttsx3
import threading
from queue import Queue, Empty, Full
import time
import numpy as np
from ultralytics import YOLO # <--- FIX 1: ADDED THIS IMPORT

# --- System Configuration ---
# Lower resolution for faster processing. 480x320 is a good balance.
CAPTURE_WIDTH = 480
CAPTURE_HEIGHT = 320

# The model will process frames at this size. 320 is fast.
YOLO_IMG_SIZE = 320

# Path to your YOLO model. 'yolov10n.pt' is the smallest and fastest.
YOLO_MODEL_PATH = 'yolov10n.pt'

# Confidence threshold: only detect objects with confidence > 0.4
CONFIDENCE_THRESHOLD = 0.4

# Cooldown in seconds before announcing the same object type again
OBJECT_COOLDOWN = 5.0

class OptimizedObjectDetector:
    """
    A real-time object detector optimized for low latency by using dedicated threads
    for video capture, YOLO inference, and text-to-speech.
    """
    def __init__(self):
        # Threading and Queue setup
        self.frame_queue = Queue(maxsize=1)
        self.detection_queue = Queue(maxsize=1)
        self.tts_queue = Queue()
        self.stop_event = threading.Event()

        # State management for TTS cooldown
        self.last_spoken_time = {}

        # Initialize hardware and models
        self.device = self._get_device()
        self.model = self._load_yolo_model()
        
        # --- FIX 2: ADDED SAFETY CHECK ---
        # If model loading failed, stop initialization immediately.
        if self.model is None:
            print("❌ Model could not be loaded. Halting initialization.")
            return

        self.tts_engine = self._init_tts_engine()
        self.class_names = self.model.names

    def _get_device(self):
        """Checks for CUDA GPU and sets the device accordingly."""
        if torch.cuda.is_available():
            print("✅ CUDA GPU detected. Using GPU for acceleration.")
            return torch.device("cuda")
        print("⚠️ CUDA GPU not found. Running on CPU (will be slower).")
        return torch.device("cpu")

    def _load_yolo_model(self):
        """Loads the YOLOv10 model from the specified path."""
        try:
            print("Loading YOLOv10 model... 🚀")
            model = YOLO(YOLO_MODEL_PATH)
            # Move model to the selected device (GPU or CPU)
            model.to(self.device)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.stop_event.set()
            return None

    def _init_tts_engine(self):
        """Initializes the text-to-speech engine."""
        try:
            print("Initializing TTS engine... 🗣️")
            engine = pyttsx3.init()
            return engine
        except Exception as e:
            print(f"❌ Could not initialize TTS engine: {e}")
            return None

    def _capture_thread(self):
        """
        Dedicated thread to continuously capture frames from the webcam.
        This runs as fast as the camera can provide frames.
        """
        print("🟢 Starting camera capture thread.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ FATAL: Cannot open webcam. Please check camera connection.")
            self.stop_event.set()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Warning: Failed to grab frame.")
                time.sleep(0.1) # Wait a bit before retrying
                continue

            # Non-blocking put: if the queue is full, discard the old frame and put the new one.
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                pass
        
        cap.release()
        print("Camera capture thread stopped.")

    def _detection_thread(self):
        """
        Dedicated thread for running YOLO model inference.
        This ensures the main loop and video display are not blocked.
        """
        print("🟢 Starting object detection thread.")
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            results = self.model(frame, imgsz=YOLO_IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            detected_objects = []
            current_time = time.time()
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                
                detected_objects.append({
                    "box": box.xyxy[0].cpu().numpy(),
                    "class_name": class_name,
                    "confidence": box.conf[0].item()
                })

                if current_time - self.last_spoken_time.get(class_name, 0) > OBJECT_COOLDOWN:
                    self.last_spoken_time[class_name] = current_time
                    self.tts_queue.put(f"{class_name} detected.")

            try:
                self.detection_queue.get_nowait()
            except Empty:
                pass
            try:
                self.detection_queue.put_nowait(detected_objects)
            except Full:
                pass
        print("Object detection thread stopped.")

    def _tts_thread(self):
        """
        Dedicated thread for text-to-speech to prevent blocking other processes.
        """
        print("🟢 Starting text-to-speech thread.")
        while not self.stop_event.is_set():
            try:
                text_to_speak = self.tts_queue.get(timeout=1)
                if self.tts_engine:
                    self.tts_engine.say(text_to_speak)
                    self.tts_engine.runAndWait()
            except Empty:
                continue
        print("Text-to-speech thread stopped.")

    def run(self):
        """
        Main function to start all threads and manage the display window.
        """
        if not hasattr(self, 'model') or self.model is None:
            print("❌ Cannot run because model is not loaded.")
            return

        threads = [
            threading.Thread(target=self._capture_thread, daemon=True),
            threading.Thread(target=self._detection_thread, daemon=True),
            threading.Thread(target=self._tts_thread, daemon=True)
        ]
        for t in threads:
            t.start()
            
        print("\n✅ System is running. Press 'q' in the window to exit.")

        window_name = "Smart Glasses Simulation"
        latest_detections = []
        
        while not self.stop_event.is_set():
            try:
                frame_from_queue = self.frame_queue.get(timeout=0.5)
            except Empty:
                # If camera thread hasn't started, show a black screen
                display_frame = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
                cv2.putText(display_frame, "Connecting to camera...", (50, CAPTURE_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check if a fatal error has occurred in another thread
                if self.stop_event.is_set():
                    break
            else:
                display_frame = frame_from_queue.copy()
                try:
                    latest_detections = self.detection_queue.get_nowait()
                except Empty:
                    pass

                for det in latest_detections:
                    x1, y1, x2, y2 = map(int, det["box"])
                    label = f'{det["class_name"]} {det["confidence"]:.2f}'
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed. Shutting down...")
                break

        self.stop_event.set()
        for t in threads:
            t.join()
        
        cv2.destroyAllWindows()
        print("All resources released. Goodbye!")

if __name__ == "__main__":
    detector = OptimizedObjectDetector()
    detector.run()
