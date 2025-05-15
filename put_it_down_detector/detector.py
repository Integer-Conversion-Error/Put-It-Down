import cv2
import mediapipe as mp
import numpy as np
import math
import json
import time
import os
import collections
# sys import for path modification is no longer needed here if DistractionDetector is not imported
# from DistractionDetector import DistractionDetector # This import is also removed

# --- Constants ---
# Make CONFIG_FILE path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_PITCH_THRESHOLD = 90.0
DEFAULT_TIME_THRESHOLD_SECONDS = 5.0
DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS = 0.5

# Landmark Indices
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10

class HeadPoseMonitor:
    def __init__(self, webcam_id=0):
        self.webcam_id = webcam_id
        self.cap = None
        self.face_mesh = None
        self.mp_drawing = None
        self.mp_face_mesh = None
        self.drawing_spec = None

        self.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        self.time_threshold_seconds = DEFAULT_TIME_THRESHOLD_SECONDS
        self.pitch_smoothing_window_seconds = DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS
        
        self._load_config()

        self.image_height = 480  # Default, will be updated
        self.image_width = 640   # Default, will be updated

        # State variables for processing
        self.status = "Initializing..."
        self.looking_down_start_time = None
        self.limbo_timer_display = 0.0
        self.pitch_history = collections.deque()
        self.raw_pitch_metric_val = 0.0
        self.smoothed_pitch_metric_val = 0.0
        
        self.total_time_overall = 0.0
        self.total_time_on_phone = 0.0
        self.total_time_on_screen = 0.0
        self.total_time_limbo = 0.0
        self.total_time_no_face = 0.0
        self.last_frame_time = time.time()
        self.start_time_overall = time.time()
        self.previous_status = "Initializing..."

        self._initialize_resources()

    def _initialize_resources(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            # TODO: Handle this error more gracefully for the GUI
            return

        success_init, init_frame = self.cap.read()
        if not success_init:
            print("Error: Could not read initial frame from webcam.")
            self.release_resources()
            return
        
        init_frame_flipped = cv2.flip(init_frame, 1)
        self.image_height, self.image_width, _ = init_frame_flipped.shape
        print(f"Webcam initialized: {self.image_width}x{self.image_height}")


    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.pitch_threshold = float(config.get("pitch_threshold", DEFAULT_PITCH_THRESHOLD))
                    self.time_threshold_seconds = float(config.get("time_threshold_seconds", DEFAULT_TIME_THRESHOLD_SECONDS))
                    self.pitch_smoothing_window_seconds = float(config.get("pitch_smoothing_window_seconds", DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS))
                    print(f"HPM Loaded config: PitchThr={self.pitch_threshold}, TimeThr={self.time_threshold_seconds}s, SmoothWin={self.pitch_smoothing_window_seconds}s")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"HPM Error loading config: {e}. Using defaults.")
                self._set_defaults_and_save()
        else:
            print("HPM Config file not found. Using defaults and creating one.")
            self._set_defaults_and_save()

    def _set_defaults_and_save(self):
        self.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        self.time_threshold_seconds = DEFAULT_TIME_THRESHOLD_SECONDS
        self.pitch_smoothing_window_seconds = DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS
        self.save_config()

    def save_config(self):
        config = {
            "pitch_threshold": self.pitch_threshold,
            "time_threshold_seconds": self.time_threshold_seconds,
            "pitch_smoothing_window_seconds": self.pitch_smoothing_window_seconds
        }
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"HPM Saved config: PitchThr={self.pitch_threshold}, TimeThr={self.time_threshold_seconds}s, SmoothWin={self.pitch_smoothing_window_seconds}s")

    def _calculate_pitch_metric(self, face_landmarks, image_shape):
        # h, w, _ = image_shape # Not strictly needed if using normalized z
        lm = face_landmarks.landmark
        z_forehead = lm[FOREHEAD_INDEX].z
        z_chin = lm[CHIN_INDEX].z
        # The scaling factor 1000 is arbitrary, depends on typical z range
        raw_pitch_metric = (z_chin - z_forehead) * 1000 
        return raw_pitch_metric

    def process_next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None, {} # Return None frame and empty status if no camera

        current_loop_time = time.time()
        delta_time = current_loop_time - self.last_frame_time
        self.last_frame_time = current_loop_time
        self.total_time_overall = current_loop_time - self.start_time_overall

        success, frame = self.cap.read()
        if not success:
            print("HPM: Ignoring empty camera frame.")
            return None, {} # Or previous frame/status?

        image_processed = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image_processed.flags.writeable = False
        results = self.face_mesh.process(image_processed)
        
        annotated_frame = cv2.cvtColor(image_processed, cv2.COLOR_RGB2BGR)
        annotated_frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

                self.raw_pitch_metric_val = self._calculate_pitch_metric(face_landmarks, annotated_frame.shape)
                self.pitch_history.append((current_loop_time, self.raw_pitch_metric_val))
                
                while self.pitch_history and self.pitch_history[0][0] < (current_loop_time - self.pitch_smoothing_window_seconds):
                    self.pitch_history.popleft()
                
                if self.pitch_history:
                    self.smoothed_pitch_metric_val = sum(p[1] for p in self.pitch_history) / len(self.pitch_history)
                else:
                    self.smoothed_pitch_metric_val = self.raw_pitch_metric_val

                is_looking_down = self.smoothed_pitch_metric_val > self.pitch_threshold
                if is_looking_down:
                    if self.looking_down_start_time is None:
                        self.looking_down_start_time = current_loop_time
                    duration_looking_down = current_loop_time - self.looking_down_start_time
                    self.limbo_timer_display = duration_looking_down
                    if duration_looking_down >= self.time_threshold_seconds:
                        self.status = "Looking at Phone"
                    else:
                        self.status = "Limbo"
                else:
                    self.looking_down_start_time = None
                    self.limbo_timer_display = 0.0
                    # Basic check for looking up, could be refined
                    if self.smoothed_pitch_metric_val < -self.pitch_threshold : # Example: if pitch is significantly negative
                        self.status = "Looking Up"
                    else:
                        self.status = "Looking at Screen"
        else:
            self.status = "No Face Detected"
            self.looking_down_start_time = None
            self.limbo_timer_display = 0.0
            self.pitch_history.clear()
            self.raw_pitch_metric_val = 0.0
            self.smoothed_pitch_metric_val = 0.0
        
        if self.previous_status == "Looking at Phone": self.total_time_on_phone += delta_time
        elif self.previous_status == "Looking at Screen" or self.previous_status == "Looking Up": self.total_time_on_screen += delta_time
        elif self.previous_status == "Limbo": self.total_time_limbo += delta_time
        elif self.previous_status == "No Face Detected": self.total_time_no_face += delta_time
        self.previous_status = self.status

        status_info = {
            "status": self.status,
            "raw_pitch": self.raw_pitch_metric_val,
            "smooth_pitch": self.smoothed_pitch_metric_val,
            "limbo_timer": self.limbo_timer_display,
            "time_threshold": self.time_threshold_seconds,
            "total_time_overall": self.total_time_overall,
            "total_time_on_phone": self.total_time_on_phone,
            "total_time_on_screen": self.total_time_on_screen,
            "total_time_limbo": self.total_time_limbo,
            "total_time_no_face": self.total_time_no_face,
            "image_width": self.image_width, # For GUI to create sidebar if needed
            "image_height": self.image_height
        }
        return annotated_frame, status_info

    def update_pitch_threshold(self, val):
        self.pitch_threshold = float(val)
        self.save_config()

    def update_time_threshold(self, val):
        self.time_threshold_seconds = float(val)
        self.save_config()

    def update_smoothing_window(self, val_0_1s): # val is in 0.1s units
        self.pitch_smoothing_window_seconds = float(val_0_1s) / 10.0
        self.save_config()
        
    def get_current_thresholds(self):
        return {
            "pitch_threshold": self.pitch_threshold,
            "time_threshold_seconds": self.time_threshold_seconds,
            "pitch_smoothing_window_seconds": self.pitch_smoothing_window_seconds
        }

    def release_resources(self):
        print("HPM: Releasing resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.face_mesh:
            self.face_mesh.close()
        # cv2.destroyAllWindows() # GUI will manage its own window
        print("HPM: Done releasing resources.")

# The old main() function and if __name__ == '__main__': block are removed
# as this script will now be used as a module.
# If you want to test HeadPoseMonitor standalone, you could add a new
# minimal if __name__ == '__main__': block here for testing purposes.
# For example:
# if __name__ == '__main__':
#     monitor = HeadPoseMonitor()
#     if monitor.cap and monitor.cap.isOpened():
#         while True:
#             frame, status_info = monitor.process_next_frame()
#             if frame is None:
#                 break
#             # Simple display for testing
#             cv2.putText(frame, status_info.get("status", ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#             cv2.imshow("Head Pose Test", frame)
#             if cv2.waitKey(5) & 0xFF == ord('q'):
#                 break
#         monitor.release_resources()
#         cv2.destroyAllWindows()
