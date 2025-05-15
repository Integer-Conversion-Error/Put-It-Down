import cv2
import mediapipe as mp
import numpy as np
import math
import json
import time
import os
import collections

# --- Configuration ---
CONFIG_FILE = "put_it_down_detector/config.json"
DEFAULT_PITCH_THRESHOLD = 90.0  # Default pitch threshold for the raw metric
DEFAULT_TIME_THRESHOLD_SECONDS = 5.0  # Default time in seconds
DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS = 0.5 # Default smoothing window

# Global variables for thresholds, to be updated by trackbars
current_pitch_threshold = DEFAULT_PITCH_THRESHOLD
current_time_threshold_seconds = DEFAULT_TIME_THRESHOLD_SECONDS
current_pitch_smoothing_window_seconds = DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS

def load_config():
    global current_pitch_threshold, current_time_threshold_seconds, current_pitch_smoothing_window_seconds
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                current_pitch_threshold = float(config.get("pitch_threshold", DEFAULT_PITCH_THRESHOLD))
                current_time_threshold_seconds = float(config.get("time_threshold_seconds", DEFAULT_TIME_THRESHOLD_SECONDS))
                current_pitch_smoothing_window_seconds = float(config.get("pitch_smoothing_window_seconds", DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS))
                print(f"Loaded config: PitchThr={current_pitch_threshold}, TimeThr={current_time_threshold_seconds}s, SmoothWin={current_pitch_smoothing_window_seconds}s")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading config file or invalid format: {e}. Using defaults.")
            current_pitch_threshold = DEFAULT_PITCH_THRESHOLD
            current_time_threshold_seconds = DEFAULT_TIME_THRESHOLD_SECONDS
            current_pitch_smoothing_window_seconds = DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS
            save_config() # Save defaults if file was bad
    else:
        print("Config file not found. Using defaults and creating one.")
        current_pitch_threshold = DEFAULT_PITCH_THRESHOLD
        current_time_threshold_seconds = DEFAULT_TIME_THRESHOLD_SECONDS
        current_pitch_smoothing_window_seconds = DEFAULT_PITCH_SMOOTHING_WINDOW_SECONDS
        save_config()

def save_config():
    config = {
        "pitch_threshold": current_pitch_threshold,
        "time_threshold_seconds": current_time_threshold_seconds,
        "pitch_smoothing_window_seconds": current_pitch_smoothing_window_seconds
    }
    # Ensure directory exists
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved config: PitchThr={current_pitch_threshold}, TimeThr={current_time_threshold_seconds}s, SmoothWin={current_pitch_smoothing_window_seconds}s")

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Landmark Indices ---
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10

# --- Function to Calculate Pitch Metric ---
def calculate_pitch_metric(face_landmarks, image_shape):
    h, w, _ = image_shape
    lm = face_landmarks.landmark
    
    # Using Z-coordinates of forehead and chin for pitch metric
    # MediaPipe Z: Smaller value is closer to the camera.
    # If looking down: chin moves away (larger z relative to origin), forehead moves closer (smaller z relative to origin).
    # So, z_chin - z_forehead should be positive when looking down.
    z_forehead = lm[FOREHEAD_INDEX].z
    z_chin = lm[CHIN_INDEX].z
    
    raw_pitch_metric = (z_chin - z_forehead) * 1000 # Arbitrary scaling
    return raw_pitch_metric

# --- OpenCV Window and Trackbar Setup ---
WINDOW_NAME = 'Put It Down Detector'

def on_pitch_thresh_trackbar(val):
    global current_pitch_threshold
    current_pitch_threshold = float(val) # Trackbar gives int, metric can be float
    # save_config() # Removed to prevent lag; will save on exit

def on_time_thresh_trackbar(val):
    global current_time_threshold_seconds
    current_time_threshold_seconds = float(val) # Trackbar gives int
    # save_config() # Removed to prevent lag; will save on exit

def on_pitch_smoothing_trackbar(val):
    global current_pitch_smoothing_window_seconds
    current_pitch_smoothing_window_seconds = float(val) / 10.0 # Trackbar 0-50 -> 0.0-5.0s
    # save_config() # Removed to prevent lag; will save on exit

# --- Main Application Logic ---
def main():
    global current_pitch_threshold, current_time_threshold_seconds, current_pitch_smoothing_window_seconds # Allow modification by trackbars

    load_config()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME)
    # Trackbar for Pitch Threshold (using the raw metric scale, e.g., 0-200)
    # The actual range might need adjustment based on observed metric values
    cv2.createTrackbar('Pitch Thr.', WINDOW_NAME, int(current_pitch_threshold), 200, on_pitch_thresh_trackbar)
    # Trackbar for Time Threshold (in seconds, e.g., 1-30s)
    cv2.createTrackbar('Time Thr. (s)', WINDOW_NAME, int(current_time_threshold_seconds), 30, on_time_thresh_trackbar)
    # Trackbar for Pitch Smoothing Window (in 0.1s increments, e.g., 0-50 for 0.0s-5.0s)
    cv2.createTrackbar('Smooth Win (0.1s)', WINDOW_NAME, int(current_pitch_smoothing_window_seconds * 10), 50, on_pitch_smoothing_trackbar)


    status = "Initializing..."
    looking_down_start_time = None # Timestamp when user started looking down
    limbo_timer_display = 0.0
    pitch_history = collections.deque() # For storing (timestamp, pitch_value)

    print("Starting webcam feed...")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        current_time = time.time() # Get current time for timer logic

        # Read current threshold values (might have been updated by trackbars via callbacks)
        # The global variables current_pitch_threshold and current_time_threshold_seconds are used directly.

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                raw_pitch_metric = calculate_pitch_metric(face_landmarks, image.shape)
                
                # Update pitch history for smoothing
                pitch_history.append((current_time, raw_pitch_metric))
                
                # Remove old pitch values outside the smoothing window
                while pitch_history and pitch_history[0][0] < (current_time - current_pitch_smoothing_window_seconds):
                    pitch_history.popleft()
                
                # Calculate smoothed pitch
                smoothed_pitch_metric = 0.0
                if pitch_history:
                    smoothed_pitch_metric = sum(p[1] for p in pitch_history) / len(pitch_history)
                else: # Should not happen if window > 0 and face detected, but good for robustness
                    smoothed_pitch_metric = raw_pitch_metric


                # --- State Logic (using smoothed pitch) ---
                is_looking_down = smoothed_pitch_metric > current_pitch_threshold

                if is_looking_down:
                    if looking_down_start_time is None:
                        looking_down_start_time = current_time
                    
                    duration_looking_down = current_time - looking_down_start_time
                    limbo_timer_display = duration_looking_down

                    if duration_looking_down >= current_time_threshold_seconds:
                        status = "Looking at Phone"
                    else:
                        status = "Limbo"
                else:
                    looking_down_start_time = None # Reset timer
                    limbo_timer_display = 0.0
                    # Using smoothed_pitch_metric for looking up as well for consistency
                    if smoothed_pitch_metric < -current_pitch_threshold: 
                        status = "Looking Up"
                    else:
                        status = "Looking at Screen"
                
                cv2.putText(image, f"Raw Pitch: {raw_pitch_metric:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Smooth Pitch: {smoothed_pitch_metric:.2f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            status = "No Face Detected"
            looking_down_start_time = None # Reset timer if no face
            limbo_timer_display = 0.0
            pitch_history.clear() # Clear history if no face

        # Display status and timer
        cv2.putText(image, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if status == "Limbo":
            cv2.putText(image, f"Timer: {limbo_timer_display:.1f}s / {current_time_threshold_seconds:.0f}s", (10, 120), # Adjusted Y for new pitch text
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)


        cv2.imshow(WINDOW_NAME, image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    print("Releasing resources...")
    save_config() # Save final configuration before exiting
    cap.release()
    cv2.destroyAllWindows()
    if 'face_mesh' in globals() and face_mesh is not None: # Check if face_mesh was initialized
        face_mesh.close()
    print("Done.")

if __name__ == '__main__':
    main()
