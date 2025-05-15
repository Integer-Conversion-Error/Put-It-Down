# Put It Down - Focus and Distraction Monitor

"Put It Down" is a desktop application designed to help users monitor their application usage and head pose to understand their focus and potential distractions. It combines application window tracking with webcam-based head pose estimation.

## Current State of the Application

The application consists of three main Python components:

1.  **`DistractionDetector.py`**:
    *   **Functionality**: Tracks currently open application windows and the duration they are active.
    *   **Block List**: Maintains a `block_config.json` file where users can specify application titles to be "blocked." Blocked applications are ignored by the time tracker.
    *   **Data**: Stores information about each tracked (non-blocked) application, including its initial start time, total open time, and current open status.
    *   **Output**: Can provide a formatted list of tracked applications and their durations.

2.  **`put_it_down_detector/detector.py` (as `HeadPoseMonitor`)**:
    *   **Functionality**: Utilizes the computer's webcam, `cv2` (OpenCV), and `mediapipe` to perform real-time head pose estimation.
    *   **Pitch Metric**: Calculates a "pitch metric" based on the relative z-depth of forehead and chin landmarks. This metric is used to infer the user's head orientation.
    *   **States**: Classifies head pose into several states:
        *   `Looking at Screen`: User is likely looking at the computer screen.
        *   `Looking at Phone`: User's head is tilted down, suggesting they might be looking at a phone.
        *   `Limbo`: A transitional state when the head starts to tilt down, before confirming "Looking at Phone."
        *   `Looking Up`: User's head is tilted upwards.
        *   `No Face Detected`: No face is found in the webcam feed.
    *   **Time Tracking**: Records the cumulative time spent in each of these states.
    *   **Configuration**: Head pose detection parameters (pitch threshold, time threshold for phone detection, smoothing window for pitch) are configurable via `put_it_down_detector/config.json` and can be adjusted live from the dashboard.

3.  **`main_dashboard.py`**:
    *   **GUI**: Provides a Tkinter-based graphical user interface to visualize data from both `DistractionDetector` and `HeadPoseMonitor`.
    *   **Layout**: The dashboard is split into two main panes:
        *   **Left Pane**:
            *   Displays the live webcam feed with MediaPipe face mesh overlay.
            *   Features a pie chart showing the distribution of time spent in different head pose states (On Screen, On Phone, Limbo, No Face).
            *   Shows detailed text-based status of the `HeadPoseMonitor`, including current state, raw and smoothed pitch values, and total time in each state.
        *   **Right Pane**:
            *   **Head Pose Controls**: Allows users to dynamically adjust the pitch threshold, time threshold, and smoothing window for the `HeadPoseMonitor` using sliders. Changes are saved to `config.json`.
            *   **Tracked Applications**: Lists applications currently being tracked by `DistractionDetector` along with their accumulated open times.
            *   **Block List Manager**:
                *   Displays a list of all detected, unblocked window titles.
                *   Displays a list of currently blocked application titles (from `block_config.json`).
                *   Provides buttons to move selected applications between the "unblocked" and "blocked" lists.
    *   **Threading**: Uses background threads to manage the webcam processing and application tracking loops, ensuring the GUI remains responsive.

### How to Run

1.  Ensure you have Python installed.
2.  Install necessary dependencies:
    ```bash
    pip install psutil pygetwindow opencv-python mediapipe Pillow matplotlib
    ```
3.  Navigate to the project directory in your terminal.
4.  Run the main dashboard:
    ```bash
    python main_dashboard.py
    ```

## Future Steps & Potential Features

Here are some potential enhancements and new features that could be added to the application:

**I. Core Functionality Enhancements:**

1.  **Productivity Scoring/Reporting:**
    *   **Concept:** Combine data from `DistractionDetector` (time on productive vs. distracting apps) and `HeadPoseMonitor` (time looking at screen vs. phone) to generate a daily/session productivity score or report.
    *   **Implementation Ideas:**
        *   Allow users to categorize applications as "productive," "neutral," or "distracting."
        *   Develop a scoring algorithm (e.g., +points for time on productive apps while looking at the screen, -points for time on distracting apps or looking at phone).
        *   Display this score/report in the dashboard or save it to a log.

2.  **"Focus Mode" / Active Blocking:**
    *   **Concept:** Beyond just tracking, actively intervene when distractions are detected.
    *   **Implementation Ideas:**
        *   If a "blocked" app is opened, or if "Looking at Phone" state persists for too long, trigger an action:
            *   **Gentle:** A visual/audio notification on screen.
            *   **Moderate:** Minimize the distracting app.
            *   **Strict:** Attempt to close the distracting app (with user confirmation or after a grace period).
        *   Implement a "Focus Session" timer where stricter rules apply.

3.  **Improved "Looking at Screen" Detection:**
    *   **Concept:** The current "Looking at Screen" is a default state. Make it more specific.
    *   **Implementation Ideas:**
        *   Correlate `HeadPoseMonitor`'s "Looking at Screen" state with the *currently active window* from `DistractionDetector` (e.g., using `pygetwindow.getActiveWindow()`). This could help differentiate between genuinely working on a productive app vs. staring blankly at the screen with a game open.

4.  **Data Logging and Visualization Over Time:**
    *   **Concept:** Store the collected data (app usage times, head pose state durations) persistently to see trends.
    *   **Implementation Ideas:**
        *   Save data periodically (e.g., every 5-15 minutes, or at the end of a session) to a CSV, SQLite database, or JSON log file.
        *   Add a new tab or section in the dashboard to display historical data using charts (e.g., bar charts for daily app usage, line charts for focus trends over a week).

**II. User Experience (UX) and Interface (UI) Improvements:**

5.  **Customizable Application Categories:**
    *   **Concept:** Allow users to define their own categories for applications (e.g., "Work," "Study," "Communication," "Entertainment") beyond just "blocked/unblocked."
    *   **Implementation Ideas:**
        *   Modify `block_config.json` or use a new config file to store app titles mapped to categories.
        *   Update the UI to manage these categories.
        *   The pie chart in `main_dashboard.py` could then show time distribution by these user-defined categories.

6.  **Notification System:**
    *   **Concept:** Provide non-intrusive notifications for certain events.
    *   **Implementation Ideas:**
        *   E.g., "You've been looking at your phone for X minutes," or "You've spent Y hours on [Distracting App] today."
        *   Could use system notifications (e.g., via the `plyer` library) or custom pop-ups within the Tkinter app.

7.  **Profile/User Management (Advanced):**
    *   **Concept:** If multiple people might use it on the same machine, or if a user wants different settings for "work" vs. "leisure."
    *   **Implementation Ideas:**
        *   Allow creating profiles, each with its own block lists, app categories, and HPM thresholds.

8.  **Refined UI for Block Management:**
    *   **Concept:** Make the block list management more intuitive.
    *   **Implementation Ideas:**
        *   Drag-and-drop functionality (can be complex in Tkinter).
        *   Clearer visual distinction or icons for blocked vs. unblocked apps.
        *   Search/filter for application lists if they become very long.

**III. Technical and Performance Enhancements:**

9.  **More Robust Error Handling & Resource Management:**
    *   **Concept:** Ensure the application is stable, especially with webcam and window interactions.
    *   **Implementation Ideas:**
        *   More detailed error logging.
        *   Graceful handling if webcam disconnects/reconnects during runtime.
        *   Ensure threads are always properly terminated on exit.

10. **Configuration for `DistractionDetector`:**
    *   **Concept:** Allow configurability for `DistractionDetector` (e.g., check interval) via its own config file or the main dashboard.
    *   **Implementation Ideas:**
        *   Add `_load_config` / `_save_config` methods to `DistractionDetector`.

11. **Performance Optimization for `HeadPoseMonitor`:**
    *   **Concept:** Ensure `process_next_frame` is as efficient as possible, especially on less powerful hardware.
    *   **Implementation Ideas:**
        *   Profile the code to identify bottlenecks.
        *   Ensure no unnecessary operations are done in the main processing loop.

**IV. New Detection Capabilities (More Ambitious):**

12. **Audio Distraction Detection (Advanced):**
    *   **Concept:** Detect if loud music, videos, or other distracting sounds are playing.
    *   **Implementation Ideas:**
        *   Would require microphone access and audio processing libraries (e.g., `librosa`, `sounddevice`).
        *   Could classify audio environments or detect specific sound events.

13. **Eye Gaze Tracking (Very Advanced):**
    *   **Concept:** Go beyond head pose to estimate where on the screen the user is looking.
    *   **Implementation Ideas:**
        *   MediaPipe offers some eye landmarks. More advanced eye-tracking often requires specialized hardware or more complex models. This would be a significant R&D effort.
