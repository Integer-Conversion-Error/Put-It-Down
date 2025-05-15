import psutil
import time
import pygetwindow as gw
import os
import json

BLOCK_CONFIG_FILE = "block_config.json" 

class DistractionDetector:
    def __init__(self):
        # app_title: {'initial_start_time': float, 'total_open_time': float, 'last_seen_time': float, 'is_currently_open': bool}
        self.open_apps = {} 
        self.block_list = []
        self._load_block_list() # Use a "private" method for internal loading

    def _load_block_list(self):
        try:
            if os.path.exists(BLOCK_CONFIG_FILE):
                with open(BLOCK_CONFIG_FILE, 'r') as f:
                    loaded_list = json.load(f)
                    if isinstance(loaded_list, list) and all(isinstance(item, str) for item in loaded_list):
                        self.block_list = loaded_list
                    else:
                        print(f"Warning: {BLOCK_CONFIG_FILE} content is not a list of strings. Using empty block list.")
                        self.block_list = []
                        self._save_block_list() # Save empty list if format was wrong
            else:
                self.block_list = []
                self._save_block_list() # Create file if it doesn't exist
        except json.JSONDecodeError:
            print(f"Error decoding {BLOCK_CONFIG_FILE}. Using empty block list and overwriting.")
            self.block_list = []
            self._save_block_list()
        except Exception as e:
            print(f"Error loading {BLOCK_CONFIG_FILE}: {e}. Using empty block list.")
            self.block_list = []

    def _save_block_list(self):
        try:
            with open(BLOCK_CONFIG_FILE, 'w') as f:
                json.dump(self.block_list, f, indent=4)
        except Exception as e:
            print(f"Error saving {BLOCK_CONFIG_FILE}: {e}")

    def add_to_block_list(self, app_title):
        """Adds an app_title to the block list and saves. Removes from active tracking."""
        if app_title not in self.block_list:
            self.block_list.append(app_title)
            self._save_block_list()
            if app_title in self.open_apps:
                del self.open_apps[app_title] # Remove from currently tracked apps
                print(f"'{app_title}' added to block list and removed from active tracking.")
            else:
                print(f"'{app_title}' added to block list.")
            return True
        print(f"'{app_title}' is already in the block list.")
        return False

    def remove_from_block_list(self, app_title):
        """Removes an app_title from the block list and saves."""
        if app_title in self.block_list:
            self.block_list.remove(app_title)
            self._save_block_list()
            print(f"'{app_title}' removed from block list. It may be tracked again if currently open.")
            return True
        print(f"'{app_title}' not found in the block list.")
        return False
    
    def get_block_list(self):
        """Returns a copy of the current block list."""
        return list(self.block_list)

    def get_all_open_window_titles(self):
        """
        Gets the titles of all currently open (and visible) windows.
        Returns a list of titles.
        """
        titles = []
        try:
            for window in gw.getAllWindows():
                # Filter out windows with no title or very short titles if necessary
                if window.title: 
                    titles.append(window.title)
        except Exception as e:
            print(f"Error getting all window titles: {e}")
        return titles

    def update_open_apps(self):
        """
        Checks all open windows and updates their tracked open times,
        ignoring apps in the block list.
        """
        current_time = time.time()
        # Get all titles, so GUI can potentially list them as candidates for blocking
        all_currently_open_titles = set(self.get_all_open_window_titles())

        # Filter out blocked titles for actual tracking
        current_window_titles = {
            title for title in all_currently_open_titles if title not in self.block_list
        }
        
        # Remove any apps from self.open_apps that might have been added to block_list externally
        # or were in block_list at init but somehow got into open_apps (defensive)
        for tracked_app_title in list(self.open_apps.keys()):
            if tracked_app_title in self.block_list:
                del self.open_apps[tracked_app_title]
                print(f"Removed '{tracked_app_title}' from tracking as it's in the block list.")

        all_tracked_titles = list(self.open_apps.keys())

        for app_title in all_tracked_titles:
            app_data = self.open_apps[app_title]
            if app_title in current_window_titles: # App is currently open
                if not app_data['is_currently_open']: # It was previously closed, now reopened
                    app_data['is_currently_open'] = True
                    app_data['last_seen_time'] = current_time # Reset its timer start for this new session
                    # total_open_time preserves old duration
                    print(f"App '{app_title}' reopened at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
                else: # It was open and is still open
                    # Add the duration of this interval to total_open_time
                    app_data['total_open_time'] += (current_time - app_data['last_seen_time'])
                app_data['last_seen_time'] = current_time # Update last seen time for the next interval
            else: # App is not in current_window_titles
                if app_data['is_currently_open']: # It was open, but now it's closed
                    # Finalize its total_open_time for this session
                    app_data['total_open_time'] += (current_time - app_data['last_seen_time'])
                    app_data['is_currently_open'] = False
                    print(f"App '{app_title}' detected as closed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}. Total open time: {app_data['total_open_time']:.0f}s")
                    # No need to update last_seen_time as it's closed.

        # Add newly detected apps
        for title in current_window_titles:
            if title not in self.open_apps:
                self.open_apps[title] = {
                    'initial_start_time': current_time,
                    'total_open_time': 0,
                    'last_seen_time': current_time,
                    'is_currently_open': True
                }
                print(f"App '{title}' newly detected at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")


    def display_app_durations(self):
        """
        Displays the total accumulated open time for each tracked app.
        """
        current_time = time.time()
        if not self.open_apps:
            print("No applications are currently being tracked.")
            return

        print("\n--- Application Open Durations ---")
        for app_title, data in self.open_apps.items():
            duration_seconds = data['total_open_time']
            if data['is_currently_open']:
                # Add time since last check for currently open apps
                duration_seconds += (current_time - data['last_seen_time'])
            
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            status = "Open" if data['is_currently_open'] else "Closed"
            print(f"App: '{app_title}', Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s ({status})")
        print("----------------------------------")

    def get_formatted_app_durations_for_display(self, current_display_time, max_title_len=30):
        """
        Returns a list of strings, each representing an app and its duration,
        formatted for display. Sorted by open status then start time.
        """
        if not self.open_apps:
            return ["No applications tracked."]

        display_strings = []
        # Sort apps: currently open ones first, then by initial start time. Closed apps after open ones.
        sorted_apps = sorted(
            self.open_apps.items(), 
            key=lambda item: (not item[1]['is_currently_open'], item[1]['initial_start_time'])
        )

        for app_title, data in sorted_apps:
            duration_seconds = data['total_open_time']
            if data['is_currently_open']:
                duration_seconds += (current_display_time - data['last_seen_time'])
            
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            status_char = "O" if data['is_currently_open'] else "C"
            
            display_title = app_title[:max_title_len] + "..." if len(app_title) > max_title_len else app_title
            
            # Format: App Title - 0h0m0s (Status)
            display_strings.append(f"{display_title} - {int(hours)}h{int(minutes)}m{int(seconds)}s ({status_char})")
        return display_strings

    def run(self, check_interval=5):
        # Initial scan
        print("Performing initial scan of open applications...")
        self.update_open_apps() 
        self.display_app_durations()

        """
        Main loop to periodically check active applications.
        """
        print("Distraction Detector started. Press Ctrl+C to stop.")
        try:
            while True:
                self.update_open_apps()
                self.display_app_durations()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\nDistraction Detector stopped.")
            self.display_app_durations() # Final display

if __name__ == "__main__":
    detector = DistractionDetector()
    detector.run()
