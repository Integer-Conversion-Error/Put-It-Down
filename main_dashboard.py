import tkinter as tk
from tkinter import ttk, Listbox, Scrollbar, Button, Label, Frame, messagebox
import threading
import time
import os
import cv2 
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib # Matplotlib backend setting for Tkinter
matplotlib.use('TkAgg')


# Assuming DistractionDetector.py is in the same directory (project root)
from DistractionDetector import DistractionDetector
# Assuming detector.py (now HeadPoseMonitor) is in put_it_down_detector subdirectory
from put_it_down_detector.detector import HeadPoseMonitor


class MainDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Comprehensive Monitoring Dashboard")
        self.geometry("1000x700") 

        self.distraction_detector = DistractionDetector()
        self.head_pose_monitor = HeadPoseMonitor()
        
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane ---
        self.left_pane = ttk.Frame(self.paned_window, width=640) 
        self.paned_window.add(self.left_pane, weight=3) 

        Label(self.left_pane, text="Webcam Feed", font=("Arial", 14)).pack(padx=5, pady=5, side=tk.TOP)
        self.video_label = Label(self.left_pane, relief=tk.SUNKEN, bg="lightgrey")
        self.video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5) # Video expands

        # Pie chart will be at the bottom of left_pane, then HPM info below it.
        # --- Pie Chart for Time Distribution (Fixed Size Container) ---
        self.pie_chart_frame = ttk.LabelFrame(self.left_pane, text="Time Distribution")
        
        FIGURE_DPI = 100
        FIGURE_WIDTH_INCHES = 3.5  # Approx 350px wide
        FIGURE_HEIGHT_INCHES = 2.6 # Approx 260px tall
        
        frame_width_px = int(FIGURE_WIDTH_INCHES * FIGURE_DPI)
        frame_height_px = int(FIGURE_HEIGHT_INCHES * FIGURE_DPI)
        
        self.pie_chart_frame.config(width=frame_width_px, height=frame_height_px) 
        self.pie_chart_frame.pack_propagate(False) 
        self.pie_chart_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=5) 

        self.fig_pie = Figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), dpi=FIGURE_DPI) 
        self.ax_pie = self.fig_pie.add_subplot(111)

        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, master=self.pie_chart_frame)
        self.canvas_pie_widget = self.canvas_pie.get_tk_widget()
        self.canvas_pie_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Head Pose Info Frame (Packed below pie chart)
        hpm_info_frame = ttk.LabelFrame(self.left_pane, text="Head Pose Analysis")
        hpm_info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5) 
        
        self.hpm_status_label = Label(hpm_info_frame, text="Status: Initializing...", font=("Arial", 10))
        self.hpm_status_label.pack(anchor=tk.W, padx=5, pady=(5,0)) 
        self.hpm_pitch_label = Label(hpm_info_frame, text="Pitch (S/R): N/A / N/A", font=("Arial", 9))
        self.hpm_pitch_label.pack(anchor=tk.W, padx=5)

        self.hpm_time_overall_label = Label(hpm_info_frame, text="Overall: 0.0s", font=("Arial", 9))
        self.hpm_time_overall_label.pack(anchor=tk.W, padx=5, pady=(5,0))
        self.hpm_time_on_screen_label = Label(hpm_info_frame, text="On Screen: 0.0s", font=("Arial", 9))
        self.hpm_time_on_screen_label.pack(anchor=tk.W, padx=5)
        self.hpm_time_on_phone_label = Label(hpm_info_frame, text="On Phone: 0.0s", font=("Arial", 9))
        self.hpm_time_on_phone_label.pack(anchor=tk.W, padx=5)
        self.hpm_time_limbo_label = Label(hpm_info_frame, text="Limbo: 0.0s", font=("Arial", 9))
        self.hpm_time_limbo_label.pack(anchor=tk.W, padx=5)
        self.hpm_time_no_face_label = Label(hpm_info_frame, text="No Face: 0.0s", font=("Arial", 9))
        self.hpm_time_no_face_label.pack(anchor=tk.W, padx=5, pady=(0,5))

        # --- Right Pane ---
        self.right_pane = ttk.Frame(self.paned_window, width=350) 
        self.paned_window.add(self.right_pane, weight=1)

        hpm_controls_frame = ttk.LabelFrame(self.right_pane, text="Head Pose Controls")
        hpm_controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        current_thresholds = self.head_pose_monitor.get_current_thresholds()

        Label(hpm_controls_frame, text="Pitch Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pitch_scale = tk.Scale(hpm_controls_frame, from_=0, to=200, orient=tk.HORIZONTAL,
                                    command=lambda v: self.head_pose_monitor.update_pitch_threshold(v))
        self.pitch_scale.set(current_thresholds["pitch_threshold"])
        self.pitch_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        Label(hpm_controls_frame, text="Time Threshold (s):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_scale = tk.Scale(hpm_controls_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                   command=lambda v: self.head_pose_monitor.update_time_threshold(v))
        self.time_scale.set(current_thresholds["time_threshold_seconds"])
        self.time_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        Label(hpm_controls_frame, text="Smoothing (0.1s):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.smooth_scale = tk.Scale(hpm_controls_frame, from_=0, to=50, orient=tk.HORIZONTAL, 
                                     command=lambda v: self.head_pose_monitor.update_smoothing_window(v))
        self.smooth_scale.set(current_thresholds["pitch_smoothing_window_seconds"] * 10)
        self.smooth_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        hpm_controls_frame.columnconfigure(1, weight=1)

        tracked_apps_frame = ttk.LabelFrame(self.right_pane, text="Tracked Applications")
        tracked_apps_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5) 

        self.tracked_apps_listbox = Listbox(tracked_apps_frame, height=6) 
        self.tracked_apps_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), pady=5)
        tracked_apps_scrollbar = Scrollbar(tracked_apps_frame, orient=tk.VERTICAL, command=self.tracked_apps_listbox.yview)
        tracked_apps_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,5), pady=5)
        self.tracked_apps_listbox.config(yscrollcommand=tracked_apps_scrollbar.set)

        block_manager_frame = ttk.LabelFrame(self.right_pane, text="Block List Manager")
        block_manager_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5) 
        
        Label(block_manager_frame, text="All Detected Windows (unblocked):").pack(anchor=tk.W, padx=5)
        
        all_windows_frame = Frame(block_manager_frame)
        all_windows_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        self.all_windows_listbox = Listbox(all_windows_frame, height=5, selectmode=tk.EXTENDED)
        self.all_windows_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        all_windows_scrollbar = Scrollbar(all_windows_frame, orient=tk.VERTICAL, command=self.all_windows_listbox.yview)
        all_windows_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.all_windows_listbox.config(yscrollcommand=all_windows_scrollbar.set)
        
        buttons_frame_block = Frame(block_manager_frame)
        buttons_frame_block.pack(fill=tk.X, pady=2)
        Button(buttons_frame_block, text="Block Selected >>", command=self._block_selected).pack(side=tk.LEFT, expand=True, padx=5)
        Button(buttons_frame_block, text="<< Unblock Selected", command=self._unblock_selected).pack(side=tk.RIGHT, expand=True, padx=5)

        Label(block_manager_frame, text="Currently Blocked Applications:").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        blocked_list_frame = Frame(block_manager_frame)
        blocked_list_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        self.blocked_listbox = Listbox(blocked_list_frame, height=5, selectmode=tk.EXTENDED)
        self.blocked_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        blocked_list_scrollbar = Scrollbar(blocked_list_frame, orient=tk.VERTICAL, command=self.blocked_listbox.yview)
        blocked_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.blocked_listbox.config(yscrollcommand=blocked_list_scrollbar.set)
        
        self.running = True
        self.hpm_thread = threading.Thread(target=self._hpm_loop, daemon=True)
        self.hpm_thread.start()
        self.app_tracking_thread = threading.Thread(target=self._app_tracking_loop, daemon=True)
        self.app_tracking_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._update_block_management_ui() 

    def _hpm_loop(self):
        while self.running:
            if self.head_pose_monitor and self.head_pose_monitor.cap and self.head_pose_monitor.cap.isOpened():
                frame, status_info = self.head_pose_monitor.process_next_frame()
                if frame is not None:
                    try:
                        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv2image)
                        
                        label_width = self.video_label.winfo_width()
                        label_height = self.video_label.winfo_height()
                        if label_width > 1 and label_height > 1: 
                            img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)

                        imgtk = ImageTk.PhotoImage(image=img)
                        self.after(0, self._update_video_label, imgtk) 
                    except Exception as e:
                        print(f"Error updating video label: {e}")
                
                if status_info:
                    self.after(0, self._update_hpm_status_labels, status_info)
            else:
                print("HPM loop: Webcam not available. Retrying in 5s...")
                time.sleep(5)
                if self.running and not (self.head_pose_monitor.cap and self.head_pose_monitor.cap.isOpened()):
                     self.head_pose_monitor._initialize_resources()
            time.sleep(0.03) 

    def _update_video_label(self, imgtk):
        if not self.running or not self.video_label.winfo_exists(): return
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def _update_hpm_status_labels(self, status_info):
        if not self.running: return
        if self.hpm_status_label.winfo_exists():
            self.hpm_status_label.config(text=f"Status: {status_info.get('status', 'N/A')}")
        if self.hpm_pitch_label.winfo_exists():
            self.hpm_pitch_label.config(text=f"Pitch (S/R): {status_info.get('smooth_pitch', 0.0):.1f} / {status_info.get('raw_pitch', 0.0):.1f}")
        if self.hpm_time_overall_label.winfo_exists():
            self.hpm_time_overall_label.config(text=f"Overall: {status_info.get('total_time_overall', 0.0):.1f}s")
        if self.hpm_time_on_screen_label.winfo_exists():
            self.hpm_time_on_screen_label.config(text=f"On Screen: {status_info.get('total_time_on_screen', 0.0):.1f}s")
        if self.hpm_time_on_phone_label.winfo_exists():
            self.hpm_time_on_phone_label.config(text=f"On Phone: {status_info.get('total_time_on_phone', 0.0):.1f}s")
        if self.hpm_time_limbo_label.winfo_exists():
            self.hpm_time_limbo_label.config(text=f"Limbo: {status_info.get('total_time_limbo', 0.0):.1f}s")
        if self.hpm_time_no_face_label.winfo_exists():
            self.hpm_time_no_face_label.config(text=f"No Face: {status_info.get('total_time_no_face', 0.0):.1f}s")
        
        self._update_pie_chart(status_info)

    def _update_pie_chart(self, status_info):
        if not self.running or not self.canvas_pie_widget.winfo_exists(): return

        labels = ['On Screen', 'On Phone', 'Limbo', 'No Face']
        times = [
            status_info.get('total_time_on_screen', 0.0),
            status_info.get('total_time_on_phone', 0.0),
            status_info.get('total_time_limbo', 0.0),
            status_info.get('total_time_no_face', 0.0)
        ]
        
        active_labels = []
        active_times = []
        for i, time_val in enumerate(times):
            if time_val > 0.01: 
                active_labels.append(labels[i])
                active_times.append(time_val)

        self.ax_pie.clear() 

        if not active_times or sum(active_times) == 0:
            self.ax_pie.text(0.5, 0.5, 'No time data yet', horizontalalignment='center', verticalalignment='center', transform=self.ax_pie.transAxes)
        else:
            color_map = {'On Screen': '#4CAF50', 'On Phone': '#FFC107', 'Limbo': '#2196F3', 'No Face': '#9E9E9E'}
            pie_colors = [color_map.get(label, '#CCCCCC') for label in active_labels]

            wedges, texts, autotexts = self.ax_pie.pie(
                active_times, 
                labels=None, 
                autopct='%1.1f%%', 
                startangle=90,
                colors=pie_colors,
                pctdistance=0.85 
            )
            for text_obj in autotexts:
                text_obj.set_fontsize(7)
                text_obj.set_color("white")
            
            # Legend temporarily removed for diagnosing shrinking issue
            # legend = self.ax_pie.legend(wedges, active_labels, title="States", loc="best", fontsize='x-small')
            # try:
            #     if legend: legend.get_title().set_fontsize('x-small')
            # except AttributeError:
            #     pass 

        self.ax_pie.axis('equal')  
        # Removed tight_layout() call
        
        if self.canvas_pie_widget.winfo_exists():
            self.canvas_pie.draw_idle()

    def _app_tracking_loop(self):
        while self.running:
            self.distraction_detector.update_open_apps()
            if self.running: self.after(0, self._update_tracked_apps_listbox)
            if self.running: self.after(0, self._update_block_management_ui)
            time.sleep(2)

    def _update_tracked_apps_listbox(self):
        if not self.running or not self.tracked_apps_listbox.winfo_exists(): return
        self.tracked_apps_listbox.delete(0, tk.END)
        tracked_apps_formatted = self.distraction_detector.get_formatted_app_durations_for_display(time.time())
        for item in tracked_apps_formatted:
            self.tracked_apps_listbox.insert(tk.END, item)

    def _update_block_management_ui(self):
        if not self.running: return

        def update_listbox_preserve_selection(listbox_widget, new_items_sorted):
            selected_texts = []
            if listbox_widget.winfo_exists():
                try:
                    selected_indices = listbox_widget.curselection()
                    selected_texts = [listbox_widget.get(i) for i in selected_indices]
                except tk.TclError: pass
            
            if not listbox_widget.winfo_exists(): return

            listbox_widget.delete(0, tk.END)
            new_selection_indices = []
            for idx, item_text in enumerate(new_items_sorted):
                if item_text:
                    listbox_widget.insert(tk.END, item_text)
                    if item_text in selected_texts:
                        new_selection_indices.append(idx)
            
            if listbox_widget.winfo_exists():
                for sel_idx in new_selection_indices:
                    try: listbox_widget.selection_set(sel_idx)
                    except tk.TclError: pass
        
        all_titles = self.distraction_detector.get_all_open_window_titles()
        current_block_list = self.distraction_detector.get_block_list()
        
        candidate_titles_sorted = sorted(list(set(all_titles) - set(current_block_list)))
        if self.all_windows_listbox.winfo_exists():
             update_listbox_preserve_selection(self.all_windows_listbox, candidate_titles_sorted)

        blocked_list_sorted = sorted(current_block_list)
        if self.blocked_listbox.winfo_exists():
            update_listbox_preserve_selection(self.blocked_listbox, blocked_list_sorted)

    def _block_selected(self):
        if not self.all_windows_listbox.winfo_exists(): return
        selected_indices = self.all_windows_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Block Apps", "No application selected from 'All Detected Windows'.")
            return
        for i in selected_indices[::-1]: 
            app_title = self.all_windows_listbox.get(i)
            self.distraction_detector.add_to_block_list(app_title)
        self._update_block_management_ui()

    def _unblock_selected(self):
        if not self.blocked_listbox.winfo_exists(): return
        selected_indices = self.blocked_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Unblock Apps", "No application selected from 'Currently Blocked Applications'.")
            return
        for i in selected_indices[::-1]:
            app_title = self.blocked_listbox.get(i)
            self.distraction_detector.remove_from_block_list(app_title)
        self._update_block_management_ui()

    def _on_closing(self):
        self.running = False
        time.sleep(0.1) 
        if hasattr(self, 'head_pose_monitor') and self.head_pose_monitor:
            self.head_pose_monitor.release_resources()
        self.destroy()

if __name__ == "__main__":
    app = MainDashboard()
    app.mainloop()
